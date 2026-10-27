"""Candidate2 native FlashSAC precision refinement with fresh 4M GPU replay.

Training and evaluation are separate processes. This command never evaluates a
checkpoint while its training allocation is resident. The outer overnight
orchestrator owns GPU exclusivity and the total budget across commands.
"""

from __future__ import annotations

import argparse
import bisect
import copy
import dataclasses
import datetime as dt
import importlib.util
import json
import os
from pathlib import Path
import signal
import struct
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.native_chord_diagnostic import gpu_memory, utc_now, write_json

WORLDS = 2048
REPLAY_CAPACITY = 4_000_000
OPTIMIZER_HORIZON = 683592
MAX_PHASE_SECONDS = 7200


def read_header(path):
    """Read metadata without importing a GPU framework or loading tensor bytes."""
    with Path(path).open("rb") as stream:
        raw = stream.read(8)
        if len(raw) != 8:
            raise ValueError("Truncated safetensors header")
        size = struct.unpack("<Q", raw)[0]
        if not 0 < size <= 32 * 1024 * 1024:
            raise ValueError("Invalid safetensors header length")
        payload = stream.read(size)
    header = json.loads(payload)
    metadata = header.get("__metadata__", {})
    if not isinstance(metadata, dict):
        raise ValueError("Checkpoint metadata must be a mapping")
    if metadata.get("format") != "learner_state" or metadata.get("version") != "1":
        raise ValueError("Expected the pinned native learner_state/version1 envelope")
    return {name.removeprefix("user."): value for name, value in metadata.items()
            if name.startswith("user.")}


def checkpoint_pair(path):
    path = Path(path).resolve()
    if path.name.startswith("policy_"):
        policy, state = path, path.with_name(path.name.replace("policy_", "state_", 1))
    elif path.name.startswith("state_"):
        state, policy = path, path.with_name(path.name.replace("state_", "policy_", 1))
    else:
        raise ValueError("Checkpoint must be a native policy_ or state_ safetensors file")
    metadata = read_header(state)
    actor_metadata = read_header(policy)
    for current, kind in ((metadata, "training_state"), (actor_metadata, "policy")):
        if current.get("algorithm") != "flash_sac" or current.get("checkpoint_kind") != kind:
            raise ValueError(f"Expected a native FlashSAC {kind} checkpoint")
    for name in ("checkpoint_set_id", "environment_steps", "policy_schema_json", "critic_schema_json"):
        if not metadata.get(name) or metadata[name] != actor_metadata.get(name):
            raise ValueError(f"Policy/state checkpoint pair differs in {name}")
    return state, policy, metadata


def transition_budget(start, additional, worlds=WORLDS):
    if start < 0 or start % worlds or additional < worlds:
        raise ValueError("Start must align with worlds and budget must contain one collection")
    rounded = additional // worlds * worlds
    return start + rounded, rounded


def deadline_timestamp(text):
    value = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
    if value.tzinfo is None:
        raise ValueError("Deadline must include an explicit UTC offset")
    return value.timestamp()


def refinement_curriculum(objective, start, end, phase, ramp_transitions=10_000_000):
    weights = {name.removesuffix("_weight"): value for name, value in objective.items()
               if name.endswith("_weight")}
    weights["object_orientation"] = 0.0
    fractions = [1.0] if phase == "consolidate" else [i / 10 for i in range(11)]
    interval = max(WORLDS, (ramp_transitions // 10 // WORLDS) * WORLDS)
    stages = []
    for fraction in fractions:
        stage_weights = {**weights, "object_position": 1.0 + 2.0 * fraction,
                         "force_closure": 5.0 - 4.0 * fraction,
                         "object_orientation": 2.0 * fraction}
        stages.append({"_target_": "flash_chord.configuration.make_curriculum_stage",
                       "voc_scale": 0.0, "objective_weights": stage_weights,
                       "immediate_first_frame_probability": 1.0 if phase == "consolidate" else 0.5})
    thresholds = ([end] if phase == "consolidate" else
                  [start + interval * (i + 1) for i in range(10)] + [max(end, start + 11 * interval)])
    return {"_target_": "flash_chord.configuration.make_fixed_curriculum",
            "thresholds": thresholds, "stages": stages}


def build_config(saved, *, state, output, start, end, phase, checkpoint_interval):
    """Preserve saved network, observation, action, source and simulator contracts."""
    cfg = copy.deepcopy(saved)
    if int(cfg["scene"]["world_count"]) != WORLDS:
        raise ValueError("Native compact warm start requires a 2048-world source checkpoint")
    objective = copy.deepcopy(cfg["env"]["objective"])
    objective.update({"_target_": "training.overnight_refinement.OvernightObjectiveConfig",
                      "object_orientation_weight": 0.0 if phase == "refine" else 2.0,
                      "object_orientation_std": 0.6, "contact_blend_frames": 10,
                      "hand_position_tolerance": 0.03, "relative_hand_position_std": 0.15,
                      "hand_orientation_std": 0.8, "hand_orientation_factor": 0.25,
                      "object_position_fine_fraction": 1.0 / 3.0,
                      "object_position_fine_std": 0.05,
                      "use_object_relative_hands": True})
    objective["object_position_weight"] = 1.0 if phase == "refine" else 3.0
    objective["force_closure_weight"] = 5.0 if phase == "refine" else 1.0
    termination = copy.deepcopy(cfg["env"]["termination"])
    termination.update({"_target_": "training.overnight_refinement.OvernightTerminationConfig",
                        "palm_position_threshold": 0.25, "palm_orientation_threshold": 2.2,
                        "object_position_threshold": 0.20, "palm_dwell_steps": 10,
                        "object_dwell_steps": 5, "palm_gross_position": 0.60,
                        "palm_gross_orientation": 2.8, "object_gross_position": 0.40,
                        "object_gross_orientation": 2.8})
    reset = copy.deepcopy(cfg["env"]["reset"])
    reset.update({"reset_voc_scale": 0.0 if phase == "consolidate" else 1.0,
                  "voc_decay_steps": 0 if phase == "consolidate" else 10,
                  "always_reset_to_first_frame": phase == "consolidate",
                  "reset_to_first_frame_probability": 0.0,
                  "immediate_first_frame_probability": 1.0 if phase == "consolidate" else 0.5})
    if phase == "consolidate":
        reset.update(reset_freeze_steps=0, shoulder_spread=0.0)
        termination["reset_freeze_steps"] = 0
    else:
        reset["reset_freeze_steps"] = 50
        termination["reset_freeze_steps"] = 50
    for name, value in (("objective", objective), ("termination", termination), ("reset", reset)):
        cfg[name] = copy.deepcopy(value)
        cfg["env"][name] = value
    cfg["env"]["voc_scale"] = 0.0
    cfg["voc_scale_override"] = 0.0
    cfg["scene"]["world_count"] = WORLDS
    prior = cfg.get("overnight", {})
    ramp_start = int(prior.get("refinement_start_steps", start))
    if ramp_start > start:
        raise ValueError("Refinement ramp origin is after the restored checkpoint")
    if prior.get("phase") == "consolidate" and phase == "refine":
        raise ValueError("Cannot restart refinement after exact-start consolidation")
    curriculum = refinement_curriculum(objective, ramp_start, end, phase)
    active_index = min(bisect.bisect_right(curriculum["thresholds"], start), len(curriculum["stages"]) - 1)
    for name, weight in curriculum["stages"][active_index]["objective_weights"].items():
        cfg["env"]["objective"][name + "_weight"] = weight
        cfg["objective"][name + "_weight"] = weight
    cfg["overnight"] = {"phase": phase, "refinement_start_steps": ramp_start,
                        "method": "candidate2_position_precision_mixture",
                        "fresh_replay": True, "optimizer_horizon": OPTIMIZER_HORIZON,
                        "replay_capacity": REPLAY_CAPACITY, "replay_backend": "gpu",
                        "recovery_reset_assistance": phase == "refine"}
    cfg["curriculum"] = curriculum
    training = cfg["training"]
    training.update(world_count=WORLDS, total_environment_steps=end, resume=True,
                    checkpoint_path=str(state), curriculum=copy.deepcopy(curriculum))
    training["replay"].update(capacity=REPLAY_CAPACITY, batch_size=WORLDS,
                              clear_on_stage_change=False)
    training.setdefault("exploration", {})["warmup_action"] = "policy"
    training["optimization"]["learning_rate_schedule_updates"] = OPTIMIZER_HORIZON
    training["checkpoint"].update(save_interval_environment_steps=checkpoint_interval,
                                  save_replay=False, load_replay_path=None,
                                  load_optimizer=True, load_reward_normalizer=True)
    cfg["logging"].update(mode="offline", log_interval=25,
                          upload_checkpoints=False, upload_training_state=False)
    cfg["output_dir"] = str(output / "checkpoints")
    return cfg


def remap_job(cfg, job, output):
    """Rebase identical portable input; refuse a hidden trajectory change on resume."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from pxr import Usd

    job = Path(job).resolve()
    previous = Path(cfg["task"]["parquet"])
    if not previous.is_file():
        raise FileNotFoundError("Saved reference is missing; automatic unverified reference remap is refused")
    row = pq.read_table(job / "motion.parquet").to_pylist()[0]
    old = pq.read_table(previous).to_pylist()[0]
    for field in ("fps", "robot_joint_names", "robot_joint_positions", "robot_root_position",
                  "robot_root_wxyz", "object_body_position", "object_body_wxyz"):
        if row.get(field) != old.get(field):
            raise ValueError(f"Warm-start source trajectory differs in {field}")
    sequence = row["sequence_id"]
    if not isinstance(sequence, str) or "/" in sequence or "\\" in sequence or sequence in (".", ".."):
        raise ValueError("Invalid sequence path component")
    folder = output / "input/retargeted" / f"sequence_id={sequence}" / "robot_name=g1"
    folder.mkdir(parents=True)
    row["object_mesh_paths"] = [str(job / "assets/object.obj")]
    row["object_urdf_paths"] = [str(job / "assets/object.urdf")]
    motion = folder / "motion.parquet"
    pq.write_table(pa.Table.from_pylist([row]), motion, compression="zstd")
    support = job / "assets/support.usda"
    stage = Usd.Stage.Open(str(support))
    if stage is None:
        raise ValueError("Cannot read prepared support surface")
    has_support = any(p.GetTypeName() in ("Cube", "Cylinder") for p in stage.Traverse())
    if has_support:
        support_dir = output / "input/reconstructed_stage"
        support_dir.mkdir()
        (support_dir / f"{sequence}_support.usda").symlink_to(support)
    cfg["task"]["parquet"] = str(motion)
    cfg["scene"]["support"] = has_support


class ControlledStop(Exception):
    pass


def signal_child(pid, signum, *, shared_group):
    """An outer guard owns the whole group; never detach its GPU worker."""
    if shared_group:
        os.kill(pid, signum)
    else:
        os.killpg(pid, signum)


def checkpoint_stage_metadata(curriculum, environment_steps):
    index = min(bisect.bisect_right(curriculum["thresholds"], environment_steps), len(curriculum["stages"]) - 1)
    stage = curriculum["stages"][index]
    metadata = {"curriculum_stage": index, "voc_scale": stage["voc_scale"]}
    for name in ("reset_to_first_frame_probability", "immediate_first_frame_probability"):
        if stage.get(name) is not None:
            metadata[name] = stage[name]
    return metadata


def run_child(args):
    from training import RUNTIME_COMMIT
    from training.bundles import sha256

    output, runtime = Path(args.output).resolve(), Path(args.runtime).resolve()
    commit = subprocess.check_output(["git", "-C", str(runtime), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(runtime), "status", "--porcelain", "--untracked-files=no"], text=True).strip()
    if commit != RUNTIME_COMMIT or dirty:
        raise ValueError(f"Expected clean native runtime at {RUNTIME_COMMIT}")
    source = runtime / "robotic_grounding/flash_chord/src"
    sys.path.insert(0, str(source))
    state, policy, metadata = checkpoint_pair(args.checkpoint)
    saved = json.loads(metadata["resolved_config_json"])
    start = int(metadata["environment_steps"])
    end, additional = transition_budget(start, args.additional_transitions)
    cfg = build_config(saved, state=state, output=output, start=start, end=end,
                       phase=args.phase, checkpoint_interval=args.checkpoint_interval)
    if args.job:
        remap_job(cfg, args.job, output)
    if not Path(cfg["task"]["parquet"]).is_file():
        raise FileNotFoundError(f"Saved task parquet is unavailable: {cfg['task']['parquet']}")
    write_json(output / "resolved_config.json", cfg)
    write_json(output / "provenance.json", {
        "runtime_commit": commit, "source_state": str(state), "source_policy": str(policy),
        "source_state_sha256": sha256(state), "source_policy_sha256": sha256(policy),
        "source_environment_steps": start, "additional_transition_cap": additional,
        "absolute_transition_cap": end, "optimizer_horizon": OPTIMIZER_HORIZON,
        "replay_capacity": REPLAY_CAPACITY, "replay_is_fresh": True,
        "source_replay_capacity": saved["training"]["replay"].get("capacity"),
        "replay_backend": "gpu", "memory_limit_mib": args.memory_limit_mib,
        "jax_memory_fraction": os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION"),
        "jax_preallocate": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
        "replay_warmup_action": cfg["training"]["exploration"]["warmup_action"],
        "source_replay_warmup_action": saved["training"].get("exploration", {}).get("warmup_action", "uniform"),
        "phase": args.phase, "created_utc": utc_now(), "harness_sha256": sha256(__file__),
        "training_voc_scale": 0.0, "recovery_reset_voc_scale": cfg["env"]["reset"]["reset_voc_scale"],
        "reset_freeze_steps": cfg["env"]["reset"]["reset_freeze_steps"],
    })
    import jax
    import jax.numpy as jnp
    import numpy as np
    from omegaconf import OmegaConf
    from flash_chord.training.flash_sac import runner
    from flash_chord.training.flash_sac.checkpoint import actor_inference_state, load_actor_checkpoint
    from flash_chord.training.flash_sac.network import Actor, deterministic_action
    import training.overnight_refinement  # noqa: F401 - Hydra checkpoint targets

    entry = source.parent / "scripts/train_flash_sac.py"
    spec = importlib.util.spec_from_file_location("overnight_native_entry", entry)
    native = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = native
    spec.loader.exec_module(native)
    stop = {"reason": None}
    context = {}
    child_started = time.monotonic()

    def memory_snapshot(stage, learner=None, **details):
        """Best-effort telemetry after native device setup; never alter allocators."""
        record = {"time_utc": utc_now(), "elapsed_s": time.monotonic() - child_started,
                  "stage": stage, **details}
        try:
            record["gpu"] = gpu_memory()
        except Exception as exc:
            record["gpu_error"] = f"{type(exc).__name__}: {exc}"
        try:
            device = jax.devices()[0]  # Native setup has initialized JAX before these hooks.
            stats = device.memory_stats()
            record["jax_device"] = str(device)
            record["jax_memory_stats"] = None if stats is None else {
                key: value.item() if isinstance(value, np.generic) else value
                for key, value in stats.items()
            }
        except Exception as exc:
            record["jax_error"] = f"{type(exc).__name__}: {exc}"
        try:
            import warp as wp
            device = context["env"].env.device
            pool = {"device": str(device)}
            for name in ("get_mempool_used_mem_current", "get_mempool_used_mem_high",
                         "get_mempool_release_threshold"):
                try:
                    pool[name] = int(getattr(wp, name)(device))
                except Exception as exc:
                    pool[name + "_error"] = f"{type(exc).__name__}: {exc}"
            record["warp_mempool"] = pool
        except Exception as exc:
            record["warp_error"] = f"{type(exc).__name__}: {exc}"
        if learner is not None:
            try:
                # Metadata only: do not materialize replay on the host or retain
                # learner buffers beyond this call. This is logical array size.
                arrays = jax.tree.leaves(learner.replay)
                record["replay_array_count"] = len(arrays)
                record["replay_logical_array_bytes"] = sum(int(value.nbytes) for value in arrays)
            except Exception as exc:
                record["replay_error"] = f"{type(exc).__name__}: {exc}"
        try:
            with (output / "memory_stages.jsonl").open("a") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
        except Exception as exc:
            print(f"Memory telemetry unavailable at {stage}: {type(exc).__name__}: {exc}", flush=True)

    def request_stop(signum, _frame):
        stop["reason"] = f"signal_{signal.Signals(signum).name}"

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    class LocalMetrics:
        def __init__(self, config, **kwargs):
            self.stream = (output / "metrics.jsonl").open("w", buffering=1)
            write_json(output / "training_config.json", dataclasses.asdict(config))
            write_json(output / "environment_config.json", kwargs["environment_config"])

        def log(self, metrics, step):
            record = {"time_utc": utc_now(), "transitions": step,
                      "additional_transitions": step - start,
                      "training_elapsed_s": time.monotonic() - child_started, **metrics}
            self.stream.write(json.dumps(record, allow_nan=False) + "\n")
            write_json(output / "progress.json", record)
            memory_snapshot("metrics_log", environment_steps=step)

        def log_checkpoint(self, paths, *, environment_steps, metadata):
            record = {"time_utc": utc_now(), "environment_steps": environment_steps,
                      "additional_transitions": environment_steps - start,
                      "policy": str(paths.policy), "state": str(paths.training_state),
                      "phase": args.phase, **metadata}
            write_json(output / "latest_checkpoint.json", record)
            with (output / "checkpoints.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")

        def close(self):
            self.stream.close()

    native.WandbLogger = LocalMetrics
    original_run = native.run_training

    def recorded_run(env, config, **kwargs):
        context.update(env=env, config=config, kwargs=kwargs)
        memory_snapshot("scene_ready")
        return original_run(env, config, **kwargs)

    native.run_training = recorded_run
    original_restore = runner.restore_learner_checkpoint

    def checked_restore(path, learner, **kwargs):
        memory_snapshot("restore_before", learner)
        restored, info = original_restore(path, learner, **kwargs)
        memory_snapshot("restored_before_parity", restored)
        actor, _ = load_actor_checkpoint(policy, actor_inference_state(restored))
        leaves_a, structure_a = jax.tree.flatten(actor)
        leaves_b, structure_b = jax.tree.flatten(actor_inference_state(restored))
        equal = structure_a == structure_b and all(
            np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(leaves_a, leaves_b, strict=True))
        config = context["config"]
        model = Actor(action_dim=context["env"].action_dim, config=config.network,
                      dtype=jnp.dtype(config.compute_dtype) if config.mixed_precision else jnp.float32)
        observations = restored.observation[:16]
        def infer(value):
            distribution = model.apply({"params": value.params, "batch_stats": value.batch_stats},
                                       observations, training=False)
            return deterministic_action(distribution, action_transform=config.network.action_transform,
                                        action_scale=config.network.action_scale)
        difference = float(np.max(np.abs(np.asarray(infer(actor)) - np.asarray(infer(restored.actor)))))
        replay_size = int(jax.device_get(restored.replay.size))
        parity = {"parameter_exact_match": equal, "deterministic_action_max_abs_error": difference,
                  "observation_count": len(observations), "fresh_replay_size": replay_size,
                  "environment_steps": int(jax.device_get(restored.environment_steps)),
                  "global_update_step": int(jax.device_get(restored.global_update_step)),
                  "actor_schedule_step": int(jax.device_get(restored.actor.schedule_step)),
                  "critic_schedule_step": int(jax.device_get(restored.critic.schedule_step))}
        write_json(output / "warmstart_parity.json", parity)
        if not equal or difference != 0 or replay_size != 0:
            raise ValueError(f"Warm-start parity failed: {parity}")
        memory_snapshot("restored_after_parity", restored)
        return restored, info

    runner.restore_learner_checkpoint = checked_restore

    def save_and_stop(learner):
        jax.block_until_ready(learner.global_update_step)
        steps = int(jax.device_get(learner.environment_steps))
        info = dict(context["kwargs"]["checkpoint_metadata"])
        stage_info = checkpoint_stage_metadata(cfg["curriculum"], steps)
        info.update(environment_steps=steps, global_update_step=int(jax.device_get(learner.global_update_step)),
                    voc_scale=0.0, replay_included=False, overnight_stop_reason=stop["reason"])
        info.update(stage_info)
        paths = runner.save_learner_checkpoint(output / "checkpoints", learner, metadata=info, save_replay=False)
        context["kwargs"]["logger"].log_checkpoint(paths, environment_steps=steps,
                                                   metadata={**stage_info, "stop_reason": stop["reason"]})
        write_json(output / "graceful_stop.json", {"reason": stop["reason"], "environment_steps": steps,
                                                   "policy": str(paths.policy), "saved_utc": utc_now()})
        raise ControlledStop(stop["reason"])

    original_compile = runner._compile_executables

    def bounded_compile(executables, learner, **kwargs):
        if stop["reason"]:
            save_and_stop(learner)
        memory_snapshot("compile_before", learner)
        compiled = original_compile(executables, learner, **kwargs)
        memory_snapshot("compile_after", learner)
        if stop["reason"]:
            save_and_stop(learner)
        first_consume = True
        def consume(*values):
            nonlocal first_consume
            if stop["reason"]:
                save_and_stop(values[0])
            if first_consume:
                memory_snapshot("first_consume_before", values[0])
            result = compiled.consume_transition(*values)
            if first_consume:
                memory_snapshot("first_consume_after", result[0])
                first_consume = False
            if stop["reason"]:
                save_and_stop(result[0])
            return result
        return dataclasses.replace(compiled, consume_transition=consume)

    runner._compile_executables = bounded_compile
    try:
        native.main.__wrapped__(OmegaConf.create(cfg))
    except ControlledStop as error:
        print(f"Graceful bounded stop: {error}", flush=True)


def supervise(args):
    output = Path(args.output).resolve()
    state, policy, metadata = checkpoint_pair(args.checkpoint)
    start_steps = int(metadata["environment_steps"])
    end_steps, additional = transition_budget(start_steps, args.additional_transitions)
    remaining = min(float(args.max_seconds), deadline_timestamp(args.deadline_utc) - time.time())
    if remaining <= args.stop_grace_seconds + 5:
        raise ValueError("Insufficient time before deadline to start safely")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    status = {"state": "starting", "started_utc": utc_now(), "source_checkpoint": str(state),
              "source_policy": str(policy), "initial_steps": start_steps, "absolute_transition_cap": end_steps,
              "additional_transition_cap": additional, "wall_seconds_cap": remaining,
              "deadline_utc": args.deadline_utc, "phase": args.phase, "arguments": vars(args)}
    write_json(output / "status.json", status)
    environment = os.environ.copy()
    environment.update(XLA_PYTHON_CLIENT_PREALLOCATE="false", PYTHONUNBUFFERED="1",
                       WANDB_MODE="disabled", HYDRA_FULL_ERROR="1")
    stop_reason = None
    stop_sent = None
    shared_group = os.environ.get("V2D_OVERNIGHT_GUARD") == "1"
    status["inherits_outer_process_group"] = shared_group
    requested = {"signal": None}
    def ask_stop(signum, _frame):
        requested["signal"] = signal.Signals(signum).name
    signal.signal(signal.SIGTERM, ask_stop)
    signal.signal(signal.SIGINT, ask_stop)
    with (output / "train.log").open("w") as log, (output / "gpu_memory.jsonl").open("w") as memory_log:
        proc = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()), *sys.argv[1:], "--child"],
                                stdout=log, stderr=subprocess.STDOUT, env=environment,
                                start_new_session=not shared_group)
        status.update(state="running", pid=proc.pid)
        killed = threading.Event()
        def hard_stop():
            if proc.poll() is None:
                killed.set()
                try:
                    signal_child(proc.pid, signal.SIGKILL, shared_group=shared_group)
                except ProcessLookupError:
                    pass
        watchdog = threading.Timer(max(0.01, remaining - (time.monotonic() - started)), hard_stop)
        watchdog.daemon = True
        watchdog.start()
        try:
            while proc.poll() is None:
                elapsed = time.monotonic() - started
                sample = {"elapsed_s": elapsed, **gpu_memory()}
                memory_log.write(json.dumps(sample) + "\n")
                memory_log.flush()
                status["elapsed_s"] = elapsed
                status["peak_gpu_used_mib"] = max(status.get("peak_gpu_used_mib", 0), sample["used_mib"])
                if stop_reason is None:
                    if requested["signal"]:
                        stop_reason = f"external_{requested['signal']}"
                    elif sample["used_mib"] > args.memory_limit_mib:
                        stop_reason = "gpu_memory_guard"
                    elif elapsed >= remaining - args.stop_grace_seconds:
                        stop_reason = "wall_time_budget"
                    if stop_reason:
                        stop_sent = time.monotonic()
                        status.update(state="stopping", stop_reason=stop_reason)
                        signal_child(proc.pid, signal.SIGTERM, shared_group=shared_group)
                if elapsed >= remaining or (stop_sent is not None and time.monotonic() - stop_sent >= args.stop_grace_seconds):
                    signal_child(proc.pid, signal.SIGKILL, shared_group=shared_group)
                    status["forced_kill"] = True
                    break
                write_json(output / "status.json", status)
                time.sleep(min(1.0, max(0.05, remaining - elapsed)))
        except BaseException:
            if proc.poll() is None:
                signal_child(proc.pid, signal.SIGKILL, shared_group=shared_group)
            raise
        finally:
            proc.wait()
            watchdog.cancel()
            graceful = (output / "graceful_stop.json").exists()
            expected_stop = stop_reason in ("wall_time_budget", "external_SIGTERM", "external_SIGINT")
            success = proc.returncode == 0 and (stop_reason is None or (expected_stop and graceful))
            status.update(state="complete" if success else "failed", exit_code=proc.returncode,
                          stop_reason=stop_reason, graceful_checkpoint_saved=graceful,
                          elapsed_s=time.monotonic() - started, finished_utc=utc_now(),
                          forced_kill=status.get("forced_kill", False) or killed.is_set())
            write_json(output / "status.json", status)
    print(json.dumps(status, indent=2), flush=True)
    return 0 if success else (proc.returncode or 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train",))
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--job")
    parser.add_argument("--phase", choices=("refine", "consolidate"), default="refine")
    parser.add_argument("--additional-transitions", type=int, default=60_000_000)
    parser.add_argument("--max-seconds", type=int, default=1800)
    parser.add_argument("--deadline-utc", required=True)
    parser.add_argument("--checkpoint-interval", type=int, default=10_000_000)
    parser.add_argument("--memory-limit-mib", type=int, default=23_000)
    parser.add_argument("--stop-grace-seconds", type=int, default=30)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 0 < args.max_seconds <= MAX_PHASE_SECONDS:
        parser.error("Each command must have a positive wall budget of at most 7200 seconds")
    if not 1 <= args.stop_grace_seconds < args.max_seconds or args.checkpoint_interval <= 0:
        parser.error("Invalid checkpoint cadence or graceful-stop reserve")
    if args.additional_transitions < WORLDS:
        parser.error("Additional budget must contain at least one collection")
    deadline_timestamp(args.deadline_utc)
    if args.child:
        run_child(args)
    else:
        raise SystemExit(supervise(args))


if __name__ == "__main__":
    main()

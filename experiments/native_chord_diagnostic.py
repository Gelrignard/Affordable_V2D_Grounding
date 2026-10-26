"""Bounded native FlashCHORD training with local metrics and total-GPU monitoring.

Uses the pinned upstream entry point, learner, environment and recipe. Only the
metric sink is replaced (JSONL instead of W&B). Inputs are our prepared motions,
so this is a downstream baseline, not a reproduction of CHORD retargeting.
"""

import argparse
import dataclasses
import datetime
import importlib.metadata
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def utc_now():
    return datetime.datetime.now(datetime.UTC).isoformat()


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False))
    temporary.replace(path)


def gpu_memory():
    line = subprocess.check_output(
        [
            "nvidia-smi",
            "--id=0",
            "--query-gpu=memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=10,
    ).strip()
    used, total, utilization = (int(value.strip()) for value in line.split(","))
    return {"used_mib": used, "total_mib": total, "utilization_percent": utilization}


def validate_replay_only_change(previous, current):
    """Require the fresh comparison to change only rolling replay capacity."""
    old, new = (json.loads(json.dumps(c)) for c in (previous, current))
    capacities = [c["replay"].pop("capacity") for c in (old, new)]
    if old != new:
        raise ValueError("Comparison changes learner parameters beyond replay capacity")
    return {
        "previous_capacity": capacities[0],
        "current_capacity": capacities[1],
        "all_other_learner_parameters_identical": True,
    }


def run_child(args):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from pxr import Usd

    from training import RUNTIME_COMMIT
    from training.bundles import sha256

    # Avoid importing our PyTorch adapter into the JAX/native baseline process.
    runtime = Path(args.runtime).resolve()
    commit = subprocess.check_output(
        ["git", "-C", str(runtime), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(runtime), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    ).strip()
    if commit != RUNTIME_COMMIT or dirty:
        raise ValueError(f"Native runtime must be clean at {RUNTIME_COMMIT}")
    source = runtime / "robotic_grounding/flash_chord/src"
    sys.path.insert(0, str(source))
    urdf_sha = sha256(source / "flash_chord/assets/urdfs/g1/main_with_hand.urdf")
    if urdf_sha not in {
        "8c7f768dc8da6c969d8a4b4efb3dcbe3d9d2b05fca5104df12205a69879d19e9",
        "ab13c2fb8f51ee92f33e015e264b9826b92bc4bae0f757aa2947f6b5e66972a6",
    }:
        raise ValueError(
            "Unrecognized G1 URDF; only verified LF/CRLF variants are allowed"
        )
    job = Path(args.job).resolve()
    out = Path(args.output).resolve()
    supervisor_started = datetime.datetime.fromisoformat(
        json.loads((out / "status.json").read_text())["started_utc"]
    )
    started = (
        time.monotonic()
        - (datetime.datetime.now(datetime.UTC) - supervisor_started).total_seconds()
    )
    row = pq.read_table(job / "motion.parquet").to_pylist()[0]
    sequence = str(row["sequence_id"])
    if "/" in sequence or "\\" in sequence or sequence in (".", ".."):
        raise ValueError("sequence id must be a path component")
    # Match upstream's documented support-asset discovery convention.
    motion_dir = out / "input/retargeted" / f"sequence_id={sequence}" / "robot_name=g1"
    motion_dir.mkdir(parents=True, exist_ok=True)
    motion = motion_dir / "motion.parquet"
    row["object_mesh_paths"] = [str(job / "assets/object.obj")]
    row["object_urdf_paths"] = [str(job / "assets/object.urdf")]
    pq.write_table(pa.Table.from_pylist([row]), motion, compression="zstd")
    support = job / "assets/support.usda"
    stage = Usd.Stage.Open(str(support))
    if stage is None:
        raise ValueError(f"Cannot open support asset: {support}")
    has_support = any(p.GetTypeName() in ("Cube", "Cylinder") for p in stage.Traverse())
    if has_support:
        support_dir = out / "input/reconstructed_stage"
        support_dir.mkdir(parents=True, exist_ok=True)
        (support_dir / f"{sequence}_support.usda").symlink_to(support)

    # Keep the full recipe's optimizer-clock horizon even for truncated tests.
    # Fixed upstream curriculum thresholds likewise remain uncompressed.
    full_schedule_updates = (349999104 // args.worlds) * 4
    overrides = [
        "experiment=g1_recon_body_flash_sac",
        f"embodiment.config.asset_sha256={urdf_sha}",
        f"task.parquet={json.dumps(str(motion))}",
        f"scene.support={str(has_support).lower()}",
        f"scene.world_count={args.worlds}",
        f"training.replay.batch_size={args.batch_size}",
        f"training.replay.capacity={args.replay_capacity}",
        f"training.total_environment_steps={args.transitions}",
        f"training.optimization.learning_rate_schedule_updates={full_schedule_updates}",
        f"training.checkpoint.save_interval_environment_steps={args.checkpoint_interval}",
        f"output_dir={out / 'checkpoints'}",
        f"hydra.run.dir={out / 'hydra'}",
        "logging.mode=offline",
        "logging.log_interval=50",
        "logging.upload_checkpoints=false",
        "logging.upload_training_state=false",
    ]
    extension = None
    if args.method != "native":
        from training.native_experiment import NativeExperiment

        extension = NativeExperiment(args, out)
        overrides += [
            "objective._target_=training.native_guidance.DynamicsObjectiveConfig",
            f"+objective.dynamics_weight={args.dynamics_weight}",
        ]
        overrides += [
            f"+curriculum.stages.{index}.objective_weights.dynamics_wrench={args.dynamics_weight}"
            for index in range(9)
        ]
    entry = source.parent / "scripts/train_flash_sac.py"
    spec = importlib.util.spec_from_file_location("native_flash_chord_entry", entry)
    native = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = native
    spec.loader.exec_module(native)
    if extension is not None:
        from training.native_guidance import GuidedRLEnv

        native.RLEnv = GuidedRLEnv

    class LocalMetrics:
        def __init__(self, config, **kwargs):
            if args.comparison_config:
                previous_config = Path(args.comparison_config)
                verification = validate_replay_only_change(
                    json.loads(previous_config.read_text()), dataclasses.asdict(config)
                )
                previous_provenance = json.loads(
                    (previous_config.parent / "provenance.json").read_text()
                )
                if previous_provenance["job_motion_sha256"] != sha256(
                    job / "motion.parquet"
                ):
                    raise ValueError("Comparison input trajectory differs")
                write_json(out / "comparison_verification.json", verification)
            self.stream = (out / "metrics.jsonl").open("w", buffering=1)
            write_json(out / "training_config.json", dataclasses.asdict(config))
            write_json(out / "environment_config.json", kwargs["environment_config"])

        def log(self, metrics, step):
            if extension is not None:
                extension.log(metrics, step)
            record = {
                "time_utc": utc_now(),
                "transitions": step,
                "training_elapsed_s": time.monotonic() - started,
                **metrics,
            }
            self.stream.write(json.dumps(record, allow_nan=False) + "\n")
            write_json(
                out / "progress.json",
                {
                    "state": "training",
                    "transitions": step,
                    "training_elapsed_s": record["training_elapsed_s"],
                    "assistance_scale": metrics.get("curriculum/voc_scale"),
                    "curriculum_stage": metrics.get("curriculum/stage_index"),
                    "throughput": metrics.get("throughput/env_steps_per_second"),
                    "replay_size": metrics.get("replay/size_transitions"),
                    "actor_skipped_fraction": metrics.get("sac/actor_skipped_fraction"),
                    "critic_skipped_fraction": metrics.get(
                        "sac/critic_skipped_fraction"
                    ),
                },
            )

        def close(self):
            self.stream.close()

        def log_checkpoint(self, paths, **kwargs):
            if extension is not None:
                extension.log_checkpoint(paths, **kwargs)
            else:
                write_json(
                    Path(paths.policy).with_suffix(".experiment.json"),
                    {
                        "environment_steps": kwargs["environment_steps"],
                        "training_elapsed_s": time.monotonic() - started,
                        "method": "native_flash_sac",
                        **kwargs.get("metadata", {}),
                    },
                )

    native.WandbLogger = LocalMetrics
    original_run = native.run_training

    def recorded_run(env, config, **kwargs):
        from flash_chord.training.flash_sac.replay import ReplaySpec, replay_memory

        sessions = (env.env.action.encoder_session, env.env.action.decoder_session)
        if any(session.provider != "CUDAExecutionProvider" for session in sessions):
            raise RuntimeError(
                "Native SONIC must use CUDA for both encoder and decoder"
            )
        replay_spec = ReplaySpec.from_config(
            config.replay,
            discount=config.discount,
            world_count=env.world_count,
            observation_dim=env.observation_dim,
            action_dim=env.action_dim,
            objective_term_count=len(env.objective_term_names),
            critic_context_dim=env.critic_context_dim,
        )
        layout = dataclasses.asdict(replay_memory(replay_spec))
        used = gpu_memory()
        write_json(
            out / "replay_memory.json",
            {
                "spec": dataclasses.asdict(replay_spec),
                "logical_array_bytes": layout,
                "gpu_before_learner": used,
                "note": "Logical replay bytes exclude network/compiler/workspace allocations.",
            },
        )
        ring_mib = layout["ring_bytes"] / 2**20
        # Observed native JAX allocation peaks exceeded logical replay bytes
        # substantially. Budget allocator/compiler temporaries conservatively.
        if (
            used["used_mib"]
            + args.preflight_ring_multiplier * ring_mib
            + args.preflight_workspace_mib
            > args.memory_limit_mib
        ):
            raise MemoryError(
                "Replay preflight exceeds the configured ring/workspace allowance"
            )
        if extension is not None:
            config = extension.bind(env, config)
        return original_run(env, config, **kwargs)

    native.run_training = recorded_run
    write_json(
        out / "provenance.json",
        {
            "created_utc": utc_now(),
            "runtime_commit": subprocess.check_output(
                ["git", "-C", str(Path(args.runtime).resolve()), "rev-parse", "HEAD"],
                text=True,
            ).strip(),
            "upstream_entry_sha256": sha256(entry),
            "job": str(job),
            "job_motion_sha256": sha256(job / "motion.parquet"),
            "materialized_motion_sha256": sha256(motion),
            "overrides": overrides,
            "learner": "native_flash_chord_flash_sac",
            "logger_only_replacement": extension is None,
            "method": args.method,
            "dynamics_weight": args.dynamics_weight,
            "harness_sha256": sha256(__file__),
            "extension_source_sha256": {
                name: sha256(Path(__file__).resolve().parents[1] / "training" / name)
                for name in (
                    "adaptive_chord.py",
                    "dynamics_targets.py",
                    "native_guidance.py",
                    "native_experiment.py",
                    "leaderboard.py",
                    "leaderboard_metrics.py",
                )
            }
            if extension is not None
            else {},
            "versions": {
                package: importlib.metadata.version(package)
                for package in (
                    "jax",
                    "jaxlib",
                    "flax",
                    "optax",
                    "onnxruntime-gpu",
                    "warp-lang",
                )
            },
            "world_model": False,
            "input_is_our_prepared_motion": True,
            "full_recipe_replay_capacity": 10000000,
            "full_recipe_transitions": 349999104,
            "budget_note": (
                "Full native transition budget; reduced replay is a learning-parameter change."
                if args.transitions == 349999104
                else "Truncated diagnostic; does not establish convergence."
            ),
        },
    )
    # The entry is imported to attach a local metric sink, so give Hydra its
    # absolute config directory instead of module-relative discovery.
    sys.argv = [
        str(entry),
        "--config-path",
        str(source / "flash_chord/configs"),
        *overrides,
    ]
    if extension is None:
        native.main()
    else:
        from training.native_experiment import CandidateStalled

        try:
            native.main()
        except CandidateStalled as error:
            print(f"Controlled candidate stop: {error}", flush=True)


def supervise(args):
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    env = os.environ.copy()
    env.update(
        {
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "PYTHONUNBUFFERED": "1",
            "WANDB_MODE": "disabled",
            "HYDRA_FULL_ERROR": "1",
        }
    )
    status = {
        "state": "running",
        "started_utc": utc_now(),
        "arguments": vars(args),
        "peak_gpu_used_mib": 0,
        "memory_poll_seconds": args.memory_poll_seconds,
        "memory_limit_note": "Sampled guard plus replay preflight; not an OOM guarantee.",
    }
    write_json(out / "status.json", status)
    stop_reason = None
    stop_time = None
    with (
        (out / "train.log").open("w") as log,
        (out / "gpu_memory.jsonl").open("w") as mem,
    ):
        proc = subprocess.Popen(
            [
                sys.executable,
                "-u",
                str(Path(__file__).resolve()),
                *sys.argv[1:],
                "--child",
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        status["pid"] = proc.pid
        try:
            while proc.poll() is None:
                elapsed = time.monotonic() - start
                sample = {"elapsed_s": elapsed, **gpu_memory()}
                mem.write(json.dumps(sample) + "\n")
                mem.flush()
                status["elapsed_s"] = elapsed
                status["peak_gpu_used_mib"] = max(
                    status["peak_gpu_used_mib"], sample["used_mib"]
                )
                if stop_reason is None:
                    if sample["used_mib"] > args.memory_limit_mib:
                        stop_reason = "memory_guard"
                    elif elapsed > args.max_seconds:
                        stop_reason = "time_budget"
                    if stop_reason:
                        stop_time = time.monotonic()
                        os.killpg(proc.pid, signal.SIGINT)
                elif time.monotonic() - stop_time > 30:
                    os.killpg(proc.pid, signal.SIGKILL)
                write_json(out / "status.json", status)
                time.sleep(args.memory_poll_seconds)
        except BaseException as error:  # noqa: BLE001 - preserve status and stop child on interruption
            stop_reason = f"supervisor_exception: {type(error).__name__}: {error}"
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
        status.update(
            {
                "state": "complete"
                if proc.returncode == 0 and stop_reason is None
                else "failed",
                "exit_code": proc.returncode,
                "stop_reason": stop_reason,
                "finished_utc": utc_now(),
                "elapsed_s": time.monotonic() - start,
                "checkpoints": [
                    str(p) for p in sorted((out / "checkpoints").glob("*.safetensors"))
                ],
            }
        )
        write_json(out / "status.json", status)
    print(json.dumps(status, indent=2), flush=True)
    return proc.returncode or (1 if stop_reason else 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--worlds", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--replay-capacity", type=int, default=1000000)
    parser.add_argument(
        "--comparison-config",
        help="Require identical learner config except replay capacity",
    )
    parser.add_argument("--transitions", type=int, default=2000896)
    parser.add_argument("--checkpoint-interval", type=int, default=2000896)
    parser.add_argument("--memory-limit-mib", type=int, default=21504)
    parser.add_argument("--max-seconds", type=int, default=1800)
    parser.add_argument("--preflight-ring-multiplier", type=float, default=2.5)
    parser.add_argument("--preflight-workspace-mib", type=int, default=4096)
    parser.add_argument("--memory-poll-seconds", type=float, default=1.0)
    parser.add_argument(
        "--method", choices=("native", "adaptive", "adaptive_wrench"), default="native"
    )
    parser.add_argument("--dynamics-weight", type=float, default=0.0)
    parser.add_argument("--evaluation-worlds", type=int, default=16)
    parser.add_argument("--stall-transitions", type=int, default=30001152)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.transitions % args.worlds:
        parser.error("transition budget must be divisible by world count")
    if args.preflight_ring_multiplier < 1 or args.preflight_workspace_mib < 0:
        parser.error("Replay preflight must include at least the logical ring bytes")
    if args.memory_poll_seconds <= 0:
        parser.error("Memory poll interval must be positive")
    if args.method == "adaptive" and args.dynamics_weight != 0:
        parser.error("adaptive control must have zero dynamics weight")
    if args.method == "adaptive_wrench" and args.dynamics_weight <= 0:
        parser.error("adaptive_wrench requires a positive dynamics weight")
    if args.child:
        run_child(args)
    else:
        raise SystemExit(supervise(args))


if __name__ == "__main__":
    main()

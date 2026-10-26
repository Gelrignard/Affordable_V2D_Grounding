"""Bounded pinned CHORD PPO, optionally with a DexMachina-inspired curriculum.

Upstream optimizer, rollout, losses, SONIC actions and physics stay unchanged.
Evaluation is deferred until the learner exits, so GPU allocations never overlap.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.native_chord_diagnostic import gpu_memory, utc_now, write_json


class TrainingBudgetReached(Exception):
    pass


def run_child(args):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from pxr import Usd

    from training import RUNTIME_COMMIT
    from training.bundles import sha256

    runtime, out, job = (
        Path(p).resolve() for p in (args.runtime, args.output, args.job)
    )
    commit = subprocess.check_output(
        ["git", "-C", str(runtime), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(runtime), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    ).strip()
    if commit != RUNTIME_COMMIT or dirty:
        raise ValueError("Native runtime pin or cleanliness mismatch")
    source = runtime / "robotic_grounding/flash_chord/src"
    sys.path.insert(0, str(source))
    resume = None
    prior_elapsed = 0.0
    if args.resume:
        checkpoint = Path(args.resume).resolve()
        resume = json.loads(checkpoint.with_suffix(".experiment.json").read_text())
        previous = checkpoint.parent.parent
        provenance = json.loads((previous / "provenance.json").read_text())
        if (
            provenance["runtime_commit"] != commit
            or provenance["method"] != args.method
            or provenance["job_motion_sha256"] != sha256(job / "motion.parquet")
            or resume["environment_steps"] != resume["iteration"] * args.worlds * 24
            or not 0 < resume["iteration"] < args.iterations
        ):
            raise ValueError("Resume source, method, worlds or budget mismatch")
        prior_elapsed = float(resume["training_elapsed_s"])
    urdf_sha = sha256(source / "flash_chord/assets/urdfs/g1/main_with_hand.urdf")
    if urdf_sha not in {
        "8c7f768dc8da6c969d8a4b4efb3dcbe3d9d2b05fca5104df12205a69879d19e9",
        "ab13c2fb8f51ee92f33e015e264b9826b92bc4bae0f757aa2947f6b5e66972a6",
    }:
        raise ValueError("Unrecognized native G1 asset")
    row = pq.read_table(job / "motion.parquet").to_pylist()[0]
    sequence = str(row["sequence_id"])
    if "/" in sequence or "\\" in sequence or sequence in (".", ".."):
        raise ValueError("Invalid sequence identifier")
    motion_dir = out / "input/retargeted" / f"sequence_id={sequence}" / "robot_name=g1"
    motion_dir.mkdir(parents=True, exist_ok=True)
    motion = motion_dir / "motion.parquet"
    row["object_mesh_paths"] = [str(job / "assets/object.obj")]
    row["object_urdf_paths"] = [str(job / "assets/object.urdf")]
    pq.write_table(pa.Table.from_pylist([row]), motion, compression="zstd")
    support = job / "assets/support.usda"
    stage = Usd.Stage.Open(str(support))
    if stage is None:
        raise ValueError("Cannot open support stage")
    has_support = any(p.GetTypeName() in ("Cube", "Cylinder") for p in stage.Traverse())
    if has_support:
        support_dir = out / "input/reconstructed_stage"
        support_dir.mkdir(parents=True, exist_ok=True)
        (support_dir / f"{sequence}_support.usda").symlink_to(support)
    overrides = [
        "experiment=g1_recon_body_ppo",
        f"embodiment.config.asset_sha256={urdf_sha}",
        f"task.parquet={json.dumps(str(motion))}",
        f"scene.support={str(has_support).lower()}",
        f"scene.world_count={args.worlds}",
        f"training.max_iterations={args.iterations}",
        f"training.save_interval={args.save_interval}",
        f"output_dir={out / 'checkpoints'}",
        f"hydra.run.dir={out / 'hydra'}",
        "logging.mode=offline",
        "logging.log_interval=1",
        "logging.upload_checkpoints=false",
    ]
    if args.resume:
        overrides.extend(
            [
                "training.resume=true",
                f"training.checkpoint={json.dumps(str(checkpoint))}",
            ]
        )
    entry = source.parent / "scripts/train_rl.py"
    spec = importlib.util.spec_from_file_location("native_chord_ppo_entry", entry)
    native = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = native
    spec.loader.exec_module(native)
    from flash_chord.training.ppo import runner

    context = {"started": time.monotonic(), "latest": None, "curriculum": None}

    class LocalMetrics:
        def __init__(self, config, **kwargs):
            if resume:
                old_config = json.loads((previous / "training_config.json").read_text())
                new_config = json.loads(json.dumps(dataclasses.asdict(config)))
                for key in ("max_iterations", "save_interval", "resume", "checkpoint"):
                    old_config.pop(key, None)
                    new_config.pop(key, None)
                if old_config != new_config:
                    raise ValueError("Resume changes native learner configuration")
            self.stream = (out / "metrics.jsonl").open("w", buffering=1)
            context["logger"] = self
            write_json(out / "training_config.json", dataclasses.asdict(config))
            write_json(out / "environment_config.json", kwargs["environment_config"])

        def log(self, metrics, step):
            elapsed = time.monotonic() - context["started"]
            iteration = step // (args.worlds * 24)
            curriculum = context["curriculum"]
            metrics = dict(metrics)
            context["last_assistance"] = metrics.get("curriculum/voc_scale")
            metrics["episode/completed_count_per_iteration"] = float(
                context["episode_count"]
            )
            decision = None
            if curriculum is not None:
                decision = curriculum.observe(metrics, iteration)
            self.stream.write(
                json.dumps(
                    {
                        "time_utc": utc_now(),
                        "transitions": step,
                        "training_elapsed_s": prior_elapsed + elapsed,
                        "session_elapsed_s": elapsed,
                        **metrics,
                    },
                    allow_nan=False,
                )
                + "\n"
            )
            write_json(
                out / "progress.json",
                {
                    "state": "training",
                    "transitions": step,
                    "iteration": iteration,
                    "training_elapsed_s": prior_elapsed + elapsed,
                    "session_elapsed_s": elapsed,
                    "assistance_scale": metrics.get("curriculum/voc_scale"),
                    "throughput": metrics.get("throughput/env_steps_per_second"),
                    "learning_rate": metrics.get("ppo/learning_rate"),
                    "decision": decision,
                },
            )
            if elapsed >= args.training_seconds:
                raise TrainingBudgetReached("training_wall_budget")

        def log_checkpoint(self, path, *, iteration, metadata):
            curriculum = context["curriculum"]
            write_json(
                Path(path).with_suffix(".experiment.json"),
                {
                    "iteration": iteration,
                    **metadata,
                    "training_elapsed_s": prior_elapsed
                    + time.monotonic()
                    - context["started"],
                    "method": args.method,
                    "curriculum_state": curriculum.state_dict() if curriculum else None,
                },
            )

        def close(self):
            if not self.stream.closed:
                self.stream.close()

    native.WandbLogger = LocalMetrics
    original_load = runner.load_checkpoint

    def checked_load(path, template):
        learner, metadata = original_load(path, template)
        if int(learner.iteration) != resume["iteration"]:
            raise ValueError("Restored learner iteration disagrees with sidecar")
        write_json(
            out / "resume_validation.json",
            {
                "iteration": int(learner.iteration),
                "environment_steps": int(metadata["environment_steps"]),
                "learning_rate": float(learner.learning_rate),
                "checkpoint_sha256": sha256(Path(path)),
                "sidecar_sha256": sha256(Path(path).with_suffix(".experiment.json")),
                "all_learner_leaves_restored": True,
                "curriculum_state_restored": context["curriculum"].state_dict()
                == resume["curriculum_state"]
                if context["curriculum"]
                else True,
                "native_environment_reset_on_resume": True,
            },
        )
        return learner, metadata

    if resume:
        runner.load_checkpoint = checked_load
    original_iteration = runner.train_iteration

    def captured_iteration(env, learner, config):
        result = original_iteration(env, learner, config)
        context["latest"] = result[0]
        context["episode_count"] = result[1].rollout.episode_count
        return result

    runner.train_iteration = captured_iteration
    original_run = native.run_training

    def bounded_run(env, config, **kwargs):
        sessions = (env.env.action.encoder_session, env.env.action.decoder_session)
        if any(s.provider != "CUDAExecutionProvider" for s in sessions):
            raise RuntimeError("SONIC requires CUDA provider")
        if args.method == "dex_ppo":
            from training.dex_curriculum import DexCurriculum

            context["curriculum"] = DexCurriculum(
                config.curriculum, frame_dt=env.frame_dt
            )
            if resume:
                context["curriculum"].load_state_dict(
                    resume["curriculum_state"], iteration=resume["iteration"]
                )
            config = dataclasses.replace(config, curriculum=context["curriculum"])
            write_json(
                out / "dex_curriculum_spec.json", context["curriculum"].state_dict()
            )
        stop_reason = "transition_budget"
        try:
            learner = original_run(env, config, **kwargs)
        except (TrainingBudgetReached, KeyboardInterrupt) as error:
            learner = context["latest"]
            if learner is None:
                raise
            stop_reason = (
                str(error)
                if isinstance(error, TrainingBudgetReached)
                else "supervisor_interrupt"
            )
        iteration = int(learner.iteration)
        metadata = dict(kwargs.get("checkpoint_metadata") or {})
        metadata.update(
            iteration=iteration,
            environment_steps=iteration * args.worlds * config.rollout_steps,
        )
        active = config.curriculum.stages[
            config.curriculum.stage_index(max(iteration - 1, 0))
        ]
        metadata["voc_scale"] = context.get("last_assistance", active.voc_scale)
        path = out / "checkpoints" / f"model_{iteration}.safetensors"
        runner.save_checkpoint(path, learner, metadata=metadata)
        context["logger"].log_checkpoint(path, iteration=iteration, metadata=metadata)
        context["logger"].close()
        write_json(
            out / "training_result.json",
            {
                "state": "budget_complete",
                "stop_reason": stop_reason,
                "iterations": iteration,
                "transitions": metadata["environment_steps"],
                "training_elapsed_s": prior_elapsed
                + time.monotonic()
                - context["started"],
                "session_elapsed_s": time.monotonic() - context["started"],
                "resumed_from_transitions": resume["environment_steps"]
                if resume
                else 0,
                "checkpoint": str(path),
                "method": args.method,
                "world_model": False,
            },
        )
        return learner

    native.run_training = bounded_run
    write_json(
        out / "provenance.json",
        {
            "created_utc": utc_now(),
            "runtime_commit": commit,
            "method": args.method,
            "learner": "native_CHORD_PPO",
            "world_model": False,
            "resume_checkpoint": str(checkpoint) if resume else None,
            "overrides": overrides,
            "upstream_entry_sha256": sha256(entry),
            "job_motion_sha256": sha256(job / "motion.parquet"),
            "source_sha256": {
                str(p): sha256(p)
                for p in (
                    Path(__file__),
                    Path(__file__).resolve().parents[1] / "training/dex_curriculum.py",
                )
            },
            "scope": "corrected Tier 1 episode 5; seed 42; local input reference",
            "recipe_changes": [
                "2048 instead of 4096 worlds",
                "bounded diagnostic training budget",
            ]
            + (
                [
                    "rolling reward/completion curriculum",
                    "25% immediate frame-zero starts; 50% when unassisted",
                ]
                if args.method == "dex_ppo"
                else []
            ),
            "note": "DexMachina-inspired curriculum on CHORD PPO, not a full DexMachina reproduction. Native PPO optimizer parameters unchanged.",
        },
    )
    sys.argv = [
        str(entry),
        "--config-path",
        str(source / "flash_chord/configs"),
        *overrides,
    ]
    native.main()


def supervise(args):
    def terminate(_signum, _frame):
        raise KeyboardInterrupt("supervisor terminated")

    signal.signal(signal.SIGTERM, terminate)
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    env = os.environ.copy()
    env.update(
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
        PYTHONUNBUFFERED="1",
        HYDRA_FULL_ERROR="1",
    )
    env.pop("XLA_PYTHON_CLIENT_ALLOCATOR", None)
    env.pop("XLA_PYTHON_CLIENT_MEM_FRACTION", None)
    state = {
        "state": "running",
        "started_utc": utc_now(),
        "arguments": vars(args),
        "peak_gpu_used_mib": 0,
    }
    stop = None
    interrupted = None
    with (
        (out / "train.log").open("w") as log,
        (out / "gpu_memory.jsonl").open("w") as mem,
    ):
        proc = subprocess.Popen(
            [sys.executable, "-u", __file__, *sys.argv[1:], "--child"],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        state["pid"] = proc.pid
        try:
            while proc.poll() is None:
                elapsed = time.monotonic() - started
                sample = {"elapsed_s": elapsed, **gpu_memory()}
                mem.write(json.dumps(sample) + "\n")
                mem.flush()
                state.update(
                    elapsed_s=elapsed,
                    peak_gpu_used_mib=max(
                        state["peak_gpu_used_mib"], sample["used_mib"]
                    ),
                )
                if stop is None:
                    if sample["used_mib"] > args.memory_limit_mib:
                        stop = "memory_guard"
                    elif elapsed > args.training_seconds + 180:
                        stop = "supervisor_wall_guard"
                    if stop:
                        interrupted = time.monotonic()
                        os.killpg(proc.pid, signal.SIGINT)
                elif time.monotonic() - interrupted > 45:
                    os.killpg(proc.pid, signal.SIGKILL)
                write_json(out / "status.json", state)
                time.sleep(1)
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
        state.update(
            state="complete" if proc.returncode == 0 and stop is None else "failed",
            exit_code=proc.returncode,
            stop_reason=stop,
            elapsed_s=time.monotonic() - started,
            finished_utc=utc_now(),
        )
        write_json(out / "status.json", state)
    return 0 if state["state"] == "complete" else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runtime", required=True)
    p.add_argument("--job", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--method", choices=("native_ppo", "dex_ppo"), required=True)
    p.add_argument("--resume", help="Native learner checkpoint with experiment sidecar")
    p.add_argument("--worlds", type=int, default=2048)
    p.add_argument("--iterations", type=int, default=1221)
    p.add_argument("--save-interval", type=int, default=204)
    p.add_argument("--training-seconds", type=int, default=2700)
    p.add_argument("--memory-limit-mib", type=int, default=22000)
    p.add_argument("--child", action="store_true")
    args = p.parse_args()
    if (
        min(args.worlds, args.iterations, args.save_interval, args.training_seconds)
        <= 0
    ):
        p.error("Positive budgets required")
    if args.child:
        run_child(args)
    else:
        raise SystemExit(supervise(args))


if __name__ == "__main__":
    main()

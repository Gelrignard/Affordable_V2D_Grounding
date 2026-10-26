"""Run a bounded original/corrected pair through native CHORD and its evaluator.

Training and evaluation processes run sequentially to release GPU allocations.
All native algorithm settings are provided by native_chord_diagnostic.py.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from native_chord_diagnostic import utc_now, write_json


def run(args):
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    runtime = Path(args.runtime).resolve()
    harness = Path(__file__).with_name("native_chord_diagnostic.py")
    evaluator = runtime / "robotic_grounding/flash_chord/scripts/evaluate_policy.py"
    jobs = (
        Path(args.jobs).resolve() / f"tier{args.tier}" / f"episode_{args.episode:06d}"
    )
    env = os.environ.copy()
    env.update({"XLA_PYTHON_CLIENT_PREALLOCATE": "false", "HYDRA_FULL_ERROR": "1"})
    # Retain the exact launch sources alongside this run's artifacts.
    for source in (Path(__file__).resolve(), harness):
        shutil.copy2(source, out / f"executed_{source.name}")
    start = time.monotonic()
    status = {
        "state": "running",
        "started_utc": utc_now(),
        "pid": os.getpid(),
        "arguments": vars(args),
        "runs": [],
        "metric_reference": f"tier{args.tier}_input_reference",
        "official_leaderboard_score": False,
        "transition_budget_per_policy": args.transitions,
        "replay_capacity": 2000000,
        "planned_runs": 2,
        "scope": (
            "Full-budget native downstream comparison with reduced 2M replay."
            if args.transitions == 349999104
            else "Short native downstream comparison; not a convergence claim."
        ),
    }
    write_json(out / "plan.json", status)
    write_json(out / "queue_status.json", status)
    try:
        for variant in ("original", "corrected"):
            entry = {"variant": variant, "state": "training"}
            status["runs"].append(entry)
            write_json(out / "queue_status.json", status)
            run_dir = out / variant
            command = [
                sys.executable,
                "-u",
                str(harness),
                "--runtime",
                str(runtime),
                "--job",
                str(jobs / variant),
                "--output",
                str(run_dir),
                "--worlds",
                "2048",
                "--batch-size",
                "2048",
                "--replay-capacity",
                "2000000",
                "--transitions",
                str(args.transitions),
                "--checkpoint-interval",
                str(args.checkpoint_interval),
                "--max-seconds",
                str(args.max_seconds),
            ]
            with (out / f"{variant}_launcher.log").open("w") as log:
                subprocess.run(
                    command, stdout=log, stderr=subprocess.STDOUT, env=env, check=True
                )
            entry["training"] = json.loads((run_dir / "status.json").read_text())
            entry["state"] = "evaluating"
            write_json(out / "queue_status.json", status)
            checkpoint = (
                run_dir / "checkpoints" / f"policy_{args.transitions}.safetensors"
            )
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            entry["checkpoint"] = str(checkpoint)
            evaluation = run_dir / "evaluation.json"
            command = [
                sys.executable,
                "-u",
                str(evaluator),
                f"evaluation.checkpoint={checkpoint}",
                f"evaluation.world_count={args.evaluation_worlds}",
                "evaluation.reset_mode=explicit",
                "logging.mode=disabled",
                f"evaluation.metrics_output={evaluation}",
                f"evaluation.object_trajectories_output={run_dir / 'object_rollouts.parquet'}",
                f"hydra.run.dir={run_dir / 'evaluation_hydra'}",
            ]
            with (run_dir / "evaluation.log").open("w") as log:
                subprocess.run(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=env,
                    check=True,
                    timeout=600,
                )
            entry["evaluation"] = json.loads(evaluation.read_text())
            if args.render_video:
                entry["state"] = "rendering"
                write_json(out / "queue_status.json", status)
                video = run_dir / "policy.mp4"
                command = [
                    sys.executable,
                    "-u",
                    str(runtime / "robotic_grounding/flash_chord/scripts/view_policy.py"),
                    f"evaluation.checkpoint={checkpoint}",
                    "evaluation.reset_mode=explicit",
                    "viewer.backend=mp4",
                    f"viewer.output_path={video}",
                    f"viewer.num_frames={args.video_frames}",
                    "viewer.video_fps=50",
                    "viewer.width=1280",
                    "viewer.height=720",
                    "viewer.realtime=false",
                    "markers.enabled=false",
                    "logging.mode=disabled",
                    f"hydra.run.dir={run_dir / 'video_hydra'}",
                ]
                # A rendering failure must not discard trained policies or
                # prevent the second policy from training. Record it explicitly.
                try:
                    with (run_dir / "video.log").open("w") as log:
                        subprocess.run(
                            command,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            env=env,
                            check=True,
                            timeout=600,
                        )
                    if not video.is_file() or video.stat().st_size == 0:
                        raise RuntimeError("Renderer produced no video")
                    entry["video"] = str(video)
                except (OSError, subprocess.SubprocessError, RuntimeError) as error:
                    entry["video_error"] = f"{type(error).__name__}: {error}"
            entry["state"] = "complete"
            status["elapsed_s"] = time.monotonic() - start
            write_json(out / "queue_status.json", status)
    except BaseException as error:
        status.update({"state": "failed", "error": f"{type(error).__name__}: {error}"})
        raise
    else:
        status["state"] = "complete"
    finally:
        status.update(
            {"finished_utc": utc_now(), "elapsed_s": time.monotonic() - start}
        )
        write_json(out / "queue_status.json", status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--jobs", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tier", type=int, choices=(1, 2), default=1)
    parser.add_argument("--episode", type=int, choices=(5, 16), default=5)
    parser.add_argument("--transitions", type=int, default=10000384)
    parser.add_argument("--checkpoint-interval", type=int, default=5001216)
    parser.add_argument("--max-seconds", type=int, default=1800)
    parser.add_argument("--evaluation-worlds", type=int, default=16)
    parser.add_argument("--render-video", action="store_true")
    parser.add_argument("--video-frames", type=int, default=1100)
    args = parser.parse_args()
    if args.transitions < 409600 or args.transitions % 2048:
        parser.error("Budget must reach replay warmup and be divisible by 2048")
    if (
        args.checkpoint_interval <= 0
        or args.checkpoint_interval % 2048
        or min(args.max_seconds, args.evaluation_worlds, args.video_frames) <= 0
    ):
        parser.error("Positive bounds and a world-aligned checkpoint interval are required")
    run(args)


if __name__ == "__main__":
    main()

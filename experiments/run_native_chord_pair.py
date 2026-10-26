"""Run a bounded original/corrected pair through native CHORD and its evaluator.

Training and evaluation processes run sequentially to release GPU allocations.
All native algorithm settings are provided by native_chord_diagnostic.py.
"""

import argparse
import json
import os
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
    start = time.monotonic()
    status = {
        "state": "running",
        "started_utc": utc_now(),
        "pid": os.getpid(),
        "arguments": vars(args),
        "runs": [],
        "metric_reference": f"tier{args.tier}_input_reference",
        "official_leaderboard_score": False,
        "scope": "Short native downstream comparison; not a convergence claim.",
    }
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
                "5001216",
                "--max-seconds",
                "1800",
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
                "evaluation.world_count=16",
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
    args = parser.parse_args()
    if args.transitions < 409600 or args.transitions % 2048:
        parser.error("Budget must reach replay warmup and be divisible by 2048")
    run(args)


if __name__ == "__main__":
    main()

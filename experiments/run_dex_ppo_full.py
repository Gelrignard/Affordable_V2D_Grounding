"""Continue the tested Dex-inspired PPO candidate to the matched 350M budget."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.native_chord_diagnostic import utc_now, write_json
from training.cloud_status import read_json
from training.leaderboard import leaderboard_row, write_report


def run(args):
    root = Path(__file__).resolve().parents[1]
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    if shutil.disk_usage(out).free < 4 * 2**30:
        raise RuntimeError("Need 4 GiB for preserved checkpoints and evaluations")
    source = out / "executed_source"
    for name in ("training", "experiments"):
        shutil.copytree(
            root / name, source / name, ignore=shutil.ignore_patterns("__pycache__")
        )
    env = os.environ.copy()
    env.update(
        PYTHONPATH=str(source)
        + os.pathsep
        + str(Path(args.runtime) / "robotic_grounding/flash_chord/src"),
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
        PYTHONUNBUFFERED="1",
    )
    started = time.monotonic()
    status = {
        "state": "running",
        "pid": os.getpid(),
        "started_utc": utc_now(),
        "arguments": vars(args),
        "world_model": False,
        "runs": [],
        "scope": "corrected Tier 1 episode 5, seed 42",
        "total_transition_budget": args.iterations * 2048 * 24,
        "resume_checkpoint": args.resume,
    }

    def save():
        status["elapsed_s"] = time.monotonic() - started
        write_json(out / "queue_status.json", status)

    def execute(command, log_path, timeout):
        with log_path.open("w") as log:
            proc = subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT, cwd=source, env=env
            )
            launch = time.monotonic()
            try:
                while proc.poll() is None:
                    if time.monotonic() - launch > timeout:
                        raise TimeoutError(f"Child exceeded {timeout}s: {log_path}")
                    save()
                    time.sleep(5)
            finally:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
            return proc.returncode

    rows = []
    try:
        processes = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
            text=True,
        ).strip()
        if processes:
            raise RuntimeError(f"Unexpected compute processes: {processes}")
        dest = out / "dex_ppo_full"
        entry = {"variant": "dex_ppo_full", "state": "training", "path": str(dest)}
        status["runs"].append(entry)
        status["active"] = entry["variant"]
        save()
        code = execute(
            [
                sys.executable,
                "-u",
                str(source / "experiments/native_ppo_diagnostic.py"),
                "--runtime",
                args.runtime,
                "--job",
                args.job,
                "--output",
                str(dest),
                "--method",
                "dex_ppo",
                "--resume",
                args.resume,
                "--iterations",
                str(args.iterations),
                "--save-interval",
                "204",
                "--training-seconds",
                "10800",
                "--memory-limit-mib",
                "22000",
            ],
            out / "training_launcher.log",
            11160,
        )
        entry["training"] = read_json(dest / "training_result.json")
        entry["supervisor"] = read_json(dest / "status.json")
        if code != 0 or not entry["training"]:
            raise RuntimeError("Training failed; checkpoints and logs preserved")
        entry.update(state="evaluating", evaluations=[])
        save()
        checkpoints = sorted(
            (dest / "checkpoints").glob("model_*.safetensors"),
            key=lambda p: int(p.stem.split("_")[-1]),
        )
        # Keep all ~10M checkpoints; evaluate ~100/150/200/250/300/350M after training.
        selected = {checkpoints[-1]}
        for target in range(100_000_000, args.iterations * 2048 * 24, 50_000_000):
            candidates = [
                p
                for p in checkpoints
                if int(p.stem.split("_")[-1]) * 2048 * 24 >= target
            ]
            if candidates:
                selected.add(candidates[0])
        for checkpoint in checkpoints:
            if checkpoint not in selected:
                continue
            sidecar = read_json(checkpoint.with_suffix(".experiment.json"))
            target = dest / "evaluations" / checkpoint.stem
            code = execute(
                [
                    sys.executable,
                    "-u",
                    str(source / "experiments/evaluate_chord_candidate.py"),
                    "--checkpoint",
                    str(checkpoint),
                    "--output",
                    str(target),
                    "--worlds",
                    "16",
                    "--variant",
                    entry["variant"],
                    "--training-seconds",
                    str(sidecar["training_elapsed_s"]),
                ],
                dest / f"{checkpoint.stem}_evaluation.log",
                360,
            )
            data = read_json(target / "evaluation.json")
            if code != 0 or not data or not data.get("metric_verification"):
                entry["evaluations"].append(
                    {
                        "checkpoint": str(checkpoint),
                        "state": "failed",
                        "exit_code": code,
                    }
                )
                entry["evaluation_failed"] = True
                save()
                continue
            row = leaderboard_row(
                data,
                tier=1,
                episode=5,
                variant=entry["variant"],
                training_s=sidecar["training_elapsed_s"],
                transitions=sidecar["environment_steps"],
            )
            rows.append(row)
            entry["evaluations"].append(
                {
                    "checkpoint": str(checkpoint),
                    "state": "complete",
                    "leaderboard": row,
                    "evaluation_seconds": data["elapsed_s"],
                    "reference_tracking": data.get("reference_tracking"),
                    "probe_assistance": data["probe_assistance"],
                }
            )
            write_report(out, rows)
            save()
        if entry["training"]["stop_reason"] != "transition_budget":
            entry["state"] = "stopped_at_wall_guard"
        else:
            entry["state"] = (
                "complete_with_evaluation_errors"
                if entry.get("evaluation_failed")
                else "complete"
            )
        status["state"] = entry["state"]
    except BaseException as error:
        status.update(state="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        status["finished_utc"] = utc_now()
        save()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("runtime", "job", "output", "resume"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--iterations", type=int, default=7121)
    run(p.parse_args())

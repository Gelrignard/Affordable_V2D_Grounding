"""Run a bounded native-CHORD extension pair and publish every scored checkpoint.

Existing native checkpoints supply the baseline, with no baseline retraining.
Both candidates start fresh on the same corrected Tier 1 episode 5, seed 42.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.leaderboard import leaderboard_row, write_report
from training.native_experiment import atomic_json


def run(args):
    root = Path(__file__).resolve().parents[1]
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    if shutil.disk_usage(out).free < 4 * 2**30:
        raise RuntimeError("Need 4 GiB free for bounded checkpoints and evaluations")
    shutil.copytree(
        root / "training",
        out / "executed_source/training",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    for name in (
        "native_chord_diagnostic.py",
        "evaluate_chord_candidate.py",
        "run_chord_candidates.py",
    ):
        shutil.copy2(root / "experiments" / name, out / "executed_source" / name)
    started = time.monotonic()
    status = {
        "state": "running",
        "pid": os.getpid(),
        "arguments": vars(args),
        "runs": [],
        "world_model": False,
    }
    rows = []

    def save():
        status["elapsed_s"] = time.monotonic() - started
        atomic_json(out / "queue_status.json", status)

    def execute(command, log):
        with Path(log).open("w") as stream:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)

    atomic_json(out / "plan.json", status)
    save()
    try:
        for step in (20000768, 40001536, 60002304):
            entry = {
                "variant": "native_corrected",
                "transitions": step,
                "state": "evaluating",
            }
            status["runs"].append(entry)
            save()
            dest = out / f"baseline_{step}"
            checkpoint = (
                Path(args.baseline)
                / "corrected/checkpoints"
                / f"policy_{step}.safetensors"
            )
            cached = root / f"baseline_{step}"
            cached_report = cached / "evaluation.json"
            reusable = (
                cached_report.exists()
                and json.loads(cached_report.read_text()).get("checkpoint_sha256")
                == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            )
            if reusable:
                shutil.copytree(cached, dest)
                entry["reused_saved_evaluation"] = True
            else:
                execute(
                    [
                        sys.executable,
                        "-u",
                        str(root / "experiments/evaluate_chord_candidate.py"),
                        "--checkpoint",
                        str(
                            Path(args.baseline)
                            / "corrected/checkpoints"
                            / f"policy_{step}.safetensors"
                        ),
                        "--output",
                        str(dest),
                        "--variant",
                        "native_corrected",
                    ],
                    out / f"baseline_{step}.log",
                )
            evaluation = json.loads((dest / "evaluation.json").read_text())
            # Recover wall-time from saved checkpoint log timestamps, not an extrapolated speed.
            train_dir = Path(args.baseline) / "corrected"
            from datetime import datetime

            train_start = datetime.fromisoformat(
                json.loads((train_dir / "status.json").read_text())["started_utc"]
            )
            records = [
                json.loads(line)
                for line in (train_dir / "metrics.jsonl").read_text().splitlines()
            ]
            record = min(
                (r for r in records if r["transitions"] >= step),
                key=lambda r: r["transitions"],
            )
            train_s = (
                datetime.fromisoformat(record["time_utc"]) - train_start
            ).total_seconds()
            row = leaderboard_row(
                evaluation,
                tier=1,
                episode=5,
                variant="native_corrected",
                transitions=step,
                training_s=train_s,
            )
            rows.append(row)
            entry.update(
                state="complete",
                leaderboard=row,
                evaluation_seconds=evaluation["elapsed_s"],
            )
            write_report(out, rows)
            save()
        for method, weight in (("adaptive", 0), ("adaptive_wrench", 1)):
            entry = {"variant": method, "state": "training"}
            status["runs"].append(entry)
            save()
            dest = out / method
            execute(
                [
                    sys.executable,
                    "-u",
                    str(root / "experiments/native_chord_diagnostic.py"),
                    "--runtime",
                    args.runtime,
                    "--job",
                    args.job,
                    "--output",
                    str(dest),
                    "--worlds",
                    "2048",
                    "--batch-size",
                    "2048",
                    "--replay-capacity",
                    "2000000",
                    "--transitions",
                    "60002304",
                    "--checkpoint-interval",
                    "10000384",
                    "--max-seconds",
                    "5400",
                    "--memory-limit-mib",
                    "23000",
                    "--method",
                    method,
                    "--dynamics-weight",
                    str(weight),
                    "--stall-transitions",
                    "30001152",
                ],
                out / f"{method}_launcher.log",
            )
            result = json.loads((dest / "extension_status.json").read_text())
            entry.update(state=result["state"], result=result)
            rows.extend(json.loads((dest / "leaderboard.json").read_text())["rows"])
            write_report(out, rows)
            save()
    except BaseException as error:
        status.update(state="failed", error=f"{type(error).__name__}: {error}")
        raise
    else:
        status["state"] = "complete"
    finally:
        save()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runtime", required=True)
    p.add_argument("--job", required=True)
    p.add_argument("--baseline", required=True)
    p.add_argument("--output", required=True)
    run(p.parse_args())

"""Fresh native CHORD FlashSAC comparison with 5M rolling replay capacity."""

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
from training.cloud_status import last_metric, read_json
from training.leaderboard import leaderboard_row, write_report

BUDGET = 349999104


def run(args):
    root = Path(__file__).resolve().parents[1]
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    if shutil.disk_usage(out).free < 7 * 2**30:
        raise RuntimeError("Need 7 GiB to preserve all native 10M checkpoints")
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
        XLA_PYTHON_CLIENT_MEM_FRACTION="0.84",
        PYTHONUNBUFFERED="1",
    )
    env.pop("XLA_PYTHON_CLIENT_ALLOCATOR", None)
    started = time.monotonic()
    capacity = getattr(args, "replay_capacity", 5000000)
    backend = getattr(args, "replay_backend", "gpu")
    max_seconds = getattr(args, "max_seconds", 14400)
    name = "native_5m" if backend == "gpu" else "native_host_10m"
    variant = "native_flash_sac_5m" if backend == "gpu" else "native_flash_sac_host_10m"
    status = {
        "state": "running",
        "pid": os.getpid(),
        "started_utc": utc_now(),
        "arguments": vars(args),
        "world_model": False,
        "runs": [],
        "scope": "corrected Tier 1 episode 5, seed 42",
        "total_transition_budget": BUDGET,
        "replay_capacity": capacity,
        "replay_backend": backend,
        "optimizer_batch_size": 2048,
        "fresh_training": True,
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
        dest = out / name
        entry = {
            "variant": variant,
            "state": "training",
            "path": str(dest),
        }
        status["runs"].append(entry)
        status["active"] = entry["variant"]
        save()
        code = execute(
            [
                sys.executable,
                "-u",
                str(source / "experiments/native_chord_diagnostic.py"),
                "--runtime",
                args.runtime,
                "--job",
                args.job,
                "--output",
                str(dest),
                "--method",
                "native",
                "--worlds",
                "2048",
                "--batch-size",
                "2048",
                "--replay-capacity",
                str(capacity),
                "--replay-backend",
                backend,
                "--transitions",
                str(BUDGET),
                "--checkpoint-interval",
                "10000384",
                "--max-seconds",
                str(max_seconds),
                "--memory-limit-mib",
                "24200",
                "--memory-poll-seconds",
                "0.2",
                "--preflight-ring-multiplier",
                "1.1",
                "--preflight-workspace-mib",
                "2048",
                "--comparison-config",
                args.comparison_config,
                *(["--prune-intermediate-states"] if backend == "host" else []),
            ],
            out / "training_launcher.log",
            max_seconds + 360,
        )
        entry["supervisor"] = read_json(dest / "status.json")
        metric = last_metric(dest / "metrics.jsonl")
        entry["training"] = {
            "transitions": metric.get("transitions"),
            "training_elapsed_s": entry["supervisor"].get("elapsed_s"),
            "assistance_scale": metric.get("curriculum/voc_scale"),
        }
        if (
            code != 0
            or entry["supervisor"]["state"] != "complete"
            or metric.get("transitions") != BUDGET
        ):
            raise RuntimeError(
                "Training did not reach the full budget; checkpoints/logs preserved"
            )
        entry.update(state="evaluating", evaluations=[])
        save()
        checkpoints = sorted(
            (dest / "checkpoints").glob("policy_*.safetensors"),
            key=lambda p: int(p.stem.split("_")[-1]),
        )
        selected = {checkpoints[-1]}
        for target in (
            60_000_000,
            100_000_000,
            150_000_000,
            200_000_000,
            250_000_000,
            300_000_000,
        ):
            selected.add(
                next(p for p in checkpoints if int(p.stem.split("_")[-1]) >= target)
            )
        for checkpoint in checkpoints:
            if checkpoint not in selected:
                continue
            sidecar = read_json(checkpoint.with_suffix(".experiment.json"))
            if not sidecar:
                raise RuntimeError(f"Missing checkpoint timing: {checkpoint}")
            training_s = sidecar["training_elapsed_s"]
            if checkpoint == checkpoints[-1]:
                training_s = entry["training"]["training_elapsed_s"]
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
                    str(training_s),
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
                training_s=training_s,
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
    for name in ("runtime", "job", "output", "comparison-config"):
        p.add_argument("--" + name, required=True)
    run(p.parse_args())

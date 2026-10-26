"""Serial capacity-dependent PPO integration check and bounded learning comparison."""

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


def run(args):
    root = Path(__file__).resolve().parents[1]
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    if shutil.disk_usage(out).free < 8 * 2**30:
        raise RuntimeError("Need 8 GiB for preserved checkpoints/evaluations")
    source = out / "executed_source"
    shutil.copytree(
        root / "training",
        source / "training",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    (source / "experiments").mkdir()
    for name in (
        "native_chord_diagnostic.py",
        "native_ppo_diagnostic.py",
        "run_dex_ppo_tests.py",
        "evaluate_chord_candidate.py",
        "replay_capacity_sweep.py",
    ):
        shutil.copy2(root / "experiments" / name, source / "experiments" / name)
    # Execute the preserved source snapshot, not mutable files in a development checkout.
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        str(source)
        + os.pathsep
        + str(Path(args.runtime) / "robotic_grounding/flash_chord/src")
    )
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    env.pop("XLA_PYTHON_CLIENT_ALLOCATOR", None)
    env.pop("XLA_PYTHON_CLIENT_MEM_FRACTION", None)
    started = time.monotonic()
    status = {
        "state": "waiting_for_capacity",
        "pid": os.getpid(),
        "started_utc": utc_now(),
        "arguments": vars(args),
        "world_model": False,
        "runs": [],
        "scope": "corrected Tier 1 episode 5, seed 42",
    }
    rows = []

    def save():
        status["elapsed_s"] = time.monotonic() - started
        write_json(out / "queue_status.json", status)

    def execute(command, log_path, timeout):
        with Path(log_path).open("w") as log:
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

    def train(name, method, iterations, seconds, interval):
        dest = out / name
        entry = {
            "variant": name,
            "method": method,
            "state": "training",
            "path": str(dest),
        }
        status["runs"].append(entry)
        status.update(state="running", active=name)
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
                method,
                "--iterations",
                str(iterations),
                "--training-seconds",
                str(seconds),
                "--save-interval",
                str(interval),
                "--memory-limit-mib",
                "22000",
            ],
            out / f"{name}_launcher.log",
            seconds + 300,
        )
        entry["training"] = read_json(dest / "training_result.json")
        entry["supervisor"] = read_json(dest / "status.json")
        if code != 0 or not entry["training"]:
            entry.update(state="failed", exit_code=code)
            save()
            raise RuntimeError(
                f"{name} training failed; logs and prior checkpoints preserved"
            )
        entry["state"] = "trained"
        save()
        return entry, dest

    def evaluate(entry, dest):
        entry.update(state="evaluating", evaluations=[])
        save()
        checkpoints = sorted(
            (dest / "checkpoints").glob("model_*.safetensors"),
            key=lambda p: int(p.stem.split("_")[-1]),
        )
        for checkpoint in checkpoints:
            sidecar = read_json(checkpoint.with_suffix(".experiment.json"))
            if not sidecar:
                raise RuntimeError(f"Missing checkpoint provenance: {checkpoint}")
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
        entry["state"] = (
            "complete_with_evaluation_errors"
            if entry.get("evaluation_failed")
            else "complete"
        )
        save()

    write_json(out / "plan.json", status)
    save()
    try:
        while True:
            capacity = read_json(Path(args.capacity_status))
            if capacity and capacity.get("state") in ("complete", "failed"):
                status["capacity_result"] = capacity
                break
            if time.monotonic() - started > 3600:
                raise TimeoutError("Capacity sweep did not finish within one hour")
            save()
            time.sleep(10)
        # No overlapping GPU jobs: profiling and capacity tests must remain meaningful.
        processes = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
            text=True,
        ).strip()
        if processes:
            raise RuntimeError(f"Unexpected existing compute processes: {processes}")
        profile, profile_dir = train("ppo_integration", "dex_ppo", 8, 360, 8)
        latest = last_metric(profile_dir / "metrics.jsonl")
        if latest.get("transitions") != 8 * 2048 * 24:
            raise RuntimeError("PPO integration did not complete eight iterations")
        evaluate(
            profile, profile_dir
        )  # Verifies checkpoint loading and independent metric recomputation.
        if profile.get("evaluation_failed"):
            raise RuntimeError("PPO evaluation integration failed; long tests withheld")
        for variant, iterations, seconds in (
            ("native_ppo", 408, 1200),
            ("dex_ppo", 1221, 2700),
        ):
            entry, dest = train(variant, variant, iterations, seconds, 204)
            evaluate(entry, dest)
        baseline = read_json(Path(args.baseline_report))
        if baseline:
            baseline_rows = [
                r
                for r in baseline.get("rows", [])
                if r.get("variant") == "native_corrected"
            ]
            rows.extend(baseline_rows)
            status["baseline_evaluations_reused"] = len(baseline_rows)
        write_report(out, rows)
        # Selection is explicitly exploratory on these evaluation worlds, not held-out performance.
        valid = [
            r
            for r in rows
            if r.get("variant") == "dex_ppo" and r.get("add_auc") is not None
        ]
        status["best_dex_checkpoint_by_auc"] = (
            max(valid, key=lambda r: r["add_auc"]) if valid else None
        )
        status["selection_note"] = (
            "Exploratory best AUC on the same 16 evaluation worlds; no held-out generalization claim."
        )
        status["state"] = (
            "complete_with_evaluation_errors"
            if any(r.get("evaluation_failed") for r in status["runs"])
            else "complete"
        )
    except BaseException as error:
        status.update(state="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        status["finished_utc"] = utc_now()
        save()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("runtime", "job", "output", "capacity-status", "baseline-report"):
        p.add_argument("--" + name, required=True)
    run(p.parse_args())

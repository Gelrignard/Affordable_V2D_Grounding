"""Distinguish a conservative memory-guard stop from actual replay capacity.

Serial isolated probes use a smaller JAX allocation fraction and a higher guard.
The held PPO orchestrator is resumed in finally; no learner is paused or changed.
"""

from __future__ import annotations

import argparse
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.native_chord_diagnostic import utc_now, write_json
from training.cloud_status import last_metric, read_json


def run(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    status = {
        "state": "running",
        "pid": os.getpid(),
        "started_utc": utc_now(),
        "results": [],
        "memory_guard_mib": 24200,
        "jax_memory_fraction": 0.84,
        "scope": "Maximum observed training-only fit, not recommended long-run headroom.",
        "resume_ppo_queue_pid": args.resume_queue_pid,
    }

    def save():
        status["elapsed_s"] = time.monotonic() - started
        write_json(out / "queue_status.json", status)

    def resume():
        command = Path(f"/proc/{args.resume_queue_pid}/cmdline")
        if command.exists() and b"run_dex_ppo_tests.py" in command.read_bytes():
            os.kill(args.resume_queue_pid, signal.SIGCONT)
            status["ppo_queue_resumed"] = True

    save()
    try:
        # The existing eight-iteration evaluation may still be exiting.
        while subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
            text=True,
        ).strip():
            if time.monotonic() - started > 180:
                raise TimeoutError("GPU did not become idle for isolated probes")
            save()
            time.sleep(5)
        for capacity in (4000000, 5000000, 6000000, 7000000, 8000000):
            if time.monotonic() - started > 1500:
                break
            dest = out / f"default_{capacity}"
            transitions = math.ceil((capacity + 2048 * 256) / 2048) * 2048
            status["active"] = dest.name
            save()
            env = os.environ.copy()
            env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
            env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.84"
            env.pop("XLA_PYTHON_CLIENT_ALLOCATOR", None)
            command = [
                sys.executable,
                "-u",
                str(Path(__file__).with_name("native_chord_diagnostic.py")),
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
                str(capacity),
                "--transitions",
                str(transitions),
                "--checkpoint-interval",
                str(transitions),
                "--memory-limit-mib",
                "24200",
                "--max-seconds",
                "480",
                "--memory-poll-seconds",
                "0.2",
                "--preflight-ring-multiplier",
                "1.1",
                "--preflight-workspace-mib",
                "2048",
            ]
            with (out / f"{dest.name}.log").open("w") as log:
                proc = subprocess.Popen(
                    command, stdout=log, stderr=subprocess.STDOUT, env=env
                )
                while proc.poll() is None:
                    save()
                    time.sleep(5)
            detail = read_json(dest / "status.json") or {}
            metric = last_metric(dest / "metrics.jsonl")
            tail = (
                (dest / "train.log").read_text(errors="replace")[-6000:]
                if (dest / "train.log").exists()
                else ""
            )
            passed = (
                proc.returncode == 0
                and detail.get("state") == "complete"
                and metric.get("transitions", 0) >= transitions
            )
            status["results"].append(
                {
                    "capacity": capacity,
                    "allocator": "default_fraction_0.84",
                    "passed": passed,
                    "transitions": metric.get("transitions"),
                    "target_transitions": transitions,
                    "ring_filled_and_wrapped": passed,
                    "peak_gpu_used_mib": detail.get("peak_gpu_used_mib"),
                    "elapsed_s": detail.get("elapsed_s"),
                    "throughput": metric.get("throughput/env_steps_per_second"),
                    "exit_code": proc.returncode,
                    "stop_reason": detail.get("stop_reason"),
                    "path": str(dest),
                    "allocation_error": any(
                        s in tail.lower()
                        for s in (
                            "resource_exhausted",
                            "out of memory",
                            "failed to allocate",
                        )
                    ),
                    "error_tail": tail[-1800:] if not passed else None,
                }
            )
            save()
            if not passed:
                status["failing_default_upper_bound"] = capacity
                break
        passed = [r["capacity"] for r in status["results"] if r["passed"]]
        status.update(
            state="complete",
            active=None,
            largest_passing_tested=max(passed, default=None),
        )
    except BaseException as error:
        status.update(state="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        resume()
        status["finished_utc"] = utc_now()
        save()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("runtime", "job", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--resume-queue-pid", type=int, required=True)
    run(p.parse_args())

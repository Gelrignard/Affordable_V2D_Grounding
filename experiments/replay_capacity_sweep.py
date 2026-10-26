"""Serial, isolated CHORD capacity probes on the actual learner, not allocation alone."""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path


def save(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--guard-mib", type=int, default=23500)
    parser.add_argument("--max-total-seconds", type=int, default=2700)
    args = parser.parse_args()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    status = {
        "state": "running",
        "pid": os.getpid(),
        "started_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "baseline": "native CHORD FlashSAC; 2048 worlds; batch 2048; float16 observation replay",
        "memory_guard_mib": args.guard_mib,
        "results": [],
        "resolution_transitions": 250000,
        "note": "Subprocess/allocator failures are capacity results. Training/evaluation are serialized. No task convergence is inferred.",
    }
    status_path = out / "queue_status.json"

    def probe(capacity, allocator):
        if time.monotonic() - started >= args.max_total_seconds:
            return None
        name = f"{allocator}_{capacity}"
        target = out / name
        # Fill the whole ring, then advance another 256 collections to test rollover.
        transitions = math.ceil((capacity + 2048 * 256) / 2048) * 2048
        status.update(active=name, elapsed_s=time.monotonic() - started)
        save(status_path, status)
        env = dict(
            os.environ,
            XLA_PYTHON_CLIENT_PREALLOCATE="false",
            XLA_PYTHON_CLIENT_MEM_FRACTION="0.88",
        )
        if allocator == "platform":
            env["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
        else:
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
            str(target),
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
            str(args.guard_mib),
            "--max-seconds",
            "480",
            "--preflight-ring-multiplier",
            "1.1",
            "--preflight-workspace-mib",
            "2048",
            "--memory-poll-seconds",
            "0.2",
        ]
        with (out / f"{name}.log").open("w") as stream:
            proc = subprocess.Popen(
                command, stdout=stream, stderr=subprocess.STDOUT, env=env
            )
            while proc.poll() is None:
                status["elapsed_s"] = time.monotonic() - started
                save(status_path, status)
                time.sleep(5)
        detail = (
            json.loads((target / "status.json").read_text())
            if (target / "status.json").exists()
            else {}
        )
        metrics = []
        if (target / "metrics.jsonl").exists():
            metrics = [
                json.loads(line)
                for line in (target / "metrics.jsonl").read_text().splitlines()
            ]
        last = metrics[-1] if metrics else {}
        tail = (
            (target / "train.log").read_text(errors="replace")[-6000:]
            if (target / "train.log").exists()
            else ""
        )
        passed = (
            proc.returncode == 0
            and detail.get("state") == "complete"
            and last.get("transitions", 0) >= transitions
        )
        result = {
            "capacity": capacity,
            "allocator": allocator,
            "passed": passed,
            "exit_code": proc.returncode,
            "state": detail.get("state", "missing_status"),
            "stop_reason": detail.get("stop_reason"),
            "peak_gpu_used_mib": detail.get("peak_gpu_used_mib"),
            "elapsed_s": detail.get("elapsed_s"),
            "transitions": last.get("transitions"),
            "target_transitions": transitions,
            "ring_filled_and_wrapped": passed,
            "throughput": last.get("throughput/env_steps_per_second"),
            "path": str(target),
            "allocation_error": any(
                s in tail.lower()
                for s in ["resource_exhausted", "out of memory", "failed to allocate"]
            ),
            "error_tail": tail[-1800:] if not passed else None,
        }
        status["results"].append(result)
        save(status_path, status)
        return passed

    try:
        # Start where the user specifically asked. Locate the largest passing
        # default-allocator capacity to 250k, with a measured failing upper bound.
        lower, upper = 2000000, None
        for capacity in (4000000, 5000000, 6000000, 7000000, 8000000):
            ok = probe(capacity, "default")
            if ok is None:
                break
            if ok:
                lower = capacity
            else:
                upper = capacity
                break
        while upper is not None and upper - lower > 250000:
            middle = ((upper + lower) // 2 // 250000) * 250000
            ok = probe(middle, "default")
            if ok is None:
                break
            if ok:
                lower = middle
            else:
                upper = middle
        # Platform allocation can release compiler temporaries. Measure its
        # actual speed as well; do not silently trade throughput for capacity.
        if upper is not None and lower < 5000000:
            probe(5000000, "platform")
        passed = [r for r in status["results"] if r["passed"]]
        status.update(
            state="complete",
            active=None,
            elapsed_s=time.monotonic() - started,
            largest_passing_default=lower
            if any(r["passed"] and r["allocator"] == "default" for r in passed)
            else None,
            failing_default_upper_bound=upper,
            largest_passing_tested=max((r["capacity"] for r in passed), default=None),
        )
    except BaseException as exc:
        status.update(
            state="failed",
            error=f"{type(exc).__name__}: {exc}",
            elapsed_s=time.monotonic() - started,
        )
        save(status_path, status)
        raise
    save(status_path, status)
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()

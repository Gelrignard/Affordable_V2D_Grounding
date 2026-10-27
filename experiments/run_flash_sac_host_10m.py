"""Launch the full 10M RAM replay comparison only after GPU validation passes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.run_flash_sac_5m import run


def validate(root, verification_path, profile_path):
    verification = json.loads(Path(verification_path).read_text())
    digest = hashlib.sha256((root / "training/host_replay.py").read_bytes()).hexdigest()
    if (
        not verification.get("passed")
        or verification.get("host_replay_sha256") != digest
    ):
        raise ValueError("Current host replay source has not passed GPU parity tests")
    if len(verification["replay"]) != 4 or len(verification["learner"]) != 2:
        raise ValueError("Incomplete replay/learner verification")
    profile = Path(profile_path)
    status = json.loads((profile / "status.json").read_text())
    provenance = json.loads((profile / "provenance.json").read_text())
    ring = json.loads((profile / "host_replay.json").read_text())["rings"][0]
    config = json.loads((profile / "training_config.json").read_text())
    metrics = [
        json.loads(line)
        for line in (profile / "metrics.jsonl").read_text().splitlines()
    ]
    if status["state"] != "complete" or status.get("stop_reason"):
        raise ValueError("Host capacity profile failed or hit a guard")
    if provenance.get("host_replay_source_sha256") != digest:
        raise ValueError("Capacity profile used a different replay implementation")
    if ring["capacity"] != 10_000_000 or ring["wraps"] < 1:
        raise ValueError("Profile did not fill and wrap a genuine 10M host buffer")
    if config["world_count"] != 2048 or config["replay"]["batch_size"] != 2048:
        raise ValueError("Profile must use the full training world and batch counts")
    window = metrics[-20:]
    speed = statistics.median(m["throughput/env_steps_per_second"] for m in window)
    # 21k/s projects to 4.63h, leaving setup/checkpoint headroom under the 5h cap.
    if len(window) != 20 or not math.isfinite(speed) or speed < 21_000:
        raise ValueError(f"RAM replay throughput {speed:.0f}/s misses the 21k/s gate")
    if any(
        m["sac/actor_skipped_fraction"] or m["sac/critic_skipped_fraction"]
        for m in window
    ):
        raise ValueError("Profile has skipped optimizer updates")
    return {
        "verification_report": str(Path(verification_path).resolve()),
        "profile": str(profile.resolve()),
        "host_replay_sha256": digest,
        "median_last_20_throughput": speed,
        "projected_training_hours": 349999104 / speed / 3600,
        "peak_gpu_used_mib": status["peak_gpu_used_mib"],
        "peak_host_rss_mib": status["peak_host_rss_mib"],
        "host_ring": ring,
        "passed": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "runtime",
        "job",
        "output",
        "comparison-config",
        "verification",
        "profile",
    ):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    report = validate(root, args.verification, args.profile)
    # Place gate evidence outside the queue directory, which must be new.
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".validation.json").write_text(json.dumps(report, indent=2))
    args.replay_capacity = 10_000_000
    args.replay_backend = "host"
    args.max_seconds = 18000
    run(args)


if __name__ == "__main__":
    main()

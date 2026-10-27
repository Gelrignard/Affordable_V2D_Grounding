import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from experiments.run_flash_sac_host_10m import validate


class HostReplayGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "training").mkdir()
        source = self.root / "training/host_replay.py"
        source.write_text("verified test source")
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        self.verification = self.root / "verification.json"
        self.verification.write_text(
            json.dumps(
                {
                    "passed": True,
                    "host_replay_sha256": digest,
                    "replay": [{}] * 4,
                    "learner": [{}] * 2,
                }
            )
        )
        self.profile = self.root / "profile"
        self.profile.mkdir()
        for name, data in {
            "status": {
                "state": "complete",
                "stop_reason": None,
                "peak_gpu_used_mib": 9000,
                "peak_host_rss_mib": 30000,
            },
            "provenance": {"host_replay_source_sha256": digest},
            "host_replay": {"rings": [{"capacity": 10000000, "wraps": 1}]},
            "training_config": {"world_count": 2048, "replay": {"batch_size": 2048}},
        }.items():
            (self.profile / (name + ".json")).write_text(json.dumps(data))
        self.metrics()

    def metrics(self, speed=24000, skips=0):
        row = {
            "throughput/env_steps_per_second": speed,
            "sac/actor_skipped_fraction": skips,
            "sac/critic_skipped_fraction": 0,
        }
        (self.profile / "metrics.jsonl").write_text((json.dumps(row) + "\n") * 20)

    def check(self):
        return validate(self.root, self.verification, self.profile)

    def test_passed_evidence_allows_launch(self):
        self.assertTrue(self.check()["passed"])

    def test_source_edit_invalidates_old_verification(self):
        (self.root / "training/host_replay.py").write_text("changed after validation")
        with self.assertRaisesRegex(ValueError, "source"):
            self.check()

    def test_incomplete_capacity_is_rejected(self):
        (self.profile / "host_replay.json").write_text(
            '{"rings": [{"capacity": 10000000, "wraps": 0}]}'
        )
        with self.assertRaisesRegex(ValueError, "fill and wrap"):
            self.check()

    def test_slow_nonfinite_or_skipped_updates_rejected(self):
        for speed, skips in (
            (20000, 0),
            (float("nan"), 0),
            (float("inf"), 0),
            (24000, 0.1),
        ):
            with self.subTest(speed=speed, skips=skips):
                self.metrics(speed=speed, skips=skips)
                with self.assertRaises(ValueError):
                    self.check()

    def test_guard_stop_is_not_success(self):
        (self.profile / "status.json").write_text(
            '{"state": "failed", "stop_reason": "host_memory_guard"}'
        )
        with self.assertRaisesRegex(ValueError, "guard"):
            self.check()


if __name__ == "__main__":
    unittest.main()

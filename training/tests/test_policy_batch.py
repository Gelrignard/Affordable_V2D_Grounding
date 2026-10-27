"""CPU checks for batch identity, partitioning and budget accounting."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from contextlib import nullcontext
from types import SimpleNamespace
import io

from training import policy_batch as batch


class PolicyBatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def job(self, tier, episode, folder=None):
        path = self.root / (folder or f"jobs/tier{tier}/{episode}/corrected") / "job.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"schema": "v2d_policy_job_v1", "tier": tier,
                                    "episode_index": episode, "variant": "corrected"}))
        return path

    def args(self, *extra):
        return batch.parser().parse_args(["--jobs", str(self.root / "jobs"), "--runtime", str(self.root / "runtime"),
                                         "--output", str(self.root / "runs"), "--deadline-utc", "2099-01-01T00:00:00Z",
                                         "--gpu-hours", "24", *extra])

    def test_shards_are_disjoint_complete_and_stable(self):
        for tier in (3, 1, 2):
            for episode in (25, 3):
                self.job(tier, episode)
        a = batch.make_plan(self.args("--shard-count", "2", "--shard-index", "0"))
        b = batch.make_plan(self.args("--shard-count", "2", "--shard-index", "1"))
        keys = lambda plan: {(j["tier"], j["episode_index"]) for j in plan["jobs"]}
        self.assertFalse(keys(a) & keys(b))
        self.assertEqual(len(keys(a) | keys(b)), 6)
        self.assertEqual([j["episode_index"] for j in a["jobs"]], [3, 3, 3])

    def test_duplicate_or_invalid_identity_is_rejected(self):
        self.job(1, 3)
        self.job(1, 3, "jobs/collision")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            batch.discover_jobs(self.root / "jobs")

    def test_missing_roster_rejected_before_run(self):
        self.job(2, 3)
        with self.assertRaisesRegex(ValueError, "Missing"):
            batch.make_plan(self.args("--tiers", "2", "--episodes", "3", "25", "--require-complete"))

    def test_failed_jobs_remain_charged(self):
        path = self.root / "runs/tier2/episode_000003/ledger.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"jobs": [{"state": "failed", "elapsed_seconds": 10},
                                            {"state": "complete", "elapsed_seconds": 20}]}))
        self.assertEqual(batch.charged_seconds(self.root / "runs"), 30)

    def test_command_has_no_clean_reference_or_comparison(self):
        self.job(2, 3)
        plan = batch.make_plan(self.args())
        command = batch.episode_command(plan, plan["jobs"][0], plan["deadline_utc"])
        self.assertIn("training.policy_pipeline", command)
        self.assertNotIn("--dataset-clean", command)
        self.assertNotIn("--evaluate-bootstrap", command)

    def test_completed_batch_revalidates_after_deadline_without_budget(self):
        self.job(2, 3)
        args = self.args("--deadline-utc", "2000-01-01T00:00:00Z", "--gpu-hours", "0.001")
        output = self.root / "runs/tier2/episode_000003"
        output.mkdir(parents=True)
        (output / "pipeline_status.json").write_text(json.dumps({"state": "complete"}))
        (output / "ledger.json").write_text(json.dumps({"jobs": [{"elapsed_seconds": 50}]}))
        with patch.object(batch, "batch_lock", return_value=nullcontext()), patch.object(
            batch.subprocess, "run", return_value=SimpleNamespace(returncode=0)
        ) as child:
            self.assertEqual(batch.run(args), 0)
        self.assertEqual(child.call_count, 1)  # Child verifies evidence; marker alone is insufficient.
        self.assertEqual(json.loads((self.root / "runs/batch_status.json").read_text())["state"], "complete")

    def test_incomplete_batch_stops_before_child_if_budget_insufficient(self):
        self.job(2, 3)
        with patch.object(batch, "batch_lock", return_value=nullcontext()), patch.object(batch.subprocess, "run") as child:
            self.assertEqual(batch.run(self.args("--gpu-hours", "0.001")), 2)
        child.assert_not_called()

    def test_failed_child_stops_shard_without_running_next_episode(self):
        self.job(2, 3)
        self.job(2, 25)
        with patch.object(batch, "batch_lock", return_value=nullcontext()), patch.object(
            batch.subprocess, "run", return_value=SimpleNamespace(returncode=7)
        ) as child:
            self.assertEqual(batch.run(self.args()), 7)
        self.assertEqual(child.call_count, 1)
        status = json.loads((self.root / "runs/batch_status.json").read_text())
        self.assertEqual(status["state"], "failed")
        self.assertEqual(len(status["episodes"]), 1)

    def test_prepare_all_does_not_silently_drop_tier_three_bundle(self):
        from training.__main__ import main
        (self.root / "manifest.json").write_text(json.dumps({"entries": [
            {"tier": 1, "episode_index": 3}, {"tier": 2, "episode_index": 3},
            {"tier": 3, "episode_index": 3}]}))
        with patch("training.prepare.prepare_episode", return_value={"source_frames": 100}) as prepare, \
             patch("sys.stdout", new_callable=io.StringIO):
            main(["prepare", "--bundle", str(self.root), "--urdf", "robot.urdf",
                  "--output", str(self.root / "jobs"), "--all"])
        self.assertEqual([call.args[1] for call in prepare.call_args_list], [1, 2, 3])


if __name__ == "__main__":
    unittest.main()

import json
import tempfile
import unittest
from pathlib import Path

from training.cloud_status import snapshot


class CloudStatusTests(unittest.TestCase):
    def test_new_queue_does_not_reuse_completed_pilot_status(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            output = root / "release/outputs/full/runs/example"
            output.mkdir(parents=True)
            logs = root / "logs"
            logs.mkdir()
            (logs / "pilot_exit_code.txt").write_text("0")
            (root / "monitor_config.json").write_text(
                json.dumps(
                    {
                        "plan": "release/outputs/full/plan.json",
                        "log_prefix": "full",
                        "script": "run_full.sh",
                    }
                )
            )
            (output.parent.parent / "plan.json").write_text(
                json.dumps(
                    {
                        "entries": [{"output": "runs/example"}],
                        "config": {"total_transitions": 350000000},
                    }
                )
            )
            (output / "metrics.jsonl").write_text(
                '{"transitions": 20000, "curriculum_stage": 1, "assistance_scale": 0.75}\n'
            )
            result = snapshot(root, alive=True)
            self.assertEqual(result["state"], "running")
            self.assertEqual(result["queue"], "full")
            self.assertEqual(result["runs"][0]["metrics"]["assistance_scale"], 0.75)

    def test_partial_metric_and_unexpected_process_exit(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            plan = root / "project/outputs/policy_pilot_two"
            output = plan / "runs/example"
            output.mkdir(parents=True)
            (plan / "plan.json").write_text(
                json.dumps({"entries": [{"output": "runs/example"}]})
            )
            (output / "metrics.jsonl").write_text(
                '{"transitions": 128, "model_ready": true}\n{"transitions":'
            )
            result = snapshot(root, alive=False)
            self.assertEqual(result["state"], "stopped_without_exit_record")
            self.assertEqual(result["runs"][0]["metrics"]["transitions"], 128)

    def test_zero_exit_requires_evaluations_and_failures_are_not_hidden(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            plan = root / "project/outputs/policy_pilot_two"
            output = plan / "runs/example/evaluation"
            output.mkdir(parents=True)
            logs = root / "logs"
            logs.mkdir()
            (plan / "plan.json").write_text(
                json.dumps({"entries": [{"output": "runs/example"}]})
            )
            (logs / "pilot_exit_code.txt").write_text("0\n")
            self.assertEqual(snapshot(root, alive=False)["state"], "failed")
            (output / "evaluation.json").write_text(
                json.dumps({"valid_worlds": [True], "ever_terminated": [True]})
            )
            result = snapshot(root, alive=False)
            self.assertEqual(result["state"], "complete")
            self.assertTrue(result["runs"][0]["evaluation"]["ever_terminated"][0])
            (logs / "pilot_exit_code.txt").write_text("1\n")
            self.assertEqual(snapshot(root, alive=False)["state"], "failed")


if __name__ == "__main__":
    unittest.main()

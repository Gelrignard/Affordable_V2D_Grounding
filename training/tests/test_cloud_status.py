import json
import tempfile
import unittest
from pathlib import Path

from training.cloud_status import diagnostic_snapshot, snapshot


class CloudStatusTests(unittest.TestCase):
    def test_native_leaderboard_report_is_exposed_separately(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            run = root / "native/original"
            run.mkdir(parents=True)
            (root / "native/queue_status.json").write_text(
                json.dumps(
                    {
                        "state": "complete",
                        "runs": [{"variant": "original", "state": "complete"}],
                    }
                )
            )
            board = {
                "rows": [
                    {
                        "add_auc": 0.5,
                        "spider_sr": 0.25,
                        "maniptrans_sr": 0,
                        "mppe_cm": 4.0,
                    }
                ]
            }
            (run / "leaderboard.json").write_text(json.dumps(board))
            result = diagnostic_snapshot(
                root, "native/queue_status.json", now=0, alive=False
            )
            self.assertEqual(result["runs"][0]["leaderboard"], board)

    def test_native_diagnostics_report_failure_and_training_progress_separately(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            run = root / "native/original"
            run.mkdir(parents=True)
            path = root / "native/queue_status.json"
            queue = {
                "state": "running",
                "runs": [{"variant": "original", "state": "training"}],
            }
            path.write_text(json.dumps(queue))
            (run / "metrics.jsonl").write_text(
                '{"transitions": 2048, "curriculum/voc_scale": 1.0}\n'
            )
            result = diagnostic_snapshot(
                root, "native/queue_status.json", now=0, alive=True
            )
            self.assertEqual(result["runs"][0]["transitions"], 2048)
            self.assertEqual(result["runs"][0]["assistance_scale"], 1.0)
            self.assertEqual(result["state"], "running")
            dead = diagnostic_snapshot(
                root, "native/queue_status.json", now=0, alive=False
            )
            self.assertEqual(dead["state"], "stopped_without_exit_record")
            queue["state"] = "failed"
            queue["error"] = "evaluation failed"
            path.write_text(json.dumps(queue))
            failed = diagnostic_snapshot(
                root, "native/queue_status.json", now=0, alive=False
            )
            self.assertEqual(failed["state"], "failed")
            self.assertEqual(failed["error"], "evaluation failed")

    def test_native_status_path_must_stay_within_cloud_root(self):
        with (
            tempfile.TemporaryDirectory() as folder,
            self.assertRaisesRegex(ValueError, "escapes"),
        ):
            diagnostic_snapshot(Path(folder), "../outside.json", now=0, alive=False)

    def test_intentional_pause_requires_matching_checkpoint_and_interrupted_run(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            plan = root / "project/outputs/policy_pilot_two"
            output = plan / "runs/example"
            output.mkdir(parents=True)
            logs = root / "logs"
            logs.mkdir()
            (plan / "plan.json").write_text(
                json.dumps({"entries": [{"output": "runs/example"}]})
            )
            (logs / "pilot_exit_code.txt").write_text("143")
            checkpoint = output / "latest.pt"
            checkpoint.write_bytes(b"checkpoint-preserved")
            (output / "report.json").write_text(
                json.dumps({"stop_reason": "interrupted", "transitions": 128})
            )
            pause = {
                "state": "paused_by_user",
                "checkpoint": str(checkpoint),
                "transitions": 128,
            }
            (logs / "pilot_pause.json").write_text(json.dumps(pause))
            status = snapshot(root, alive=False)
            self.assertEqual(status["state"], "paused_by_user")
            self.assertEqual(status["runs"][0]["phase"], "paused")
            pause["transitions"] = 64
            (logs / "pilot_pause.json").write_text(json.dumps(pause))
            self.assertEqual(snapshot(root, alive=False)["state"], "failed")

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

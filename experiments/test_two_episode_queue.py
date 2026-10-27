"""CPU-only checks for the two-episode queue's budget and provenance gates."""
import datetime as dt
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

from experiments import two_episode_queue as queue


def plan(root):
    return {"schema": "v2d_two_episode_trial_v1", "root": str(root), "runtime": "/runtime",
            "source_commit": "a" * 40, "source_archive_sha256": "b" * 64,
            "source_file_sha256": {name: "c" * 64 for name in queue.REQUIRED_SOURCES},
            "deadline_utc": (dt.datetime.now(dt.timezone.utc) + dt.timedelta(days=1)).isoformat(),
            "wait_queue": "/prior_queue", "total_budget_seconds": 23400,
            "candidate_training_budget_seconds": 10800, "maximum_candidates": 2,
            "bootstrap_transitions": queue.BOOTSTRAP_STEPS, "bootstrap_wall_seconds": 9000,
            "refinement_transitions": 60_000_000, "refinement_wall_seconds": 1800,
            "episodes": [{"episode_index": ep, "tier": 1, "candidate_id": f"episode_{ep:06d}",
                          "job": str(root / f"job{ep}"), "sequence_id": f"sequence{ep}",
                          "source_frames": frames, "bootstrap_config": str(root / f"config{ep}.json"),
                          "bootstrap_config_sha256": "d" * 64,
                          "bootstrap_output": str(root / f"ep{ep}_bootstrap"),
                          "refinement_output": str(root / f"ep{ep}_refinement")}
                         for ep, frames in [(16, 477), (18, 858)]]}


class TwoEpisodeQueueTests(unittest.TestCase):
    def test_plan_rejects_duplicate_existing_paths_and_wrong_episodes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            p = plan(root)
            self.assertEqual(queue.validate_plan(p)["maximum_candidates"], 2)
            p["episodes"][1]["bootstrap_output"] = p["episodes"][0]["bootstrap_output"]
            with self.assertRaisesRegex(ValueError, "overlap"):
                queue.validate_plan(p)
            p = plan(root)
            Path(p["episodes"][0]["bootstrap_output"]).mkdir()
            with self.assertRaisesRegex(ValueError, "already"):
                queue.validate_plan(p)
            p["episodes"][0]["episode_index"] = 5
            with self.assertRaisesRegex(ValueError, "16 and 18"):
                queue.validate_plan(p)

    def test_budget_reserves_grace_evaluation_and_cumulative_training(self):
        p = plan(Path("unused"))
        ledger = {"budget_policy": queue.budget_policy(p), "jobs": []}
        self.assertEqual(queue.training_caps(p, ledger, "episode_000016", 9000), (9000, 9060))
        ledger["jobs"] = [{"candidate": "episode_000016", "kind": "training", "state": "complete", "elapsed_seconds": 9000}]
        self.assertEqual(queue.training_caps(p, ledger, "episode_000016", 1800), (1740, 1800))
        deadline = queue.parse_deadline(p["deadline_utc"])
        self.assertEqual(queue.training_caps(p, ledger, "episode_000016", 1800, now=deadline - 800), (140, 200))
        with self.assertRaisesRegex(RuntimeError, "Insufficient"):
            queue.training_caps(p, ledger, "episode_000016", 1800, now=deadline - 650)

    def test_every_guard_receives_shared_ledger_candidate_and_custom_caps(self):
        p = plan(Path("/trial"))
        with patch.object(queue.subprocess, "run") as run:
            for phase in ("bootstrap", "refinement", "evaluation"):
                queue.guarded(p, "episode_000016", phase, "training" if phase != "evaluation" else phase, 600, ["python", "child.py"])
        for call in run.call_args_list:
            command = call.args[0]
            for key, value in [("--candidate", "episode_000016"), ("--total-budget-seconds", "23400"),
                               ("--candidate-training-budget-seconds", "10800"), ("--maximum-candidates", "2")]:
                self.assertEqual(command[command.index(key) + 1], value)
            self.assertEqual(Path(command[command.index("--ledger") + 1]).name, "ledger.json")
            self.assertTrue(call.kwargs["check"])

    def test_source_sentinel_and_live_file_hashes_are_checked(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            p = plan(root)
            (root / "SOURCE_COMMIT").write_text(p["source_commit"] + "\n")
            for name in queue.REQUIRED_SOURCES:
                source = root / name
                source.parent.mkdir(parents=True, exist_ok=True)
                source.write_text(name)
                p["source_file_sha256"][name] = queue.file_hash(source)
            with patch.object(queue, "SOURCE", root):
                queue.verify_sources(p)
                (root / "experiments/overnight_policy.py").write_text("changed")
                with self.assertRaisesRegex(ValueError, "Source hash mismatch"):
                    queue.verify_sources(p)

    def test_input_manifest_allows_only_asset_rebasing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            e = plan(root)["episodes"][0]
            job = Path(e["job"])
            (job / "assets").mkdir(parents=True)
            files = {}
            for asset in ["object.obj", "object.urdf", "support.usda"]:
                path = job / "assets" / asset
                path.write_text(asset)
                files["assets/" + asset] = queue.file_hash(path)
            source = {"sequence_id": e["sequence_id"], "robot_joint_positions": [[0.0]] * e["source_frames"],
                      "object_body_names": ["object"], "object_mesh_paths": ["old/object.obj"],
                      "object_urdf_paths": ["old/object.urdf"]}
            pq.write_table(pa.Table.from_pylist([source]), job / "motion.parquet")
            files["motion.parquet"] = queue.file_hash(job / "motion.parquet")
            metadata = {key: e[key] for key in ["episode_index", "tier", "sequence_id", "source_frames"]}
            (job / "job.json").write_text(json.dumps({**metadata, "variant": "corrected", "files": files}))
            # Native materialization uses Hive-shaped folders, but this is one
            # file: dataset discovery would merge string/dictionary sequence IDs.
            runtime = root / f"sequence_id={e['sequence_id']}" / "robot_name=g1" / "motion.parquet"
            runtime.parent.mkdir(parents=True)
            rebased = {**source, "object_mesh_paths": [str(job / "assets/object.obj")],
                       "object_urdf_paths": [str(job / "assets/object.urdf")]}
            pq.write_table(pa.Table.from_pylist([rebased]), runtime)
            config = Path(e["bootstrap_config"])
            config.write_text(json.dumps({"task": {"parquet": str(runtime)}}))
            e["bootstrap_config_sha256"] = queue.file_hash(config)
            self.assertEqual(queue.verify_episode_inputs(e)["runtime_motion"], str(runtime.resolve()))
            rebased["robot_joint_positions"] = [[1.0]] * e["source_frames"]
            pq.write_table(pa.Table.from_pylist([rebased]), runtime)
            with self.assertRaisesRegex(ValueError, "changes source trajectory"):
                queue.verify_episode_inputs(e)

    def test_bootstrap_shortfall_and_config_drift_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = {"training": {"total_environment_steps": queue.BOOTSTRAP_STEPS}}
            (root / "status.json").write_text(json.dumps({"state": "complete", "exit_code": 0}))
            (root / "resolved_config.json").write_text(json.dumps(config))
            state, actor = root / "state.safetensors", root / "policy.safetensors"
            record = {"state": str(state), "policy": str(actor), "environment_steps": queue.BOOTSTRAP_STEPS - 2048}
            (root / "latest_checkpoint.json").write_text(json.dumps(record))
            with patch.object(queue, "checkpoint_pair", return_value=(state, actor, {"environment_steps": str(record["environment_steps"])})):
                with self.assertRaisesRegex(RuntimeError, "required 349999104"):
                    queue.completed_checkpoint(root, config, required_steps=queue.BOOTSTRAP_STEPS)
            (root / "resolved_config.json").write_text("{}")
            with self.assertRaisesRegex(RuntimeError, "configuration|config differs"):
                queue.completed_checkpoint(root, config)


if __name__ == "__main__":
    unittest.main()

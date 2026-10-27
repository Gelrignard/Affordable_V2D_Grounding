"""CPU-only checks for the two-episode queue's budget and provenance gates."""
import datetime as dt
from contextlib import nullcontext
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

from experiments import two_episode_queue as queue
from experiments import prepare_two_episode_policy as prepare


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


def tier2_plan(root):
    value = plan(root)
    value.update(prepare.trial_settings(2, [3, 25]))
    for item, episode, frames in zip(value["episodes"], [3, 25], [781, 859]):
        item.update(episode_index=episode, tier=2, candidate_id=f"tier2_episode_{episode:06d}",
                    source_frames=frames, sequence_id=f"sequence{episode}")
    return value


class TwoEpisodeQueueTests(unittest.TestCase):
    def test_preparer_preserves_legacy_scope_and_explicitly_scopes_tier2(self):
        self.assertEqual(prepare.trial_settings(1, [16, 18]), {"schema": "v2d_two_episode_trial_v1"})
        self.assertIs(prepare.trial_settings(2, [3, 25])["evaluate_bootstrap"], False)
        for tier, episodes in [(1, [3, 25]), (2, [3, 3]), (2, [-1, 25]),
                               (2, [3, 30]), (2, [True, 25]), (2, [3]), (3, [3, 25])]:
            with self.subTest(tier=tier, episodes=episodes), self.assertRaises(ValueError):
                prepare.trial_settings(tier, episodes)

    def test_tier2_rejects_wrong_tier_ids_collisions_and_comparator_evaluation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            valid = tier2_plan(root)
            self.assertEqual(queue.validate_plan(valid)["maximum_candidates"], 2)
            for key, value in [("tier", 1), ("tier", True), ("episode_index", -1),
                               ("episode_index", 30), ("episode_index", 3.0),
                               ("episode_index", True), ("candidate_id", "episode_000003")]:
                bad = copy.deepcopy(valid)
                bad["episodes"][0][key] = value
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    queue.validate_plan(bad)
            for change in ("duplicate_id", "candidate_collision", "output_collision", "evaluate", "missing_flag"):
                bad = copy.deepcopy(valid)
                if change == "duplicate_id":
                    bad["episodes"][1]["episode_index"] = 3
                elif change == "candidate_collision":
                    bad["episodes"][1]["candidate_id"] = bad["episodes"][0]["candidate_id"]
                elif change == "output_collision":
                    bad["episodes"][1]["bootstrap_output"] = bad["episodes"][0]["bootstrap_output"]
                elif change == "evaluate":
                    bad["evaluate_bootstrap"] = True
                else:
                    del bad["evaluate_bootstrap"]
                with self.subTest(change=change), self.assertRaises(ValueError):
                    queue.validate_plan(bad)
            legacy = plan(root)
            legacy["evaluate_bootstrap"] = False
            with self.assertRaisesRegex(ValueError, "Legacy"):
                queue.validate_plan(legacy)

    def test_preparer_job_tier_defaults_to_one_and_checks_declared_tier(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = {"schema": "v2d_policy_job_v1", "tier": 2, "episode_index": 3,
                    "variant": "corrected", "runtime_commit": prepare.RUNTIME_COMMIT, "files": {}}
            (root / "job.json").write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "identity"):
                prepare.validate_job(root, 3)
            self.assertEqual(prepare.validate_job(root, 3, 2)[1], data)

    def test_actual_queue_path_skips_only_tier2_bootstrap_evaluation(self):
        for tier in (1, 2):
            with self.subTest(tier=tier), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                p = plan(root) if tier == 1 else tier2_plan(root)
                registry, calls = {}, []
                for episode in p["episodes"]:
                    Path(episode["bootstrap_config"]).write_text("{}")
                plan_path = root / "plan.json"
                plan_path.write_text(json.dumps(p))

                def saved_pair(output, steps, config):
                    output.mkdir(parents=True, exist_ok=True)
                    state, actor = output / f"state_{steps}.safetensors", output / f"policy_{steps}.safetensors"
                    state.write_bytes(f"state {output} {steps}".encode())
                    actor.write_bytes(f"actor {output} {steps}".encode())
                    metadata = {"environment_steps": str(steps), "resolved_config_json": json.dumps(config)}
                    registry[str(state.resolve())] = (state.resolve(), actor.resolve(), metadata)
                    return state, actor

                def fake_guard(plan_value, candidate, name, kind, seconds, command):
                    calls.append((candidate, name, kind, command))
                    output = Path(command[command.index("--output") + 1])
                    episode = next(e for e in p["episodes"] if e["candidate_id"] == candidate)
                    if kind == "training":
                        config = queue.read_json(root / (name + "_planned_config.json"))
                        steps = config["training"]["total_environment_steps"]
                        state, actor = saved_pair(output, steps, config)
                        queue.write_json(output / "status.json", {"state": "complete", "exit_code": 0})
                        queue.write_json(output / "resolved_config.json", config)
                        queue.write_json(output / "latest_checkpoint.json", {
                            "state": str(state), "policy": str(actor), "environment_steps": steps})
                        if "--fresh-config" in command:
                            initial, _ = saved_pair(output, 0, config)
                            queue.write_json(output / "initial_checkpoint.json", {"state": str(initial)})
                    else:
                        actor = Path(command[command.index("--checkpoint") + 1])
                        steps = int(actor.stem.split("_")[-1])
                        output.mkdir()
                        report = {"checkpoint_sha256": queue.file_hash(actor), "environment_steps": steps,
                                  "world_count": 16, "start_frame": 0, "reset_mode": "explicit", "settling_steps": 0,
                                  "step_count": episode["source_frames"], "non_finite_worlds": [],
                                  "evaluation_contract": {"frame_zero": True, "explicit_reset": True,
                                      "within_attempt_resets": 0, "voc_scale": 0.0, "reset_voc_scale": 0.0,
                                      "settling_steps": 0, "training_termination_disabled_during_rollout": True},
                                  "leaderboard": {"scores": {"add_auc": .5, "mppe_cm": 5.0,
                                                               "spider_sr": .5, "maniptrans_sr": .25}},
                                  "phase_metrics": {"test": "complete"}}
                        queue.write_json(output / "evaluation.json", report)

                with patch.object(queue, "exclusive_lock", side_effect=lambda _: nullcontext()), \
                     patch.object(queue, "verify_sources"), \
                     patch.object(queue, "verify_episode_inputs", return_value={"verified": True}), \
                     patch.object(queue, "build_fresh_config", side_effect=lambda config, **kw: {
                         "training": {"total_environment_steps": kw["end"]}}), \
                     patch.object(queue, "build_config", side_effect=lambda config, **kw: {
                         "training": {"total_environment_steps": kw["end"]}}), \
                     patch.object(queue, "checkpoint_pair", side_effect=lambda path: registry[str(Path(path).resolve())]), \
                     patch.object(queue, "guarded", side_effect=fake_guard):
                    queue.run(plan_path)
                per_episode = ["training", "evaluation", "training", "evaluation"] if tier == 1 else ["training", "training", "evaluation"]
                self.assertEqual([c[2] for c in calls], per_episode * 2)
                for episode in p["episodes"]:
                    candidate = episode["candidate_id"]
                    manifest = queue.read_json(root / (candidate + "_refinement_manifest.json"))
                    self.assertEqual(manifest["parent_policy_sha256"], queue.file_hash(manifest["parent_policy"]))
                    self.assertEqual(manifest["parent_state_sha256"], queue.file_hash(manifest["parent_state"]))
                    suffix = "_comparison.json" if tier == 1 else "_result.json"
                    result = queue.read_json(root / (candidate + suffix))
                    if tier == 1:
                        self.assertIn("parent", result)
                        self.assertIn("parent_evaluation_sha256", manifest)
                    else:
                        self.assertNotIn("parent", result)
                        self.assertNotIn("parent_evaluation", manifest)
                        self.assertIs(result["bootstrap_evaluation_performed"], False)
                        self.assertEqual(result["published_baseline"], p["published_baseline"])
                        self.assertFalse(Path(episode["bootstrap_output"] + "_evaluation").exists())
                self.assertEqual(queue.read_json(root / "queue_status.json")["state"], "awaiting_agent_diagnosis")

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

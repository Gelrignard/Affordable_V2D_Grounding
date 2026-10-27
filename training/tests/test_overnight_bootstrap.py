"""Selection rejects unusable sources and ranks measured quality over recency."""

import copy
import json
from pathlib import Path
import struct
import tempfile
import unittest

from experiments.overnight_bootstrap import eligible_sources, verify_parent_repeat


def source_run(root, name, step, auc, *, mppe=5.0, assistance=0.0, missing_state=False):
    run = root / name
    checkpoint_dir = run / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    policy = checkpoint_dir / f"policy_{step}.safetensors"
    state = checkpoint_dir / f"state_{step}.safetensors"
    fields = {"algorithm": "flash_sac", "checkpoint_set_id": f"pair_{name}_{step}",
              "environment_steps": str(step), "policy_schema_json": "{}", "critic_schema_json": "{}"}
    for path, kind in ((policy, "policy"), (state, "training_state")):
        if missing_state and kind == "training_state":
            continue
        metadata = {"format": "learner_state", "version": "1", "manifest": "[]",
                    **{"user." + key: value for key, value in {**fields, "checkpoint_kind": kind}.items()}}
        header = json.dumps({"__metadata__": metadata}).encode()
        path.write_bytes(struct.pack("<Q", len(header)) + header)
    report = run / "evaluations" / f"{step}_unassisted" / "evaluation.json"
    report.parent.mkdir(parents=True)
    report.write_text(json.dumps({
        "checkpoint": str(policy), "environment_steps": step,
        "non_finite_worlds": [], "world_count": 16,
        "start_frame": 0, "reset_mode": "explicit",
        "probe_assistance": assistance, "settling_steps": 0,
        "leaderboard": {"scores": {"add_auc": auc, "mppe_cm": mppe,
                                     "spider_sr": .95, "maniptrans_sr": .80}},
    }))
    return run, policy


class OvernightBootstrapTests(unittest.TestCase):
    def test_rejects_missing_resumable_state_even_with_high_score(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            broken, _ = source_run(root, "missing_state", 200_001_536, .99, missing_state=True)
            valid, policy = source_run(root, "valid", 100_001_792, .80)
            candidates, rejected = eligible_sources([broken, valid])
            self.assertEqual([item["policy"] for item in candidates], [str(policy)])
            self.assertEqual(len(rejected), 1)
            self.assertIn("state_200001536.safetensors", rejected[0]["reason"])
            with self.assertRaisesRegex(ValueError, "No evaluated compact state"):
                eligible_sources([broken])

    def test_rejects_assisted_evaluation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            assisted, _ = source_run(root, "assisted", 200_001_536, .99, assistance=.05)
            valid, policy = source_run(root, "unassisted", 100_001_792, .80)
            candidates, rejected = eligible_sources([assisted, valid])
            self.assertEqual(candidates[0]["policy"], str(policy))
            self.assertEqual(len(candidates), 1)
            self.assertIn("unassisted", rejected[0]["reason"])

    def test_selects_best_auc_instead_of_newest_step(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            older, old_policy = source_run(root, "older", 100_001_792, .93)
            newer, _ = source_run(root, "newer", 349_999_104, .81)
            candidates, rejected = eligible_sources([newer, older])
            self.assertEqual(candidates[0]["policy"], str(old_policy))
            self.assertEqual(candidates[0]["environment_steps"], 100_001_792)
            self.assertEqual(rejected, [])

    def test_auc_ties_prefer_lower_mppe(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            older, old_policy = source_run(root, "older", 100_001_792, .93, mppe=3)
            newer, _ = source_run(root, "newer", 349_999_104, .93, mppe=7)
            candidates, _ = eligible_sources([newer, older])
            self.assertEqual(candidates[0]["policy"], str(old_policy))


class ParentRepeatTests(unittest.TestCase):
    def report(self):
        return {
            "checkpoint_sha256": "a" * 64, "world_count": 16, "step_count": 200,
            "start_frame": 0, "reset_mode": "explicit", "settling_steps": 0,
            "sim_control_fps": 50, "sim_physics_fps": 200,
            "parquet": "/run/input/motion.parquet", "non_finite_worlds": [],
            "leaderboard": {"scores": {"add_auc": .35575, "mppe_cm": 6.0,
                                         "spider_sr": .9375, "maniptrans_sr": .8125}},
        }

    def test_allows_small_repeat_drift_with_identical_contract(self):
        expected = self.report()
        actual = copy.deepcopy(expected)
        actual["leaderboard"]["scores"].update(add_auc=.35295, mppe_cm=6.088)
        before = copy.deepcopy((expected, actual))
        drift = verify_parent_repeat(expected, actual)
        self.assertIsInstance(drift, dict)
        self.assertEqual((expected, actual), before)

    def test_rejects_changed_checkpoint_reference_and_rollout_contract(self):
        changes = {
            "checkpoint_sha256": "b" * 64,
            "parquet": "/different/input/motion.parquet",
            "world_count": 8,
            "step_count": 199,
            "start_frame": 1,
            "reset_mode": "random",
            "settling_steps": 1,
            "sim_control_fps": 30,
            "sim_physics_fps": 100,
        }
        for key, value in changes.items():
            with self.subTest(key=key):
                expected = self.report()
                actual = copy.deepcopy(expected)
                actual[key] = value
                with self.assertRaises(RuntimeError):
                    verify_parent_repeat(expected, actual)

    def test_rejects_non_finite_rollout_worlds(self):
        expected = self.report()
        actual = copy.deepcopy(expected)
        actual["non_finite_worlds"] = [3]
        with self.assertRaises(RuntimeError):
            verify_parent_repeat(expected, actual)

    def test_rejects_large_repeat_drift_in_either_direction(self):
        for metric, delta in (("add_auc", .0101), ("mppe_cm", .5001)):
            for sign in (-1, 1):
                with self.subTest(metric=metric, sign=sign):
                    expected = self.report()
                    actual = copy.deepcopy(expected)
                    actual["leaderboard"]["scores"][metric] += sign * delta
                    with self.assertRaises(RuntimeError):
                        verify_parent_repeat(expected, actual)

    def test_requires_exact_success_rates(self):
        for metric in ("spider_sr", "maniptrans_sr"):
            with self.subTest(metric=metric):
                expected = self.report()
                actual = copy.deepcopy(expected)
                actual["leaderboard"]["scores"][metric] -= 1 / 16
                with self.assertRaises(RuntimeError):
                    verify_parent_repeat(expected, actual)

    def test_rejects_non_finite_scores(self):
        for metric in ("add_auc", "mppe_cm", "spider_sr", "maniptrans_sr"):
            for value in (float("nan"), float("inf")):
                with self.subTest(metric=metric, value=value):
                    expected = self.report()
                    actual = copy.deepcopy(expected)
                    actual["leaderboard"]["scores"][metric] = value
                    with self.assertRaises(RuntimeError):
                        verify_parent_repeat(expected, actual)


if __name__ == "__main__":
    unittest.main()

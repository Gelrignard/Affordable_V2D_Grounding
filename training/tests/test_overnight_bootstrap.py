"""Selection rejects unusable sources and ranks measured quality over recency."""

import json
from pathlib import Path
import struct
import tempfile
import unittest

from experiments.overnight_bootstrap import eligible_sources


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


if __name__ == "__main__":
    unittest.main()

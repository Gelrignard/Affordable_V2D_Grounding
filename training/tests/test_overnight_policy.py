"""CPU checks for restart accounting and preservation of policy contracts."""

import bisect
import copy
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock

from experiments.overnight_policy import (
    OPTIMIZER_HORIZON,
    build_config,
    checkpoint_pair,
    checkpoint_stage_metadata,
    deadline_timestamp,
    signal_child,
    transition_budget,
)


def saved_config():
    objective = {name + "_weight": value for name, value in {
        "termination": -300, "action_rate_l2": -.0001, "action_l2": -.000001,
        "soft_joint_limit": -.001, "pelvis_position": 1, "pelvis_orientation": 1,
        "body_joint_position": 5, "object_position": 1, "progress": 1,
        "palm_position": 1, "palm_orientation": 1, "force_closure": 5,
    }.items()}
    return {
        "task": {"parquet": "original.parquet", "source_frame_playback": True},
        "scene": {"world_count": 2048},
        "env": {"objective": objective, "termination": {"object_orientation_threshold": 1.5},
                "reset": {"reset_freeze_steps": 50, "shoulder_spread": .2},
                "action": {"input_mode": "raw", "residual_scale": .15},
                "observation": {"marker": "same"}},
        "training": {"network": {"actor_hidden_dim": 256}, "replay": {"n_step": 7, "minimum_size": 100_000},
                     "exploration": {"warmup_action": "uniform", "target_sigma": .15,
                                     "maximum_noise_repeat": 16},
                     "optimization": {"learning_rate_peak": .0003}, "checkpoint": {}},
        "logging": {},
    }


class OvernightPolicyTests(unittest.TestCase):
    def build(self, phase="refine"):
        return build_config(saved_config(), state=Path("state_10000384.safetensors"),
                            output=Path("new-output"), start=10_000_384, end=70_000_128,
                            phase=phase, checkpoint_interval=10_000_000)

    def test_budget_counts_additional_steps_and_rounds_down(self):
        end, additional = transition_budget(10_000_384, 60_000_000)
        self.assertEqual(additional, 59_998_208)
        self.assertEqual(end - 10_000_384, additional)
        self.assertLessEqual(additional, 60_000_000)
        self.assertEqual(end % 2048, 0)
        with self.assertRaises(ValueError):
            transition_budget(1, 60_000_000)

    def test_saved_contracts_unchanged_and_replay_fresh(self):
        saved = saved_config()
        original = copy.deepcopy(saved)
        cfg = build_config(saved, state=Path("state"), output=Path("output"), start=0,
                           end=60_000_000, phase="refine", checkpoint_interval=10_000_000)
        self.assertEqual(saved, original)
        for name in ("action", "observation"):
            self.assertEqual(cfg["env"][name], saved["env"][name])
        self.assertEqual(cfg["task"], saved["task"])
        self.assertEqual(cfg["training"]["network"], saved["training"]["network"])
        self.assertEqual(cfg["training"]["optimization"]["learning_rate_schedule_updates"], OPTIMIZER_HORIZON)
        self.assertEqual(cfg["training"]["replay"]["capacity"], 5_000_000)
        self.assertEqual(cfg["training"]["replay"]["minimum_size"], saved["training"]["replay"]["minimum_size"])
        self.assertEqual(cfg["training"]["exploration"],
                         {**saved["training"]["exploration"], "warmup_action": "policy"})
        self.assertIsNone(cfg["training"]["checkpoint"]["load_replay_path"])
        self.assertTrue(cfg["training"]["checkpoint"]["load_optimizer"])

    def test_reward_ramp_is_relative_to_resume_and_no_global_assistance(self):
        cfg = self.build()
        curriculum = cfg["curriculum"]
        index = min(bisect.bisect_right(curriculum["thresholds"], 10_000_384), len(curriculum["stages"]) - 1)
        self.assertEqual(index, 0)
        initial = curriculum["stages"][0]["objective_weights"]
        final = curriculum["stages"][-1]["objective_weights"]
        self.assertEqual((initial["object_position"], initial["force_closure"], initial["object_orientation"]), (1, 5, 0))
        self.assertEqual((final["object_position"], final["force_closure"], final["object_orientation"]), (3, 1, 2))
        self.assertAlmostEqual(sum(initial.values()), sum(final.values()))
        self.assertTrue(all(stage["voc_scale"] == 0 for stage in curriculum["stages"]))
        self.assertEqual(cfg["env"]["reset"]["reset_voc_scale"], 1)
        self.assertEqual(cfg["env"]["reset"]["reset_freeze_steps"], 50)
        self.assertEqual(cfg["env"]["voc_scale"], 0)
        self.assertEqual(cfg["env"]["reset"], cfg["reset"])

    def test_consolidation_uses_unassisted_exact_start(self):
        cfg = self.build("consolidate")
        reset = cfg["env"]["reset"]
        self.assertEqual(reset["immediate_first_frame_probability"], 1)
        self.assertEqual(reset["reset_freeze_steps"], 0)
        self.assertEqual(reset["reset_voc_scale"], 0)
        self.assertEqual(reset["shoulder_spread"], 0)
        self.assertEqual(cfg["env"]["termination"]["reset_freeze_steps"], 0)

    def test_continuation_does_not_restart_reward_ramp(self):
        previous = self.build()
        continued = build_config(previous, state=Path("state"), output=Path("next"),
                                 start=40_001_536, end=60_000_256, phase="refine",
                                 checkpoint_interval=10_000_000)
        self.assertEqual(continued["overnight"]["refinement_start_steps"], 10_000_384)
        self.assertEqual(continued["env"]["objective"]["object_orientation_weight"], 2)
        self.assertEqual(continued["env"]["objective"]["force_closure_weight"], 1)

    def test_deadline_requires_offset(self):
        self.assertEqual(deadline_timestamp("2026-10-27T13:30:00Z"), deadline_timestamp("2026-10-27T09:30:00-04:00"))
        with self.assertRaises(ValueError):
            deadline_timestamp("2026-10-27T09:30:00")

    def test_outer_guard_uses_direct_child_signal(self):
        with mock.patch("experiments.overnight_policy.os.kill") as direct, \
             mock.patch("experiments.overnight_policy.os.killpg", create=True) as grouped:
            signal_child(123, 15, shared_group=True)
            direct.assert_called_once_with(123, 15)
            grouped.assert_not_called()
            direct.reset_mock()
            signal_child(123, 15, shared_group=False)
            grouped.assert_called_once_with(123, 15)
            direct.assert_not_called()

    def test_graceful_checkpoint_contains_native_evaluator_stage_metadata(self):
        cfg = self.build()
        metadata = checkpoint_stage_metadata(cfg["curriculum"], 40_001_536)
        self.assertEqual(metadata["curriculum_stage"], 10)
        self.assertEqual(metadata["voc_scale"], 0)
        self.assertEqual(metadata["immediate_first_frame_probability"], .5)

    def test_mismatched_policy_state_pair_rejected_without_tensor_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / name for name in ("state_2048.safetensors", "policy_2048.safetensors")]
            common = {"algorithm": "flash_sac", "environment_steps": "2048", "checkpoint_set_id": "same",
                      "policy_schema_json": "{}", "critic_schema_json": "{}"}
            def envelope(fields):
                return {"__metadata__": {"format": "learner_state", "version": "1", "manifest": "[]",
                                         **{"user." + name: value for name, value in fields.items()}}}
            for path, kind in zip(paths, ("training_state", "policy")):
                payload = json.dumps(envelope({**common, "checkpoint_kind": kind})).encode()
                path.write_bytes(struct.pack("<Q", len(payload)) + payload)
            state, policy, _ = checkpoint_pair(paths[1])
            self.assertEqual((state, policy), tuple(paths))
            common["checkpoint_set_id"] = "different"
            payload = json.dumps(envelope({**common, "checkpoint_kind": "policy"})).encode()
            paths[1].write_bytes(struct.pack("<Q", len(payload)) + payload)
            with self.assertRaises(ValueError):
                checkpoint_pair(paths[0])


if __name__ == "__main__":
    unittest.main()

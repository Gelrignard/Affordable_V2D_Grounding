"""CPU checks for restart accounting and preservation of policy contracts."""

import bisect
import copy
import json
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from experiments.overnight_policy import (
    OPTIMIZER_HORIZON,
    build_config,
    build_fresh_config,
    checkpoint_pair,
    checkpoint_stage_metadata,
    deadline_timestamp,
    signal_child,
    save_initial_checkpoint_once,
    source_config,
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
        "training": {"network": {"actor_hidden_dim": 256},
                     "replay": {"capacity": 10_000_000, "n_step": 7, "minimum_size": 100_000},
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
        self.assertEqual(cfg["training"]["replay"]["capacity"], 4_000_000)
        self.assertEqual(saved["training"]["replay"]["capacity"], 10_000_000)
        self.assertEqual(cfg["overnight"]["replay_capacity"], 4_000_000)
        self.assertEqual(cfg["overnight"]["replay_backend"], "gpu")
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


class FreshNativePolicyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)
        self.parquet = self.output / "motion.parquet"
        self.parquet.write_bytes(b"prepared job is checked by native scene setup")
        self.saved = saved_config()
        self.saved["task"].update(parquet=str(self.parquet), motion_start_frame=0, motion_end_frame=-1)
        self.saved["env"]["objective"]["_target_"] = "flash_chord.objectives.recon_body.ReconBodyObjectiveConfig"
        self.saved["env"]["reset"].update(seed=42, reset_voc_scale=1.0)
        self.saved["training"].update(seed=42, world_count=2048, resume=False, checkpoint_path=None)
        self.saved["training"]["optimization"]["learning_rate_schedule_updates"] = OPTIMIZER_HORIZON
        self.saved["voc_scale_override"] = None
        self.saved["curriculum"] = {
            "thresholds": [35000320, 70000640, 104998912, 139999232, 174999552,
                           209999872, 288749568, 306249728, 349999104],
            "stages": [{"voc_scale": scale, "objective_weights": {"object_position": 1},
                        "immediate_first_frame_probability": probability}
                       for scale, probability in zip([1., .75, .5, .25, .1, .05, 0., 0., 0.],
                                                     [0.] * 7 + [.25, .5])],
        }
        self.saved["training"]["curriculum"] = copy.deepcopy(self.saved["curriculum"])

    def build(self, saved=None):
        return build_fresh_config(self.saved if saved is None else saved, output=self.output,
                                  end=349999104, checkpoint_interval=50_000_000)

    def test_fresh_preserves_native_method_and_does_not_restore_any_state(self):
        original = copy.deepcopy(self.saved)
        cfg = self.build()
        self.assertEqual(self.saved, original)
        for key in ("env", "task", "curriculum", "scene"):
            self.assertEqual(cfg[key], original[key])
        self.assertNotIn("overnight", cfg)
        self.assertFalse(cfg["training"]["resume"])
        self.assertIsNone(cfg["training"]["checkpoint_path"])
        self.assertFalse(cfg["training"]["checkpoint"]["load_optimizer"])
        self.assertFalse(cfg["training"]["checkpoint"]["load_reward_normalizer"])
        self.assertEqual(cfg["training"]["replay"]["capacity"], 4_000_000)
        self.assertEqual(cfg["training"]["network"], original["training"]["network"])
        self.assertEqual(cfg["training"]["seed"], 42)
        self.assertEqual(cfg["training"]["checkpoint"]["save_interval_environment_steps"], 50_000_000)
        self.assertEqual(checkpoint_stage_metadata(cfg["curriculum"], 0)["voc_scale"], 1)
        self.assertEqual(checkpoint_stage_metadata(cfg["curriculum"], 70_000_640)["voc_scale"], .5)
        self.assertEqual(checkpoint_stage_metadata(cfg["curriculum"], 209_999_872)["voc_scale"], 0)

    def test_rejects_stale_resume_refinement_and_wrong_input_contracts(self):
        changes = [
            ("training", "resume", True), ("training", "checkpoint_path", "state_123.safetensors"),
            ("training", "world_count", 16), ("scene", "world_count", 16),
            ("training", "seed", 0), ("task", "source_frame_playback", False),
            ("task", "motion_start_frame", 10), ("task", "parquet", "missing.parquet"),
        ]
        for section, key, value in changes:
            with self.subTest(section=section, key=key):
                cfg = copy.deepcopy(self.saved)
                cfg[section][key] = value
                with self.assertRaises(ValueError):
                    self.build(cfg)
        for key, value in (("overnight", {"refinement_start_steps": 0}),
                           ("reference", {}), ("voc_scale_override", 0.0)):
            with self.subTest(key=key):
                cfg = copy.deepcopy(self.saved)
                cfg[key] = value
                with self.assertRaises(ValueError):
                    self.build(cfg)
        cfg = copy.deepcopy(self.saved)
        cfg["training"]["checkpoint"]["load_replay_path"] = "replay.safetensors"
        with self.assertRaises(ValueError):
            self.build(cfg)
        cfg = copy.deepcopy(self.saved)
        cfg["training"]["optimization"]["learning_rate_schedule_updates"] = 100
        with self.assertRaises(ValueError):
            self.build(cfg)

    def test_resolved_source_starts_at_zero_without_reading_checkpoint(self):
        path = self.output / "fresh.json"
        path.write_text(json.dumps(self.saved), encoding="utf-8")
        args = SimpleNamespace(fresh_config=str(path), checkpoint=None, phase="refine",
                               additional_transitions=60_000_000, checkpoint_interval=50_000_000)
        with mock.patch("experiments.overnight_policy.checkpoint_pair") as pair:
            cfg, _, state, policy, start, end, additional = source_config(args, self.output)
        pair.assert_not_called()
        self.assertEqual((state, policy, start), (None, None, 0))
        self.assertEqual((end, additional), (59_998_208, 59_998_208))
        self.assertFalse(cfg["training"]["resume"])
        args.phase = "consolidate"
        with self.assertRaises(ValueError):
            source_config(args, self.output)

    def test_initial_native_checkpoint_saved_once_before_learning_with_actual_voc(self):
        cfg = self.build()
        learner = SimpleNamespace(environment_steps=0, global_update_step=0,
                                  actor=SimpleNamespace(schedule_step=0), critic=SimpleNamespace(schedule_step=0),
                                  replay=SimpleNamespace(size=0))
        paths = SimpleNamespace(policy=self.output / "policy_0.safetensors",
                                training_state=self.output / "state_0.safetensors")
        logger = mock.Mock()
        context = {"kwargs": {"checkpoint_metadata": {"policy_schema_json": "actor"}, "logger": logger}}
        save = mock.Mock(return_value=paths)
        first = save_initial_checkpoint_once(context, learner, cfg, self.output, scalar=int, save_checkpoint=save)
        second = save_initial_checkpoint_once(context, learner, cfg, self.output, scalar=int, save_checkpoint=save)
        self.assertEqual(first, second)
        save.assert_called_once()
        logger.log_checkpoint.assert_called_once()
        metadata = save.call_args.kwargs["metadata"]
        self.assertEqual((metadata["voc_scale"], metadata["environment_steps"], metadata["seed"]), (1, 0, 42))
        self.assertFalse(save.call_args.kwargs["save_replay"])
        self.assertEqual(json.loads((self.output / "initial_checkpoint.json").read_text())["fresh_replay_size"], 0)
        learner.environment_steps = 2048
        with self.assertRaises(ValueError):
            save_initial_checkpoint_once({"kwargs": context["kwargs"]}, learner, cfg, self.output,
                                         scalar=int, save_checkpoint=save)


if __name__ == "__main__":
    unittest.main()

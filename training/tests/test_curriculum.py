import tempfile
import unittest
from pathlib import Path

import torch

from training.curriculum import CHORD_THRESHOLDS, CHORD_TRANSITIONS, stage_at
from training.learner import Learner
from training.replay import Replay
from training.runner import train
from training.tests.test_learning import collect, small_config
from training.toy import ToyEnv


class CurriculumTests(unittest.TestCase):
    def test_pinned_schedule_reaches_zero_at_sixty_percent(self):
        self.assertEqual(stage_at(0)["assistance_scale"], 1)
        for i, threshold in enumerate(CHORD_THRESHOLDS):
            self.assertEqual(stage_at(threshold - 1)["stage"], i)
            self.assertEqual(stage_at(threshold)["stage"], i + 1)
        self.assertEqual(stage_at(CHORD_THRESHOLDS[5])["assistance_scale"], 0)
        self.assertEqual(stage_at(CHORD_TRANSITIONS)["first_frame_probability"], 0.5)

    def test_clear_removes_old_dynamics_and_sequences(self):
        replay = Replay(256, 4, 4, 1, 2)
        collect(ToyEnv(), replay)
        replay.clear()
        self.assertEqual(replay.size, 0)
        with self.assertRaises(ValueError):
            replay.sequences(4, 3, "cpu")

    def test_assistance_normalization_is_fixed_for_unseen_stages(self):
        env = ToyEnv()
        env.spec["interface"] = {"goal_blocks": {"assistance": [1, 2]}}
        replay = Replay(256, 4, 4, 1, 2)
        collect(env, replay)
        learner = Learner(small_config(), env.spec)
        learner.fit_normalizers(replay)
        for module in (learner.actor, learner.critic, learner.target, learner.model):
            self.assertEqual(float(module.norm.mean[-1]), 0)
            self.assertEqual(float(module.norm.std[-1]), 1)

    def test_training_changes_stages_and_evaluates_saved_checkpoints(self):
        torch.set_num_threads(1)

        class CurriculumToy(ToyEnv):
            def set_training_stage(self, stage):
                self.stages.append(stage.copy())

        env = CurriculumToy()
        env.stages = []
        evaluations = []

        def evaluate(path, output):
            self.assertTrue(path.is_file())
            evaluations.append(output)

        config = small_config(
            total_transitions=640,
            curriculum="chord",
            curriculum_transitions=640,
            evaluation_interval=320,
        )
        with tempfile.TemporaryDirectory() as folder:
            result = train(env, config, Path(folder), evaluate_checkpoint=evaluate)
        self.assertEqual([s["stage"] for s in env.stages], list(range(9)))
        self.assertEqual(result["assistance_scale"], 0)
        self.assertGreater(result["model_updates"], 0)
        self.assertEqual(len(evaluations), 2)


if __name__ == "__main__":
    unittest.main()

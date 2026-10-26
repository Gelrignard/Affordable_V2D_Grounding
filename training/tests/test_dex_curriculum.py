import dataclasses
import unittest
from types import SimpleNamespace

from training.dex_curriculum import DexCurriculum, DexGate


@dataclasses.dataclass
class Stage:
    voc_scale: float
    objective_weights: dict
    immediate_first_frame_probability: float = 0


def curriculum():
    weights = {
        "object_position": 1,
        "force_closure": 5,
        "body_joint_position": 5,
        "palm_position": 1,
        "palm_orientation": 1,
    }
    native = SimpleNamespace(
        stages=[Stage(s, weights.copy()) for s in (1, 0.75, 0.5, 0.25, 0.1, 0.05, 0)]
    )
    return DexCurriculum(
        native,
        frame_dt=0.02,
        gate=DexGate(
            window=4,
            warmup_iterations=4,
            minimum_stage_iterations=4,
            recovery_iterations=4,
        ),
    )


def observations(c, *, value=0.8, progress=0.99, completion=0.95, count=100):
    result = {
        f"reward_components/{k}_weighted_mean_per_env_step": 0.02 * w * value
        for k, w in c.stages[c.index].objective_weights.items()
    }
    result.update(
        {
            "termination/reference_end_fraction_per_completed_episode": completion,
            "episode/reference_progress_mean_per_completed_episode": progress,
            "episode/completed_count_per_iteration": count,
        }
    )
    return result


class DexCurriculumTests(unittest.TestCase):
    def test_normalized_rewards_advance_only_after_window_and_clear_old_evidence(self):
        c = curriculum()
        for i in range(1, 4):
            self.assertEqual(c.observe(observations(c), i)["action"], "hold")
        result = c.observe(observations(c), 4)
        self.assertEqual(result["action"], "advance")
        self.assertAlmostEqual(result["averages"]["contact"], 0.8)
        self.assertEqual(c.index, 1)
        self.assertEqual(len(c.history), 0)
        self.assertEqual(c.observe(observations(c), 5)["action"], "hold")

    def test_no_completed_episodes_is_not_success(self):
        c = curriculum()
        for i in range(1, 9):
            result = c.observe(observations(c, count=0), i)
        self.assertEqual(c.index, 0)
        self.assertFalse(result["gates"]["completed_episodes"])

    def test_recovery_and_completion_weighting(self):
        c = curriculum()
        for i in range(1, 5):
            c.observe(observations(c), i)
        for i in range(5, 9):
            result = c.observe(observations(c, progress=0.1, completion=0), i)
        self.assertEqual(result["action"], "recover")
        self.assertEqual(c.index, 0)
        self.assertEqual(c.stages[-1].voc_scale, 0)
        self.assertEqual(c.stages[-1].immediate_first_frame_probability, 0.5)

    def test_rising_rewards_and_nonfinite_inputs(self):
        c = curriculum()
        for i in range(1, 5):
            result = c.observe(observations(c, value=0.5 + 0.1 * i), i)
        self.assertEqual(result["action"], "hold")
        self.assertFalse(result["gates"]["plateau"])
        with self.assertRaises(ValueError):
            c.observe(observations(c, value=float("nan")), 5)


if __name__ == "__main__":
    unittest.main()

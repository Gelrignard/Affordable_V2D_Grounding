"""CPU checks for phase boundaries, failed cohorts and fixed diagnostic criteria."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

from experiments.evaluate_overnight_policy import (
    Trace,
    common_tracking_analysis,
    episode_phase_windows,
    first_bad_frame,
    main,
    phase_analysis,
    quat_multiply,
    quat_rotate,
)


def poses(frames=1003, worlds=2):
    achieved = np.zeros((frames, worlds, 1, 7), dtype=np.float32)
    achieved[..., 6] = 1.0
    return achieved, achieved[:, 0].copy(), np.zeros((1, 1, 3)), np.array([0])


class OvernightEvaluationTests(unittest.TestCase):
    def test_trace_detects_within_attempt_reset_and_is_bounded(self):
        class Array:
            def __init__(self, value):
                self.value = np.asarray(value)

            def numpy(self):
                return self.value

        env = SimpleNamespace(
            objective=object(), contact=None, action=object(),
            timestep=Array([1, 1]),
            reset_policy=SimpleNamespace(applied_voc_scale=Array([0.0, 0.0])),
        )
        trace = Trace(env, steps=1003, stride=1, max_samples=10)
        for frame in range(1003):
            env.timestep.value[:] = min(frame + 1, 1002)
            trace.after_step()
        self.assertLessEqual(len(trace.data["reference_frame"]), 10)
        self.assertEqual(int(trace.data["reference_frame"][-1]), 1002)
        trace = Trace(env, steps=1003, stride=5, max_samples=400)
        env.timestep.value[:] = 0
        with self.assertRaisesRegex(RuntimeError, "reset/froze"):
            trace.after_step()

    def test_window_edges_and_late_failed_world_are_preserved(self):
        a, r, v, ids = poses()
        a[850:, 1, 0, 0] = 0.20
        result = phase_analysis(a, r, v, ids)
        handover = result["phases"]["handover_730_850"]
        self.assertEqual((handover["frame_start"], handover["frame_stop"]), (730, 851))
        self.assertAlmostEqual(handover["position_mean_cm"], 10 / 121, places=6)
        self.assertEqual(handover["every_frame_object_threshold_pass_fraction"], 0.5)
        self.assertEqual(result["first_object_10cm_or_1p5rad_frame_by_world"], [None, 850])
        self.assertEqual(result["phases"]["final_200"]["frame_start"], 803)
        self.assertGreater(result["phases"]["final_200"]["position_mean_cm"], 7)

    def test_native_invalid_world_flag_cannot_be_erased_by_zero_filling(self):
        a, r, v, ids = poses(frames=20)
        result = phase_analysis(a, r, v, ids, invalid_worlds=(1,))
        self.assertEqual(result["invalid_worlds"], [1])
        self.assertFalse(result["phases"]["full_sequence"]["available"])
        self.assertNotIn("scores", result["phases"]["full_sequence"])

    def test_short_reference_does_not_fabricate_handover_metrics(self):
        result = phase_analysis(*poses(frames=100))
        self.assertFalse(result["phases"]["handover_730_850"]["available"])
        self.assertEqual(result["phases"]["final_200"]["frame_start"], 0)

    def test_other_episode_thirds_partition_all_frames(self):
        for episode, frames, boundaries in [(16, 477, (159, 318)), (18, 858, (286, 572))]:
            with self.subTest(episode=episode):
                first, second = boundaries
                self.assertEqual(episode_phase_windows(frames, episode), {
                    "full_sequence": (0, frames), "early_third": (0, first),
                    "middle_third": (first, second), "late_third": (second, frames),
                    "final_200": (frames - 200, frames),
                })
                a, r, v, ids = poses(frames=frames)
                a[first:second, :, 0, 0] = .03
                a[second:, :, 0, 0] = .06
                result = phase_analysis(a, r, v, ids, episode=episode)
                self.assertNotIn("handover_730_850", result["phases"])
                for phase, expected in [("early_third", 0), ("middle_third", 3), ("late_third", 6)]:
                    self.assertTrue(result["phases"][phase]["available"])
                    self.assertAlmostEqual(result["phases"][phase]["position_mean_cm"], expected, places=5)
                legacy = phase_analysis(a, r, v, ids)
                self.assertEqual(result["phases"]["full_sequence"], legacy["phases"]["full_sequence"])
                self.assertEqual(result["phases"]["final_200"], legacy["phases"]["final_200"])

    def test_explicit_episode_five_preserves_default_report(self):
        arrays = poses()
        self.assertEqual(phase_analysis(*arrays), phase_analysis(*arrays, episode=5))

    def test_generic_phase_invalid_cohort_remains_unscored(self):
        result = phase_analysis(*poses(frames=477), episode=16, invalid_worlds=(1,))
        self.assertEqual(result["invalid_worlds"], [1])
        for phase in result["phases"].values():
            self.assertFalse(phase["available"])
            self.assertNotIn("scores", phase)

    def test_cpu_cli_accepts_tier_one_episode_eighteen(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            a, r, v, ids = poses(frames=858)
            source = root / "rollout.npz"
            np.savez(source, achieved_pose_w=a, reference_pose_w=r, object_vertices_o=v,
                     body_object_ids=ids, invalid_worlds=np.array([], dtype=int))
            output = root / "analysis"
            with patch("sys.argv", ["evaluate_overnight_policy", "--tier", "1", "--episode", "18",
                                    "--analyze-rollout", str(source), "--output", str(output)]):
                with contextlib.redirect_stdout(io.StringIO()):
                    main()
            result = json.loads((output / "phase_metrics.json").read_text())
            self.assertEqual(result["phases"]["late_third"]["frame_start"], 572)
            self.assertNotIn("handover_730_850", result["phases"])

    def test_frame_zero_failure_and_threshold_activation(self):
        self.assertEqual(first_bad_frame([[True, False], [False, True]]), [0, 1])
        errors = np.zeros((6, 2, 2))
        errors[0, 0, 0] = 0.71  # pelvis active immediately
        errors[0, 1, 1] = 0.11  # object ignored during the declared freeze
        errors[4, 1, 1] = 0.11
        result = common_tracking_analysis(errors, ["pelvis_position_m", "object_position_m"], freeze_steps=3)
        self.assertEqual(result["first_failure_frame_by_world"], [0, 4])
        self.assertEqual(result["success_fraction"], 0)

    def test_pose_composition_uses_xyzw(self):
        q = np.array([0, 0, np.sqrt(0.5), np.sqrt(0.5)])
        np.testing.assert_allclose(quat_rotate(q, np.array([1, 0, 0])), [0, 1, 0], atol=1e-7)
        np.testing.assert_allclose(quat_multiply(q, np.array([0, 0, 0, 1])), q, atol=1e-7)


if __name__ == "__main__":
    unittest.main()

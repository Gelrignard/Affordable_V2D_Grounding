import dataclasses
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from training.adaptive_chord import AdaptiveCurriculum, ProbeGate
from training.dynamics_targets import required_wrench


@dataclasses.dataclass
class Stage:
    voc_scale: float
    objective_weights: dict
    immediate_first_frame_probability: float = 0.0


class NativeExtensionTests(unittest.TestCase):
    def test_required_wrench_static_accelerating_and_offset_com(self):
        time = np.arange(101) * 0.02
        q = np.tile([0, 0, 0, 1.0], (len(time), 1))
        position = np.zeros((len(time), 3))
        position[:, 0] = time**2  # a_x = 2 m/s^2
        result = required_wrench(
            position,
            q,
            dt=0.02,
            mass=2,
            inertia=np.diag([0.1, 0.2, 0.3]),
            com=np.array([0.1, 0, 0]),
        )
        np.testing.assert_allclose(
            result["force_w"], np.tile([4, 0, 19.62], (101, 1)), atol=1e-5
        )
        np.testing.assert_allclose(result["torque_b"], 0, atol=1e-7)
        np.testing.assert_allclose(result["confidence"][11:-11], 1, atol=1e-6)
        # Stationary body origin with an offset COM rotating around z.
        rot = Rotation.from_rotvec(np.column_stack([time * 0, time * 0, time]))
        result = required_wrench(
            position * 0,
            rot.as_quat(),
            dt=0.02,
            mass=2,
            inertia=np.eye(3) * 0.1,
            com=np.array([0.2, 0, 0]),
        )
        expected = -0.4 * np.column_stack([np.cos(time), np.sin(time)])
        np.testing.assert_allclose(
            result["force_w"][12:-12, :2], expected[12:-12], atol=5e-4
        )

    def test_quaternion_sign_and_inertial_torque(self):
        time = np.arange(101) * 0.02
        angle = 0.3 * time**2
        q = Rotation.from_rotvec(np.column_stack([time * 0, time * 0, angle])).as_quat()
        q[::2] *= -1
        result = required_wrench(
            np.zeros((101, 3)),
            q,
            dt=0.02,
            mass=1,
            inertia=np.diag([0.1, 0.2, 0.3]),
            com=np.zeros(3),
        )
        np.testing.assert_allclose(
            result["torque_b"][20:-20], np.tile([0, 0, 0.18], (61, 1)), atol=1e-5
        )
        np.testing.assert_allclose(result["torque_w"], result["torque_b"], atol=1e-6)

    def test_invalid_physics_is_rejected(self):
        with self.assertRaises(ValueError):
            required_wrench(
                np.zeros((20, 3)),
                np.ones((20, 4)),
                dt=0.02,
                mass=1,
                inertia=np.diag([1, 1, -1]),
                com=np.zeros(3),
            )

    def test_reward_does_not_advance_and_missing_probes_fail_closed(self):
        self.assertFalse(ProbeGate().accepts({"reward": 1e6}, 0.75))
        native = type(
            "Native",
            (),
            {
                "thresholds": (1, 2, 3),
                "stages": tuple(
                    Stage(s, {"native": 1})
                    for s in [1, 0.75, 0.5, 0.25, 0.1, 0.05, 0, 0, 0]
                ),
            },
        )()
        adaptive = AdaptiveCurriculum(native, dynamics_weight=1)
        report = {
            "probe_assistance": 0.75,
            "metrics": {
                "object_position_error_m": 0.05,
                "object_orientation_error_rad": 0.2,
            },
            "reference_tracking": {"failure_cause_fraction": {"pelvis": 0}},
            "physical_diagnostics": {
                "active_phase_contact_fraction": 0.4,
                "wrench_deficit_mean": 0.7,
            },
        }
        self.assertEqual(adaptive.stage_index(350000000), 0)  # no time-only reduction
        self.assertTrue(adaptive.observe(report, 10)["passed"])
        self.assertEqual(adaptive.index, 1)
        report["probe_assistance"] = 0.5
        report["non_finite_worlds"] = [0]
        self.assertFalse(adaptive.observe(report, 20)["passed"])
        self.assertTrue(adaptive.stalled(40, 30))
        self.assertFalse(adaptive.stalled(39, 30))
        self.assertEqual(adaptive.stages[-1].immediate_first_frame_probability, 0.5)
        self.assertEqual(adaptive.stages[0].objective_weights["dynamics_wrench"], 1)


if __name__ == "__main__":
    unittest.main()

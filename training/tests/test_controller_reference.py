import unittest
from types import SimpleNamespace

import numpy as np

from training.controller_reference import carrier_joint_coordinates


class ControllerReferenceTests(unittest.TestCase):
    def setUp(self):
        self.scene = SimpleNamespace(
            layout=SimpleNamespace(
                scalar_joints=SimpleNamespace(names=["a", "b"], q_ids=[7, 8])
            ),
            robot_reference=SimpleNamespace(joint_q=np.zeros((2, 9))),
        )
        self.task = SimpleNamespace(num_frames=2, fps=50)
        self.carrier = SimpleNamespace(
            num_frames=2,
            fps=50,
            robot_joint_names=lambda: ["b", "a"],
            robot_joint_pos=lambda: np.array([[2, 1], [4, 3]]),
            robot_root_pos_w=lambda: np.array([[0, 0, 0.8], [0.1, 0, 0.8]]),
            robot_root_quat_w=lambda: np.array([[1, 0, 0, 0], [0.5, 0.5, 0.5, 0.5]]),
        )

    def test_named_mapping_quaternion_order_and_original_unchanged(self):
        q = carrier_joint_coordinates(self.scene, self.task, self.carrier)
        np.testing.assert_array_equal(q[:, 7:], [[1, 2], [3, 4]])
        np.testing.assert_array_equal(q[0, 3:7], [0, 0, 0, 1])
        np.testing.assert_array_equal(
            self.scene.robot_reference.joint_q, np.zeros((2, 9))
        )

    def test_rejects_mismatched_timeline_and_joints(self):
        self.carrier.fps = 30
        with self.assertRaisesRegex(ValueError, "timeline"):
            carrier_joint_coordinates(self.scene, self.task, self.carrier)
        self.carrier.fps = 50
        self.carrier.robot_joint_names = lambda: ["b", "b"]
        with self.assertRaisesRegex(ValueError, "named robot joints"):
            carrier_joint_coordinates(self.scene, self.task, self.carrier)


if __name__ == "__main__":
    unittest.main()

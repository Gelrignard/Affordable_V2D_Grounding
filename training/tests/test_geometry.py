import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from training.bundles import inside
from training.kinematics import URDFKinematics
from training.runtime import canonical_history, canonical_urdf_sha
from training.wrench import basis_directions, capability_reward, wrench_support


class GeometryTests(unittest.TestCase):
    def test_fk_named_joint_and_root_frame(self):
        xml = """<robot name="test"><link name="root"/><link name="arm"/><link name="tip"/>
        <joint name="rotation" type="revolute"><parent link="root"/><child link="arm"/><axis xyz="0 0 1"/></joint>
        <joint name="tip_fixed" type="fixed"><parent link="arm"/><child link="tip"/><origin xyz="1 0 0"/></joint></robot>"""
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "robot.urdf"
            path.write_text(xml)
            pose = URDFKinematics(path).poses(
                np.array([[np.pi / 2]]),
                np.array([[2, 3, 4]]),
                np.array([[1, 0, 0, 0]]),
                ["rotation"],
                ["tip"],
            )
            np.testing.assert_allclose(pose[0, 0, :3], [2, 4, 4], atol=1e-7)
            path.write_bytes(xml.encode().replace(b"\n", b"\r\n"))
            self.assertEqual(
                canonical_urdf_sha(path),
                __import__("hashlib").sha256(xml.encode()).hexdigest(),
            )

    def test_wrench_direction_missing_contacts_and_scale(self):
        points = torch.tensor([[[0.0, 1, 0]]])
        normals = torch.tensor([[[1.0, 0, 0]]])
        active = torch.tensor([[True]])
        basis = torch.eye(6)
        support = wrench_support(points, normals, active, basis, 1, friction=0)
        self.assertAlmostEqual(float(support[0, 0]), 1)
        self.assertAlmostEqual(float(support[0, 5]), -1)
        larger = wrench_support(points * 2, normals, active, basis, 2, friction=0)
        torch.testing.assert_close(support, larger)
        none = wrench_support(points, normals, ~active, basis, 1)
        self.assertTrue(torch.equal(none, torch.zeros_like(none)))
        self.assertEqual(basis_directions(7).shape, (7, 6))
        self.assertGreater(
            float(capability_reward(support, support, torch.ones(1))),
            float(capability_reward(none, support, torch.ones(1))),
        )

    def test_history_rotation_and_reset_stale_memory(self):
        history = torch.arange(24).reshape(2, 4, 3).float()
        actual = canonical_history(history, torch.tensor([1, 0]), torch.tensor([1, 0]))
        torch.testing.assert_close(actual[0], history[0, [2, 3, 0, 1]])
        self.assertTrue(torch.all(actual[1] == 0))

    def test_manifest_cannot_escape_root(self):
        with self.assertRaises(ValueError):
            inside(Path.cwd(), "../outside.npz")


if __name__ == "__main__":
    unittest.main()

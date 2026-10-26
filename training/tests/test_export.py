import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq
import torch
from scipy.spatial.transform import Rotation

from training.bundles import arrays
from training.config import TrainConfig
from training.evaluate import physical_rollout_summary, resample_poses, write_trajectory
from training.prepare import load_job
from training.runtime import SonicEnv, materialize_motion, support_geometry
from training.wrench import basis_directions


class ExportTests(unittest.TestCase):
    def test_physical_diagnostics_keep_failure_visible_and_preserve_invalid_worlds(
        self,
    ):
        times = np.array([0, 0.02, 0.04])
        joints = np.zeros((3, 2, 7))
        joints[:, 0, 2] = [0.8, 0.7, 0.2]
        joints[:, 1, 2] = [0.8, 0.7, np.nan]
        forces = np.zeros((3, 2, 1, 3))
        forces[1, 0, 0, 0] = 1
        active = np.ones((3, 2, 1))
        terminated = np.array([[True, False], [True, True]])
        result = physical_rollout_summary(times, joints, forces, active, terminated)
        self.assertEqual(result[0]["first_tracking_failure_s"], 0.02)
        self.assertEqual(result[0]["first_root_below_threshold_s"], 0.04)
        self.assertEqual(result[0]["measured_hand_object_contact_fraction"], 0.5)
        self.assertFalse(result[1]["physical_diagnostics_finite"])
        self.assertIsNone(result[1]["minimum_root_height_m"])

    def test_physical_time_alignment_and_failed_world_retained(self):
        times = np.arange(6) / 50
        pose = np.zeros((6, 2, 1, 7))
        pose[:, :, 0, 0] = times[:, None] * 2
        pose[:, :, 0, 3:] = Rotation.from_euler("z", (times * 3)[:, None]).as_quat()[
            :, None
        ]
        pose[2, 1, 0, 0] = np.nan
        target_times = np.array([0, 1 / 30, 2 / 30, 0.1])
        actual, valid = resample_poses(times, pose, target_times)
        self.assertEqual(valid.tolist(), [True, False])
        np.testing.assert_allclose(actual[:, 0, 0, 0], 2 * target_times)
        np.testing.assert_allclose(
            Rotation.from_quat(actual[:, 0, 0, 3:]).as_rotvec()[:, 2], target_times * 3
        )
        reference = actual[:, 0].copy()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "episode_000018.parquet"
            write_trajectory(
                path,
                actual,
                reference,
                target_times,
                np.arange(4),
                18,
                30,
                valid,
                {
                    "eval_script_sha256": "a" * 64,
                    "eval_code_sha256": "b" * 64,
                    "checkpoint_sha256": "c" * 64,
                },
            )
            table = pq.read_table(path)
            self.assertEqual(table.num_rows, 8)
            self.assertEqual(table["environment_valid"].to_pylist(), [True, False] * 4)
            kit = (
                Path(__file__).resolve().parents[2]
                / "vendor/submission-kit/v2d_submission_kit"
            )
            if kit.exists():
                sys.path.insert(0, str(kit))
                from v2dlb.submission_parquet import read_trajectory

                read = read_trajectory(path)
                self.assertEqual(read.episode_index, 18)
                np.testing.assert_array_equal(
                    read.frame_index, np.repeat(np.arange(4), 2)
                )
                np.testing.assert_allclose(
                    read.pose_xyzw[::2], reference[:, 0], atol=1e-6
                )
        with self.assertRaises(ValueError):
            resample_poses(times, pose, np.array([0, 0.11]))

    @unittest.skipUnless(
        os.environ.get("V2D_PREPARED_JOBS") and os.environ.get("V2D_FLASH_RUNTIME"),
        "optional real pilot/runtime integration",
    )
    def test_all_real_jobs_load_in_pinned_runtime(self):
        runtime = Path(os.environ["V2D_FLASH_RUNTIME"])
        sys.path.insert(0, str(runtime / "robotic_grounding/flash_chord/src"))
        from flash_chord.data import load_reference

        jobs = list(Path(os.environ["V2D_PREPARED_JOBS"]).rglob("job.json"))
        self.assertTrue(jobs)
        with tempfile.TemporaryDirectory() as temp:
            for path in jobs:
                job = load_job(path.parent)
                motion = materialize_motion(path.parent, Path(temp) / "motion.parquet")
                reference = load_reference(
                    motion, control_fps=50, source_frame_playback=False
                )
                self.assertEqual(reference.robot_joint_pos().shape[1], 43)
                self.assertEqual(reference.object_body_pos_w().shape[1], 1)
                self.assertAlmostEqual(
                    (reference.num_frames - 1) / 50,
                    (job["source_frames"] - 1) / job["source_fps"],
                    delta=0.011,
                )
                self.assertTrue(np.isfinite(reference.robot_joint_pos()).all())
                self.assertTrue(
                    np.isfinite(
                        support_geometry(path.parent / "assets/support.usda")
                    ).all()
                )
                # Rebased paths must not invalidate the portable source job.
                load_job(path.parent)
            # Exercise the real reference -> model/actor goal seam without GPU inference.
            adapter = SonicEnv.__new__(SonicEnv)
            adapter.device, adapter.worlds, adapter.frames = (
                torch.device("cpu"),
                2,
                reference.num_frames,
            )
            adapter.reference, adapter.control_fps = reference, 50
            adapter.cfg = TrainConfig(device="cpu", worlds=2, wrench_basis=8)
            adapter.source_reference = arrays(path.parent / "reference.npz")
            adapter.guidance = arrays(path.parent / "guidance.npz")
            adapter.geometry = arrays(path.parent / "geometry.npz")
            adapter.support_descriptor = support_geometry(
                path.parent / "assets/support.usda"
            )
            adapter.radius = float(adapter.geometry["radius"])
            adapter.basis = basis_directions(8)
            adapter.tensor = lambda value: value
            adapter.synchronize = lambda: None
            adapter.object_ids = torch.tensor([[0], [1]])
            action = SimpleNamespace(
                encoder_input=torch.zeros(2, 1762),
                encoder_reference=torch.zeros(reference.num_frames, 1762),
                latent=torch.zeros(2, 64),
                encoder_session=SimpleNamespace(run=lambda: None),
            )
            adapter.env = SimpleNamespace(action=action)
            adapter.scene = SimpleNamespace(
                model=SimpleNamespace(
                    body_mass=torch.ones(2), body_inertia=torch.eye(3).repeat(2, 1, 1)
                )
            )
            adapter._make_goals()
            self.assertTrue(torch.isfinite(adapter.goals).all())
            self.assertEqual(adapter.target_object.shape, (reference.num_frames, 7))


if __name__ == "__main__":
    unittest.main()

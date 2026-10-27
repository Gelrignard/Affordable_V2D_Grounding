"""CPU integration tests against the pinned official kit (no simulator imports).

Set V2D_SUBMISSION_KIT to the unpacked v2d_submission_kit directory. Tests fail
explicitly when an appointed kit is invalid; only absence skips kit integration.
"""

import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np

from experiments import score_clean_reference as scorer


KIT_ROOT = Path(os.environ.get("V2D_SUBMISSION_KIT", "vendor/submission-kit/v2d_submission_kit"))


@unittest.skipUnless(KIT_ROOT.is_dir(), "Set V2D_SUBMISSION_KIT to run official-code integration tests")
class CleanReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kit = scorer.load_official_kit(KIT_ROOT)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.input_root, self.clean_root = self.root / "tier2", self.root / "tier1"
        self.frames, self.worlds = 4, 2
        self.clean = np.zeros((self.frames, 1, 7))
        self.clean[:, 0, 0] = np.arange(self.frames) * 0.01
        self.clean[..., 6] = 1.0
        self.input = self.clean.copy()
        self.input[..., 0] += 0.05  # Source reconstruction error must survive scoring.
        self.write_dataset(self.clean_root, self.clean, 1)
        self.write_dataset(self.input_root, self.input, 2)
        self.rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        self.offset = np.array([1., 2., 3.])
        self.q = np.array([0., 0., np.sqrt(0.5), np.sqrt(0.5)])
        self.reference = self.to_sim(self.input)
        achieved_source = np.broadcast_to(self.input[:, None], (self.frames, self.worlds, 1, 7)).copy()
        achieved_source[:, 1, :, 0] += 0.10
        self.arrays = {"achieved_pose_w": self.to_sim(achieved_source),
                       "reference_pose_w": self.reference, "object_vertices_o": np.zeros((1, 6, 3)),
                       "body_object_ids": np.array([0]), "invalid_worlds": np.array([], dtype=int)}
        self.arrays_path = self.root / "rollout_arrays.npz"
        self.report = {"schema": "v2d_overnight_evaluation_v1", "start_frame": 0,
                       "reset_mode": "explicit", "settling_steps": 0, "world_count": 2,
                       "step_count": 4, "motion_start_frame": 0, "motion_end_frame": -1,
                       "non_finite_worlds": [], "metric_reference": "tier2_input_reference",
                       "checkpoint_sha256": "a" * 64, "environment_steps": 123,
                       "parquet": "/input/sequence_id=test_sequence/robot_name=g1/motion.parquet",
                       "evaluation_contract": {"frame_zero": True, "explicit_reset": True,
                           "within_attempt_resets": 0, "voc_scale": 0., "reset_voc_scale": 0.,
                           "settling_steps": 0, "training_termination_disabled_during_rollout": True}}
        self.reference_path = self.root / "reference.npz"
        transform = np.eye(4)
        transform[:3, :3], transform[:3, 3] = self.rotation, self.offset
        self.stored = {"source_frame_indices": np.arange(4), "object_position": self.reference[:, 0, :3],
                       "object_wxyz": self.reference[:, 0, [6, 3, 4, 5]],
                       "world_from_source_cv": transform, "coordinate_frame": "world_z_up",
                       "quaternion_order": "wxyz", "units": "metres_seconds_radians"}
        self.save()

    def to_sim(self, pose):
        pose = pose.copy()
        pose[..., :3] = pose[..., :3] @ self.rotation.T + self.offset
        pose[..., 3:] = self.kit.op._quaternion_multiply(np.broadcast_to(self.q, pose[..., 3:].shape), pose[..., 3:])
        return pose

    def write_dataset(self, root, pose, tier, *, names=None, visible=None):
        import pyarrow as pa
        import pyarrow.parquet as pq
        (root / "meta").mkdir(parents=True, exist_ok=True)
        path = root / "data/chunk-000/episode_000005.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        values = pose[:, 0, [0, 1, 2, 6, 3, 4, 5]].tolist()
        table = pa.table({"frame_index": np.arange(4), "episode_index": np.full(4, 5),
                          "observation.object.pose": pa.array(values, type=pa.list_(pa.float64(), 7)),
                          "observation.object.visible": [True] * 4 if visible is None else visible})
        table = table.replace_schema_metadata({b"source_sequence_id": b"test_sequence",
                                              b"objects": json.dumps(names or ["box"]).encode()})
        pq.write_table(table, path)
        manifest = {"repository": "nvidia/video_to_data_challenge", "revision": "test_revision",
                    "tier": tier, "files": {path.relative_to(root).as_posix(): {"sha256": scorer.sha256(path)}}}
        (root / "download_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    def save(self):
        np.savez(self.arrays_path, **self.arrays)
        np.savez(self.reference_path, **self.stored)
        (self.root / "evaluation.json").write_text(json.dumps(self.report), encoding="utf-8")

    def score(self, *, reference=True):
        return scorer.score_saved_rollout(dataset_input=self.input_root, dataset_clean=self.clean_root,
                    episode=5, arrays=self.arrays_path, kit_root=KIT_ROOT,
                    reference_npz=self.reference_path if reference else None)

    def test_input_noise_and_failing_world_survive_clean_scoring(self):
        result = self.score()
        self.assertAlmostEqual(result["clean_reference_metrics"]["mppe_cm"], 10.0, places=10)
        self.assertEqual(result["clean_reference_metrics"]["spider_sr"], 0.5)
        self.assertEqual(result["clean_reference_metrics"]["maniptrans_sr"], 0.0)
        np.testing.assert_allclose(result["all_world_mean_position_error_m"], [0.05, 0.15])
        self.assertEqual(result["dropped_worlds"], [])
        self.assertFalse(result["initial_achieved_to_clean_alignment"])
        self.assertTrue(result["reference_frame_conversion"]["stored_transform_verified"])
        self.assertEqual(self.score(reference=False)["clean_reference_metrics"], result["clean_reference_metrics"])
        analysis = result["clean_reference_phase_analysis"]
        self.assertEqual(analysis["full_sequence_tail"]["worlds_ever_origin_error_over_10cm"], [1])
        self.assertEqual(analysis["full_sequence_tail"]["first_origin_error_over_10cm_source_frame_by_world"], [None, 0])
        self.assertAlmostEqual(analysis["full_sequence_tail"]["origin_error_max_cm"], 15.0, places=10)
        for phase in analysis["phases"].values():
            self.assertAlmostEqual(phase["clean_reference_metrics"]["mppe_cm"], 10.0, places=10)

    def test_clean_target_does_not_determine_conversion(self):
        first = self.score()
        changed_clean = self.clean.copy()
        changed_clean[..., 0] += 0.02
        self.write_dataset(self.clean_root, changed_clean, 1)
        second = self.score()
        self.assertEqual(first["reference_frame_conversion"], second["reference_frame_conversion"])
        self.assertAlmostEqual(second["clean_reference_metrics"]["mppe_cm"], 8.0, places=10)

    def test_invalid_cohort_is_rejected_instead_of_filtered(self):
        self.arrays["invalid_worlds"] = np.array([1])
        self.save()
        with self.assertRaisesRegex(ValueError, "no failed worlds may be dropped"):
            self.score()

    def test_nonrigid_reference_and_wrong_stored_transform_rejected(self):
        self.arrays["reference_pose_w"][1, 0, 0] += 0.02
        self.save()
        with self.assertRaisesRegex(ValueError, "one rigid transform"):
            self.score(reference=False)
        self.arrays["reference_pose_w"][1, 0, 0] -= 0.02
        self.stored["world_from_source_cv"][0, 3] += 0.01
        self.save()
        with self.assertRaisesRegex(ValueError, "disagrees with world_from_source_cv"):
            self.score()

    def test_names_sequence_frames_and_reset_contract_checked(self):
        self.write_dataset(self.clean_root, self.clean, 1, names=["different_object"])
        with self.assertRaisesRegex(ValueError, "sequence/object/frame"):
            self.score()
        self.write_dataset(self.clean_root, self.clean, 1)
        self.report["parquet"] = "/sequence_id=wrong_sequence/motion.parquet"
        self.save()
        with self.assertRaisesRegex(ValueError, "sequence identities"):
            self.score()
        self.report["parquet"] = "/sequence_id=test_sequence/motion.parquet"
        self.report["evaluation_contract"]["within_attempt_resets"] = 1
        self.save()
        with self.assertRaisesRegex(ValueError, "uninterrupted"):
            self.score()
        self.report["evaluation_contract"]["within_attempt_resets"] = 0
        self.stored["source_frame_indices"] = np.array([0, 1, 2, 4])
        self.save()
        with self.assertRaisesRegex(ValueError, "source frame grid"):
            self.score()

    def test_dataset_and_kit_tampering_rejected(self):
        path = self.input_root / "data/chunk-000/episode_000005.parquet"
        with path.open("ab") as stream:
            stream.write(b"tamper")
        with self.assertRaisesRegex(ValueError, "hash does not match"):
            self.score()
        copy = self.root / "kit"
        copy.mkdir()
        shutil.copyfile(KIT_ROOT / "KIT_MANIFEST.json", copy / "KIT_MANIFEST.json")
        for name in scorer.PINNED_CODE:
            (copy / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(KIT_ROOT / name, copy / name)
        with (copy / "tools/pack_submission.py").open("a", encoding="utf-8") as stream:
            stream.write("\n# altered\n")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            scorer.load_official_kit(copy)

    def test_only_official_clean_visibility_controls_scored_frames(self):
        self.write_dataset(self.clean_root, self.clean, 1, visible=[True, False, True, True])
        result = self.score()
        self.assertEqual(result["rollout_frame_count"], 4)
        self.assertEqual(result["scored_source_frame_indices"], [0, 2, 3])
        self.assertEqual(result["official_clean_invisible_frames"], [1])
        self.assertEqual(result["world_count"], 2)
        self.assertAlmostEqual(result["clean_reference_metrics"]["mppe_cm"], 10.0, places=10)
        phases = result["clean_reference_phase_analysis"]["phases"]
        self.assertFalse(phases["middle_third"]["available"])
        self.assertEqual(phases["middle_third"]["scored_source_frame_indices"], [])
        self.assertEqual(phases["late_third"]["scored_source_frame_indices"], [2, 3])


if __name__ == "__main__":
    unittest.main()

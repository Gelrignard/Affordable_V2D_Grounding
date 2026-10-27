"""CPU-only official-schema export checks. Set V2D_SUBMISSION_KIT to run."""

import json
from pathlib import Path
import shutil
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from experiments import export_policy_rollout as exporter
from experiments import test_score_clean_reference as fixtures

KIT_ROOT = fixtures.KIT_ROOT


@unittest.skipUnless(KIT_ROOT.is_dir(), "Set V2D_SUBMISSION_KIT to run official-code integration tests")
class PolicyRolloutExportTests(unittest.TestCase):
    def setUp(self):
        # Reuse the independently scored Tier2 fixture with 5cm reconstruction
        # bias and different outcomes in its two worlds.
        self.fixture = fixtures.CleanReferenceTests("test_input_noise_and_failing_world_survive_clean_scoring")
        self.fixture.kit = exporter.validation.load_official_kit(KIT_ROOT)
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.reader = exporter.load_submission_reader(KIT_ROOT)
        self.job = self.fixture.root / "job"
        self.job.mkdir()
        self.output = self.fixture.root / "export"
        self.fixture.report["sim_control_fps"] = 50.
        self.fixture.save()
        self.make_job()
        self.write_native()

    def make_job(self):
        f = self.fixture
        shutil.copyfile(f.reference_path, self.job / "reference.npz")
        row = {"sequence_id": "test_sequence", "robot_name": "g1", "object_body_names": ["object"],
               "object_body_position": f.reference[..., :3].tolist(),
               "object_body_wxyz": f.reference[..., [6, 3, 4, 5]].tolist()}
        pq.write_table(pa.Table.from_pylist([row]), self.job / "motion.parquet")
        (self.job / "assets").mkdir()
        for name in ("object.obj", "object.urdf", "support.usda"):
            (self.job / "assets" / name).write_text("fixture", encoding="utf-8")
        files = {p.relative_to(self.job).as_posix(): exporter.validation.sha256(p)
                 for p in self.job.rglob("*") if p.is_file()}
        manifest = {"schema": "v2d_policy_job_v1", "tier": 2, "episode_index": 5,
                    "sequence_id": "test_sequence", "source_frames": 4,
                    "runtime_commit": exporter.validation.RUNTIME_COMMIT, "variant": "corrected",
                    "source_sha256": exporter.validation.sha256(f.input_root / "data/chunk-000/episode_000005.parquet"),
                    "files": files}
        (self.job / "job.json").write_text(json.dumps(manifest), encoding="utf-8")

    def write_native(self):
        f = self.fixture
        achieved = f.arrays["achieved_pose_w"].astype(np.float32)
        reference = np.broadcast_to(f.reference[:, None], achieved.shape).astype(np.float32)
        columns = {"environment_id": np.tile(np.arange(2, dtype=np.int32), 4),
                   "step": np.repeat(np.arange(4, dtype=np.int32), 2),
                   "reference_frame": np.repeat(np.arange(4, dtype=np.int32), 2),
                   "object_body_id": np.zeros(8, dtype=np.int32), "object_id": np.zeros(8, dtype=np.int32),
                   "object_body_name": ["object"] * 8, "environment_valid": np.ones(8, dtype=bool)}
        for i, suffix in enumerate(exporter.POSE_SUFFIXES):
            columns[f"achieved_{suffix}"] = achieved.reshape(-1, 7)[:, i]
            columns[f"reference_{suffix}"] = reference.reshape(-1, 7)[:, i]
        self.native = f.root / "object_rollouts.parquet"
        self.reader.write_example(self.native, columns, episode_index=5, worlds=2,
                                  eval_hash="b" * 64, code_hash="c" * 64,
                                  checkpoint_hash=f.report["checkpoint_sha256"])
        table = pq.read_table(self.native)
        metadata = dict(table.schema.metadata)
        metadata.update({b"flash_chord.start_frame": b"0", b"flash_chord.step_count": b"4",
                         b"flash_chord.object_body_count": b"1", b"flash_chord.control_fps": b"50.0",
                         b"flash_chord.reference_parquet": f.report["parquet"].encode()})
        pq.write_table(table.replace_schema_metadata(metadata), self.native)

    def export(self, **kwargs):
        return exporter.export_rollout(arrays=self.fixture.arrays_path, job=self.job,
                    dataset_input=self.fixture.input_root, episode=5, kit_root=KIT_ROOT,
                    output_dir=self.output, **kwargs)

    def make_tier3_input(self):
        f = self.fixture
        f.input_root = f.root / "tier3_input_frames"
        f.input_root.mkdir()
        path = f.input_root / "episode_000005.npz"
        np.savez(path, episode_index=5, coordinate_frame="released_track2_world",
                 frame_index=np.arange(4), pose_xyzw=f.input)
        job_path = self.job / "job.json"
        job = json.loads(job_path.read_text())
        job["tier"] = 3
        job_path.write_text(json.dumps(job), encoding="utf-8")
        manifest = {"schema": "v2d_track2_input_frames_v1", "tier": 3,
                    "coordinate_frame": "released_track2_world",
                    "files": {path.name: exporter.validation.sha256(path)},
                    "episodes": {"5": {"sequence_id": "test_sequence", "object_names": ["box"],
                                       "source_sha256": job["source_sha256"]}}}
        (f.input_root / "input_frame_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        f.report["metric_reference"] = "tier3_input_reference"
        f.save()
        self.write_native()

    def test_preflight_hashes_exact_inputs_without_artifacts_or_clean_reference(self):
        shutil.rmtree(self.fixture.clean_root)
        before = sorted(str(p) for p in self.fixture.root.rglob("*"))
        result = exporter.validate_export_inputs(self.job, self.fixture.input_root, 5, KIT_ROOT)
        self.assertEqual(result["state"], "verified")
        self.assertEqual(result["tier"], 2)
        self.assertEqual(result["frame_count"], 4)
        self.assertFalse(result["clean_reference_used"])
        self.assertEqual(before, sorted(str(p) for p in self.fixture.root.rglob("*")))
        for path, digest in result["file_sha256"].items():
            self.assertTrue(Path(path).is_absolute())
            self.assertEqual(exporter.validation.sha256(path), digest)
        self.assertIn(str((self.fixture.input_root / "download_manifest.json").resolve()), result["file_sha256"])
        manifest_path = self.job / "job.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["source_sha256"] = "0" * 64
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "source hash"):
            exporter.validate_export_inputs(self.job, self.fixture.input_root, 5, KIT_ROOT)

    def test_prepared_tier3_mapping_preflights_and_exports_with_official_converter(self):
        self.make_tier3_input()
        shutil.rmtree(self.fixture.clean_root)
        result = exporter.validate_export_inputs(self.job, self.fixture.input_root, 5, KIT_ROOT)
        self.assertEqual(result["tier"], 3)
        self.assertTrue(result["conversion"]["stored_transform_verified"])
        report = self.export()
        self.assertEqual(report["tier"], 3)
        self.assertFalse(report["clean_reference_used"])
        trajectory = self.reader.read_trajectory(self.output / "episode_000005.parquet")
        np.testing.assert_allclose(trajectory.pose_xyzw, self.fixture.input[:, 0], atol=1e-6)
        self.assertEqual(trajectory.frame_index.tolist(), [0, 1, 2, 3])
        self.assertTrue(report["validation"]["official_packer_conversion_check"])

    def test_tier3_missing_or_nonrigid_mapping_fails_preflight(self):
        self.make_tier3_input()
        f = self.fixture
        path, manifest_path = f.input_root / "episode_000005.npz", f.input_root / "input_frame_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        changed = f.input.copy()
        changed[2, 0, 0] += .02
        np.savez(path, episode_index=5, coordinate_frame="released_track2_world",
                 frame_index=np.arange(4), pose_xyzw=changed)
        with self.assertRaisesRegex(ValueError, "NPZ hash"):
            exporter.validate_export_inputs(self.job, f.input_root, 5, KIT_ROOT)
        manifest["files"][path.name] = exporter.validation.sha256(path)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "one rigid transform"):
            exporter.validate_export_inputs(self.job, f.input_root, 5, KIT_ROOT)
        manifest_path.unlink()
        with self.assertRaisesRegex(ValueError, "input_frame_manifest"):
            exporter.validate_export_inputs(self.job, f.input_root, 5, KIT_ROOT)
        self.assertFalse(self.output.exists())

    def test_fixed_world_zero_export_keeps_bad_outcome_and_input_noise(self):
        # World0 is deliberately worse than world1. It must still be exported.
        self.fixture.arrays["achieved_pose_w"][:, 0, 0, 1] += 1.
        self.fixture.save()
        self.write_native()
        report = self.export()
        artifact = self.output / "episode_000005.parquet"
        trajectory = self.reader.read_trajectory(artifact)
        np.testing.assert_allclose(trajectory.pose_xyzw[:, 0], np.arange(4) * .01 + 1.05, atol=1e-6)
        np.testing.assert_allclose(trajectory.reference_xyzw[:, 0], np.arange(4) * .01 + .05, atol=1e-6)
        self.assertEqual(trajectory.world_index.tolist(), [0] * 4)
        self.assertEqual(trajectory.frame_index.tolist(), [0, 1, 2, 3])
        self.assertEqual(trajectory.object_name.tolist(), ["box"] * 4)
        self.assertEqual(report["source_world_count"], 2)
        self.assertFalse(report["clean_reference_used"])
        self.assertFalse(report["validation"]["packer_changes_coordinate_frame"])
        self.assertEqual(self.reader.provenance(trajectory), report["recorded_evaluator_provenance"])
        self.assertEqual(report["artifact"]["sha256"], exporter.validation.sha256(artifact))
        self.assertEqual(json.loads((self.output / "export_manifest.json").read_text())["state"], "complete")
        table = pq.read_table(artifact)
        self.assertEqual(table["achieved_position_x"].type, pa.float32())
        self.assertEqual(table["reference_frame"].type, pa.int32())
        np.testing.assert_allclose(table["time_s"].to_numpy(), np.arange(4) / 50.)

    def test_nonzero_selection_and_output_overwrite_rejected(self):
        with self.assertRaisesRegex(ValueError, "fixes world 0"):
            self.export(world_id=1)
        self.export()
        before = exporter.validation.sha256(self.output / "episode_000005.parquet")
        with self.assertRaises(FileExistsError):
            self.export()
        self.assertEqual(before, exporter.validation.sha256(self.output / "episode_000005.parquet"))

    def test_missing_evaluator_hash_cannot_be_fabricated(self):
        table = pq.read_table(self.native)
        metadata = dict(table.schema.metadata)
        metadata.pop(b"flash_chord.eval_code_sha256")
        pq.write_table(table.replace_schema_metadata(metadata), self.native)
        with self.assertRaisesRegex(ValueError, "lacks recorded eval_code_sha256"):
            self.export()
        self.assertFalse(self.output.exists())

    def test_npz_and_portable_job_tampering_rejected(self):
        self.fixture.arrays["achieved_pose_w"][1, 1, 0, 0] += .01
        self.fixture.save()
        with self.assertRaisesRegex(ValueError, "complete NPZ cohort"):
            self.export()
        self.write_native()
        (self.job / "assets/object.obj").write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "hash differs"):
            self.export()
        self.assertFalse(self.output.exists())

    def test_wrong_checkpoint_file_and_invalid_world_rejected(self):
        checkpoint = self.fixture.root / "wrong.safetensors"
        checkpoint.write_bytes(b"wrong weights")
        with self.assertRaisesRegex(ValueError, "checkpoint hash"):
            self.export(checkpoint=checkpoint)
        self.fixture.arrays["invalid_worlds"] = np.array([1])
        self.fixture.save()
        with self.assertRaisesRegex(ValueError, "no failed worlds may be dropped"):
            self.export()
        self.assertFalse(self.output.exists())

    def test_selected_manifest_binds_hashes_and_uses_pack_friendly_paths(self):
        f = self.fixture
        policy = f.root / "policy_123.safetensors"
        policy.write_bytes(b"fixture policy bytes")
        f.report["checkpoint_sha256"] = exporter.validation.sha256(policy)
        f.save()
        self.write_native()
        selected = {"schema": "v2d_selected_policy_v1", "clean_reference_used": False,
                    "world0_export_required": True, "tier": 2, "episode_index": 5,
                    "evaluation_directory": str(f.root), "job": str(self.job), "policy": str(policy),
                    "environment_steps": 123, "checkpoint_sha256": exporter.validation.sha256(policy),
                    "evaluation_sha256": exporter.validation.sha256(f.root / "evaluation.json"),
                    "rollout_arrays_sha256": exporter.validation.sha256(f.arrays_path),
                    "input_job_sha256": exporter.validation.sha256(self.job / "job.json")}
        path = f.root / "selected_checkpoint.json"
        path.write_text(json.dumps(selected), encoding="utf-8")
        result = exporter.export_selected(path, f.input_root, KIT_ROOT, self.output)
        self.assertEqual(Path(result["parquet"]), self.output / "tier2/episode_000005/episode_000005.parquet")
        manifest = json.loads(Path(result["manifest"]).read_text())
        self.assertEqual(manifest["selected_policy_manifest"]["sha256"], exporter.validation.sha256(path))
        selected["rollout_arrays_sha256"] = "0" * 64
        path.write_text(json.dumps(selected), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "evidence hash differs"):
            exporter.export_selected(path, f.input_root, KIT_ROOT, f.root / "another_export")

    def test_collection_verifies_completed_runs_and_copies_only_fixed_world_artifacts(self):
        f = self.fixture
        selected = f.root / "selected_checkpoint.json"
        selected.write_text('{"schema":"fixture_selection"}', encoding="utf-8")
        report = self.export(selection_provenance=exporter.validation.identity(selected))
        artifact, manifest = self.output / "episode_000005.parquet", self.output / "export_manifest.json"
        record = {"parquet": str(artifact), "manifest": str(manifest), "tier": 2, "episode": 5,
                  "parquet_sha256": exporter.validation.sha256(artifact), "manifest_sha256": exporter.validation.sha256(manifest),
                  "selected_manifest_sha256": exporter.validation.sha256(selected), "source_world_id": 0, "state": "complete"}
        (f.root / "export_complete.json").write_text(json.dumps(record), encoding="utf-8")
        (f.root / "pipeline_status.json").write_text(json.dumps({"state": "complete", "export": record,
                     "selected_manifest_sha256": record["selected_manifest_sha256"]}), encoding="utf-8")
        incomplete = f.root / "unfinished"
        incomplete.mkdir()
        (incomplete / "pipeline_status.json").write_text('{"state":"running"}', encoding="utf-8")
        collection = f.root / "collected"
        result = exporter.collect_runs([f.root], collection, KIT_ROOT)
        copied = collection / "tier2/episode_000005.parquet"
        self.assertEqual(list(collection.rglob("*.parquet")), [copied])
        self.assertEqual(exporter.validation.sha256(copied), record["parquet_sha256"])
        self.assertEqual(result["complete_30_episode_tiers"], [])
        self.assertEqual(result["pack_argv_templates"], {})
        self.assertEqual(len(result["skipped_incomplete_runs"]), 1)
        self.assertFalse(result["clean_reference_used"])
        with self.assertRaises(FileExistsError):
            exporter.collect_exports([record], collection, KIT_ROOT)
        rejected = f.root / "rejected"
        with self.assertRaisesRegex(ValueError, "Duplicate export"):
            exporter.collect_exports([record, record], rejected, KIT_ROOT)
        with self.assertRaisesRegex(ValueError, "expected roster"):
            exporter.collect_exports([record], rejected, KIT_ROOT, {2: [5, 6]})
        artifact.write_bytes(b"corrupted")
        with self.assertRaisesRegex(ValueError, "parquet hash"):
            exporter.collect_exports([record], rejected, KIT_ROOT)
        self.assertFalse(rejected.exists())


if __name__ == "__main__":
    unittest.main()

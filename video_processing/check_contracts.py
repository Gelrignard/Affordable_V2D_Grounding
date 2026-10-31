"""CPU contract checks. Run: python -m video_processing.check_contracts"""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

from video_processing.common import digest
from video_processing.correcting.core.algorithm import Candidate, assess, select_path
from video_processing.correcting.core.repair import generate
from video_processing.correcting.input import DEFAULTS, validate_plan
from video_processing.processing.input import validate_prompts
from video_processing.processing.output.geometry import apply_human_transform, object_source_mesh_poses
from video_processing.processing.output.soma import load_calibration


def pose(x=0):
    result = np.eye(4)
    result[0, 3] = x
    result[2, 3] = 2
    return result


def measured(p, origin, eligible=True, score=.9, anchor=None):
    return Candidate(p.copy(), origin, {"observed_visible_pixels": 100}, eligible, score, [], anchor)


class Contracts(unittest.TestCase):
    def test_review_represents_each_unresolved_run(self):
        from video_processing.correcting.output.review import representative_frames
        self.assertEqual(representative_frames([1, 2, 3, 4, 5, 8, 10, 11]), {1, 3, 5, 8, 10, 11})

    def test_cli_help_is_model_free(self):
        for module in ("processing", "correcting"):
            result = subprocess.run([sys.executable, "-m", f"video_processing.{module}.run", "--help"], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("--help", result.stdout)

    def test_prompts_are_image_coordinates(self):
        value = {"prompts": [{"object_id": i, "frame_index": 3, "points": [{"x": 100, "y": 200}], "point_labels": [1]} for i in (0, 1)]}
        validate_prompts(value, 5)
        value["prompts"][0]["frame_index"] = 5
        with self.assertRaises(ValueError):
            validate_prompts(value, 5)

    def test_anchor_binds_exact_reconstruction_and_review(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            image = root / "review.jpg"
            image.write_bytes(b"fixture")
            plan = {"schema": "v2d.reviewed_anchor_repair.v1", "baseline_sha256": "a" * 64,
                    "anchors": [{"id": "middle", "frame": 3, "review_image": "review.jpg", "review_image_sha256": digest(image), "reviewer": "operator", "note": "Image-supported reconstructed pose"}],
                    "intervals": [{"start": 1, "end": 4, "anchor_ids": ["middle"]}]}
            indices, _, _ = validate_plan(plan, 6, "a" * 64, root)
            self.assertEqual(indices, [1, 2, 3, 4])
            plan["tier1_trajectory"] = "forbidden.npz"
            with self.assertRaises(ValueError):
                validate_plan(plan, 6, "a" * 64, root)
            del plan["tier1_trajectory"]
            image.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                validate_plan(plan, 6, "a" * 64, root)

    def test_bidirectional_tracking_from_one_anchor(self):
        class Backend:
            def __init__(self):
                self.visited = []
            def measure(self, frame, poses, origin, anchor_id=None):
                good = origin != "baseline" or frame == 3
                return [measured(p, origin, good, .9, anchor_id) for p in poses]
            def track(self, frame, previous):
                self.visited.append(frame)
                return pose(frame * .01)
        backend = Backend()
        track = {"fp_poses_cam": np.asarray([pose(i * .01) for i in range(7)])[:, None]}
        plan = {"anchors": [{"id": "mid", "frame": 3}], "intervals": [{"start": 1, "end": 5, "anchor_ids": ["mid"]}]}
        _, selected, unresolved, _, reasons, _ = generate(backend, plan, track, [1, 2, 3, 4, 5], DEFAULTS)
        self.assertEqual(backend.visited, [4, 5, 2, 1])
        self.assertEqual(set(selected), {1, 2, 3, 4, 5})
        self.assertEqual(unresolved, [])
        self.assertEqual(reasons, [])

    def test_unsupported_tracking_stops_and_remains_unresolved(self):
        class Backend:
            def __init__(self):
                self.visited = []
            def measure(self, frame, poses, origin, anchor_id=None):
                return [measured(p, origin, frame == 0, anchor=anchor_id) for p in poses]
            def track(self, frame, previous):
                self.visited.append(frame)
                return pose(frame * .01)
        backend = Backend()
        track = {"fp_poses_cam": np.asarray([pose() for _ in range(10)])[:, None]}
        plan = {"anchors": [{"id": "a", "frame": 0}], "intervals": [{"start": 0, "end": 9, "anchor_ids": ["a"]}]}
        _, _, unresolved, _, reasons, _ = generate(backend, plan, track, list(range(10)), DEFAULTS)
        self.assertEqual(backend.visited, [1, 2, 3])
        self.assertEqual(unresolved, list(range(1, 10)))
        self.assertTrue(reasons)

    def test_selection_returns_actual_estimates_without_interpolation(self):
        a, b = measured(pose(.01), "forward"), measured(pose(.02), "backward")
        selected, unresolved = select_path({0: [a], 1: [], 2: [b]}, [0, 1, 2], np.asarray([pose() for _ in range(3)]), DEFAULTS)
        self.assertIs(selected[0], a)
        self.assertIs(selected[2], b)
        self.assertEqual(unresolved, [1])

    def test_bad_depth_cannot_pass_good_silhouette(self):
        metrics = {"iou": .95, "precision": .95, "recall": .95, "observed_visible_pixels": 500,
                   "depth_pixels": 200, "depth_median_error_m": 2., "observed_depth_median_m": 3.}
        eligible, _, reasons = assess(metrics, DEFAULTS)
        self.assertFalse(eligible)
        self.assertIn("depth_disagreement", reasons)

    def test_calibration_is_explicit_rigid_and_frame_zero_only(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "calibration.json"
            data = {"schema": "v2d.camera_world_calibration.v1", "source": "manual camera calibration", "world_from_camera": np.eye(4).tolist()}
            path.write_text(json.dumps(data))
            transform, _ = load_calibration(path)
            self.assertTrue(np.array_equal(transform, np.eye(4)))
            data["world_from_camera"][0][0] = 2
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                load_calibration(path)
            data["world_from_camera"][0][0] = 1
            data["benchmark_frame_indices_used"] = [0, 5]
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                load_calibration(path)

    def test_human_and_original_mesh_share_world_transform(self):
        world = np.eye(4)
        world[:3, 3] = [1, 2, 3]
        positions, quat = apply_human_transform(np.zeros((2, 77, 3)), np.tile([1., 0, 0, 0], (2, 77, 1)), world)
        obj = object_source_mesh_poses(np.tile(np.eye(4), (2, 1, 1)), np.eye(4), world)
        np.testing.assert_allclose(positions[:, 0], obj[:, :3, 3])
        np.testing.assert_allclose(quat, np.tile([1, 0, 0, 0], (2, 77, 1)))

    def test_native_boundary_stops_before_refinement(self):
        from video_processing.processing.core import native
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            fake = types.ModuleType("v2d.cari4d.lib.run_inference")
            fake.__file__ = str(root / "reconstruction/modules/v2d_cari4d/lib/run_inference.py")
            seen = []
            def stage(stage_name, command, *args):
                seen.append((stage_name, command))
                return {"name": stage_name}
            fake._stage_run = stage
            def run(*args, **kwargs):
                for stage_name in ("01_depth", "05_object_pose", "06_coconet", "07_refinement", "08_render"):
                    fake._stage_run(stage_name, ["python", "source.py"], [], [], root, {}, False)
            fake.run_inference = run
            lib = types.ModuleType("v2d.cari4d.lib")
            lib.run_inference = fake
            request = {k: str(root) for k in ("upstream_root", "weights_root", "video", "masks", "mesh", "output", "result_root")}
            request.update(stop_after="coconet", frames=1, identity_sha256="fixture")
            with patch.object(native, "verify_upstream", return_value=root), patch.object(native, "activate_runtime"), patch.dict(sys.modules, {"v2d.cari4d.lib": lib}):
                native.execute(request)
            self.assertEqual([x[0] for x in seen], ["01_depth", "05_object_pose", "06_coconet"])
            self.assertIn("video_processing.processing.core.foundationpose", seen[1][1])
            self.assertIs(fake._stage_run, stage)

    def test_complete_native_merge_preserves_outside_frames(self):
        from video_processing.correcting.output.native import merge_selected, METRICS_TO_NATIVE
        count = 5
        camera = np.asarray([pose(i * .01) for i in range(count)], dtype=np.float32)
        original = {"frames": [f"{i:06d}" for i in range(count)], "camera_id": 0,
                    "fp_poses_cam": camera[:, None], "obj_pose_world": camera.copy(),
                    "obj_rot_init": camera[:, :3, :3].copy(), "obj_t_init": camera[:, :3, 3].copy(),
                    "metadata": {"object_pose_fallback_reasons": [""] * count}}
        fields = list(METRICS_TO_NATIVE.values()) + ["fp_selected_iou", "fp_selected_visibility", "fp_reliable", "fp_candidate_count", "fp_selected_candidate_index", "fp_retry_count", "fp_translation_threshold_m", "fp_selected_translation_distance_m", "fp_confirmed_pose_source", "fp_emitted_pose_source", "fp_pending_pose_source", "fp_tracking_state", "fp_gap_length", "fp_confirmation_count", "fp_rejection_mask", "fp_reacquisition_latency"]
        for key in fields:
            original[key] = np.zeros(count, np.float32)
        original["fp_observed_visible_pixels"] = np.full(count, 100)
        value = measured(pose(.07).astype(np.float32), "forward_track")
        value.metrics.update({key: .8 for key in METRICS_TO_NATIVE})
        value.metrics["rendered_visible_pixels"] = 100
        native = {"PoseSource": types.SimpleNamespace(REACQUISITION_CANDIDATE=2, FOUNDATIONPOSE_TRACK=1, NONE=0), "TrackingState": types.SimpleNamespace(TRACKING=1)}
        merged = merge_selected(original, {2: value}, [1, 2, 3], [1, 3], np.eye(4), native)
        np.testing.assert_array_equal(merged["fp_poses_cam"][[0, 4]], original["fp_poses_cam"][[0, 4]])
        np.testing.assert_array_equal(merged["fp_poses_cam"][2, 0], value.pose)
        np.testing.assert_array_equal(merged["targeted_repair_unresolved"], [False, True, False, True, False])

    def test_soma_output_roundtrips_actual_repository_reader(self):
        from video_processing.processing.output import soma
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "retargeting/src"))
        from v2d_retarget.dataprocessing.readers import load_npz
        class FakeConverter:
            names = np.asarray([f"joint_{i}" for i in range(77)])
            def __init__(self, weights):
                pass
            def convert(self, params):
                n = len(params["mhr_trans"])
                return {"positions": np.zeros((n, 77, 3)), "rotations_wxyz": np.tile([1., 0, 0, 0], (n, 77, 1)),
                        "fit_error_mean_m": np.zeros(n), "fit_error_max_m": np.zeros(n)}
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            mesh, source, calibration = root / "object.glb", root / "native.npz", root / "calibration.json"
            mesh.write_bytes(b"mesh fixture; reader checks local identity")
            np.savez(source, timestamps=np.arange(3) / 30, predicted_mhr_trans=np.zeros((3, 3)),
                     predicted_object_pose_camera=np.tile(np.eye(4), (3, 1, 1)),
                     source_object_mesh_to_pose_body_transform=np.eye(4), sequence_id="fixture", hand_contact_logits=np.ones((3, 2)))
            source.with_suffix(".json").write_text(json.dumps({"artifact": {"sha256": digest(source)}, "source_mesh": {"sha256": digest(mesh)}, "mesh_transform_status": "upstream_transform_and_current_local_mesh_identities_matched"}))
            calibration.write_text(json.dumps({"schema": "v2d.camera_world_calibration.v1", "source": "manual fixture", "world_from_camera": np.eye(4).tolist()}))
            args = types.SimpleNamespace(native=source, mesh=mesh, calibration=calibration, output=root / "converted", upstream_root=root, weights_root=root, batch_size=2, dry_run=False)
            fake_module = types.SimpleNamespace(Converter=FakeConverter, ASSET_REVISION="fixture")
            with patch.object(soma, "verify_upstream", return_value=root), patch.object(soma, "activate_runtime"), patch.dict(sys.modules, {"torch": types.SimpleNamespace(from_numpy=lambda a: a), "video_processing.processing.output.mhr_soma": fake_module}):
                soma.convert(args)
            motion = load_npz(args.output / "human_motion.npz")
            self.assertEqual(motion.positions.shape, (3, 77, 3))
            self.assertEqual(motion.metadata["calibration"]["source"], "manual fixture")
            with np.load(args.output / "human_motion.npz", allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["hand_contact_logits"], np.ones((3, 2)))


if __name__ == "__main__":
    unittest.main(verbosity=2)

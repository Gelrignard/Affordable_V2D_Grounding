"""Export trusted, locally produced CARI4D bundles as compact pickle-free NPZ.

This preserves native MHR parameters and camera-space rigid object motion. It
does not invent SOMA joint rotations, camera calibration, or a challenge-world
transform. Only load bundles produced by the pinned runner: torch's upstream
bundle format contains Python pickle and is not safe for arbitrary downloads.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np


UPSTREAM_COMMIT = "193382a184308f6800331f3db829f76dc4294cfe"
BUNDLE_SCHEMA = "cari4d.mhr_wild_inference.v1"
ARTIFACT_SCHEMA = "v2dchallenge.native_cari4d_reconstruction.v1"
MHR_DIMS = {"mhr_global_rot6d": 6, "mhr_trans": 3,
            "mhr_body_pose_cont": 260, "mhr_hand": 108, "mhr_shape": 45,
            "mhr_scale": 28, "mhr_face": 72}
POSE_FRAME = "centered_axis_aligned"
POSE_FRAME_REVISION = "cari4d.object_pose_frame.centered_axis_aligned.v1"


def file_identity(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": digest.hexdigest()}


def _array(value: Any, name: str, shape: tuple[int, ...]) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    value = np.asarray(value)
    if value.shape != shape or value.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be a numerical array of shape {shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values")
    return value.astype(np.float64, copy=False)


def _rigid(value: Any, name: str, count: int | None = None) -> np.ndarray:
    array = _array(value, name, (4, 4) if count is None else (count, 4, 4))
    rotation = array[..., :3, :3]
    if not np.allclose(array[..., 3, :], [0, 0, 0, 1], atol=1e-5, rtol=0):
        raise ValueError(f"{name} has invalid homogeneous bottom rows")
    if not np.allclose(np.swapaxes(rotation, -1, -2) @ rotation,
                       np.eye(3), atol=1e-4, rtol=0):
        raise ValueError(f"{name} contains a non-orthonormal rotation")
    if not np.allclose(np.linalg.det(rotation), 1, atol=1e-4, rtol=0):
        raise ValueError(f"{name} contains a reflection or scaled rotation")
    return array


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float) and np.isfinite(value):
        return value
    raise ValueError(f"Unsupported or non-finite metadata value: {type(value).__name__}")


def validate_bundle(bundle: Mapping[str, Any], expected_frames: int) -> dict[str, np.ndarray]:
    """Validate the pinned all-frame camera-space bundle without decoding MHR."""
    if expected_frames < 1 or bundle.get("schema") != BUNDLE_SCHEMA:
        raise ValueError("Expected a current native CARI4D wild bundle and positive frame count")
    if bundle.get("kid") != 0:
        raise ValueError("Only the official monocular camera-0 export is supported")
    names = bundle.get("frames", [])
    if len(names) != expected_frames or any(not isinstance(n, str) or not n.isdecimal() for n in names):
        raise ValueError("frames must contain one decimal source-frame name per video frame")
    indices = np.array([int(n) for n in names], dtype=np.int64)
    if not np.array_equal(indices, np.arange(expected_frames)):
        raise ValueError("Only complete, zero-based source frame identity is accepted; no reindexing")
    frame_meta = bundle.get("frame_meta", [])
    if len(frame_meta) != expected_frames:
        raise ValueError("frame_meta must cover every source frame")
    for name, index, record in zip(names, indices, frame_meta):
        if (record.get("frame") != name or record.get("src_frame") != int(index)
                or record.get("kid") != 0):
            raise ValueError("frame_meta disagrees with exact source frame identity")
    metadata = bundle.get("metadata", {})
    if metadata.get("ground_truth_used") is not False:
        raise ValueError("Bundle must explicitly declare ground_truth_used=false")
    if (metadata.get("object_pose_frame") != POSE_FRAME
            or metadata.get("object_pose_frame_revision") != POSE_FRAME_REVISION):
        raise ValueError("Object-body frame is missing, stale, or unsupported")
    arrays: dict[str, np.ndarray] = {"frame_names": np.array(names), "source_frame_indices": indices}
    for key in ("object_pose_storage_to_training_transform", "object_mesh_to_training_transform"):
        if key not in metadata:
            raise ValueError(f"Missing explicit object frame transform: {key}")
        arrays[key] = _rigid(metadata[key], key)
    for block_name, prefix in (("pr", "predicted"), ("in", "initialized")):
        block = bundle.get(block_name, {})
        arrays[f"{prefix}_object_pose_camera"] = _rigid(
            block.get("pose_abs"), f"{block_name}.pose_abs", expected_frames)
        for key, dim in MHR_DIMS.items():
            array = _array(block.get(key), f"{block_name}.{key}", (expected_frames, dim))
            # Validate 6D axes can define a proper rotation; coefficients are not
            # quaternions and need not have unit norm before decoder projection.
            if key == "mhr_global_rot6d":
                # Native global_rot6d flattens the first two matrix columns
                # row-major (interleaved), unlike the body's 6D controls.
                first, second = array[:, 0::2], array[:, 1::2]
                if np.any(np.linalg.norm(np.cross(first, second), axis=-1) < 1e-8):
                    raise ValueError(f"{block_name}.{key} contains degenerate rotation axes")
            arrays[f"{prefix}_{key}"] = array.astype(np.float32)
    if "contact_logits" in bundle["pr"]:
        logits = _array(bundle["pr"]["contact_logits"], "pr.contact_logits", (expected_frames, 2))
        arrays["hand_contact_logits"] = logits.astype(np.float32)
        arrays["network_positive_hand_contact"] = logits > 0
    postopt = bundle.get("postopt", {})
    if postopt and postopt.get("frame_indices") != indices.tolist():
        raise ValueError("Post-optimization did not retain the full original frame selection")
    return arrays


def _trusted_load(path: Path) -> Mapping[str, Any]:
    import torch
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(bundle, Mapping):
        raise ValueError("Upstream bundle must be a mapping")
    return bundle


def export_artifact(bundle_path: Path, output: Path, *, fps: float,
                    expected_frames: int, sequence_id: str, source_mesh: Path,
                    upstream_commit: str = UPSTREAM_COMMIT,
                    aligned_mesh: Path | None = None,
                    export_metadata: Path | None = None) -> Path:
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    if upstream_commit != UPSTREAM_COMMIT:
        raise ValueError("Exporter contract is only verified for the pinned upstream commit")
    if not sequence_id or output.suffix.lower() != ".npz":
        raise ValueError("Supply a sequence ID and an .npz output path")
    report_path = output.with_suffix(".json")
    if output.exists() or report_path.exists():
        raise FileExistsError(f"Refusing to overwrite {output} or {report_path}")
    bundle_path = bundle_path.resolve()
    bundle = _trusted_load(bundle_path)
    arrays = validate_bundle(bundle, expected_frames)
    arrays["timestamps"] = arrays["source_frame_indices"].astype(np.float64) / fps
    arrays["fps"] = np.array(fps, dtype=np.float64)
    arrays["schema"] = np.array(ARTIFACT_SCHEMA)
    arrays["sequence_id"] = np.array(sequence_id)
    arrays["coordinate_frame"] = np.array("source_camera_opencv")
    arrays["object_body_frame"] = np.array(POSE_FRAME)
    arrays["units"] = np.array("metres_seconds; native MHR parameters retain decoder-defined units")
    report: dict[str, Any] = {
        "schema": ARTIFACT_SCHEMA, "sequence_id": sequence_id,
        "upstream_commit": upstream_commit, "source_bundle": file_identity(bundle_path),
        "source_mesh": file_identity(source_mesh),
        "aligned_mesh": None if aligned_mesh is None else file_identity(aligned_mesh),
        "frame_count": expected_frames, "fps": fps,
        "timestamp_source": "zero-based source frame index divided by caller-supplied constant video fps",
        "coordinate_frame": "source_camera_opencv",
        "retarget_ready": False, "evaluation_world_transform": None,
        "human_representation": "Native CARI4D MHR decoder parameters; not SOMA-X pose columns or decoded joint orientations.",
        "object_pose_semantics": "T_camera_from_centered_axis_aligned_body; translation in metres, no camera-to-evaluation-world alignment applied.",
        "mesh_transform_status": "unverified_source_mesh_to_aligned_correspondence",
        "contact_semantics": "Two hand-contact logits; logit>0 is network-positive only. The final fixed geometry eligibility gate is described in postopt metadata. Contacts are not forces, confidence, visibility, or feasibility certificates.",
        "bundle_metadata": _jsonable(bundle.get("metadata", {})),
        "checkpoint": _jsonable(bundle.get("checkpoint", {})),
        "mhr_inference": _jsonable(bundle.get("mhr_inference", {})),
        "postopt": _jsonable(bundle.get("postopt", {})),
    }
    if export_metadata is not None:
        raw_metadata = json.loads(export_metadata.read_text(encoding="utf-8"))
        if (raw_metadata.get("schema") != "cari4d.mhr_wild_export.v2"
                or raw_metadata.get("sequence") != sequence_id
                or raw_metadata.get("frame_count") != expected_frames
                or raw_metadata.get("camera_id") != 0):
            raise ValueError("Wild export metadata does not match the sequence/frame/camera contract")
        for key in ("object_mesh_to_training_transform", "object_pose_storage_to_training_transform"):
            transform = _rigid(raw_metadata.get(key), f"wild_export.{key}")
            if not np.allclose(transform, arrays[key], atol=1e-6, rtol=0):
                raise ValueError(f"Export metadata and reconstruction disagree on {key}")
        report["wild_export_metadata"] = _jsonable(raw_metadata)
        report["wild_export_source"] = file_identity(export_metadata)
        original_to_aligned = _rigid(raw_metadata.get("source_object_mesh_to_aligned_transform"),
                                     "source_object_mesh_to_aligned_transform")
        arrays["source_object_mesh_to_aligned_transform"] = original_to_aligned
        arrays["source_object_mesh_to_pose_body_transform"] = (
            arrays["object_mesh_to_training_transform"] @ original_to_aligned)
        # Metadata carries the transform, but verifying it belongs to the exact
        # supplied original mesh requires an explicit source identity comparison.
        report["mesh_transform_status"] = "upstream_transform_preserved; caller_mesh_correspondence_requires_verification"
        source_identity = raw_metadata.get("source_object_mesh", {})
        source_stat = source_mesh.stat()
        original_matches = (Path(source_identity.get("path", "")).resolve() == source_mesh.resolve()
                            and source_identity.get("size") == source_stat.st_size
                            and source_identity.get("mtime_ns") == source_stat.st_mtime_ns)
        aligned_matches = (aligned_mesh is not None
                           and Path(raw_metadata.get("object_mesh_file", "")).resolve() == aligned_mesh.resolve())
        report["mesh_identity_checks"] = {
            "original_matches_upstream_path_size_mtime": original_matches,
            "aligned_matches_upstream_path": aligned_matches,
            "note": "Upstream export records path/size/mtime, not content hashes. This export hashes current mesh files."}
        if original_matches and aligned_matches:
            report["mesh_transform_status"] = "upstream_transform_and_current_local_mesh_identities_matched"
    arrays["metadata_json"] = np.array(json.dumps(report, sort_keys=True, allow_nan=False))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        # Confirm the delivered file can be opened with object deserialization disabled.
        with np.load(temporary, allow_pickle=False) as result:
            for key in result.files:
                result[key]
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    report["artifact"] = file_identity(output)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument("--expected-frames", type=int, required=True)
    parser.add_argument("--sequence-id", required=True)
    parser.add_argument("--source-mesh", type=Path, required=True)
    parser.add_argument("--aligned-mesh", type=Path)
    parser.add_argument("--export-metadata", type=Path)
    parser.add_argument("--upstream-commit", default=UPSTREAM_COMMIT)
    args = vars(parser.parse_args())
    args["bundle_path"] = args.pop("bundle")
    print(export_artifact(**args))


if __name__ == "__main__":
    main()

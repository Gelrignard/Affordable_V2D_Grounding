"""Export one fixed-world Track 2 trajectory for the official submission packer.

CPU only: no clean scoring reference, simulator, checkpoint execution, or upload.
Inputs are training.evaluate.policy's arrays/evaluation.json/native trajectory,
a hash-verified portable job, the released input-tier dataset, and the pinned kit.
World 0 is fixed for every episode; it is never selected by an outcome metric.
The output directory must be new. A complete export_manifest.json is written last.

The official packer needs all episodes for a tier and a code commit URL. This
helper makes one validated trajectory, not a complete or submitted competition
entry. Single-object Tier 1/2 use released LeRobot input. Prepared Tier 3 uses
the official packer's NPZ input-frame format with explicit hash/identity binding;
an arbitrary reconstruction camera frame is not a released-input frame mapping.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import sys
from types import ModuleType, SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from training.evaluate import score_clean_reference as validation


SUBMISSION_READER = "v2dlb/submission_parquet.py"
SUBMISSION_READER_SHA256 = "cc2808c3c1b28a662c2f1ca508d4edde8e7bd4e34d0892cd33cb6199eb828bba"
POSE_SUFFIXES = ("position_x", "position_y", "position_z", "quaternion_x",
                 "quaternion_y", "quaternion_z", "quaternion_w")


def load_submission_reader(kit_root):
    """Execute the pinned official reader in memory without writing into the kit."""
    root = Path(kit_root).resolve()
    manifest = json.loads((root / "KIT_MANIFEST.json").read_text(encoding="utf-8"))
    if (validation.sha256(root / SUBMISSION_READER) != SUBMISSION_READER_SHA256
            or manifest.get("files", {}).get(SUBMISSION_READER) != SUBMISSION_READER_SHA256):
        raise ValueError("Official submission reader hash/manifest mismatch")
    name = "_policy_export_official_submission_parquet"
    module = ModuleType(name)
    module.__file__ = str(root / SUBMISSION_READER)
    sys.modules[name] = module
    try:
        exec(compile((root / SUBMISSION_READER).read_text(encoding="utf-8"), module.__file__, "exec"), module.__dict__)
    finally:
        sys.modules.pop(name, None)
    return module


def read_input_reference(root, episode, kit):
    """Read released inputs or explicitly prepared Tier3 packer references.

    A Tier3 directory has episode_NNNNNN.npz in the official packer format and
    input_frame_manifest.json with schema v2d_track2_input_frames_v1, tier 3,
    coordinate_frame released_track2_world, files {filename: sha256}, and
    episodes {str(episode): {sequence_id, object_names, source_sha256}}. The
    source hash binds the reconstruction input used by the portable job. This
    function verifies that declaration; it cannot calibrate a camera frame.
    """
    root = Path(root).resolve()
    if (root / "meta").is_dir():
        data = validation.read_dataset(root, episode, kit)
        data.source_sha256 = data.provenance["parquet"]["sha256"]
        return data
    manifest_path = root / "input_frame_manifest.json"
    if not manifest_path.is_file():
        raise ValueError("Input must be a released Tier1/2 dataset or a prepared Tier3 input_frame_manifest.json directory")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("schema") != "v2d_track2_input_frames_v1" or manifest.get("tier") != 3
            or manifest.get("coordinate_frame") != "released_track2_world"):
        raise ValueError("Tier3 input-frame manifest must explicitly declare the released input frame")
    record = manifest.get("episodes", {}).get(str(episode), {})
    sequence, names, source_hash = record.get("sequence_id"), record.get("object_names"), record.get("source_sha256", "")
    if (not isinstance(sequence, str) or not sequence or not isinstance(names, list)
            or len(names) != 1 or not isinstance(names[0], str) or not names[0]
            or not re.fullmatch(r"[0-9a-f]{64}", source_hash)):
        raise ValueError("Tier3 input-frame identity must bind one sequence/object and portable source hash")
    path = root / f"episode_{episode:06d}.npz"
    if manifest.get("files", {}).get(path.name) != validation.sha256(path):
        raise ValueError("Tier3 input-frame NPZ hash differs from manifest")
    with np.load(path, allow_pickle=False) as data:
        required = {"episode_index", "coordinate_frame", "frame_index", "pose_xyzw"}
        if not required.issubset(data.files):
            raise ValueError("Tier3 input-frame NPZ lacks official packer fields")
        if (np.asarray(data["episode_index"]).shape != () or data["episode_index"].item() != episode
                or np.asarray(data["coordinate_frame"]).shape != ()
                or data["coordinate_frame"].item() != "released_track2_world"):
            raise ValueError("Tier3 input-frame NPZ episode/coordinate frame differs")
        frames, pose = np.asarray(data["frame_index"]), np.asarray(data["pose_xyzw"], dtype=np.float64)
    if (frames.ndim != 1 or frames.dtype.kind not in "iu" or not len(frames)
            or not np.array_equal(frames, np.arange(len(frames))) or pose.shape != (len(frames), 1, 7)
            or not np.isfinite(pose).all() or np.any(np.linalg.norm(pose[..., 3:], axis=-1) < 1e-8)):
        raise ValueError("Tier3 input-frame NPZ must contain finite poses on the complete ordered frame grid")
    return SimpleNamespace(root=root, path=path, sequence=sequence, frames=frames.astype(np.int64),
                           reference=SimpleNamespace(frame_index=frames, pose_xyzw=pose, object_names=names),
                           tier=3, source_sha256=source_hash,
                           provenance={"input_frame_archive": validation.identity(path),
                                       "input_frame_manifest": validation.identity(manifest_path),
                                       "tier": 3, "source_sequence_id": sequence, "object_names": names,
                                       "source_sha256": source_hash, "declared_coordinate_frame": "released_track2_world",
                                       "mapping_source": "prepared input-only mapping; no calibration inferred by exporter"})


def verify_job(job, episode, input_data, recorded_reference):
    import pyarrow.parquet as pq

    job = Path(job).resolve()
    manifest = json.loads((job / "job.json").read_text(encoding="utf-8"))
    expected = {"schema": "v2d_policy_job_v1", "tier": input_data.tier,
                "episode_index": episode, "sequence_id": input_data.sequence,
                "source_frames": len(input_data.frames), "runtime_commit": validation.RUNTIME_COMMIT,
                "source_sha256": input_data.source_sha256}
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("Portable job identity/runtime/source hash does not match released input")
    files = manifest.get("files", {})
    required = {"reference.npz", "motion.parquet", "assets/object.obj",
                "assets/object.urdf", "assets/support.usda"}
    if not required.issubset(files):
        raise ValueError("Portable job is missing required input assets")
    for name, digest in files.items():
        path = (job / name).resolve()
        if not path.is_relative_to(job) or validation.sha256(path) != digest:
            raise ValueError(f"Portable job asset escapes root or hash differs: {name}")
    rows = pq.ParquetFile(job / "motion.parquet").read().to_pylist()
    if (len(rows) != 1 or rows[0].get("sequence_id") != input_data.sequence
            or rows[0].get("robot_name") != "g1" or len(rows[0].get("object_body_names", [])) != 1):
        raise ValueError("Portable motion must describe the same single-object G1 episode")
    row = rows[0]
    position, quaternion = np.asarray(row["object_body_position"]), np.asarray(row["object_body_wxyz"])
    if position.shape != recorded_reference[..., :3].shape or quaternion.shape != recorded_reference[..., 3:].shape:
        raise ValueError("Portable motion object/frame shape differs from evaluation")
    pose = np.concatenate((position, quaternion[..., [1, 2, 3, 0]]), axis=-1)
    if not np.allclose(pose, recorded_reference, atol=1e-7, rtol=0):
        raise ValueError("Portable motion object trajectory differs from evaluation")
    transform, reference_provenance = validation.stored_reference_check(
        job / "reference.npz", recorded_reference, input_data.frames)
    return transform, {"root": str(job), "manifest": validation.identity(job / "job.json"),
                       "file_sha256": files, "variant": manifest.get("variant"),
                       "reference_archive": reference_provenance}


def verify_native_trajectory(path, reader, evaluation, achieved, reference, episode, object_name):
    """Bind native evaluator hashes to the complete NPZ cohort, before selection."""
    trajectory = reader.read_trajectory(path)
    steps, worlds = achieved.shape[:2]
    metadata = trajectory.metadata
    expected_meta = {"flash_chord.schema": "flash_chord_object_trajectories_v1",
                     "flash_chord.position_unit": "meter", "flash_chord.quaternion_order": "xyzw",
                     "flash_chord.start_frame": "0", "flash_chord.step_count": str(steps),
                     "flash_chord.world_count": str(worlds), "flash_chord.object_body_count": "1"}
    if any(metadata.get(key) != value for key, value in expected_meta.items()):
        raise ValueError("Native trajectory metadata differs from evaluation contract")
    if (trajectory.episode_index != episode
            or not np.array_equal(trajectory.world_index, np.tile(np.arange(worlds), steps))
            or not np.array_equal(trajectory.frame_index, np.repeat(np.arange(steps), worlds))
            or not np.all(trajectory.object_slot == 0) or not np.all(trajectory.object_id == 0)
            or not trajectory.world_valid.all() or trajectory.reference_xyzw is None):
        raise ValueError("Native trajectory has an invalid/incomplete full-cohort frame/world grid")
    if trajectory.object_name is None or not np.isin(trajectory.object_name, ["object", object_name]).all():
        raise ValueError("Native trajectory object name disagrees with the released input")
    expected_a = achieved.astype(np.float32).reshape(-1, 7).astype(np.float64)
    expected_r = np.broadcast_to(reference[:, None], achieved.shape).astype(np.float32).reshape(-1, 7).astype(np.float64)
    if (not np.array_equal(trajectory.pose_xyzw, expected_a)
            or not np.array_equal(trajectory.reference_xyzw, expected_r)):
        raise ValueError("Native trajectory poses do not match the complete NPZ cohort")
    fps = float(evaluation.get("sim_control_fps", 0))
    if not np.isfinite(fps) or fps <= 0 or float(metadata.get("flash_chord.control_fps", 0)) != fps:
        raise ValueError("Native trajectory/evaluation control rates differ or are invalid")
    if metadata.get("flash_chord.reference_parquet") != evaluation.get("parquet"):
        raise ValueError("Native trajectory/evaluation reference paths differ")
    provenance = reader.provenance(trajectory)
    for name in ("checkpoint_sha256", "eval_script_sha256", "eval_code_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", provenance.get(name, "")):
            raise ValueError(f"Native trajectory lacks recorded {name}; do not fabricate it")
    if provenance["checkpoint_sha256"] != evaluation["checkpoint_sha256"]:
        raise ValueError("Native trajectory and evaluation checkpoint hashes differ")
    return provenance


def input_frame_conversion(kit, achieved, reference, released, stored_transform):
    """Apply the official reference-only transform to both exported pose columns."""
    op = kit.op
    converted, position_error, angle_error, changed = kit.packer.convert_track2_frame(
        achieved, reference, released)
    converted_reference, _, _, _ = kit.packer.convert_track2_frame(reference, reference, released)
    first = op._normalise_quaternions(reference[:1], "reference", ValueError)
    target = op._normalise_quaternions(released[:1], "input", ValueError)
    q = op._quaternion_multiply(target[:, 3:], first[:, 3:] * [-1, -1, -1, 1])
    transform = np.eye(4)
    transform[:3, :3] = op._rotation_matrix(q)[0]
    transform[:3, 3] = target[0, :3] - transform[:3, :3] @ first[0, :3]
    if not np.allclose(transform, stored_transform, rtol=0, atol=1e-5):
        raise ValueError("Official conversion disagrees with the portable reference transform")
    return converted, converted_reference, {
        "function": "tools/pack_submission.py::convert_track2_frame",
        "basis": "evaluator reference -> released INPUT reference only",
        "released_input_from_simulator_world": transform.tolist(),
        "max_reference_position_residual_m": position_error,
        "max_reference_orientation_residual_rad": angle_error,
        "changed_coordinate_frame": changed, "stored_transform_verified": True,
        "achieved_alignment": False, "clean_reference_used": False}


def export_frame_grid(data):
    """Tier1's own released visibility is its official scoring grid.

    The full source rollout remains mandatory. Tier2/3 input visibility cannot
    redefine the clean scoring grid, so those inputs still require every frame.
    """
    frames = np.asarray(data.reference.frame_index if data.tier == 1 else data.frames, dtype=np.int64)
    if (frames.ndim != 1 or not len(frames) or np.any(np.diff(frames) <= 0)
            or not np.isin(frames, data.frames).all()):
        raise ValueError("Released export frame grid is empty or malformed")
    return frames


def validate_export_inputs(job, dataset_input, episode, kit_root):
    """CPU preflight with absolute-path hashes; creates no files or artifacts.

    Call before training, persist this result, and recheck file_sha256 before a
    resume/launch. No clean target, simulator, policy, or selected rollout is used.
    """
    import pyarrow.parquet as pq

    if not isinstance(episode, int) or isinstance(episode, bool) or episode < 0:
        raise ValueError("Episode must be a nonnegative integer")
    job, kit_root = Path(job).resolve(), Path(kit_root).resolve()
    kit = validation.load_official_kit(kit_root)
    load_submission_reader(kit_root)
    data = read_input_reference(dataset_input, episode, kit)
    rows = pq.ParquetFile(job / "motion.parquet").read().to_pylist()
    if len(rows) != 1:
        raise ValueError("Portable motion must contain exactly one episode")
    position = np.asarray(rows[0]["object_body_position"])
    quaternion = np.asarray(rows[0]["object_body_wxyz"])
    if position.shape != (len(data.frames), 1, 3) or quaternion.shape != (len(data.frames), 1, 4):
        raise ValueError("Portable motion object/frame shape differs from released input")
    reference = np.concatenate((position, quaternion[..., [1, 2, 3, 0]]), axis=-1)
    transform, job_provenance = verify_job(job, episode, data, reference)
    frames = export_frame_grid(data)
    released = kit.packer.input_frame_reference(data.root, episode, frames, 1, data.reference.object_names)
    _, _, conversion = input_frame_conversion(kit, reference[frames, 0], reference[frames, 0], released, transform)
    paths = [job / "job.json", *(job / name for name in job_provenance["file_sha256"]),
             kit_root / "KIT_MANIFEST.json", *(kit_root / name for name in validation.PINNED_CODE),
             kit_root / SUBMISSION_READER, Path(__file__), Path(validation.__file__)]
    paths.extend(Path(value["path"]) for value in data.provenance.values()
                 if isinstance(value, dict) and "path" in value and "sha256" in value)
    return {"schema": "v2d_policy_export_preflight_v1", "state": "verified",
            "tier": data.tier, "episode": episode, "sequence_id": data.sequence,
            "frame_count": len(data.frames), "clean_reference_used": False,
            "exported_frame_count": len(frames), "exported_source_frame_indices": frames.tolist(),
            "officially_unscored_source_frames": np.setdiff1d(data.frames, frames).tolist(),
            "portable_job": job_provenance, "released_input": data.provenance,
            "conversion": conversion, "official_code": {**kit.provenance,
                "submission_reader_sha256": SUBMISSION_READER_SHA256},
            "file_sha256": {str(path.resolve()): validation.sha256(path) for path in paths}}


def write_parquet(path, achieved, reference, *, episode, object_name, fps, provenance, metadata,
                  frame_indices=None, source_frame_count=None):
    """Native v1 object-trajectory columns; no robot columns are required by the kit."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    steps = len(achieved)
    frame = (np.arange(steps, dtype=np.int32) if frame_indices is None
             else np.asarray(frame_indices, dtype=np.int32))
    if frame.shape != (steps,) or np.any(frame < 0) or np.any(np.diff(frame) <= 0):
        raise ValueError("Export frame indices must preserve the ordered source grid")
    values = {"environment_id": np.zeros(steps, dtype=np.int32), "step": frame,
              "reference_frame": frame, "time_s": frame.astype(np.float64) / fps,
              "object_id": np.zeros(steps, dtype=np.int32), "object_body_id": np.zeros(steps, dtype=np.int32),
              "object_body_name": pa.DictionaryArray.from_arrays(pa.array(np.zeros(steps, dtype=np.int32)), pa.array([object_name])),
              "environment_valid": np.ones(steps, dtype=bool)}
    for index, suffix in enumerate(POSE_SUFFIXES):
        values[f"achieved_{suffix}"] = achieved[:, index].astype(np.float32)
        values[f"reference_{suffix}"] = reference[:, index].astype(np.float32)
    table = pa.table(values)
    native = {"schema": "flash_chord_object_trajectories_v1", "episode_index": str(episode),
              "position_unit": "meter", "quaternion_order": "xyzw",
              "row_order": "step,environment_id,object_body_id", "control_fps": str(fps),
              "start_frame": "0", "step_count": str(steps if source_frame_count is None else source_frame_count), "world_count": "1",
              "object_body_count": "1", "pose_width": "7", **provenance}
    encoded = {f"flash_chord.{key}".encode(): str(value).encode() for key, value in native.items()}
    encoded.update({f"v2d_export.{key}".encode(): str(value).encode() for key, value in metadata.items()})
    schema = pa.schema([pa.field(field.name, field.type, nullable=False) for field in table.schema], metadata=encoded)
    table = table.cast(schema)
    pq.write_table(table, path, compression="zstd", use_dictionary=True)


def validate_export(path, reader, kit, released, *, episode, expected_achieved, expected_reference, provenance,
                    frame_indices=None):
    trajectory = reader.read_trajectory(path)
    frames = np.arange(len(released)) if frame_indices is None else np.asarray(frame_indices)
    if (trajectory.episode_index != episode or not np.array_equal(trajectory.frame_index, frames)
            or not np.all(trajectory.world_index == 0) or not np.all(trajectory.object_slot == 0)
            or not np.all(trajectory.object_id == 0) or not trajectory.world_valid.all()
            or reader.provenance(trajectory) != provenance):
        raise ValueError("Export failed the official trajectory reader/grid/provenance round trip")
    if (not np.array_equal(trajectory.pose_xyzw, expected_achieved.astype(np.float32).astype(np.float64))
            or not np.array_equal(trajectory.reference_xyzw, expected_reference.astype(np.float32).astype(np.float64))):
        raise ValueError("Exported poses changed during parquet serialization")
    # Run exactly the packer's frame-conversion check again on serialized values.
    checked, pos, angle, changed = kit.packer.convert_track2_frame(
        trajectory.pose_xyzw, trajectory.reference_xyzw, released)
    if changed or np.max(np.abs(checked[:, :3] - trajectory.pose_xyzw[:, :3])) > 1e-6:
        raise ValueError("Export is not already in the released input coordinate frame")
    return {"official_reader": "v2dlb/submission_parquet.py::read_trajectory", "passed": True,
            "official_packer_conversion_check": True, "packer_changes_coordinate_frame": changed,
            "max_serialized_reference_position_residual_m": pos,
            "max_serialized_reference_orientation_residual_rad": angle}


def export_rollout(*, arrays, job, dataset_input, episode, kit_root, output_dir,
                   native_trajectory=None, checkpoint=None, world_id=0, selection_provenance=None):
    if world_id != 0:
        raise ValueError("This exporter fixes world 0 for every episode; outcome-based selection is unsupported")
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError("Export output directory already exists; prior artifacts are immutable")
    arrays = Path(arrays).resolve()
    sidecar = arrays.parent / "evaluation.json"
    native_path = Path(native_trajectory).resolve() if native_trajectory else arrays.parent / "object_rollouts.parquet"
    kit, reader = validation.load_official_kit(kit_root), load_submission_reader(kit_root)
    input_data = read_input_reference(dataset_input, episode, kit)
    evaluation = json.loads(sidecar.read_text(encoding="utf-8"))
    with np.load(arrays, allow_pickle=False) as archive:
        achieved, reference, _, _ = validation.validate_rollout(archive, evaluation, input_data)
    transform, job_provenance = verify_job(job, episode, input_data, reference)
    provenance = verify_native_trajectory(native_path, reader, evaluation, achieved, reference,
                                           episode, input_data.reference.object_names[0])
    checkpoint_provenance = None
    if checkpoint is not None:
        checkpoint_provenance = validation.identity(checkpoint)
        if checkpoint_provenance["sha256"] != provenance["checkpoint_sha256"]:
            raise ValueError("Supplied policy checkpoint hash does not match the evaluated policy")
    # The complete source cohort was authenticated above. Only Tier1's released
    # invisible frames may be absent from the official artifact; no outcome or
    # world selection affects this grid and no missing pose is manufactured.
    frames = export_frame_grid(input_data)
    released = kit.packer.input_frame_reference(input_data.root, episode, frames, 1,
                                                input_data.reference.object_names)
    exported, exported_reference, conversion = input_frame_conversion(
        kit, achieved[frames, 0, 0], reference[frames, 0], released, transform)
    input_identities = {"arrays": validation.identity(arrays), "evaluation": validation.identity(sidecar),
                        "native_trajectory": validation.identity(native_path), "portable_job": job_provenance,
                        "released_input": input_data.provenance, "policy_checkpoint": checkpoint_provenance}
    helper_hashes = {"export_policy_rollout.py": validation.sha256(__file__),
                     "score_clean_reference.py": validation.sha256(validation.__file__)}
    output_dir.mkdir(parents=True, exist_ok=False)
    temporary = output_dir / ".trajectory.pending.parquet"
    artifact = output_dir / f"episode_{episode:06d}.parquet"
    try:
        write_parquet(temporary, exported, exported_reference, episode=episode,
                      frame_indices=frames, source_frame_count=len(input_data.frames),
                      object_name=input_data.reference.object_names[0], fps=float(evaluation["sim_control_fps"]),
                      provenance={**provenance, "reference_parquet": str(input_data.path)},
                      metadata={"coordinate_frame": "released_track2_world", "input_tier": input_data.tier,
                                "source_world_id": 0, "world_selection": "fixed_world_zero_no_metric_selection",
                                "source_arrays_sha256": input_identities["arrays"]["sha256"],
                                "source_native_trajectory_sha256": input_identities["native_trajectory"]["sha256"],
                                "exporter_sha256": helper_hashes["export_policy_rollout.py"],
                                "validation_helper_sha256": helper_hashes["score_clean_reference.py"]})
        checks = validate_export(temporary, reader, kit, released, episode=episode,
                                 expected_achieved=exported, expected_reference=exported_reference,
                                 provenance=provenance, frame_indices=frames)
        temporary.rename(artifact)
        manifest = {"schema": "v2d_policy_rollout_export_v1", "state": "complete",
                    "created_utc": datetime.now(timezone.utc).isoformat(), "episode": episode,
                    "tier": input_data.tier, "sequence_id": input_data.sequence,
                    "source_world_count": achieved.shape[1], "exported_world_count": 1,
                    "source_world_id": 0, "world_selection": "fixed_world_zero_no_metric_selection",
                    "frame_count": len(released), "source_frame_count": len(input_data.frames),
                    "source_frame_indices": frames.tolist(),
                    "officially_unscored_source_frames": np.setdiff1d(input_data.frames, frames).tolist(),
                    "dropped_frames": [], "clean_reference_used": False, "achieved_alignment": False,
                    "artifact": {"filename": artifact.name, "sha256": validation.sha256(artifact),
                                 "bytes": artifact.stat().st_size}, "recorded_evaluator_provenance": provenance,
                    "inputs": input_identities, "conversion": conversion, "validation": checks,
                    "selected_policy_manifest": selection_provenance,
                    "official_code": {**kit.provenance, "submission_reader_sha256": SUBMISSION_READER_SHA256},
                    "export_helper_sha256": helper_hashes, "uploaded": False,
                    "scope": "Single-episode official-schema artifact, not a complete submission or training-lineage certificate",
                    "next_step": "Collect one validated export per required episode within each input tier, then run the official packer with a pinned code commit URL. Packing and upload are separate actions."}
        with (output_dir / "export_manifest.json").open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        return manifest
    finally:
        temporary.unlink(missing_ok=True)


def export_selected(manifest_path, dataset_input, kit_root, output_root):
    """Export a pipeline-selected policy; return absolute artifact paths/hashes.

    ``output_root/tierN/episode_XXXXXX`` must be new. Selection is fixed by the
    hash-bound v2d_selected_policy_v1 manifest; this function does not rerank it.
    """
    manifest_path = Path(manifest_path).resolve()
    selected = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (selected.get("schema") != "v2d_selected_policy_v1"
            or selected.get("clean_reference_used") is not False
            or selected.get("world0_export_required") is not True
            or selected.get("tier") not in (1, 2, 3)):
        raise ValueError("Expected a selected Tier1/2/3 policy with no clean-reference selection and fixed world0")
    episode, tier = int(selected["episode_index"]), int(selected["tier"])
    if episode < 0:
        raise ValueError("Selected episode must be nonnegative")
    directory, job = Path(selected["evaluation_directory"]), Path(selected["job"])
    for path, key in ((directory / "evaluation.json", "evaluation_sha256"),
                      (directory / "rollout_arrays.npz", "rollout_arrays_sha256"),
                      (Path(selected["policy"]), "checkpoint_sha256"),
                      (job / "job.json", "input_job_sha256")):
        if validation.sha256(path) != selected.get(key):
            raise ValueError(f"Selected policy evidence hash differs: {key}")
    evaluation = json.loads((directory / "evaluation.json").read_text(encoding="utf-8"))
    if (evaluation.get("environment_steps") != selected.get("environment_steps")
            or evaluation.get("checkpoint_sha256") != selected["checkpoint_sha256"]
            or evaluation.get("metric_reference") != f"tier{tier}_input_reference"):
        raise ValueError("Selected policy step/checkpoint/tier differs from evaluation")
    destination = Path(output_root).resolve() / f"tier{tier}" / f"episode_{episode:06d}"
    report = export_rollout(arrays=directory / "rollout_arrays.npz", job=job,
                            dataset_input=dataset_input, episode=episode, kit_root=kit_root,
                            output_dir=destination, checkpoint=selected["policy"],
                            selection_provenance=validation.identity(manifest_path))
    artifact = destination / report["artifact"]["filename"]
    exported_manifest = destination / "export_manifest.json"
    return {"parquet": str(artifact), "parquet_sha256": report["artifact"]["sha256"],
            "manifest": str(exported_manifest), "manifest_sha256": validation.sha256(exported_manifest),
            "selected_manifest_sha256": validation.sha256(manifest_path), "episode": episode,
            "tier": tier, "source_world_id": 0, "state": "complete"}


def collect_exports(records, output_root, kit_root, expected_by_tier=None):
    """Collect existing fixed-world exports only; never rerun or select policies.

    records are export_selected return values (also pipeline_status['export']).
    expected_by_tier, when supplied by an immutable batch roster, requires an
    exact identity match. The official packer still validates its full roster.
    """
    output_root = Path(output_root).resolve()
    if output_root.exists():
        raise FileExistsError("Collection directory already exists; use a new immutable collection")
    kit = validation.load_official_kit(kit_root)
    reader = load_submission_reader(kit_root)
    verified, identities, build_hashes = [], set(), set()
    for record in records:
        tier, episode = record.get("tier"), record.get("episode")
        if (type(tier) is not int or tier not in (1, 2, 3) or type(episode) is not int
                or not 0 <= episode < 30 or record.get("state") != "complete" or record.get("source_world_id") != 0):
            raise ValueError("Collection requires completed fixed-world Track2 exports with episodes 0..29")
        if (tier, episode) in identities:
            raise ValueError(f"Duplicate export identity tier{tier}/episode{episode}")
        identities.add((tier, episode))
        artifact, manifest_path = Path(record["parquet"]).resolve(), Path(record["manifest"]).resolve()
        for key, path in (("parquet", artifact), ("manifest", manifest_path)):
            if validation.sha256(path) != record.get(key + "_sha256"):
                raise ValueError(f"Collection source {key} hash differs")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        selected = manifest.get("selected_policy_manifest") or {}
        if (manifest.get("schema") != "v2d_policy_rollout_export_v1" or manifest.get("state") != "complete"
                or manifest.get("tier") != tier or manifest.get("episode") != episode
                or manifest.get("source_world_id") != 0 or manifest.get("exported_world_count") != 1
                or manifest.get("clean_reference_used") is not False or manifest.get("achieved_alignment") is not False
                or manifest.get("dropped_frames") != [] or manifest.get("validation", {}).get("passed") is not True
                or selected.get("sha256") != record.get("selected_manifest_sha256") or not selected.get("sha256")
                or manifest.get("artifact", {}).get("sha256") != record["parquet_sha256"]
                or (manifest_path.parent / manifest["artifact"]["filename"]).resolve() != artifact):
            raise ValueError("Export manifest violates collection identity/selection/fixed-world contract")
        trajectory = reader.read_trajectory(artifact)
        frames = np.asarray(manifest["source_frame_indices"], dtype=np.int64)
        source_frames = np.arange(manifest.get("source_frame_count", manifest["frame_count"]))
        omitted = manifest.get("officially_unscored_source_frames", [])
        if (len(frames) != manifest["frame_count"] or not len(frames) or np.any(np.diff(frames) <= 0)
                or not np.isin(frames, source_frames).all()
                or np.setdiff1d(source_frames, frames).tolist() != omitted):
            raise ValueError("Collected export has an invalid source/visibility frame grid")
        if omitted:
            if tier != 1:
                raise ValueError("Only Tier1 released visibility may omit source frames")
            released = manifest["inputs"]["released_input"]
            for key in ("parquet", "download_manifest"):
                if validation.sha256(released[key]["path"]) != released[key]["sha256"]:
                    raise ValueError("Released Tier1 visibility provenance changed")
            data = read_input_reference(Path(released["download_manifest"]["path"]).parent, episode, kit)
            if not np.array_equal(export_frame_grid(data), frames) or not np.array_equal(data.frames, source_frames):
                raise ValueError("Export omissions differ from released Tier1 visibility")
        provenance = reader.provenance(trajectory)
        if (trajectory.episode_index != episode or not np.array_equal(trajectory.frame_index, frames)
                or manifest.get("source_frame_indices") != frames.tolist()
                or not np.all(trajectory.world_index == 0) or not trajectory.world_valid.all()
                or provenance != manifest["recorded_evaluator_provenance"]
                or any(not re.fullmatch(r"[0-9a-f]{64}", provenance.get(key, ""))
                       for key in ("eval_script_sha256", "eval_code_sha256", "checkpoint_sha256"))):
            raise ValueError("Collected parquet violates full-frame world0/evaluator provenance contract")
        build_hashes.add((provenance["eval_script_sha256"], provenance["eval_code_sha256"]))
        verified.append((tier, episode, artifact, record, manifest_path))
    if not verified:
        raise ValueError("No completed exports found to collect")
    if len(build_hashes) != 1:
        raise ValueError("Collection mixes evaluator builds; official packer requires consistent hashes")
    if expected_by_tier is not None:
        expected = {(int(tier), int(episode)) for tier, episodes in expected_by_tier.items() for episode in episodes}
        if identities != expected:
            raise ValueError(f"Collection differs from expected roster: missing={sorted(expected-identities)}, extra={sorted(identities-expected)}")
    output_root.mkdir(parents=True, exist_ok=False)
    items = []
    for tier, episode, source, record, manifest_path in sorted(verified, key=lambda item: item[:2]):
        destination = output_root / f"tier{tier}" / f"episode_{episode:06d}.parquet"
        destination.parent.mkdir(exist_ok=True)
        shutil.copyfile(source, destination)
        if validation.sha256(destination) != record["parquet_sha256"]:
            raise ValueError("Collected copy hash differs; no complete collection manifest written")
        items.append({"tier": tier, "episode": episode, "filename": destination.relative_to(output_root).as_posix(),
                      "sha256": record["parquet_sha256"], "source_export": validation.identity(manifest_path),
                      "selected_manifest_sha256": record["selected_manifest_sha256"]})
    tiers = sorted({tier for tier, _ in identities})
    missing = {str(tier): sorted(set(range(30)) - {episode for item_tier, episode in identities if item_tier == tier})
               for tier in tiers}
    report = {"schema": "v2d_policy_export_collection_v1", "state": "complete",
              "created_utc": datetime.now(timezone.utc).isoformat(), "artifacts": items,
              "world_selection": "fixed_world_zero_no_metric_selection", "clean_reference_used": False,
              "missing_episodes_by_tier": missing, "complete_30_episode_tiers": [int(tier) for tier, ids in missing.items() if not ids],
              "official_code": kit.provenance, "uploaded": False,
              "packing_note": "Only complete tiers can be submitted. Official packing separately reads clean Tier1 for scoring and verifies the required roster; collection never reads it.",
              "pack_argv_templates": {str(tier): [sys.executable, str(Path(kit_root).resolve() / "tools/pack_submission.py"),
                  "--track", f"track_2_tier{tier}", "--results", str(output_root / f"tier{tier}"),
                  "--dataset-root", "<challenge-dataset-root>", "--frame-references", f"<tier{tier}-input-reference-root>",
                  "--code-commit", "<GitHub-commit-URL>", "-o", str(output_root / f"track2_tier{tier}.csv")]
                  for tier in tiers if not missing[str(tier)]}}
    with (output_root / "collection_manifest.json").open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def collect_runs(run_roots, output_root, kit_root):
    """Find completed pipeline outputs across shards; incomplete runs are listed."""
    paths = set()
    for root in run_roots:
        root = Path(root).resolve()
        if not root.is_dir():
            raise ValueError(f"Run root does not exist: {root}")
        paths.update(root.rglob("pipeline_status.json"))
    records, skipped = [], []
    for path in sorted(paths):
        status = json.loads(path.read_text(encoding="utf-8"))
        if status.get("state") != "complete":
            skipped.append({"path": str(path), "state": status.get("state")})
            continue
        record = status.get("export")
        completion = path.parent / "export_complete.json"
        if (not record or not completion.is_file()
                or json.loads(completion.read_text(encoding="utf-8")) != record
                or status.get("selected_manifest_sha256") != record.get("selected_manifest_sha256")
                or validation.sha256(path.parent / "selected_checkpoint.json") != record.get("selected_manifest_sha256")):
            raise ValueError(f"Completed pipeline lacks matching immutable export/selection records: {path}")
        records.append(record)
    report = collect_exports(records, output_root, kit_root)
    report["skipped_incomplete_runs"] = skipped
    # Collection is immutable; discovery notes have their own manifest.
    with (Path(output_root) / "collection_sources.json").open("x", encoding="utf-8") as stream:
        stream.write(json.dumps({"pipeline_status": [validation.identity(path) for path in sorted(paths)],
                                 "skipped_incomplete_runs": skipped}, indent=2) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("kit", "output-dir"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--dataset-input", type=Path)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--arrays", type=Path)
    source.add_argument("--selected-manifest", type=Path,
                        help="Pipeline selection; --output-dir becomes the tier/episode collection root")
    source.add_argument("--collect-runs", type=Path, nargs="+",
                        help="Collect completed pipeline exports from run/shard roots; no re-export or training")
    parser.add_argument("--job", type=Path)
    parser.add_argument("--episode", type=int)
    parser.add_argument("--native-trajectory", type=Path, help="Defaults to sibling object_rollouts.parquet; evaluator hashes required")
    parser.add_argument("--checkpoint", type=Path, help="Optional local policy file to verify against the recorded checkpoint hash")
    parser.add_argument("--world-id", type=int, default=0, choices=(0,), help="Fixed world0 across all episodes")
    args = parser.parse_args(argv)
    if args.collect_runs:
        if any(value is not None for value in (args.dataset_input, args.job, args.episode, args.native_trajectory, args.checkpoint)):
            parser.error("collect-runs uses completed export records; do not supply direct export inputs")
        print(json.dumps(collect_runs(args.collect_runs, args.output_dir, args.kit)))
        return
    if args.dataset_input is None:
        parser.error("arrays/selected-manifest export requires --dataset-input")
    if args.selected_manifest:
        if any(value is not None for value in (args.job, args.episode, args.native_trajectory, args.checkpoint)):
            parser.error("selected-manifest supplies its own job, episode, trajectory and checkpoint")
        print(json.dumps(export_selected(args.selected_manifest, args.dataset_input, args.kit, args.output_dir)))
        return
    if args.job is None or args.episode is None:
        parser.error("--arrays requires --job and --episode")
    if args.episode < 0:
        parser.error("episode must be non-negative")
    manifest = export_rollout(arrays=args.arrays, job=args.job, dataset_input=args.dataset_input,
                              episode=args.episode, kit_root=args.kit, output_dir=args.output_dir,
                              native_trajectory=args.native_trajectory, checkpoint=args.checkpoint,
                              world_id=args.world_id)
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "artifact": manifest["artifact"], "state": manifest["state"]}))


if __name__ == "__main__":
    main()

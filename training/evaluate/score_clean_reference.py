"""Offline Track 2 scoring of saved uninterrupted rollouts against Tier 1 poses.

Requires NumPy and PyArrow, not the simulator, JAX, CUDA, or pandas. The sibling
evaluation.json authenticates the rollout contract. Dataset roots are released
LeRobot tier directories with this project's download_manifest.json. Conversion
uses only the selected input tier; clean poses are used only for scoring.

The current evaluator exports one rigid object and no object-name column. Its
sequence id plus the input/clean parquet object names establish that mapping;
ambiguous multi-body exports are deliberately rejected. This is local official-
code scoring, not a submission or an exact hosted AUC (host mesh samples differ).
"""

from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
from types import ModuleType, SimpleNamespace

import numpy as np


METRICS = ("add_auc", "spider_sr", "maniptrans_sr", "mppe_cm")
RUNTIME_COMMIT = "8c778f905cb35f76f8e6cd6e2ea486da4fb9e0ef"
# Pin the functions inspected for this scorer, as well as checking KIT_MANIFEST.
PINNED_CODE = {
    "tools/pack_submission.py": "49b13052e3bbdd056d0cb97761d670cbd94d1663e5463aa30db00b68529411e0",
    "v2dlb/object_pose.py": "c6915924a322540f00c69c75b2ae93ae455bb6efb8a41cc0a96c967f64d7e2c0",
    "v2dlb/trajectory.py": "564fbe5469f17a36804800f979b725c32b76a4ef59c4fa451fcfd7a1d3eac251",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def load_official_kit(root):
    """Load verified modules and exact packer functions without its pandas CLI."""
    root = Path(root).resolve()
    manifest_path = root / "KIT_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("evaluator", {}).get("commit") != RUNTIME_COMMIT:
        raise ValueError("Submission kit evaluator commit is not the audited version")
    hashes = {}
    for name, pinned in PINNED_CODE.items():
        actual = sha256(root / name)
        if actual != pinned or manifest.get("files", {}).get(name) != actual:
            raise ValueError(f"Submission kit code/manifest hash mismatch: {name}")
        hashes[name] = actual

    def load(name, relative):
        # Register dataclass modules without importing another installed v2dlb.
        module = ModuleType(name)
        module.__file__ = str(root / relative)
        sys.modules[name] = module
        try:
            # Compile in memory so the read-only kit receives no __pycache__.
            exec(compile((root / relative).read_text(encoding="utf-8"), module.__file__, "exec"), module.__dict__)
        finally:
            sys.modules.pop(name, None)
        return module

    op = load("_clean_score_official_object_pose", "v2dlb/object_pose.py")
    trajectory = load("_clean_score_official_trajectory", "v2dlb/trajectory.py")
    names = {"convert_track2_frame", "input_frame_reference"}
    constants = {"_TRACK2_POSITION_TOLERANCE_M", "_TRACK2_ANGLE_TOLERANCE_RAD"}
    nodes = []
    for node in ast.parse((root / "tools/pack_submission.py").read_text(encoding="utf-8")).body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            nodes.append(node)
        elif isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id in constants for target in node.targets):
            nodes.append(node)
    if len(nodes) != 4:
        raise ValueError("Audited packer functions/constants were not found")
    namespace = {"np": np, "op": op, "Path": Path,
                 "read_episode": trajectory.read_episode,
                 "episode_parquet": trajectory.episode_parquet}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(root / "tools/pack_submission.py"), "exec"), namespace)
    provenance = {"root": str(root), "manifest": identity(manifest_path),
                  "evaluator_commit": RUNTIME_COMMIT, "file_sha256": hashes,
                  "all_executed_kit_files_match_manifest_and_pins": True,
                  "packer_loading": "Unmodified AST functions and tolerance constants; no CLI import"}
    return SimpleNamespace(op=op, trajectory=trajectory,
                           packer=SimpleNamespace(**namespace), provenance=provenance)


def read_dataset(root, episode, kit, *, clean=False):
    import pyarrow.parquet as pq

    root = Path(root).resolve()
    if not (root / "meta").is_dir():
        raise ValueError(f"Expected a released LeRobot tier root: {root}")
    paths = sorted((root / "data").rglob(f"episode_{episode:06d}.parquet"))
    if len(paths) != 1:
        raise ValueError(f"Expected exactly one parquet for episode {episode}: {root}")
    path = paths[0]
    manifest_path = root / "download_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("repository") != "nvidia/video_to_data_challenge"
            or manifest.get("tier") not in ((1,) if clean else (1, 2))
            or not manifest.get("revision")):
        raise ValueError("Expected clean Tier 1 / input Tier 1 or 2 download provenance")
    record = manifest.get("files", {}).get(path.relative_to(root).as_posix(), {})
    if record.get("sha256") != sha256(path):
        raise ValueError(f"Dataset parquet hash does not match download manifest: {path}")
    table = pq.read_table(path)
    meta = {k.decode(): v.decode() for k, v in (table.schema.metadata or {}).items()}
    sequence = meta.get("source_sequence_id")
    declared = json.loads(meta.get("objects", "[]"))
    if not sequence or len(declared) != 1 or not isinstance(declared[0], str):
        raise ValueError("Require an explicit sequence and one rigid object name")
    frames = np.asarray(table["frame_index"].to_numpy(), dtype=np.int64)
    if frames.ndim != 1 or len(frames) < 1 or not np.array_equal(frames, np.arange(len(frames))):
        raise ValueError("Released episode must contain its complete ordered frame grid from zero")
    if not np.all(np.asarray(table["episode_index"].to_numpy()) == episode):
        raise ValueError("Episode column does not match the requested episode")
    reference = kit.trajectory.read_episode(path, episode)
    if reference.object_names != declared:
        raise ValueError("Parquet and official reader disagree about object names")
    return SimpleNamespace(root=root, path=path, sequence=sequence, frames=frames,
                           reference=reference, tier=manifest["tier"],
                           provenance={"parquet": identity(path), "download_manifest": identity(manifest_path),
                                       "revision": manifest["revision"], "tier": manifest["tier"],
                                       "source_sequence_id": sequence, "object_names": declared,
                                       "parquet_matches_download_manifest": True})


def validate_rollout(arrays, evaluation, input_data):
    """Reject incomplete/invalid cohorts rather than selecting surviving worlds."""
    a = np.asarray(arrays["achieved_pose_w"], dtype=np.float64)
    r = np.asarray(arrays["reference_pose_w"], dtype=np.float64)
    vertices = np.asarray(arrays["object_vertices_o"], dtype=np.float64)
    ids = np.asarray(arrays["body_object_ids"])
    if (a.ndim != 4 or a.shape[-2:] != (1, 7) or min(a.shape[:2]) < 1
            or r.shape != (a.shape[0], 1, 7)):
        raise ValueError("Expected full [frame, world, one rigid object, 7] rollout")
    if ids.shape != (1,) or ids.dtype.kind not in "iu" or int(ids[0]) != 0:
        raise ValueError("Single rigid object must map to body_object_ids=[0]")
    if vertices.ndim != 3 or vertices.shape[0] != 1 or vertices.shape[-1] != 3 or vertices.shape[1] < 1:
        raise ValueError("Malformed object-local AUC vertices")
    if arrays["invalid_worlds"].size or evaluation.get("non_finite_worlds") != []:
        raise ValueError("Invalid rollout cohort; no failed worlds may be dropped")
    for label, poses in (("achieved", a), ("reference", r)):
        if not np.isfinite(poses).all() or (np.linalg.norm(poses[..., 3:], axis=-1) < 1e-8).any():
            raise ValueError(f"Invalid {label} poses; no frames/worlds may be dropped")
    if not np.isfinite(vertices).all():
        raise ValueError("Nonfinite AUC vertices")
    if evaluation.get("schema") not in ("v2d_policy_evaluation_v1", "v2d_overnight_evaluation_v1"):
        raise ValueError("Unsupported evaluation schema")
    expected = {"start_frame": 0,
                "reset_mode": "explicit", "settling_steps": 0,
                "step_count": a.shape[0], "world_count": a.shape[1],
                "motion_start_frame": 0, "motion_end_frame": -1}
    if any(evaluation.get(key) != value for key, value in expected.items()):
        raise ValueError("Evaluation sidecar does not describe a complete frame-zero rollout")
    contract = evaluation.get("evaluation_contract", {})
    required = {"frame_zero": True, "explicit_reset": True, "within_attempt_resets": 0,
                "voc_scale": 0.0, "reset_voc_scale": 0.0, "settling_steps": 0,
                "training_termination_disabled_during_rollout": True}
    if any(contract.get(key) != value for key, value in required.items()):
        raise ValueError("Require uninterrupted, unassisted evaluation with training termination disabled")
    sequence = re.search(r"(?:^|[/\\])sequence_id=([^/\\]+)", str(evaluation.get("parquet", "")))
    if sequence is None or sequence.group(1) != input_data.sequence:
        raise ValueError("Evaluation and released input sequence identities differ")
    if evaluation.get("metric_reference") != f"tier{input_data.tier}_input_reference":
        raise ValueError("Evaluation sidecar input tier does not match dataset-input")
    if not re.fullmatch(r"[0-9a-f]{64}", str(evaluation.get("checkpoint_sha256", ""))):
        raise ValueError("Evaluation lacks a checkpoint SHA256")
    if a.shape[0] != len(input_data.frames):
        raise ValueError("Rollout does not cover every released input frame")
    return a, r, vertices, ids


def stored_reference_check(path, recorded, input_frames):
    """Check archive grid/poses and return source-from-simulator transform."""
    with np.load(path, allow_pickle=False) as data:
        frames = data["source_frame_indices"]
        if not np.array_equal(frames, input_frames):
            raise ValueError("Reference archive source frame grid differs from the complete rollout")
        if (str(data["coordinate_frame"].item()) != "world_z_up"
                or str(data["quaternion_order"].item()) != "wxyz"
                or str(data["units"].item()) != "metres_seconds_radians"):
            raise ValueError("Reference archive has an unsupported frame/quaternion convention")
        pose = np.concatenate((data["object_position"], data["object_wxyz"][:, [1, 2, 3, 0]]), axis=-1)
        if pose.shape != recorded[:, 0].shape or not np.isfinite(pose).all():
            raise ValueError("Malformed archived object reference")
        position_error = float(np.linalg.norm(pose[:, :3] - recorded[:, 0, :3], axis=-1).max())
        q, qr = pose[:, 3:], recorded[:, 0, 3:]
        if (np.linalg.norm(q, axis=-1) < 1e-8).any():
            raise ValueError("Degenerate archived quaternion")
        dot = np.abs(np.sum(q / np.linalg.norm(q, axis=-1, keepdims=True)
                            * qr / np.linalg.norm(qr, axis=-1, keepdims=True), axis=-1))
        angle_error = float((2 * np.arccos(np.clip(dot, 0, 1))).max())
        if position_error > 1e-5 or angle_error > 1e-5:
            raise ValueError("Saved evaluator reference does not match reference.npz")
        transform = np.asarray(data["world_from_source_cv"], dtype=float)
    if (transform.shape != (4, 4) or not np.isfinite(transform).all()
            or not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-8, rtol=0)
            or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-8, rtol=0)
            or not np.isclose(np.linalg.det(transform[:3, :3]), 1, atol=1e-8, rtol=0)):
        raise ValueError("world_from_source_cv is not a rigid transform")
    return np.linalg.inv(transform), {"archive": identity(path),
                                      "max_recorded_reference_position_error_m": position_error,
                                      "max_recorded_reference_orientation_error_rad": angle_error}


def score_arrays(kit, achieved, recorded, vertices, ids, released_input, clean,
                 *, stored_transform=None):
    """The clean target never enters the frame converter or transform fit."""
    frames, worlds, bodies, _ = achieved.shape
    long_a = achieved.transpose(1, 0, 2, 3).reshape(-1, 7)
    long_r = np.broadcast_to(recorded[None], (worlds, frames, bodies, 7)).reshape(-1, 7)
    converted, pos_residual, angle_residual, changed = kit.packer.convert_track2_frame(
        long_a, long_r, released_input)
    op = kit.op
    first_r = op._normalise_quaternions(long_r[:1], "reference", ValueError)
    first_input = op._normalise_quaternions(released_input[:1], "input", ValueError)
    q = op._quaternion_multiply(first_input[:, 3:], first_r[:, 3:] * [-1, -1, -1, 1])
    rotation = op._rotation_matrix(q)[0]
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = first_input[0, :3] - rotation @ first_r[0, :3]
    if stored_transform is not None and not np.allclose(transform, stored_transform, rtol=0, atol=1e-5):
        raise ValueError("Official input-frame conversion disagrees with world_from_source_cv")
    converted = converted.reshape(worlds, frames, bodies, 7).transpose(1, 0, 2, 3)
    clean = op._normalise_quaternions(clean[:, None], "clean reference", ValueError)
    scores = {name: float(op.episode_metric(name, converted, clean, ids, vertices,
                                          align_initial_pose=False)) for name in METRICS}
    if not all(np.isfinite(value) for value in scores.values()):
        raise ValueError("Official metric returned a nonfinite value")
    return scores, {"function": "tools/pack_submission.py::convert_track2_frame",
                    "basis": "recorded evaluator reference -> released INPUT reference only",
                    "released_input_from_simulator_world": transform.tolist(),
                    "max_reference_position_residual_m": pos_residual,
                    "max_reference_orientation_residual_rad": angle_residual,
                    "changed_coordinate_frame": changed,
                    "stored_transform_verified": stored_transform is not None}, converted, clean


def clean_phase_analysis(op, achieved, clean, vertices, ids, source_frames, source_frame_count):
    """Temporal thirds on the original source grid, never on a filtered reindexing."""
    def tail(mask):
        selected_frames = source_frames[mask]
        position = np.linalg.norm(achieved[mask, ..., :3] - clean[mask, ..., :3], axis=-1)
        bad = (position > 0.10).any(axis=-1)
        return {"origin_error_mean_cm": float(position.mean() * 100),
                "origin_error_p95_cm": float(np.quantile(position, 0.95) * 100),
                "origin_error_max_cm": float(position.max() * 100),
                "origin_error_mean_cm_by_world": (position.mean(axis=(0, 2)) * 100).tolist(),
                "origin_error_p95_cm_by_world": (np.quantile(position, 0.95, axis=(0, 2)) * 100).tolist(),
                "origin_error_max_cm_by_world": (position.max(axis=(0, 2)) * 100).tolist(),
                "worlds_ever_origin_error_over_10cm": np.flatnonzero(bad.any(axis=0)).tolist(),
                "first_origin_error_over_10cm_source_frame_by_world": [
                    int(selected_frames[np.flatnonzero(column)[0]]) if column.any() else None
                    for column in bad.T]}

    first, second = source_frame_count // 3, 2 * source_frame_count // 3
    result = {"reference": "clean_tier1_after_input_only_frame_conversion",
              "phase_definition": "temporal thirds of complete source grid, not inferred task phases",
              "tail_diagnostic": "Object-origin distance >10cm is diagnostic only, not SP-SR or MP-SR",
              "full_sequence_tail": tail(np.ones(len(source_frames), dtype=bool)), "phases": {}}
    for name, start, stop in (("early_third", 0, first), ("middle_third", first, second),
                             ("late_third", second, source_frame_count)):
        mask = (source_frames >= start) & (source_frames < stop)
        phase = {"source_frame_start_inclusive": start, "source_frame_stop_exclusive": stop,
                 "scored_source_frame_indices": source_frames[mask].tolist(),
                 "scored_frame_count": int(mask.sum()), "available": bool(mask.any())}
        if mask.any():
            phase["clean_reference_metrics"] = {
                name: float(op.episode_metric(name, achieved[mask], clean[mask], ids, vertices,
                                              align_initial_pose=False)) for name in METRICS}
            phase["tail"] = tail(mask)
        else:
            phase["reason"] = "No officially visible clean frames in this source interval"
        result["phases"][name] = phase
    return result


def score_saved_rollout(*, dataset_input, dataset_clean, episode, arrays, kit_root, reference_npz=None):
    kit = load_official_kit(kit_root)
    input_data = read_dataset(dataset_input, episode, kit)
    clean_data = read_dataset(dataset_clean, episode, kit, clean=True)
    if (input_data.sequence != clean_data.sequence
            or input_data.reference.object_names != clean_data.reference.object_names
            or not np.array_equal(input_data.frames, clean_data.frames)):
        raise ValueError("Input and clean episodes have inconsistent sequence/object/frame identities")
    arrays = Path(arrays).resolve()
    sidecar = arrays.parent / "evaluation.json"
    evaluation = json.loads(sidecar.read_text(encoding="utf-8"))
    with np.load(arrays, allow_pickle=False) as archive:
        a, r, vertices, ids = validate_rollout(archive, evaluation, input_data)
    transform, archive_provenance = None, None
    if reference_npz is not None:
        transform, archive_provenance = stored_reference_check(reference_npz, r, input_data.frames)
    # Official visibility determines this grid; there is no outcome-based filter.
    frames = clean_data.reference.frame_index
    if len(frames) == 0:
        raise ValueError("Clean episode has no officially scored frames")
    released = kit.packer.input_frame_reference(input_data.root, episode, frames,
                                                a.shape[1], input_data.reference.object_names)
    scores, conversion, converted, clean = score_arrays(
        kit, a[frames], r[frames], vertices, ids, released, clean_data.reference.pose_xyzw,
        stored_transform=transform)
    errors = kit.op._per_world_pose_error(converted, clean)
    return {"schema": "v2d_clean_reference_score_v1", "created_utc": datetime.now(timezone.utc).isoformat(),
            "classification": "local official-code clean-reference scoring; not a hosted leaderboard result",
            "official_leaderboard_score": False, "episode": episode, "input_tier": input_data.tier,
            "world_count": a.shape[1], "rollout_frame_count": a.shape[0], "scored_frame_count": len(frames),
            "scored_source_frame_indices": frames.tolist(), "dropped_worlds": [],
            "official_clean_invisible_frames": np.setdiff1d(clean_data.frames, frames).tolist(),
            "initial_achieved_to_clean_alignment": False, "clean_reference_use": "offline_scoring_only_in_this_tool",
            "checkpoint_sha256": evaluation["checkpoint_sha256"], "environment_steps": evaluation.get("environment_steps"),
            "inputs": {"rollout": identity(arrays), "evaluation": identity(sidecar),
                       "released_input": input_data.provenance, "clean_reference": clean_data.provenance,
                       "reference_archive": archive_provenance},
            "official_code": kit.provenance, "reference_frame_conversion": conversion,
            "clean_reference_metrics": scores,
            "clean_reference_phase_analysis": clean_phase_analysis(
                kit.op, converted, clean, vertices, ids, frames, len(clean_data.frames)),
            "all_world_mean_position_error_m": errors[0].reshape(-1).tolist(),
            "all_world_mean_orientation_error_rad": errors[1].reshape(-1).tolist(),
            "limitations": ["AUC uses the saved rollout's local object-frame mesh samples, not hidden hosted samples.",
                            "All saved worlds are retained; this is not a submission roster or provenance acceptance certificate.",
                            "Frames are paired by released indices without resampling; timing/physics validity is not certified.",
                            "Clean data is read only by this offline scorer; this report does not independently audit training lineage."]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset-input", "dataset-clean", "arrays", "output", "kit"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--reference-npz", type=Path)
    args = parser.parse_args(argv)
    if args.episode < 0:
        parser.error("episode must be non-negative")
    if args.output.exists():
        parser.error("output already exists; preserve prior scoring evidence")
    report = score_saved_rollout(dataset_input=args.dataset_input, dataset_clean=args.dataset_clean,
                                episode=args.episode, arrays=args.arrays, kit_root=args.kit,
                                reference_npz=args.reference_npz)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output.resolve()), "scores": report["clean_reference_metrics"]}))


if __name__ == "__main__":
    main()

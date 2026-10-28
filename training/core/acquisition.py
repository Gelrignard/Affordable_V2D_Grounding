"""Deterministic, input-only grasp regions for optional acquisition guidance.

Geometry checks establish a conservative point-level shaping prior, not force
feasibility or whole-hand collision clearance. No achieved rollout or clean
reference is accepted here. Failed windows keep the existing objective.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class AcquisitionTable:
    centers_o: np.ndarray
    radii: np.ndarray
    blend: np.ndarray
    audit: dict


def _digest(arrays, settings):
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True, allow_nan=False).encode())
    for name, value in sorted(arrays.items()):
        array = np.ascontiguousarray(value)
        digest.update(name.encode())
        digest.update(str((array.shape, array.dtype.str)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _rotate_inverse(quaternion, vectors):
    q = np.asarray(quaternion, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    xyz, w = -q[..., :3], q[..., 3:]
    return vectors + 2 * np.cross(xyz, np.cross(xyz, vectors) + w * vectors)


def build_acquisition_table(*, palm_positions_w, fingertip_positions_w,
                            object_positions_w, object_quaternions_xyzw,
                            contact_active, object_observed_mask,
                            mesh_vertices_o, mesh_faces, window_frames=25,
                            seed_frames=5, region_radius=.03,
                            contact_distance=.03, min_palm_clearance=.015,
                            max_seed_spread=.04, source_hashes=None):
    """Build immutable phase-indexed regions; rejected windows have zero blend.

    Each seed begins ``seed_frames`` after a contact onset and lasts that many
    frames. A medoid preserves an actual reference palm offset. Only pre-contact
    frames change. Radius is bounded by exterior palm clearance; the approach
    standoff is checked at every table center. This checks points, not a swept
    articulated hand volume.
    """
    import trimesh

    if (type(window_frames) is not int or type(seed_frames) is not int
            or window_frames < 4 or seed_frames < 3):
        raise ValueError("acquisition window >=4 and seed >=3 integer frames required")
    settings = dict(window_frames=window_frames, seed_frames=seed_frames,
                    region_radius=float(region_radius), contact_distance=float(contact_distance),
                    min_palm_clearance=float(min_palm_clearance), max_seed_spread=float(max_seed_spread))
    if not all(np.isfinite(v) and v > 0 for v in settings.values()):
        raise ValueError("acquisition geometry settings must be finite and positive")
    palm = np.asarray(palm_positions_w, dtype=np.float64)
    tips = np.asarray(fingertip_positions_w, dtype=np.float64)
    pos = np.asarray(object_positions_w, dtype=np.float64)
    quat = np.asarray(object_quaternions_xyzw, dtype=np.float64)
    labels = np.asarray(contact_active, dtype=np.float64)
    observed_raw = np.asarray(object_observed_mask)
    if not np.isfinite(observed_raw).all() or not np.isin(observed_raw, (0, 1)).all():
        raise ValueError("object observed mask must contain only finite boolean values")
    observed = observed_raw.astype(bool)
    vertices = np.asarray(mesh_vertices_o, dtype=np.float64)
    faces = np.asarray(mesh_faces, dtype=np.int64)
    if palm.ndim != 3 or palm.shape[1:] != (2, 3):
        raise ValueError("palm positions must be [frames,2,3]")
    n = len(palm)
    if (n == 0 or tips.ndim != 4 or tips.shape[:2] != (n, 2) or tips.shape[2] < 2
            or tips.shape[3] != 3 or pos.shape != (n, 3) or quat.shape != (n, 4)
            or labels.shape != (n, 2) or observed.shape != (n,)):
        raise ValueError("acquisition reference frame/hand/point shapes differ")
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3
            or not len(vertices) or not len(faces) or faces.min() < 0 or faces.max() >= len(vertices)):
        raise ValueError("invalid object mesh")
    arrays = dict(palms=palm, tips=tips, object_position=pos, object_quaternion=quat,
                  contact=labels, observed=observed, vertices=vertices, faces=faces)
    if not all(np.isfinite(a).all() for a in arrays.values()):
        raise ValueError("nonfinite acquisition input")
    if np.any(np.linalg.norm(quat, axis=-1) < .5) or np.any((labels < 0) | (labels > 1)):
        raise ValueError("invalid reference quaternion/contact labels")
    centers = np.zeros((n, 2, 3), dtype=np.float32)
    radii = np.zeros((n, 2), dtype=np.float32)
    blend = np.zeros((n, 2), dtype=np.float32)
    audit = dict(schema="v2d_acquisition_region_v1", settings=settings,
                 input_sha256=_digest(arrays, settings), source_hashes=source_hashes or {},
                 active_windows=0, rejected_windows=0, windows=[],
                 geometry_scope="reference fingertip proximity and exterior palm-region clearance; not whole-hand or force feasibility")
    # OBJ texture/normal seams duplicate spatial vertices. Merge ONLY exactly
    # equal coordinates for topology checks: no smoothing, hole fill or change
    # to the represented triangles is permitted.
    unique_vertices, vertex_ids = np.unique(vertices, axis=0, return_inverse=True)
    mesh = trimesh.Trimesh(vertices=unique_vertices, faces=vertex_ids[faces], process=False)
    audit["mesh_topology"] = dict(input_vertices=len(vertices), unique_vertices=len(unique_vertices),
                                  faces=len(faces), operation="merge_exact_duplicate_positions_only")
    mesh_ok = bool(mesh.is_watertight and mesh.is_winding_consistent and abs(mesh.volume) > 1e-12)
    audit["mesh_closed_consistent"] = mesh_ok
    if mesh_ok and mesh.volume < 0:
        mesh.invert()
    palm_o = _rotate_inverse(quat[:, None, :], palm - pos[:, None, :])
    tips_o = _rotate_inverse(quat[:, None, None, :], tips - pos[:, None, None, :])

    def nearest(points):
        # Small setup-time queries avoid optional rtree and any GPU work.
        chunks = [trimesh.proximity.closest_point_naive(mesh, points[i:i + 4])
                  for i in range(0, len(points), 4)]
        closest, distance, triangle = [np.concatenate([chunk[j] for chunk in chunks]) for j in range(3)]
        normals = mesh.face_normals[triangle]
        signed = np.sum((points - closest) * normals, axis=-1)
        return closest, distance, normals, signed

    for hand in range(2):
        active = labels[:, hand] > .5
        changes = np.diff(np.r_[False, active, False].astype(np.int8))
        for onset, end in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)):
            onset, end = int(onset), int(end)
            start = max(0, onset - window_frames)
            seed_start, seed_end = onset + seed_frames, onset + 2 * seed_frames
            record = dict(hand=hand, onset=onset, contact_end=end, start=start,
                          seed_start=seed_start, seed_end=seed_end, accepted=False)
            reason = None
            if not mesh_ok:
                reason = "mesh_not_closed_consistent"
            elif onset < 4 or seed_end > end:
                reason = "insufficient_precontact_or_sustained_contact"
            elif np.any(active[start:onset]):
                reason = "overlapping_previous_contact"
            elif not observed[start:seed_end].all():
                reason = "unobserved_object_in_window_or_seed"
            if reason is None:
                seed_palms = palm_o[seed_start:seed_end, hand]
                seed_tips = tips_o[seed_start:seed_end, hand]
                _, tip_distance, _, tip_signed = nearest(seed_tips.reshape(-1, 3))
                near = ((tip_distance <= contact_distance) & (tip_signed >= -.01)).reshape(seed_frames, -1)
                _, palm_distance, palm_normals, palm_signed = nearest(seed_palms)
                valid = (near.sum(axis=1) >= 2) & (palm_distance >= min_palm_clearance) & (palm_signed >= min_palm_clearance)
                record.update(valid_seed_frames=int(valid.sum()), min_tip_distance_m=float(tip_distance.min()),
                              min_palm_distance_m=float(palm_distance.min()))
                if valid.sum() < max(3, seed_frames - 1):
                    reason = "insufficient_near_fingers_or_exterior_palm_clearance"
                else:
                    candidates = seed_palms[valid]
                    medoid = int(np.argmin(np.linalg.norm(candidates[:, None] - candidates[None], axis=-1).sum(axis=1)))
                    anchor = candidates[medoid]
                    spread = float(np.linalg.norm(candidates - anchor, axis=1).max())
                    record["seed_spread_m"] = spread
                    if spread > max_seed_spread:
                        reason = "unstable_object_relative_seed"
                    else:
                        index = np.flatnonzero(valid)[medoid]
                        normal = palm_normals[index]
                        frames = np.arange(start, onset)
                        fraction = (frames - start) / max(onset - start - 1, 1)
                        # The retained palm offset, not the object surface, is the target.
                        proposed = anchor[None] + (.05 * (1 - fraction))[:, None] * normal
                        _, distance, _, signed = nearest(proposed)
                        allowed = np.minimum(region_radius, .5 * (distance - min_palm_clearance))
                        if np.any(signed < min_palm_clearance) or np.any(allowed < .005):
                            reason = "approach_region_clearance_too_small"
                        else:
                            edge = min(5, max(2, (onset - start) // 4))
                            phase_blend = np.minimum(np.minimum((frames - start) / edge,
                                                               (onset - frames) / edge), 1.0)
                            phase_blend = phase_blend ** 2 * (3 - 2 * phase_blend)
                            centers[frames, hand] = proposed
                            radii[frames, hand] = allowed
                            blend[frames, hand] = phase_blend
                            record.update(accepted=True, anchor_o_m=anchor.tolist(),
                                          anchor_source_frame=int(seed_start + index),
                                          min_region_radius_m=float(allowed.min()),
                                          max_region_radius_m=float(allowed.max()))
            if reason is not None:
                record["reason"] = reason
                audit["rejected_windows"] += 1
            else:
                audit["active_windows"] += 1
            audit["windows"].append(record)
    audit["active_frame_hands"] = int(np.count_nonzero(blend))
    audit["table_sha256"] = _digest(dict(centers=centers, radii=radii, blend=blend), settings)
    for array in (centers, radii, blend):
        array.setflags(write=False)
    return AcquisitionTable(centers, radii, blend, audit)


def table_from_native(reference, robot_reference, config):
    """Use resolved native assets/FK and an identity-checked portable mask."""
    import trimesh

    source = Path(config.acquisition_reference_path)
    source_digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if source_digest != config.acquisition_reference_sha256:
        raise ValueError("acquisition portable reference SHA256 mismatch")
    with np.load(source, allow_pickle=False) as portable:
        pos = np.asarray(reference.object_body_pos_w(), dtype=np.float64)
        quat_wxyz = np.asarray(reference.object_body_quat_w(), dtype=np.float64)
        if pos.shape != (reference.num_frames, 1, 3) or quat_wxyz.shape != (reference.num_frames, 1, 4):
            raise ValueError("acquisition supports one rigid object")
        pos, quat_wxyz = pos[:, 0], quat_wxyz[:, 0]
        source_pos, source_q = portable["object_position"], portable["object_wxyz"]
        if source_pos.shape != pos.shape or source_q.shape != quat_wxyz.shape:
            raise ValueError("acquisition portable/native frame shapes differ")
        native_q = quat_wxyz / np.linalg.norm(quat_wxyz, axis=-1, keepdims=True)
        source_q = source_q / np.linalg.norm(source_q, axis=-1, keepdims=True)
        if (not np.allclose(source_pos, pos, atol=1e-5, rtol=0)
                or not np.allclose(np.abs(np.sum(source_q * native_q, axis=-1)), 1., atol=1e-6, rtol=0)):
            raise ValueError("acquisition portable/native object reference mismatch")
        observed = portable["object_observed_mask"].copy()
    paths = reference.object_mesh_paths()
    if len(paths) != 1:
        raise ValueError("acquisition requires one resolved object mesh")
    mesh_path = Path(paths[0])
    mesh = trimesh.load(str(mesh_path), force="mesh", process=False)
    hands = [robot_reference.hand(side) for side in config.sides]
    kwargs = {name: getattr(config, "acquisition_" + name) for name in
              ("window_frames", "seed_frames", "region_radius", "contact_distance", "min_palm_clearance", "max_seed_spread")}
    return build_acquisition_table(
        palm_positions_w=np.stack([hand.wrist_pos_w for hand in hands], axis=1),
        fingertip_positions_w=np.stack([hand.fingertip_pos_w for hand in hands], axis=1),
        object_positions_w=pos, object_quaternions_xyzw=quat_wxyz[:, [1, 2, 3, 0]],
        contact_active=np.stack([reference.contact_active(side) for side in config.sides], axis=1),
        object_observed_mask=observed, mesh_vertices_o=mesh.vertices, mesh_faces=mesh.faces,
        source_hashes=dict(reference_npz=source_digest,
                           mesh=hashlib.sha256(mesh_path.read_bytes()).hexdigest()), **kwargs)


def preflight_acquisition_config(resolved_config):
    """Build the exact native reference/FK on CPU, without an RL environment.

    Run this in a CPU-only subprocess before scheduling a GPU job. Only the
    one-robot builder and native reference FK are constructed; no object scene,
    physics solver, SONIC, learner or replay is initialized.
    """
    from copy import deepcopy
    if isinstance(resolved_config, (str, Path)):
        values = json.loads(Path(resolved_config).read_text(encoding="utf-8-sig"))
    else:
        values = deepcopy(resolved_config)
    from training.core.configuration import normalize_checkpoint_config
    values = normalize_checkpoint_config(values)
    objective_values = values["env"]["objective"]
    if not objective_values.get("acquisition_enabled", False):
        return dict(enabled=False, active_windows=0, status="disabled")
    scene = values.get("scene", {})
    if scene.get("object_scale_min", 1.0) != 1.0 or scene.get("object_scale_max", 1.0) != 1.0:
        raise ValueError("acquisition preflight requires unscaled object geometry")
    import warp as wp
    import newton
    from hydra.utils import instantiate
    from flash_chord.data import frame_window, load_reference
    from flash_chord.embodiments.g1_dex3 import G1Dex3
    from training.core.refinement import RefinementObjectiveConfig

    task = values["task"]
    with wp.ScopedDevice("cpu"):
        objective = instantiate(objective_values, _convert_="all")
        if not isinstance(objective, RefinementObjectiveConfig):
            raise TypeError("acquisition preflight requires RefinementObjectiveConfig")
        embodiment = instantiate(values["embodiment"], _convert_="all")
        if not isinstance(embodiment, G1Dex3):
            raise TypeError("acquisition preflight supports the pinned G1+Dex3 embodiment")
        reference = load_reference(
            task["parquet"], control_fps=float(values["env"]["sim"]["fps"]),
            motion_speed=float(task.get("motion_speed", 1.0)),
            source_frame_playback=bool(task.get("source_frame_playback", False)),
        )
        reference = frame_window(reference, int(task.get("motion_start_frame", 0)),
                                 int(task.get("motion_end_frame", -1)))
        builder = newton.ModelBuilder()
        layout = embodiment.build(builder)
        # G1.bind_reference uses build_robot_reference_binding(device='cpu').
        binding = embodiment.bind_reference(builder, layout, reference)
        table = table_from_native(reference, binding, objective)
    if table.audit["active_windows"] == 0:
        raise ValueError("acquisition preflight found no valid windows: " + json.dumps(table.audit, sort_keys=True))
    audit = dict(table.audit, enabled=True, preflight_device="cpu", native_binding=True,
                 reference_frames=int(reference.num_frames))
    return audit


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="CPU-only preflight for optional grasp-region guidance")
    parser.add_argument("--config", required=True, help="Resolved training config JSON")
    parser.add_argument("--output", required=True, help="Write compact geometry/provenance audit JSON")
    args = parser.parse_args(argv)
    audit = preflight_acquisition_config(args.config)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: audit[key] for key in
                      ("enabled", "active_windows", "rejected_windows", "table_sha256") if key in audit}))


if __name__ == "__main__":
    main()

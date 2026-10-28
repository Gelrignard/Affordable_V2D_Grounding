"""Input-only geometric checks for a final supported release.

The table never changes object or hand targets. Its checks are necessary geometric
conditions, not a collision, force-closure, or dynamic-stability certificate.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass(frozen=True)
class PlacementTable:
    centers_w: np.ndarray
    reference_quat_xyzw: np.ndarray
    phase_mask: np.ndarray
    soften_mask: np.ndarray
    withdrawal_mask: np.ndarray
    support: dict | None
    audit: dict


def parse_native_support(path, expected_sha=None):
    """Match pinned scene/support.py: axis-aligned Cylinder/Cube, prim-local translation.

    That runtime ignores rotations/parent transforms. Reject such authoring rather
    than interpreting it differently. Coordinates are already simulation world;
    world_from_source_cv must NOT be applied a second time.
    """
    from pxr import Usd
    path = Path(path)
    if expected_sha is not None and _sha(path) != expected_sha:
        raise ValueError("placement support SHA256 mismatch")
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise ValueError("Cannot open placement support")
    result = []
    for prim in stage.Traverse():
        kind = prim.GetTypeName()
        attrs = {a.GetName(): a.Get() for a in prim.GetAttributes()}
        transforms = [k for k in attrs if k.startswith("xformOp:")]
        allowed = {"xformOp:translate", "xformOp:scale"} if kind == "Cube" else {"xformOp:translate"}
        if any(k not in allowed for k in transforms) or (kind not in ("Cylinder", "Cube") and transforms):
            raise ValueError("Unsupported support transform; native runtime does not compose it")
        if kind not in ("Cylinder", "Cube"):
            continue
        center = np.asarray(attrs.get("xformOp:translate") or (0., 0., 0.), dtype=float)
        if kind == "Cylinder":
            if attrs.get("axis", "Z") != "Z":
                raise ValueError("Only native Z cylinders are supported")
            radius, height = float(attrs["radius"]), float(attrs["height"])
            shape = dict(kind="cylinder", radius=radius, half_extents_xy=None)
            dims = [radius, height]
        else:
            dims = float(attrs.get("size", 2.)) * np.abs(np.asarray(attrs.get("xformOp:scale") or (1., 1., 1.)))
            height = float(dims[2])
            shape = dict(kind="box", radius=None, half_extents_xy=(dims[:2] * .5).tolist())
        if center.shape != (3,) or not np.isfinite(center).all() or not np.isfinite(dims).all() or np.any(np.asarray(dims) <= 0):
            raise ValueError("Invalid support dimensions")
        shape.update(name=str(prim.GetPath()), center_w=center.tolist(),
                     top_center_w=(center + [0., 0., height * .5]).tolist(), normal_w=[0., 0., 1.],
                     height=height, source_path=str(path), source_sha256=_sha(path))
        result.append(shape)
    if not result:
        raise ValueError("No native Cylinder/Cube support shapes")
    return result


def _inside(points_xy, support, margin):
    d = points_xy - np.asarray(support["top_center_w"][:2])
    if support["kind"] == "cylinder":
        return np.linalg.norm(d, axis=-1) <= support["radius"] - margin
    return np.all(np.abs(d) <= np.asarray(support["half_extents_xy"]) - margin, axis=-1)


def build_placement_table(*, object_positions_w, object_quaternions_xyzw, contact_active,
                          observed_mask, object_vertices_o, object_com_o, supports, frame_dt,
                          pre_release_frames=25, post_release_frames=100, stationary_frames=10,
                          max_linear_speed=.15, max_angular_speed=1.5, max_surface_gap=.04,
                          max_penetration=.015, com_margin=.005, source_hashes=None):
    p = np.asarray(object_positions_w, dtype=np.float64)
    q = np.asarray(object_quaternions_xyzw, dtype=np.float64)
    c = np.asarray(contact_active)
    observed = np.asarray(observed_mask)
    v = np.asarray(object_vertices_o, dtype=np.float64)
    com = np.asarray(object_com_o, dtype=np.float64)
    n = len(p)
    if (p.shape != (n, 3) or q.shape != (n, 4) or c.shape != (n, 2) or observed.shape != (n,)
            or v.ndim != 2 or v.shape[1] != 3 or len(v) < 4 or com.shape != (3,) or n < 2):
        raise ValueError("Malformed placement input arrays")
    if not all(np.isfinite(x).all() for x in (p, q, c, observed, v, com)):
        raise ValueError("Nonfinite placement inputs")
    if not np.isin(observed, (0, 1)).all() or not np.isin(c, (0, 1)).all():
        raise ValueError("Placement masks must be binary")
    if (not np.isfinite([frame_dt, max_linear_speed, max_angular_speed, max_surface_gap,
                         max_penetration, com_margin]).all()
            or any(type(x) is not int for x in (pre_release_frames, post_release_frames, stationary_frames))
            or frame_dt <= 0 or min(pre_release_frames, post_release_frames, stationary_frames) < 1
            or min(max_linear_speed, max_angular_speed, max_surface_gap, max_penetration, com_margin) <= 0):
        raise ValueError("Invalid placement settings")
    norms = np.linalg.norm(q, axis=-1)
    if np.any(norms < 1e-8):
        raise ValueError("Degenerate placement quaternion")
    q = q / norms[:, None]
    rotation = Rotation.from_quat(q).as_matrix()
    phase = np.zeros(n, dtype=np.float32)
    soften = phase.copy()
    withdrawal = np.zeros((n, 2), dtype=np.float32)
    active = (c > .5).any(axis=1)
    releases = np.flatnonzero(active[:-1] & ~active[1:]) + 1
    audit = dict(schema="v2d_supported_placement_table_v1", active_windows=0, windows=[],
                 source_hashes=source_hashes or {}, support_geometry=supports,
                 object_com_o=com.tolist(), reference_frames=n,
                 support_coordinates="native prim-local translation in simulation world; no additional transform",
                 target_correction="none; original object and hand targets retained",
                 scope="geometric compatibility only, not force or collision certification")
    settings = dict(frame_dt=frame_dt, pre_release_frames=pre_release_frames,
                    post_release_frames=post_release_frames, stationary_frames=stationary_frames,
                    max_linear_speed=max_linear_speed, max_angular_speed=max_angular_speed,
                    max_surface_gap=max_surface_gap, max_penetration=max_penetration, com_margin=com_margin)
    audit["settings"] = settings
    selected = None
    linear = np.r_[0., np.linalg.norm(np.diff(p, axis=0), axis=1) / frame_dt]
    angular = np.r_[0., 2 * np.arccos(np.clip(np.abs(np.sum(q[1:] * q[:-1], axis=1)), 0., 1.)) / frame_dt]
    for release in releases:
        record = dict(release_frame=int(release), accepted=False)
        audit["windows"].append(record)
        if active[release:].any():
            record["reason"] = "not_final_release"
            continue
        stop = min(n, int(release) + post_release_frames)
        seed_stop = int(release) + stationary_frames
        if seed_stop > stop or not observed[max(0, release - pre_release_frames):stop].all():
            record["reason"] = "short_or_unobserved_release"
            continue
        start_bound = max(0, int(release) - pre_release_frames)
        frames = np.arange(start_bound, stop)
        vertices = np.einsum("tij,vj->tvi", rotation[frames], v) + p[frames, None, :]
        com_w = np.einsum("tij,j->ti", rotation[frames], com) + p[frames]
        bottom = vertices[..., 2].min(axis=1)
        for support in supports:
            top = float(support["top_center_w"][2])
            gap = bottom - top
            low = vertices[..., 2] <= bottom[:, None] + .01
            overlap = (low & _inside(vertices[..., :2], support, 0.)).any(axis=1)
            com_inside = _inside(com_w[:, :2], support, com_margin)
            slow = (linear[frames] <= max_linear_speed) & (angular[frames] <= max_angular_speed)
            valid = (gap >= -max_penetration) & (gap <= max_surface_gap) & overlap & com_inside & slow
            seed = slice(int(release) - start_bound, seed_stop - start_bound)
            if not valid[seed].all():
                continue
            # Every active target must remain compatible, including the final hold.
            if not valid[int(release) - start_bound:].all():
                continue
            start = int(release)
            while start > start_bound and valid[start - start_bound - 1]:
                start -= 1
            phase[start:stop] = 1.
            soften[start:stop] = 1.
            ramp = min(10, stop - start)
            soften[stop-ramp:stop] = np.linspace(1., 0., ramp)
            hands = (c[max(0, int(release)-pre_release_frames):release] > .5).any(axis=0)
            withdrawal[start:stop] = hands.astype(np.float32)
            selected = support
            record.update(accepted=True, start_frame=start, end_frame_exclusive=stop,
                          seed_start=int(release), seed_stop=seed_stop, withdrawing_hands=np.flatnonzero(hands).tolist(),
                          support_name=support["name"], min_target_bottom_gap_m=float(gap[start-start_bound:].min()),
                          max_target_bottom_gap_m=float(gap[start-start_bound:].max()),
                          minimum_support_edge_clearance_m=float((support["radius"]-np.linalg.norm(com_w[:, :2]-np.asarray(support["top_center_w"][:2]), axis=1)).min()) if support["kind"] == "cylinder" else None)
            audit["active_windows"] += 1
            break
        if not record["accepted"]:
            record["reason"] = "target_not_slow_supported_with_COM_overlap"
    audit["active_frames"] = int(np.count_nonzero(phase))
    centers = p.astype(np.float32)
    quats = q.astype(np.float32)
    digest = hashlib.sha256(json.dumps(dict(audit=audit, settings=settings), sort_keys=True).encode())
    for a in (centers, quats, phase, soften, withdrawal):
        digest.update(a.tobytes())
        a.setflags(write=False)
    audit["table_sha256"] = digest.hexdigest()
    return PlacementTable(centers, quats, phase, soften, withdrawal, selected, audit)

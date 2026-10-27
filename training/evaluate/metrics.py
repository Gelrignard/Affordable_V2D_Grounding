"""Local Track 2 pose metrics, checked against the released Kaggle formulas.

No initial alignment, time centering, dropped failed worlds or early truncation.
AUC uses supplied local object-frame samples, not the hidden host sample set.
"""

import numpy as np


def _rot(q):
    x, y, z, w = np.moveaxis(q, -1, 0)
    return np.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        axis=-1,
    ).reshape(*q.shape[:-1], 3, 3)


def pose_metrics(achieved, reference, vertices, object_ids=None):
    """Score [frame,world,body,7] against [frame,body,7], xyzw in metres.

    Invalid cohorts are explicitly unavailable rather than silently scoring only
    survivors. Input and exported raw traces are never modified.
    """
    a, r = np.array(achieved, dtype=float), np.array(reference, dtype=float)
    if (
        a.ndim != 4
        or a.shape[-1] != 7
        or min(a.shape[:3]) < 1
        or r.shape != (a.shape[0], a.shape[2], 7)
    ):
        raise ValueError(
            "Expected a nonempty rectangular rollout and matching reference"
        )
    keys = ("add_auc", "spider_sr", "maniptrans_sr", "mppe_cm")
    invalid = ~np.isfinite(a).all(axis=(0, 2, 3)) | (
        np.linalg.norm(a[..., 3:], axis=-1) < 1e-8
    ).any(axis=(0, 2))
    if not np.isfinite(r).all() or (np.linalg.norm(r[..., 3:], axis=-1) < 1e-8).any():
        raise ValueError("Invalid reference poses")
    if invalid.any():
        return {
            "scores": dict.fromkeys(keys),
            "invalid_worlds": np.flatnonzero(invalid).tolist(),
            "reason": "Invalid physical rollout; no worlds discarded",
        }
    a[..., 3:] /= np.linalg.norm(a[..., 3:], axis=-1, keepdims=True)
    r[..., 3:] /= np.linalg.norm(r[..., 3:], axis=-1, keepdims=True)
    r = r[:, None]
    rotation_delta = _rot(a[..., 3:]) - _rot(r[..., 3:])
    position_delta = a[..., :3] - r[..., :3]
    position = np.linalg.norm(position_delta, axis=-1).mean(axis=0)
    dot = np.clip(np.abs((a[..., 3:] * r[..., 3:]).sum(axis=-1)), 0, 1)
    angle = (2 * np.arctan2(np.sqrt(np.maximum(0, 1 - dot * dot)), dot)).mean(axis=0)
    ids = np.arange(a.shape[2]) if object_ids is None else np.asarray(object_ids)
    if ids.shape != (a.shape[2],):
        raise ValueError("One object id per body required")
    sp = (position.mean(axis=-1) <= 0.1) & (angle.mean(axis=-1) <= 0.5)
    mp = np.ones(a.shape[1], dtype=bool)
    for oid in np.unique(ids):
        mask = ids == oid
        mp &= (position[:, mask].mean(axis=-1) < 0.03) & (
            np.degrees(angle[:, mask]).mean(axis=-1) < 30
        )
    keypoints = 0.05 * np.concatenate((np.eye(3), -np.eye(3)))
    delta_points = (
        np.einsum("...ij,kj->...ki", rotation_delta, keypoints)
        + position_delta[..., None, :]
    )
    scores = {
        "add_auc": None,
        "spider_sr": float(sp.mean()),
        "maniptrans_sr": float(mp.mean()),
        "mppe_cm": float(np.linalg.norm(delta_points, axis=-1).mean() * 100),
    }
    if vertices is not None:
        vertices = np.asarray(vertices, dtype=float)
        if (
            vertices.ndim != 3
            or vertices.shape[0] != a.shape[2]
            or vertices.shape[1] < 1
            or vertices.shape[2] != 3
            or not np.isfinite(vertices).all()
        ):
            raise ValueError("Object-frame samples must be finite [body,point,3]")
        add = np.empty(a.shape[:3])
        for body in range(a.shape[2]):
            for frame in range(0, a.shape[0], 64):
                for world in range(0, a.shape[1], 8):
                    f, w = slice(frame, frame + 64), slice(world, world + 8)
                    offset = np.einsum(
                        "twij,vj->twvi", rotation_delta[f, w, body], vertices[body]
                    )
                    offset += position_delta[f, w, body, None, :]
                    add[f, w, body] = np.linalg.norm(offset, axis=-1).mean(axis=-1)
        thresholds = np.arange(0.01, 0.10, 0.01)
        rates = np.array(
            [(add < threshold).mean(axis=(0, 1)) for threshold in thresholds]
        )
        scores["add_auc"] = float(((rates[:-1] + rates[1:]) / 2).mean())
    return {
        "scores": scores,
        "invalid_worlds": [],
        "world_count": a.shape[1],
        "step_count": a.shape[0],
        "object_body_count": a.shape[2],
        "alignment": "none",
        "auc_points": "local_visible_mesh_samples",
    }

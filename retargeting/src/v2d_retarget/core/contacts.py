from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation

SIDES = ("left", "right")
FINGERS = ("thumb", "index", "middle")
TIP_NAMES = tuple(f"{side}_{finger}_tip" for side in SIDES for finger in FINGERS)
SOURCE_TIPS = tuple(
    f"{side.capitalize()}Hand{finger.capitalize()}End"
    for side in SIDES
    for finger in FINGERS
)


@dataclass
class ContactTargets:
    source_tips: np.ndarray
    target_tips: np.ndarray
    active: np.ndarray
    blend: np.ndarray
    anchors_local: np.ndarray
    normals_local: np.ndarray
    surface_distance: np.ndarray
    relative_speed: np.ndarray
    clearance_m: float


def latch_contacts(local_points, distances, nearest, normals, timestamps, cfg):
    """Hysteresis in seconds; anchors are fixed within each continuous contact.

    These are geometric contact hypotheses, not measured forces. Release is
    triggered by distance or excessive drift so sliding/regrasp can relatch.
    """
    n, k, _ = local_points.shape
    speed = np.zeros((n, k))
    if n > 1:
        speed[1:] = (
            np.linalg.norm(np.diff(local_points, axis=0), axis=2)
            / np.diff(timestamps)[:, None]
        )
        speed[0] = speed[1]
    active = np.zeros((n, k), bool)
    anchors = nearest.copy()
    normal_out = normals.copy()
    blend = np.zeros((n, k))
    for finger in range(k):
        on = False
        enter_time = 0.0
        exit_time = 0.0
        amount = 0.0
        anchor = nearest[0, finger].copy()
        normal = normals[0, finger].copy()
        for t in range(n):
            dt = 0.0 if t == 0 else timestamps[t] - timestamps[t - 1]
            if on:
                leaving = (
                    distances[t, finger] > cfg["contact_exit_distance_m"]
                    or np.linalg.norm(local_points[t, finger] - anchor)
                    > cfg["contact_max_anchor_drift_m"]
                )
                exit_time = exit_time + dt if leaving else 0.0
                if leaving and exit_time >= cfg["contact_release_duration_s"]:
                    on = False
                    enter_time = 0.0
            else:
                entering = (
                    distances[t, finger] <= cfg["contact_enter_distance_m"]
                    and speed[t, finger] <= cfg["contact_enter_speed_m_s"]
                )
                enter_time = enter_time + dt if entering else 0.0
                if entering and enter_time >= cfg["contact_enter_duration_s"]:
                    on = True
                    exit_time = 0.0
                    anchor = nearest[t, finger].copy()
                    normal = normals[t, finger].copy()
            change = dt / max(cfg["contact_blend_duration_s"], 1e-9)
            amount = min(1.0, amount + change) if on else max(0.0, amount - change)
            active[t, finger] = on
            blend[t, finger] = amount
            anchors[t, finger] = anchor
            normal_out[t, finger] = normal
    return active, blend, anchors, normal_out, speed


def build_contacts(motion, surface, cfg):
    tips = motion.positions[:, [motion.index(n) for n in SOURCE_TIPS]]
    rotations = Rotation.from_quat(motion.object_wxyz, scalar_first=True).as_matrix()
    local = np.einsum(
        "tji,tkj->tki", rotations, tips - motion.object_positions[:, None, :]
    )
    nearest, distances, normals = surface.contact_candidates(
        local, cfg["contact_exit_distance_m"]
    )
    active, blend, anchors, normals, speed = latch_contacts(
        local, distances, nearest, normals, motion.timestamps, cfg
    )
    contact_local = anchors + cfg["tip_clearance_m"] * normals
    fixed_world = (
        np.einsum("tij,tkj->tki", rotations, contact_local)
        + motion.object_positions[:, None, :]
    )
    target = (1 - blend[..., None]) * tips + blend[..., None] * fixed_world
    return ContactTargets(
        tips.copy(),
        target,
        active,
        blend,
        anchors,
        normals,
        distances,
        speed,
        cfg["tip_clearance_m"],
    )

"""Rigid first-frame alignment for cached camera-space motion."""
import numpy as np
from scipy.spatial.transform import Rotation

BODY_JOINTS = ("Hips", "Chest", "Head", "LeftArm", "RightArm", "LeftForeArm", "RightForeArm",
               "LeftHand", "RightHand", "LeftLeg", "RightLeg", "LeftShin", "RightShin", "LeftFoot", "RightFoot")


def rigid_alignment(camera_frame0, reference_frame0, joint_names):
    indices = [list(joint_names).index(n) for n in BODY_JOINTS]
    p = np.asarray(camera_frame0, dtype=float)[indices]
    q = np.asarray(reference_frame0, dtype=float)[indices]
    if p.shape != q.shape or not np.isfinite([p, q]).all():
        raise ValueError("Invalid first-frame joints")
    a, b = p.mean(axis=0), q.mean(axis=0)
    u, singular, vt = np.linalg.svd((p - a).T @ (q - b))
    if singular[1] < singular[0] * 1e-5:
        raise ValueError("First-frame alignment landmarks are degenerate")
    handedness = np.diag([1., 1., np.linalg.det(vt.T @ u.T)])
    transform = np.eye(4)
    transform[:3, :3] = vt.T @ handedness @ u.T
    transform[:3, 3] = b - transform[:3, :3] @ a
    distances = np.linalg.norm(p @ transform[:3, :3].T + transform[:3, 3] - q, axis=1)
    return transform, {"method": "rigid Kabsch on frame 0 only; no scale fit", "joint_names": list(BODY_JOINTS),
                       "residual_metres": distances.tolist(), "rms_metres": float(np.sqrt(np.mean(distances**2))),
                       "singular_values": singular.tolist()}


def apply_human_transform(positions, quaternions_wxyz, transform):
    p = np.asarray(positions) @ transform[:3, :3].T + transform[:3, 3]
    q = np.asarray(quaternions_wxyz)
    r = Rotation.from_quat(q[..., [1, 2, 3, 0]].reshape(-1, 4)).as_matrix()
    out = Rotation.from_matrix(transform[:3, :3] @ r).as_quat().reshape(q.shape)
    return p, out[..., [3, 0, 1, 2]]


def object_source_mesh_poses(camera_body_poses, source_mesh_to_pose_body, world_from_camera):
    # Vertices remain in the original supplied GLB's frame. Transform that
    # source mesh through the native centered body frame before world alignment.
    return world_from_camera @ np.asarray(camera_body_poses) @ source_mesh_to_pose_body

"""Smoothed object-COM wrench targets; no robot-trajectory feasibility assumption."""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation


def required_wrench(position, quaternion_xyzw, *, dt, mass, inertia, com, window=11):
    """Return world-frame force and body-frame torque about the simulator COM.

    Position is the reference body origin. Gravity is excluded from the physical
    contact target. The two-window discrepancy is a heuristic confidence, not a
    calibrated probability. No clipping silently changes the specified motion.
    """
    p = np.asarray(position, dtype=np.float64)
    q = np.asarray(quaternion_xyzw, dtype=np.float64)
    inertia = np.asarray(inertia, dtype=np.float64)
    com = np.asarray(com, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3 or len(p) < 7 or q.shape != (len(p), 4):
        raise ValueError("At least seven position/quaternion frames are required")
    if not all(np.isfinite(a).all() for a in (p, q, inertia, com)):
        raise ValueError("Reference and physical parameters must be finite")
    if not np.isfinite(dt) or dt <= 0 or not np.isfinite(mass) or mass <= 0:
        raise ValueError("Positive finite dt and mass required")
    if inertia.shape != (3, 3) or com.shape != (3,):
        raise ValueError("Expected a 3x3 COM inertia and a three-vector COM")
    if not np.allclose(inertia, inertia.T) or np.linalg.eigvalsh(inertia).min() <= 0:
        raise ValueError("COM inertia must be symmetric positive definite")
    if window < 5 or window % 2 != 1 or np.any(np.linalg.norm(q, axis=1) < 1e-8):
        raise ValueError("Use an odd window >=5 and nonzero quaternions")
    rot = Rotation.from_quat(q)
    com_w = p + rot.apply(np.broadcast_to(com, p.shape).copy())
    max_window = len(p) if len(p) % 2 else len(p) - 1

    def derivatives(width):
        width = min(width, max_window)
        acceleration = savgol_filter(com_w, width, 3, deriv=2, delta=dt, axis=0)
        # Relative rotations, unlike quaternion component differences, are sign invariant.
        angular_mid = (rot[1:] * rot[:-1].inv()).as_rotvec() / dt
        omega_w = np.vstack(
            (angular_mid[0], (angular_mid[:-1] + angular_mid[1:]) / 2, angular_mid[-1])
        )
        omega_w = savgol_filter(omega_w, width, 3, axis=0)
        alpha_w = savgol_filter(omega_w, width, 3, deriv=1, delta=dt, axis=0)
        omega_b = rot.inv().apply(omega_w)
        alpha_b = rot.inv().apply(alpha_w)
        force_w = mass * (acceleration - np.array([0.0, 0.0, -9.81]))
        torque_b = alpha_b @ inertia.T + np.cross(omega_b, omega_b @ inertia.T)
        return force_w, torque_b, acceleration

    force, torque, acceleration = derivatives(window)
    wider_force, wider_torque, _ = derivatives(2 * window - 1)
    radius = max(float(np.sqrt(np.trace(inertia) / mass)), 0.01)
    discrepancy = np.linalg.norm(force - wider_force, axis=1) / (mass * 9.81)
    discrepancy += np.linalg.norm(torque - wider_torque, axis=1) / (
        mass * 9.81 * radius
    )
    confidence = 1.0 / (1.0 + discrepancy**2)
    # Boundary derivatives lack symmetric context.
    confidence[: window // 2] *= 0.5
    confidence[-(window // 2) :] *= 0.5
    return {
        "force_w": force.astype(np.float32),
        "torque_b": torque.astype(np.float32),
        "torque_w": rot.apply(torque).astype(np.float32),
        "confidence": confidence.astype(np.float32),
        "acceleration_w": acceleration.astype(np.float32),
    }

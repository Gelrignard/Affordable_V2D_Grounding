import time
import warnings
from dataclasses import dataclass
import numpy as np
import pinocchio as pin
import pink
from pink.tasks import FrameTask, PostureTask
from scipy.spatial.transform import Rotation
from .contacts import TIP_NAMES
from .constraints import OutputVelocityLimit, HullAvoidanceTask
from ..dataprocessing.completion import fill_missing


@dataclass
class SolveResult:
    q: np.ndarray
    iterations: np.ndarray
    stopped: list[str]
    elapsed_seconds: float


def repair_failed_frames(output, missing, times, kin, velocity_fraction):
    """Repair only numerically missing IK frames; retain finite solved frames."""
    result = np.asarray(output)
    if not missing:
        return result
    samples = result.copy()
    samples[missing] = np.nan
    base, _ = fill_missing(times, samples[:, :3])
    rotations, _ = fill_missing(times, samples[:, [6, 3, 4, 5]], quaternion=True)
    joints, _ = fill_missing(times, samples[:, 7:])
    result = np.concatenate((base, rotations[:, [1, 2, 3, 0]], joints), axis=1)
    low, high = kin.robot.model.lowerPositionLimit[7:], kin.robot.model.upperPositionLimit[7:]
    if np.any(joints < low - 1e-4) or np.any(joints > high + 1e-4):
        raise ValueError('Repaired IK violates joint limits')
    if len(times) > 1:
        bounds = np.diff(times)[:, None] * kin.robot.model.velocityLimit[6:] * velocity_fraction
        if np.any(np.abs(np.diff(joints, axis=0)) > bounds + 1e-6):
            raise ValueError('Repaired IK violates joint velocity limits')
    warnings.warn(f'Interpolated numerically missing IK frames {missing}; finite solved frames retained', RuntimeWarning)
    return result


def solve_contact_aware(kin, motion, body, contacts, surface, cfg, progress=print):
    tips = [
        FrameTask(n, position_cost=1.0, orientation_cost=0.0, lm_damping=0.05)
        for n in TIP_NAMES
    ]
    posture = PostureTask(cost=kin.config.posture_task.q_prev_cost, lm_damping=0.0)
    avoidance = HullAvoidanceTask(kin, surface, cfg["penetration_cost"])
    obj_rot = Rotation.from_quat(motion.object_wxyz, scalar_first=True).as_matrix()
    output = []
    iterations = []
    stopped = []
    missing = []
    previous = None
    start = time.perf_counter()
    for t in range(len(motion.timestamps)):
        initial = kin.robot.q0.copy() if previous is None else previous.copy()
        kin.configuration.update(initial)
        posture.set_target(initial)
        for j, task in enumerate(kin.frame_tasks.values()):
            task.set_target(pin.SE3(body.rotations[t, j], body.positions[t, j]))
        for j, task in enumerate(tips):
            task.set_position_cost(
                cfg["tip_free_cost"]
                + (cfg["tip_contact_cost"] - cfg["tip_free_cost"])
                * contacts.blend[t, j]
            )
            task.set_target(pin.SE3(np.eye(3), contacts.target_tips[t, j]))
        avoidance.set_object(obj_rot[t], motion.object_positions[t])
        tasks = list(kin.frame_tasks.values()) + tips + [posture, avoidance]
        dt = (
            1 / 200
            if previous is None
            else float(motion.timestamps[t] - motion.timestamps[t - 1])
        )
        limits = list(kin.configuration_limits)
        if previous is not None:
            limits.append(
                OutputVelocityLimit(
                    kin.robot.model, previous, dt, cfg["velocity_fraction"]
                )
            )
        reason = "iteration_limit"
        for iteration in range(cfg["max_iterations"]):
            velocity = pink.solve_ik(
                kin.configuration,
                tasks,
                kin.dt,
                solver=kin.solver,
                limits=limits,
                safety_break=True,
            )
            if not np.isfinite(velocity).all():
                reason = 'missing_numerical_frame'
                kin.configuration.update(initial)
                break
            kin.configuration.integrate_inplace(velocity, kin.dt)
            # QP feasibility tolerances can leave a hinge a few microradians
            # outside its bound, which Pink rejects on the next iteration.
            # Project numerical residue only; substantial violations still fail.
            bounded = kin.configuration.q.copy()
            if not np.isfinite(bounded).all():
                reason = 'missing_numerical_frame'
                kin.configuration.update(initial)
                break
            low, high = kin.robot.model.lowerPositionLimit[7:], kin.robot.model.upperPositionLimit[7:]
            violation = np.maximum(low-bounded[7:], bounded[7:]-high)
            if np.max(violation) > 1e-4:
                raise ValueError(f"IK joint bound violation at frame {t}: {np.max(violation)} rad")
            bounded[7:] = np.clip(bounded[7:], low, high)
            kin.configuration.update(bounded)
            if (
                iteration >= 4
                and np.linalg.norm(velocity * kin.dt) < cfg["step_tolerance"]
            ):
                reason = "small_step"
                break
        q = kin.configuration.q.copy()
        if reason == 'missing_numerical_frame':
            missing.append(t)
        # Same ground proxy as baseline; all diagnostics are computed AFTER it.
        soles = np.array(
            [
                p.translation[2] - kin.config.ankle_roll_offset
                for p in kin.poses(q, kin.config.foot_frames)
            ]
        )
        q[2] += max(0.0, -float(soles.min()))
        if not np.isfinite(q).all():
            raise ValueError(f"Nonfinite IK at frame {t}")
        if previous is not None:
            bound = kin.robot.model.velocityLimit[6:] * dt * cfg["velocity_fraction"]
            if np.any(np.abs(q[7:] - previous[7:]) > bound + 1e-6):
                raise ValueError(f"Output velocity constraint failed at frame {t}")
        output.append(q)
        previous = q
        iterations.append(iteration + 1)
        stopped.append(reason)
        if t == 0 or (t + 1) % 100 == 0 or t + 1 == len(motion.timestamps):
            progress(
                f"Contact IK {t + 1}/{len(motion.timestamps)} ({time.perf_counter() - start:.1f}s)"
            )
    return SolveResult(
        repair_failed_frames(output, missing, motion.timestamps, kin, cfg['velocity_fraction']),
        np.asarray(iterations), stopped, time.perf_counter() - start
    )


def solve_baseline(kin, motion, progress=print):
    """Re-run pinned NVIDIA objectives; preserve their stopping and grounding."""
    world = np.asarray(kin.config.r_world)
    source_p = motion.positions @ world
    shape = motion.rotations_wxyz.shape
    r = Rotation.from_quat(
        motion.rotations_wxyz.reshape(-1, 4), scalar_first=True
    ).as_matrix()
    source_q = (
        Rotation.from_matrix(world.T @ r).as_quat(scalar_first=True).reshape(shape)
    )
    out = []
    iterations = []
    previous = None
    start = time.perf_counter()
    for t in range(len(source_p)):
        result = kin.compute(
            source_p[t], source_q[t], qpos=previous, foot_target_ground_z=0.0
        )
        q = result["q"].copy()
        soles = [
            p.translation[2] - kin.config.ankle_roll_offset
            for p in kin.poses(q, kin.config.foot_frames)
        ]
        q[2] += max(0.0, -min(soles))
        previous = q
        out.append(q)
        iterations.append(result["num_optimization_iterations"])
        if t == 0 or (t + 1) % 100 == 0 or t + 1 == len(source_p):
            progress(
                f"Baseline IK {t + 1}/{len(source_p)} ({time.perf_counter() - start:.1f}s)"
            )
    return SolveResult(
        np.asarray(out),
        np.asarray(iterations),
        ["upstream_stopping_rule"] * len(out),
        time.perf_counter() - start,
    )

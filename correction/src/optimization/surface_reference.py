"""Build a smooth, fixed-contact candidate around one accepted static grasp.

This is an acquisition/replay experiment, not the final horizon correction.
Only robot motion is interpolated; observed object acceleration is never used.
"""
from time import perf_counter
from copy import deepcopy
from dataclasses import dataclass, asdict
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation
from .surface_hypotheses import solve_geometry, contact_kinematics


def smoothstep(x):
    x = np.clip(x, 0., 1.)
    return x*x*x*(10+x*(-15+6*x))


@dataclass(frozen=True)
class PhaseTiming:
    """Frame indices in the original uninterrupted episode, never reset times."""
    approach_start: int
    close_start: int
    contact_start: int
    force_full: int
    contact_end: int
    force_end: int
    release_end: int
    hold_end: int

    def validate(self, count):
        indices = (self.approach_start, self.close_start, self.contact_start,
                   self.force_full, self.contact_end, self.force_end, self.release_end, self.hold_end)
        if not all(isinstance(i, (int, np.integer)) for i in indices) or not (
                0 <= indices[0] < indices[1] < indices[2] < indices[3] <= indices[4] <
                indices[5] <= indices[6] <= indices[7] < count):
            raise ValueError('Phase timing must be ordered original-episode frame indices')


def build_reference(scene, raw, points, grasp_q, forces_world, grasp_frame, *,
                    first=235, last=480, stride=12, solve_seconds=2.,
                    approach=False, rigid_grasp=False,release=False, phase=None, arm_neighborhood=.9,
                    grasp_object_pose=None, continuous_ik=False):
    start = perf_counter()
    times = raw['timestamps']-raw['timestamps'][0]
    count = len(times); idx = 7+scene.movable
    timing = phase or PhaseTiming(168, 192, first, 270, last, last+21, last+45, last+45)
    timing.validate(count)
    first, last = timing.contact_start, timing.contact_end
    if not 0<=first<last<count:
        raise ValueError('Episode must include the selected grasp and complete release transition')
    lower = scene.kin.robot.model.lowerPositionLimit[idx]
    upper = scene.kin.robot.model.upperPositionLimit[idx]
    nominal = savgol_filter(raw['q'][:, idx], 21, 3, axis=0)
    nominal = np.clip(nominal, lower, upper)
    nominal[0] = raw['q'][0, idx]
    begin = timing.approach_start if approach else first
    solve_end=timing.hold_end if release else last
    knot_frames = np.unique(np.r_[np.arange(0, count, stride), count-1,
                                  begin-36, begin-24, begin-12, begin, first, last, solve_end,solve_end+24,solve_end+36])
    knot_frames = knot_frames[(knot_frames >= 0) & (knot_frames < count)]
    if rigid_grasp:
        from .rigid_grasp import RigidGrasp
        grasp_p, grasp_r = (grasp_object_pose if grasp_object_pose is not None else
                            (raw['object_positions'][grasp_frame], raw['object_wxyz'][grasp_frame]))
        rigid=RigidGrasp(scene,points,grasp_q,grasp_p,grasp_r)
        knot_frames=np.unique(np.r_[knot_frames,np.arange(begin,solve_end+1)])
    if release:
        knot_frames=knot_frames[~((knot_frames>solve_end)&(knot_frames<solve_end+24))]
    knots = nominal[knot_frames].copy()
    records = []
    previous = grasp_q.copy()
    previous_frame = None
    for row, frame in enumerate(knot_frames):
        if not begin <= frame <= solve_end: continue
        if continuous_ik and previous_frame is None:
            previous_frame = int(knot_frames[max(0, row-1)])
            previous = raw['q'][previous_frame].copy()
            previous[idx] = nominal[previous_frame]
        target_points = deepcopy(points)
        clearance=lift=0.
        if approach:
            # Approach outside each object's face, then close. Both endpoints
            # still refer to the observed object; physics never sets its pose.
            clearance = .055*(1-smoothstep((frame-timing.close_start)/(first-timing.close_start)))
            lift = .06*(1-smoothstep((frame-begin)/(first-begin)))
            up_local = Rotation.from_quat(raw['object_wxyz'][frame],scalar_first=True).inv().apply([0.,0.,lift])
            for point in target_points:
                point.object_local = point.object_local+clearance*point.object_outward+up_local
        if release and frame>last:
            clearance=.08*smoothstep((frame-last)/(timing.release_end-last))
            lift=.06*smoothstep((frame-last)/(timing.release_end-last))
            up_local=Rotation.from_quat(raw['object_wxyz'][frame],scalar_first=True).inv().apply([0.,0.,lift])
            target_points=deepcopy(points)
            for point in target_points:point.object_local=point.object_local+clearance*point.object_outward+up_local
        initial = raw['q'][frame].copy(); initial[idx] = previous[idx]
        if rigid_grasp:
            q,audit=rigid.solve(raw['q'][frame],raw['object_positions'][frame],raw['object_wxyz'][frame],initial,
                clearance=clearance if approach or release else 0.,lift=lift if approach or release else 0.,
                arm_neighborhood=arm_neighborhood,
                step_dt=max(float(times[frame]-times[previous_frame]), 1e-6) if continuous_ik else None)
        else:
            q, audit = solve_geometry(scene, target_points, raw['q'][frame], raw['object_positions'][frame],
                                      raw['object_wxyz'][frame], initial=initial, seconds=solve_seconds, max_evaluations=160)
        knots[row] = q[idx]; previous = q
        previous_frame = int(frame)
        records.append(dict(frame=int(frame), approach=bool(approach and frame<first), **audit))
    # Shape-preserving C1 interpolation avoids cubic overshoot of joint limits.
    # Dynamics replay uses these derivatives, not an unrelated reconstructed spline.
    spline = PchipInterpolator(times[knot_frames], knots, axis=0)
    q = raw['q'].copy(); q[:, idx] = spline(times)
    velocity = spline(times, 1)
    dense = np.minimum(np.arange(round(times[-1]/.001)+1)*.001, times[-1])
    dq, dv, da = (spline(dense, order) for order in range(3))
    speed = scene.kin.robot.model.velocityLimit[6+scene.movable]
    violation = np.maximum(np.maximum(lower-dq, dq-upper), 0)
    # Fixed object-frame force plan is a candidate feedforward. It is not an
    # assertion that gravity demand rotates with the object or that it is delivered.
    forces_local = Rotation.from_quat(raw['object_wxyz'][grasp_frame], scalar_first=True).inv().apply(forces_world)
    r = Rotation.from_quat(raw['object_wxyz'], scalar_first=True).as_matrix()
    forces = np.einsum('tij,kj->tki', r, forces_local)
    blend = smoothstep((np.arange(count)-first)/(timing.force_full-first))*smoothstep(
        (timing.force_end-np.arange(count))/(timing.force_end-last))
    forces *= blend[:, None, None]
    active = np.tile((blend > 1e-5)[:, None], (1, len(points)))
    normals = np.tile(-np.array([p.object_outward for p in points]), (count, 1, 1))
    gaps = []
    for frame in range(first, last+1):
        scene.set_state(q[frame], raw['object_positions'][frame], raw['object_wxyz'][frame])
        position, _, _, _ = contact_kinematics(scene, points)
        targets = raw['object_positions'][frame]+Rotation.from_quat(raw['object_wxyz'][frame], scalar_first=True).apply(
            np.array([p.object_local for p in points]))
        gaps.append(np.linalg.norm(position-targets, axis=1))
    report = dict(scope='local contact IK; movable joints are smoothed and interpolated across the full timeline; callers may compose only a selected interval',
        first_frame=first, last_frame=last, approach_start_frame=begin, approach_enabled=approach,
        rigid_grasp=rigid_grasp,
        release_enabled=release,release_end_frame=solve_end,
        timing=asdict(timing),
        arm_neighborhood_rad=arm_neighborhood,
        continuous_ik=continuous_ik,
        stride=stride, interpolation='PCHIP C1; explicit command velocities',
        max_joint_limit_violation_rad=float(violation.max()),
        max_speed_fraction=float(np.max(abs(dv)/speed)),
        max_command_acceleration_rad_s2=float(np.max(abs(da))),
        rms_command_acceleration_rad_s2=float(np.sqrt(np.mean(da*da))),
        dense_max_contact_gap_m=float(np.max(gaps)), knot_audits=records,
        wall_seconds=perf_counter()-start, ready_for_training=False)
    return dict(q=q, command_velocity=velocity, force_ff=forces, contact_active=active,
                inward_normals_local=normals, timestamps=times), report

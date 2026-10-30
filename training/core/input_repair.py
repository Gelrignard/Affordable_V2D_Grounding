"""Complete missing source guidance before allocation, never policy rollouts."""
import numpy as np
from data_tools.observation_completion import complete_samples


class UnusableReference(ValueError):
    """A reference has no usable trajectory; a separate original may be used."""


PORTABLE_PAIRS = (("joint_positions", "robot_joint_positions", False),
                  ("root_position", "robot_root_position", False), ("root_wxyz", "robot_root_wxyz", False),
                  ("object_position", "object_body_position", True), ("object_wxyz", "object_body_wxyz", True))


def merge_portable_counterparts(reference, row):
    """Fill absent samples only from the same job's finite paired representation.

    Interpolation follows separately, after both stored representations have
    supplied their observed samples. Originals and already-finite values stay
    unchanged. This does not reconcile disagreeing finite trajectories.
    """
    reference_input, motion_input, missing_frames = dict(reference), dict(row), {}
    for ref_key, row_key, object_axis in PORTABLE_PAIRS:
        a = np.asarray(reference[ref_key], dtype=float)
        b = np.asarray(row[row_key], dtype=float)
        if object_axis:
            b = b[:, 0]
        if a.shape != b.shape:
            raise ValueError(f"Portable motion/reference shape mismatch: {ref_key}")
        good_a, good_b = np.isfinite(a), np.isfinite(b)
        if ref_key.endswith("wxyz"):
            good_a = np.broadcast_to((good_a.all(-1) & (np.linalg.norm(a, axis=-1) > 1e-12))[:, None], a.shape)
            good_b = np.broadcast_to((good_b.all(-1) & (np.linalg.norm(b, axis=-1) > 1e-12))[:, None], b.shape)
        missing = ~(good_a & good_b)
        if missing.any():
            missing_frames[ref_key] = np.flatnonzero(missing.reshape(len(a), -1).any(-1)).tolist()
            aa, bb = a.copy(), b.copy()
            aa[~good_a & good_b] = b[~good_a & good_b]
            bb[~good_b & good_a] = a[~good_b & good_a]
            reference_input[ref_key] = aa.astype(reference[ref_key].dtype, copy=False)
            motion_input[row_key] = (bb[:, None] if object_axis else bb).tolist()
    object_missing = sorted(set(missing_frames.get('object_position', [])) |
                            set(missing_frames.get('object_wxyz', [])))
    if object_missing:
        reference_input['object_observed_mask'] = reference['object_observed_mask'].copy()
        reference_input['object_observed_mask'][object_missing] = False
    return reference_input, motion_input, missing_frames


def _fill(times, value, kind, name, report):
    value = np.asarray(value, dtype=float)
    try:
        if kind == 'pose':
            position, observed_p = complete_samples(times, value[..., :3])
            rotation, observed_r = complete_samples(times, value[..., 3:], rotation='wxyz')
            observed = observed_p.all(axis=-1) & observed_r
            result = np.concatenate((position, rotation), axis=-1)
        else:
            result, observed = complete_samples(times, value, rotation=kind)
        bad_frames = np.flatnonzero(~observed.reshape(len(times), -1).all(axis=1))
    except ValueError as exc:
        raise UnusableReference(f'{name}: {exc}') from exc
    if len(bad_frames):
        report[name] = bad_frames.tolist()
    return result


def repair_reference(reference):
    """Return a repaired copy and source-frame repair locations, or the input."""
    times = reference['timestamps']
    result, report = dict(reference), {}
    for name in ('joint_positions', 'joint_velocity', 'root_position', 'root_wxyz',
                 'object_position', 'object_wxyz', 'ee_pose_w'):
        if name not in reference:
            continue
        kind = 'wxyz' if name.endswith('wxyz') else 'pose' if name == 'ee_pose_w' else None
        filled = _fill(times, reference[name], kind, name, report)
        if name in report:
            result[name] = filled.astype(reference[name].dtype, copy=False)
    if not report:
        return reference, None
    missing = sorted(set(report.get('object_position', [])) | set(report.get('object_wxyz', [])))
    if missing:
        result['object_observed_mask'] = reference['object_observed_mask'].copy()
        result['object_observed_mask'][missing] = False
    return result, report


def repair_motion_row(row):
    """Complete motion_v1 references on their original uniform sample clock.

    Schema, asset identity and unit checks remain the caller's responsibility.
    A wholly absent channel is not replaced by fabricated motion.
    """
    n = len(row['robot_joint_positions'])
    fps = float(row['fps'])
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError('Motion FPS must be finite and positive')
    times = np.arange(n, dtype=float) / fps
    result, report = dict(row), {}
    kinds = {'robot_joint_positions': None, 'robot_root_position': None,
             'robot_root_wxyz': 'wxyz', 'object_body_position': None,
             'object_body_wxyz': 'wxyz', 'ee_pose_w': 'pose', 'hand_frames_w': 'pose'}
    for name, kind in kinds.items():
        if name not in row or not np.asarray(row[name]).size:
            continue
        value = np.asarray(row[name], dtype=float)
        if kind == 'pose' and value.shape[-1] != 7:
            # Unknown hand-frame representations require a declared adapter.
            if not np.isfinite(value).all():
                raise UnusableReference(f'Cannot infer the representation of {name}')
            continue
        filled = _fill(times, value, kind, name, report)
        if name in report:
            result[name] = filled.tolist()
    if not report:
        return row, None
    return result, {'method': 'same-input linear position/scalar and SLERP rotation; nearest endpoint hold',
                    'repaired_source_frame_indices': report, 'frame_count': n,
                    'valid_samples_unchanged': True, 'original_input_unchanged': True}

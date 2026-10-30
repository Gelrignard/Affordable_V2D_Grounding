"""Complete a pose timeline without turning interpolants into observations."""
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def complete_samples(times, values, *, rotation=None):
    """Fill missing source samples only; leave every valid sample unchanged.

    Rotation channels are complete xyz rotation vectors or wxyz quaternions.
    Other channels are interpolated independently. End gaps hold the nearest
    observation. Entirely absent channels cannot be inferred from neighbours.
    This helper must not be applied to simulated rollouts or learner state.
    """
    times = np.asarray(times, dtype=float)
    values = np.asarray(values)
    if (times.ndim != 1 or not len(times) or not np.isfinite(times).all()
            or np.any(np.diff(times) <= 0) or values.shape[0] != len(times)):
        raise ValueError('Completion requires the original finite, increasing timeline')
    result = values.copy()
    if rotation is None:
        flat = result.reshape(len(times), -1)
        observed = np.isfinite(flat)
        for channel in range(flat.shape[1]):
            good = observed[:, channel]
            if good.all():
                continue
            if not good.any():
                raise ValueError(f'No observed values for source channel {channel}')
            flat[~good, channel] = np.interp(times[~good], times[good], flat[good, channel])
        return result, observed.reshape(values.shape)
    size = {'rotvec': 3, 'wxyz': 4}.get(rotation)
    if size is None or values.shape[-1] != size:
        raise ValueError('Rotation completion expects rotvec or wxyz channels')
    flat = result.reshape(len(times), -1, size)
    observed = np.isfinite(flat).all(axis=-1)
    if rotation == 'wxyz':
        observed &= np.linalg.norm(flat, axis=-1) > 1e-8
    for channel in range(flat.shape[1]):
        good = observed[:, channel]
        if good.all():
            continue
        if not good.any():
            raise ValueError(f'No observed rotations for source channel {channel}')
        raw = flat[good, channel]
        rotations = (Rotation.from_rotvec(raw) if rotation == 'rotvec'
                     else Rotation.from_quat(raw[:, [1, 2, 3, 0]]))
        filled = (rotations[0] if good.sum() == 1 else
                  Slerp(times[good], rotations)(np.clip(times[~good], times[good][0], times[good][-1])))
        flat[~good, channel] = (filled.as_rotvec() if rotation == 'rotvec'
                               else filled.as_quat()[..., [3, 0, 1, 2]])
    return result, observed.reshape(values.shape[:-1])


def complete_object_poses(times, poses, visible):
    times, poses, visible = np.asarray(times), np.asarray(poses), np.asarray(visible, bool)
    if poses.shape != (len(times), 7) or visible.shape != times.shape or np.any(np.diff(times) <= 0):
        raise ValueError('Expected increasing timestamps and matching xyz/wxyz observations')
    valid = visible & np.isfinite(poses).all(axis=1) & (np.linalg.norm(poses[:, 3:], axis=1) > 1e-8)
    ids = np.flatnonzero(valid)
    if not len(ids):
        raise ValueError('No usable observed object pose; cannot create a reference')
    result = np.empty_like(poses, dtype=float)
    for j in range(3):
        result[:, j] = np.interp(times, times[ids], poses[ids, j])
    rotations = Rotation.from_quat(poses[ids, 3:], scalar_first=True)
    result[:, 3:] = (rotations[0].as_quat(scalar_first=True) if len(ids) == 1 else
        Slerp(times[ids], rotations)(np.clip(times, times[ids[0]], times[ids[-1]])).as_quat(scalar_first=True))
    # Sign alignment is representational only; retain every observed rotation.
    for k in range(1, len(result)):
        if result[k, 3:]@result[k-1, 3:] < 0:
            result[k, 3:] *= -1
    return result, valid

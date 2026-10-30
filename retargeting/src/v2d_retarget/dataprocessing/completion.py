"""Repair missing source/retarget samples without smoothing valid observations."""
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def fill_missing(times, values, *, quaternion=False):
    result = np.asarray(values).copy()
    width = result.shape[-1]
    flat = result.reshape(len(times), -1, width)
    if not quaternion:
        observed = np.isfinite(flat)
        for channel in range(flat.shape[1]):
            for axis in range(width):
                good = observed[:, channel, axis]
                if good.all():
                    continue
                if not good.any():
                    raise ValueError(f'No finite source samples in channel {channel}, axis {axis}')
                flat[~good, channel, axis] = np.interp(times[~good], times[good], flat[good, channel, axis])
        return result, observed.all(axis=-1).reshape(result.shape[:-1])
    valid = np.isfinite(flat).all(axis=-1)
    if quaternion:
        valid &= np.linalg.norm(flat, axis=-1) > 1e-8
    for channel in range(flat.shape[1]):
        good = valid[:, channel]
        if good.all():
            continue
        if not good.any():
            raise ValueError(f'No finite source samples in channel {channel}')
        if quaternion:
            rotations = Rotation.from_quat(flat[good, channel][:, [1, 2, 3, 0]])
            filled = (rotations[0] if good.sum() == 1 else
                      Slerp(times[good], rotations)(np.clip(times[~good], times[good][0], times[good][-1])))
            flat[~good, channel] = filled.as_quat()[..., [3, 0, 1, 2]]
    return result, valid.reshape(result.shape[:-1])

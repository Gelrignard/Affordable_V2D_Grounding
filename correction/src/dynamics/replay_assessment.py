"""Assess complete correction replays without importing experiment commands."""
import numpy as np

from correction.src.optimization.robot_manipulation import metrics, tracking_accepted


def assessed(record, times):
    result = metrics(record)
    acceleration = np.diff(record['joint_velocities'], axis=0)/np.diff(times)[:, None]
    eef_velocity = np.diff(record['fk_tip_positions'], axis=0)/np.diff(times)[:, None, None]
    eef_acceleration = np.diff(eef_velocity, axis=0)/(.5*(times[2:]-times[:-2]))[:, None, None]
    result.update(rms_joint_acceleration_rad_s2=float(np.sqrt(np.mean(acceleration**2))),
                  rms_eef_acceleration_m_s2=float(np.sqrt(np.mean(eef_acceleration**2))),
                  max_joint_step_rad=float(np.max(abs(np.diff(record['q'][:, 7:], axis=0)))),
                  max_eef_step_m=float(np.max(np.linalg.norm(np.diff(record['fk_tip_positions'], axis=0), axis=-1))),
                  simulated_frames=len(record['q']), full_episode=True)
    for key, value in record.items():
        if key.startswith('controller_') and np.asarray(value).ndim == 0:
            result[key] = value.item() if isinstance(value, np.ndarray) else value
    result['accepted_physics'] = bool(tracking_accepted(record) and record['max_torque_fraction_requested'] <= 1.)
    return result

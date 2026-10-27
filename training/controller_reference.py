"""Keep manipulation guidance separate from the motion sent to frozen SONIC."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np


def carrier_joint_coordinates(scene, task_reference, carrier_reference):
    """Map a same-timeline named carrier into SONIC's simulation joint order."""
    if carrier_reference.num_frames != task_reference.num_frames or not np.isclose(
        carrier_reference.fps, task_reference.fps
    ):
        raise ValueError("SONIC carrier and task must have the same timeline")
    scalar = scene.layout.scalar_joints
    names = carrier_reference.robot_joint_names()
    if len(set(names)) != len(names) or set(names) != set(scalar.names):
        raise ValueError("SONIC carrier must use the same named robot joints")
    columns = {name: i for i, name in enumerate(names)}
    q = scene.robot_reference.joint_q.copy()
    q[:, :3] = carrier_reference.robot_root_pos_w()
    q[:, 3:7] = carrier_reference.robot_root_quat_w()[:, [1, 2, 3, 0]]
    for name, qid in zip(scalar.names, scalar.q_ids, strict=True):
        q[:, qid] = carrier_reference.robot_joint_pos()[:, columns[name]]
    if not np.isfinite(q).all():
        raise ValueError("nonfinite SONIC carrier coordinates")
    return q


def install_sonic_carrier(env, scene, task_reference, carrier_reference):
    """Replace only tokenizer motion; task targets, fingers and resets stay intact.

    Call before goal-latent encoding and graph capture. This does not promise
    stability: the carrier itself still needs physical validation.
    """
    from flash_chord.runtime.actions.sonic import build_sonic_encoder_reference

    q = carrier_joint_coordinates(scene, task_reference, carrier_reference)
    binding = replace(scene.robot_reference, joint_q=q)
    encoded = build_sonic_encoder_reference(
        SimpleNamespace(robot_reference=binding, layout=scene.layout)
    )
    env.action.encoder_reference.assign(encoded)

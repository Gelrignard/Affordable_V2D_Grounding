"""Add one measured contact-wrench objective to the pinned native CHORD environment.

The simulator enforces contact/friction and actuator limits. This module scores
realized force, not a feasibility certificate or the virtual-assistance wrench.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp
from flash_chord.envs.rl import RLEnv
from flash_chord.objectives.recon_body import ReconBodyObjectiveConfig

from training.dynamics_targets import required_wrench


@wp.kernel
def accumulate_object_wrench(
    count: wp.array(dtype=wp.int32),
    shape0: wp.array(dtype=wp.int32),
    shape1: wp.array(dtype=wp.int32),
    point0: wp.array(dtype=wp.vec3),
    point1: wp.array(dtype=wp.vec3),
    force: wp.array(dtype=wp.spatial_vector),
    shape_body: wp.array(dtype=wp.int32),
    shape_hand: wp.array(dtype=wp.int32),
    shapes_per_world: int,
    body_q: wp.array(dtype=wp.transform),
    body_com: wp.array(dtype=wp.vec3),
    bodies_per_world: int,
    object_body: int,
    hand_wrench: wp.array(dtype=wp.spatial_vector),
    support_wrench: wp.array(dtype=wp.spatial_vector),
    hand_force_magnitude: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    if i >= count[0]:
        return
    s0 = int(shape0[i])
    s1 = int(shape1[i])
    b0 = int(-1)  # noqa: UP018, RUF046 - Warp mutable local
    b1 = int(-1)  # noqa: UP018, RUF046 - Warp mutable local
    if s0 >= 0:
        b0 = shape_body[s0]
    if s1 >= 0:
        b1 = shape_body[s1]
    obj = int(-1)  # noqa: UP018, RUF046 - Warp mutable local
    other = int(-1)  # noqa: UP018, RUF046 - Warp mutable local
    sign = float(1.0)  # noqa: UP018 - Warp mutable local
    point = point0[i]
    if b0 >= 0 and b0 % bodies_per_world == object_body:
        obj = b0
        other = s1
    elif b1 >= 0 and b1 % bodies_per_world == object_body:
        obj = b1
        other = s0
        sign = -1.0
        point = point1[i]
    if obj < 0:
        return
    world = obj // bodies_per_world
    f = sign * wp.spatial_top(force[i])
    p = wp.transform_point(body_q[obj], point)
    com = wp.transform_point(body_q[obj], body_com[obj])
    tau = wp.cross(p - com, f) + sign * wp.spatial_bottom(force[i])
    wrench = wp.spatial_vector(f, tau)
    is_hand = False
    if (
        other >= 0
        and other < shape_body.shape[0]
        and other // shapes_per_world == world
    ):
        is_hand = shape_hand[other % shapes_per_world] >= 0
    if is_hand:
        wp.atomic_add(hand_wrench, world, wrench)
        wp.atomic_add(hand_force_magnitude, world, wp.length(f))
    else:
        wp.atomic_add(support_wrench, world, wrench)


@wp.kernel
def compose_dynamics_term(
    native_terms: wp.array(dtype=wp.float32),
    native_score: wp.array(dtype=wp.float32),
    timestep: wp.array(dtype=wp.int32),
    body_q: wp.array(dtype=wp.transform),
    object_body: int,
    bodies_per_world: int,
    target_force_w: wp.array(dtype=wp.vec3),
    target_torque_w: wp.array(dtype=wp.vec3),
    confidence: wp.array(dtype=wp.float32),
    reference_active: wp.array(dtype=wp.float32),
    hand: wp.array(dtype=wp.spatial_vector),
    support: wp.array(dtype=wp.spatial_vector),
    hand_force_magnitude: wp.array(dtype=wp.float32),
    force_scale: float,
    torque_scale: float,
    weight: wp.array(dtype=wp.float32),
    frame_dt: float,
    terms: wp.array(dtype=wp.float32),
    score: wp.array(dtype=wp.float32),
    diagnostics: wp.array2d(dtype=wp.float32),
):
    w = wp.tid()
    t = wp.clamp(timestep[w], 0, target_force_w.shape[0] - 1)
    # Subtract measured environmental support before asking the hands for the remainder.
    needed_hand_f = target_force_w[t] - wp.spatial_top(support[w])
    needed_hand_tau = target_torque_w[t] - wp.spatial_bottom(support[w])
    df = wp.length(wp.spatial_top(hand[w]) - needed_hand_f) / force_scale
    dtau = wp.length(wp.spatial_bottom(hand[w]) - needed_hand_tau) / torque_scale
    deficit = wp.sqrt(df * df + dtau * dtau)
    # Smooth robust loss; saturation prevents rare impact spikes dominating reward.
    huber = wp.sqrt(1.0 + deficit * deficit) - 1.0
    value = confidence[t] * wp.exp(-wp.min(huber, 20.0))
    for j in range(12):
        terms[w * 13 + j] = native_terms[w * 12 + j]
    terms[w * 13 + 12] = value
    score[w] = native_score[w] + frame_dt * weight[0] * value
    diagnostics[w, 0] = deficit
    diagnostics[w, 1] = wp.where(hand_force_magnitude[w] > 0.02 * force_scale, 1.0, 0.0)
    diagnostics[w, 2] = reference_active[t]
    diagnostics[w, 3] = df
    diagnostics[w, 4] = dtau
    diagnostics[w, 5] = confidence[t]


@dataclass(frozen=True)
class DynamicsObjectiveConfig(ReconBodyObjectiveConfig):
    dynamics_weight: float = 0.0
    smoothing_window: int = 11

    def build(
        self,
        model,
        embodiment,
        robot_reference,
        reference,
        command,
        contact,
        action,
        **kwargs,
    ):
        native = super().build(
            model,
            embodiment,
            robot_reference,
            reference,
            command,
            contact,
            action,
            **kwargs,
        )
        return DynamicsObjective(native, model, reference, self)


class DynamicsObjective:
    def __init__(self, native, model, reference, config):
        self.native = native
        self.model = model
        self.config = config
        self.contacts = None
        self.world_count = native.world_count
        self.term_names = native.term_names + ("dynamics_wrench",)
        ids = (
            np.arange(native.world_count) * native.bodies_per_world
            + native.object_body_id
        )
        masses = model.body_mass.numpy()[ids]
        inertias = model.body_inertia.numpy()[ids]
        coms = model.body_com.numpy()[ids]
        for values in (masses, inertias, coms):
            if not np.allclose(values, values[0], atol=1e-6):
                raise ValueError(
                    "V1 guidance requires identical object physics across worlds"
                )
        self.mass = float(masses[0])
        self.radius = max(float(native.command.object_radius.numpy()[0]), 0.01)
        q_wxyz = np.asarray(reference.object_body_quat_w())[:, 0]
        self.targets = required_wrench(
            np.asarray(reference.object_body_pos_w())[:, 0],
            q_wxyz[:, [1, 2, 3, 0]],
            dt=native.frame_dt,
            mass=self.mass,
            inertia=inertias[0],
            com=coms[0],
            window=config.smoothing_window,
        )
        device = native.score.device
        self.target_f = wp.array(self.targets["force_w"], dtype=wp.vec3, device=device)
        self.target_tau = wp.array(
            self.targets["torque_w"], dtype=wp.vec3, device=device
        )
        self.confidence = wp.array(self.targets["confidence"], device=device)
        labels = np.asarray(native.contact_active.numpy()).reshape(-1, native.num_hands)
        self.reference_active = wp.array(
            (labels.max(axis=1) > 0.5).astype(np.float32), device=device
        )
        self.hand = wp.zeros(self.world_count, dtype=wp.spatial_vector, device=device)
        self.support = wp.zeros(
            self.world_count, dtype=wp.spatial_vector, device=device
        )
        self.terms = wp.zeros(self.world_count * 13, dtype=wp.float32, device=device)
        self.hand_magnitude = wp.zeros(
            self.world_count, dtype=wp.float32, device=device
        )
        self.score = wp.zeros(self.world_count, dtype=wp.float32, device=device)
        self.diagnostics = wp.zeros(
            (self.world_count, 6), dtype=wp.float32, device=device
        )
        self.weight = wp.array(
            [config.dynamics_weight], dtype=wp.float32, device=device
        )
        self.weights = wp.array(
            [*native.weights.numpy().tolist(), config.dynamics_weight],
            dtype=wp.float32,
            device=device,
        )

    def set_weights(self, weights):
        if set(weights) != set(self.term_names) or not all(
            np.isfinite(v) for v in weights.values()
        ):
            raise ValueError("Invalid dynamics objective weights")
        self.native.set_weights({k: weights[k] for k in self.native.term_names})
        self.weight.assign([weights["dynamics_wrench"]])
        self.weights.assign([weights[name] for name in self.term_names])

    def reset(self, mask):
        self.native.reset(mask)

    def evaluate(self, state, timestep, terminated):
        if self.contacts is None:
            raise RuntimeError(
                "Guidance must bind the solver's measured contacts before capture"
            )
        self.native.evaluate(state, timestep, terminated)
        self.hand.zero_()
        self.support.zero_()
        self.hand_magnitude.zero_()
        c, n = self.contacts, self.native
        wp.launch(
            accumulate_object_wrench,
            dim=c.rigid_contact_max,
            inputs=[
                c.rigid_contact_count,
                c.rigid_contact_shape0,
                c.rigid_contact_shape1,
                c.rigid_contact_point0,
                c.rigid_contact_point1,
                c.force,
                self.model.shape_body,
                n.contact.shape_hand_ids,
                n.contact.shapes_per_world,
                state.body_q,
                self.model.body_com,
                n.bodies_per_world,
                n.object_body_id,
            ],
            outputs=[self.hand, self.support, self.hand_magnitude],
        )
        wp.launch(
            compose_dynamics_term,
            dim=self.world_count,
            inputs=[
                n.terms,
                n.score,
                timestep,
                state.body_q,
                n.object_body_id,
                n.bodies_per_world,
                self.target_f,
                self.target_tau,
                self.confidence,
                self.reference_active,
                self.hand,
                self.support,
                self.hand_magnitude,
                self.mass * 9.81,
                self.mass * 9.81 * self.radius,
                self.weight,
                n.frame_dt,
            ],
            outputs=[self.terms, self.score, self.diagnostics],
        )


class GuidedRLEnv(RLEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(self.objective, DynamicsObjective):
            if self.contacts.force is None:
                raise ValueError("Measured solver contact forces are required")
            self.objective.contacts = self.contacts

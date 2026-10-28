"""Opt-in task-focused refinement of the pinned native CHORD runtime.

Hydra targets are ``RefinementObjectiveConfig`` and ``RefinementTerminationConfig``.
The actor, critic observations, action mapping, SONIC, and body tracking term are
unchanged. The thirteenth *raw* objective term is available to SAC replay reward
relabeling; changing only the environment score would not train SAC correctly.
Reference contact is a smoothed shaping prior, never a claim of physical contact.
"""
from __future__ import annotations

import functools
import inspect
import json
import math
from dataclasses import dataclass

import numpy as np
import warp as wp


def smooth_contact_blend(labels, radius: int = 10) -> np.ndarray:
    """Symmetric triangular smoothing, preserving constant contact/no-contact."""
    values = np.asarray(labels, dtype=np.float32)
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError("contact labels must be a nonempty finite [frames, hands] array")
    if radius < 0 or int(radius) != radius:
        raise ValueError("contact blend radius must be a non-negative integer")
    values = np.clip(values, 0.0, 1.0)
    if radius == 0:
        return values.copy()
    radius = int(radius)
    kernel = radius + 1 - np.abs(np.arange(-radius, radius + 1))
    kernel = kernel / kernel.sum()
    padded = np.pad(values, ((radius, radius), (0, 0)), mode="edge")
    return np.stack([np.convolve(padded[:, h], kernel, mode="valid")
                     for h in range(values.shape[1])], axis=1).astype(np.float32)


@wp.func
def _angle(a: wp.quat, b: wp.quat) -> float:
    qa = wp.normalize(a)
    qb = wp.normalize(b)
    return 2.0 * wp.acos(wp.clamp(wp.abs(wp.dot(qa, qb)), 0.0, 1.0))


@wp.kernel
def compose_refinement_objective(
    body_q: wp.array(dtype=wp.transform),
    timestep: wp.array(dtype=wp.int32),
    reference_wrist_pos: wp.array(dtype=wp.vec3),
    reference_wrist_quat: wp.array(dtype=wp.quat),
    reference_object_pos: wp.array(dtype=wp.vec3),
    reference_object_quat: wp.array(dtype=wp.quat),
    object_target_pos: wp.array(dtype=wp.vec3),
    object_target_quat: wp.array(dtype=wp.quat),
    wrist_body_ids: wp.array(dtype=wp.int32),
    wrist_local_pos: wp.array(dtype=wp.vec3),
    wrist_local_quat: wp.array(dtype=wp.quat),
    contact_blend: wp.array(dtype=wp.float32),
    native_terms: wp.array(dtype=wp.float32),
    weights: wp.array(dtype=wp.float32),
    num_frames: int,
    bodies_per_world: int,
    object_body_id: int,
    use_relative: int,
    position_tolerance: float,
    world_position_std: float,
    relative_position_std: float,
    hand_orientation_std: float,
    hand_orientation_factor: float,
    object_orientation_std: float,
    object_position_fine_fraction: float,
    object_position_fine_std: float,
    frame_dt: float,
    terms: wp.array(dtype=wp.float32),
    score: wp.array(dtype=wp.float32),
) -> None:
    world = wp.tid()
    frame = wp.clamp(timestep[world], 0, num_frames - 1)
    body_base = world * bodies_per_world
    object_xf = body_q[body_base + object_body_id]
    object_inv = wp.transform_inverse(object_xf)
    reference_object_xf = wp.transform(reference_object_pos[frame],
                                       wp.normalize(reference_object_quat[frame]))
    reference_object_inv = wp.transform_inverse(reference_object_xf)
    position_cost = float(0.0)
    orientation_cost = float(0.0)
    for hand in range(2):
        ref_index = frame * 2 + hand
        palm_body = body_q[body_base + wrist_body_ids[hand]]
        palm_pos = wp.transform_point(palm_body, wrist_local_pos[hand])
        palm_quat = wp.normalize(wp.mul(wp.transform_get_rotation(palm_body), wrist_local_quat[hand]))
        ref_pos = reference_wrist_pos[ref_index]
        ref_quat = wp.normalize(reference_wrist_quat[ref_index])
        blend = float(0.0)
        if use_relative != 0:
            blend = contact_blend[ref_index]
        world_distance = wp.max(wp.length(palm_pos - ref_pos) - position_tolerance, 0.0)
        live_relative = wp.transform_point(object_inv, palm_pos)
        ref_relative = wp.transform_point(reference_object_inv, ref_pos)
        relative_distance = wp.max(wp.length(live_relative - ref_relative) - position_tolerance, 0.0)
        position_cost += ((1.0 - blend) * world_distance * world_distance / (world_position_std * world_position_std)
                          + blend * relative_distance * relative_distance / (relative_position_std * relative_position_std))
        world_angle = _angle(palm_quat, ref_quat)
        relative_angle = _angle(wp.mul(wp.transform_get_rotation(object_inv), palm_quat),
                                wp.mul(wp.transform_get_rotation(reference_object_inv), ref_quat))
        orientation_cost += ((1.0 - blend) * world_angle * world_angle
                             + blend * relative_angle * relative_angle) / (hand_orientation_std * hand_orientation_std)
    base = world * 13
    for term in range(12):
        terms[base + term] = native_terms[world * 12 + term]
    # Zero leaves the copied native term bit-for-bit unchanged. Use the command
    # target, which can differ from the raw reference after scene transforms.
    if object_position_fine_fraction > 0.0:
        object_delta = wp.transform_get_translation(object_xf) - object_target_pos[world]
        fine_position = wp.exp(-wp.dot(object_delta, object_delta)
                               / (object_position_fine_std * object_position_fine_std))
        terms[base + 7] = ((1.0 - object_position_fine_fraction) * terms[base + 7]
                           + object_position_fine_fraction * fine_position)
    # Retain the native sum-over-two-hands exponential, avoiding another scale change.
    terms[base + 9] = wp.exp(-position_cost)
    terms[base + 10] = hand_orientation_factor * wp.exp(-orientation_cost)
    object_angle = _angle(wp.transform_get_rotation(object_xf), object_target_quat[world])
    terms[base + 12] = wp.exp(-object_angle * object_angle / (object_orientation_std * object_orientation_std))
    total = float(0.0)
    for term in range(13):
        total += weights[term] * terms[base + term]
    score[world] = total * frame_dt


@wp.kernel
def apply_acquisition_position(
    body_q: wp.array(dtype=wp.transform), timestep: wp.array(dtype=wp.int32),
    reference_wrist_pos: wp.array(dtype=wp.vec3),
    reference_object_pos: wp.array(dtype=wp.vec3), reference_object_quat: wp.array(dtype=wp.quat),
    wrist_body_ids: wp.array(dtype=wp.int32), wrist_local_pos: wp.array(dtype=wp.vec3),
    contact_blend: wp.array(dtype=wp.float32), centers_o: wp.array(dtype=wp.vec3),
    radii: wp.array(dtype=wp.float32), acquisition_blend: wp.array(dtype=wp.float32),
    weights: wp.array(dtype=wp.float32), num_frames: int, bodies_per_world: int,
    object_body_id: int, use_relative: int, position_tolerance: float,
    world_position_std: float, relative_position_std: float, frame_dt: float,
    terms: wp.array(dtype=wp.float32), score: wp.array(dtype=wp.float32),
) -> None:
    world = wp.tid()
    frame = wp.clamp(timestep[world], 0, num_frames - 1)
    if acquisition_blend[frame * 2] > 0.0 or acquisition_blend[frame * 2 + 1] > 0.0:
        body_base = world * bodies_per_world
        object_inv = wp.transform_inverse(body_q[body_base + object_body_id])
        ref_inv = wp.transform_inverse(wp.transform(reference_object_pos[frame],
                                                   wp.normalize(reference_object_quat[frame])))
        cost = float(0.0)
        for hand in range(2):
            i = frame * 2 + hand
            palm = wp.transform_point(body_q[body_base + wrist_body_ids[hand]], wrist_local_pos[hand])
            relative = wp.transform_point(object_inv, palm)
            ref_relative = wp.transform_point(ref_inv, reference_wrist_pos[i])
            world_distance = wp.max(wp.length(palm - reference_wrist_pos[i]) - position_tolerance, 0.0)
            relative_distance = wp.max(wp.length(relative - ref_relative) - position_tolerance, 0.0)
            blend = float(0.0)
            if use_relative != 0:
                blend = contact_blend[i]
            existing_cost = ((1.0 - blend) * world_distance * world_distance / (world_position_std * world_position_std)
                             + blend * relative_distance * relative_distance / (relative_position_std * relative_position_std))
            region_distance = wp.max(wp.length(relative - centers_o[i]) - radii[i], 0.0)
            region_cost = region_distance * region_distance / (relative_position_std * relative_position_std)
            alpha = acquisition_blend[i]
            cost += (1.0 - alpha) * existing_cost + alpha * region_cost
        base = world * 13
        terms[base + 9] = wp.exp(-cost)
        total = float(0.0)
        for term in range(13):
            total += weights[term] * terms[base + term]
        score[world] = total * frame_dt


@wp.kernel
def reset_refinement_counters(
    reset_mask: wp.array(dtype=wp.int32),
    palm_count: wp.array(dtype=wp.int32),
    object_count: wp.array(dtype=wp.int32),
    last_episode_step: wp.array(dtype=wp.int32),
) -> None:
    world = wp.tid()
    if reset_mask[world] != 0:
        palm_count[world] = 0
        object_count[world] = 0
        last_episode_step[world] = -1


@wp.kernel
def apply_recovery_termination(
    episode_step: wp.array(dtype=wp.int32),
    palm_position: wp.array(dtype=wp.float32),
    palm_orientation: wp.array(dtype=wp.float32),
    object_position: wp.array(dtype=wp.float32),
    object_orientation: wp.array(dtype=wp.float32),
    pelvis_done: wp.array(dtype=wp.int32),
    reset_freeze_steps: int,
    palm_position_threshold: float,
    palm_orientation_threshold: float,
    object_position_threshold: float,
    object_orientation_threshold: float,
    palm_dwell_steps: int,
    object_dwell_steps: int,
    palm_gross_position: float,
    palm_gross_orientation: float,
    object_gross_position: float,
    object_gross_orientation: float,
    palm_count: wp.array(dtype=wp.int32),
    object_count: wp.array(dtype=wp.int32),
    last_episode_step: wp.array(dtype=wp.int32),
    palm_done: wp.array(dtype=wp.int32),
    object_done: wp.array(dtype=wp.int32),
    terminated: wp.array(dtype=wp.int32),
    packed_diagnostics: wp.array(dtype=wp.float32),
) -> None:
    world = wp.tid()
    age = episode_step[world]
    if age < last_episode_step[world]:
        palm_count[world] = 0
        object_count[world] = 0
    # Explicit reset hook also handles a reset at the same episode age.
    last_episode_step[world] = age
    pd = int(0)
    od = int(0)
    if age <= reset_freeze_steps:
        palm_count[world] = 0
        object_count[world] = 0
    else:
        palm_bad = (palm_position[world] > palm_position_threshold
                    or palm_orientation[world] > palm_orientation_threshold)
        object_bad = (object_position[world] > object_position_threshold
                      or object_orientation[world] > object_orientation_threshold)
        if palm_bad:
            palm_count[world] = wp.min(palm_count[world] + 1, palm_dwell_steps)
        else:
            palm_count[world] = 0
        if object_bad:
            object_count[world] = wp.min(object_count[world] + 1, object_dwell_steps)
        else:
            object_count[world] = 0
        pd = int(palm_count[world] >= palm_dwell_steps
                 or palm_position[world] > palm_gross_position
                 or palm_orientation[world] > palm_gross_orientation)
        od = int(object_count[world] >= object_dwell_steps
                 or object_position[world] > object_gross_position
                 or object_orientation[world] > object_gross_orientation)
    palm_done[world] = pd
    object_done[world] = od
    terminated[world] = wp.max(pelvis_done[world], wp.max(pd, od))
    packed_diagnostics[world * 9 + 1] = float(pd)
    packed_diagnostics[world * 9 + 2] = float(od)


# Kernels remain importable on CPU-only developer machines without the simulator.
_RUNTIME_IMPORT_ERROR = None
try:
    from flash_chord.objectives.recon_body import ReconBodyObjectiveConfig as _NativeObjectiveConfig
    from flash_chord.lifecycle.termination import ReconBodyTerminationConfig as _NativeTerminationConfig
except ModuleNotFoundError as exc:
    _RUNTIME_IMPORT_ERROR = exc
    _NativeObjectiveConfig = object
    _NativeTerminationConfig = object


def _require_runtime():
    if _RUNTIME_IMPORT_ERROR is not None:
        raise ImportError("The pinned flash_chord runtime is required to build refinement configs") from _RUNTIME_IMPORT_ERROR


@dataclass(frozen=True)
class RefinementObjectiveConfig(_NativeObjectiveConfig):
    object_orientation_weight: float = 2.0
    object_orientation_std: float = 0.6
    object_position_fine_fraction: float = 0.0
    object_position_fine_std: float = 0.05
    contact_blend_frames: int = 10
    hand_position_tolerance: float = 0.03
    relative_hand_position_std: float = 0.15
    hand_orientation_std: float = 0.8
    hand_orientation_factor: float = 0.25
    use_object_relative_hands: bool = True
    acquisition_enabled: bool = False
    acquisition_reference_path: str | None = None
    acquisition_reference_sha256: str | None = None
    acquisition_window_frames: int = 25
    acquisition_seed_frames: int = 5
    acquisition_region_radius: float = 0.03
    acquisition_contact_distance: float = 0.03
    acquisition_min_palm_clearance: float = 0.015
    acquisition_max_seed_spread: float = 0.04
    placement_enabled: bool = False
    placement_reference_path: str | None = None
    placement_reference_sha256: str | None = None
    placement_support_path: str | None = None
    placement_support_sha256: str | None = None

    def __post_init__(self):
        _require_runtime()
        super().__post_init__()
        vals = (self.object_orientation_weight, self.object_orientation_std,
                self.object_position_fine_fraction, self.object_position_fine_std,
                self.hand_position_tolerance, self.relative_hand_position_std,
                self.hand_orientation_std, self.hand_orientation_factor)
        if not all(math.isfinite(v) for v in vals):
            raise ValueError("refinement objective parameters must be finite")
        if min(self.object_orientation_std, self.object_position_fine_std,
               self.relative_hand_position_std, self.hand_orientation_std) <= 0:
            raise ValueError("refinement objective standard deviations must be positive")
        if not 0.0 <= self.object_position_fine_fraction <= 1.0:
            raise ValueError("object_position_fine_fraction must be in [0, 1]")
        if self.hand_position_tolerance < 0 or not 0 <= self.hand_orientation_factor <= 1:
            raise ValueError("invalid hand tolerance or orientation factor")
        if int(self.contact_blend_frames) != self.contact_blend_frames or self.contact_blend_frames < 0:
            raise ValueError("contact_blend_frames must be a non-negative integer")
        if self.acquisition_enabled:
            if not self.acquisition_reference_path or not self.acquisition_reference_sha256:
                raise ValueError("enabled acquisition requires portable reference path and SHA256")
            if (len(self.acquisition_reference_sha256) != 64
                    or any(c not in "0123456789abcdef" for c in self.acquisition_reference_sha256)):
                raise ValueError("invalid acquisition reference SHA256")
            if (type(self.acquisition_window_frames) is not int or self.acquisition_window_frames < 4
                    or type(self.acquisition_seed_frames) is not int or self.acquisition_seed_frames < 3):
                raise ValueError("invalid acquisition window/seed frame counts")
            if not all(math.isfinite(v) and v > 0 for v in (
                self.acquisition_region_radius, self.acquisition_contact_distance,
                self.acquisition_min_palm_clearance, self.acquisition_max_seed_spread,
            )):
                raise ValueError("acquisition geometry tolerances must be positive and finite")
        if self.placement_enabled:
            if self.acquisition_enabled:
                raise ValueError("placement and acquisition are separate experimental variants")
            for kind in ("reference", "support"):
                path, digest = getattr(self, "placement_" + kind + "_path"), getattr(self, "placement_" + kind + "_sha256")
                if not path or not digest or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                    raise ValueError("enabled placement requires input/support paths and SHA256")

    def build(self, *args, **kwargs):
        table = None
        if self.acquisition_enabled:
            from training.core.acquisition import table_from_native
            bound = inspect.signature(_NativeObjectiveConfig.build).bind(self, *args, **kwargs)
            table = table_from_native(bound.arguments["reference"], bound.arguments["robot_reference"], self)
            if table.audit["active_windows"] == 0:
                raise ValueError("acquisition has no valid windows: " + json.dumps(table.audit, sort_keys=True))
        result = RefinementObjective(super().build(*args, **kwargs), self, acquisition_table=table)
        if self.placement_enabled:
            from training.core.placement import PlacementRuntime
            bound = inspect.signature(_NativeObjectiveConfig.build).bind(self, *args, **kwargs)
            result._placement = PlacementRuntime(result.native, bound.arguments["model"], self, bound.arguments["reference"])
            result.placement_audit = result._placement.audit
            result.placement_diagnostics = result._placement.diagnostics
        return result


class RefinementObjective:
    def __init__(self, native, config, acquisition_table=None):
        self.native = native
        self.config = config
        self._term_names = native.term_names + ("object_orientation",)
        device = native.terms.device
        self._weights = wp.array(tuple(config.weights) + (config.object_orientation_weight,), dtype=wp.float32, device=device)
        self._terms = wp.zeros(native.world_count * 13, dtype=wp.float32, device=device)
        self._score = wp.zeros(native.world_count, dtype=wp.float32, device=device)
        labels = native.contact_active.numpy().reshape(native.num_frames, 2)
        self.contact_blend = wp.array(smooth_contact_blend(labels, config.contact_blend_frames).reshape(-1), dtype=wp.float32, device=device)
        self.acquisition_audit = None
        self._acquisition_arrays = None
        self._placement = None
        self.placement_audit = None
        self.placement_diagnostics = {}
        if getattr(config, "acquisition_enabled", False):
            if acquisition_table is None or acquisition_table.audit["active_windows"] == 0:
                raise ValueError("enabled acquisition requires at least one validated window")
            if not np.allclose(native.command.object_scales.numpy(), 1.0, atol=1e-6, rtol=0):
                raise ValueError("acquisition currently requires unscaled object geometry")
            self.acquisition_audit = acquisition_table.audit
            self._acquisition_arrays = (
                wp.array(acquisition_table.centers_o.reshape(-1, 3), dtype=wp.vec3, device=device),
                wp.array(acquisition_table.radii.reshape(-1), dtype=wp.float32, device=device),
                wp.array(acquisition_table.blend.reshape(-1), dtype=wp.float32, device=device),
            )

    # Runtime protocols on Python 3.12 use inspect.getattr_static: delegated
    # __getattr__ attributes are invisible. Keep every protocol member explicit.
    @property
    def term_names(self):
        return self._term_names

    @property
    def weights(self):
        return self._weights

    @property
    def terms(self):
        return self._terms

    @property
    def score(self):
        return self._score

    def __getattr__(self, name):
        return getattr(self.native, name)

    def reset(self, reset_mask):
        self.native.reset(reset_mask)

    def set_weights(self, weights):
        if set(weights) != set(self.term_names):
            raise ValueError(f"refinement weights must contain exactly {self.term_names}")
        values = tuple(float(weights[name]) for name in self.term_names)
        if not all(math.isfinite(v) for v in values):
            raise ValueError("refinement weights must be finite")
        self.native.set_weights({name: weights[name] for name in self.native.term_names})
        self.weights.assign(values)

    def evaluate(self, state, timestep, terminated):
        self.native.evaluate(state, timestep, terminated)
        n, c = self.native, self.config
        wp.launch(compose_refinement_objective, dim=n.world_count, inputs=[
            state.body_q, timestep, n.reference.wrist_pos_w, n.reference.wrist_quat_w,
            n.command.reference.body_pos_w, n.command.reference.body_quat_w,
            n.command.body_target_pos_w, n.command.body_target_quat_w, n.wrist_frames.body_ids,
            n.wrist_frames.body_to_frame_pos, n.wrist_frames.body_to_frame_quat,
            self.contact_blend, n.terms, self.weights, n.num_frames,
            n.bodies_per_world, n.object_body_id, int(c.use_object_relative_hands),
            c.hand_position_tolerance, c.palm_position_std, c.relative_hand_position_std,
            c.hand_orientation_std, c.hand_orientation_factor, c.object_orientation_std,
            c.object_position_fine_fraction, c.object_position_fine_std, n.frame_dt,
        ], outputs=[self.terms, self.score], device=n.terms.device)
        if self._acquisition_arrays is not None:
            wp.launch(apply_acquisition_position, dim=n.world_count, inputs=[
                state.body_q, timestep, n.reference.wrist_pos_w,
                n.command.reference.body_pos_w, n.command.reference.body_quat_w,
                n.wrist_frames.body_ids, n.wrist_frames.body_to_frame_pos,
                self.contact_blend, *self._acquisition_arrays, self.weights, n.num_frames,
                n.bodies_per_world, n.object_body_id, int(c.use_object_relative_hands),
                c.hand_position_tolerance, c.palm_position_std, c.relative_hand_position_std, n.frame_dt,
            ], outputs=[self.terms, self.score], device=n.terms.device)
        if self._placement is not None:
            self._placement.evaluate(self, state, timestep)


@dataclass(frozen=True)
class RefinementTerminationConfig(_NativeTerminationConfig):
    palm_position_threshold: float = 0.25
    palm_orientation_threshold: float = 2.2
    object_position_threshold: float = 0.20
    palm_dwell_steps: int = 10
    object_dwell_steps: int = 5
    palm_gross_position: float = 0.60
    palm_gross_orientation: float = 2.8
    object_gross_position: float = 0.40
    object_gross_orientation: float = 2.8

    def __post_init__(self):
        _require_runtime()
        super().__post_init__()
        if min(self.palm_dwell_steps, self.object_dwell_steps) < 1:
            raise ValueError("termination dwell must be positive")
        if any(int(v) != v for v in (self.palm_dwell_steps, self.object_dwell_steps)):
            raise ValueError("termination dwell must use integer steps")
        values = (self.palm_gross_position, self.palm_gross_orientation,
                  self.object_gross_position, self.object_gross_orientation)
        if not all(math.isfinite(v) and v >= 0 for v in values):
            raise ValueError("gross failure thresholds must be finite and non-negative")

    def build(self, *args, **kwargs):
        install()
        return RefinementTermination(super().build(*args, **kwargs), self)


class RefinementTermination:
    def __init__(self, native, config):
        self.native, self.config = native, config
        self.palm_count = wp.zeros(native.world_count, dtype=wp.int32, device=native.terminated.device)
        self.object_count = wp.zeros_like(self.palm_count)
        self.last_episode_step = wp.full(native.world_count, -1, dtype=wp.int32, device=native.terminated.device)

    @property
    def terminated(self):
        return self.native.terminated

    @property
    def truncated(self):
        return self.native.truncated

    @property
    def cause_names(self):
        return self.native.cause_names

    @property
    def cause_arrays(self):
        return self.native.cause_arrays

    @property
    def error_names(self):
        return self.native.error_names

    @property
    def error_arrays(self):
        return self.native.error_arrays

    @property
    def packed_diagnostic_names(self):
        return self.native.packed_diagnostic_names

    @property
    def packed_diagnostics(self):
        return self.native.packed_diagnostics

    def __getattr__(self, name):
        return getattr(self.native, name)

    def reset_counters(self, reset_mask):
        wp.launch(reset_refinement_counters, dim=self.world_count, inputs=[reset_mask],
                  outputs=[self.palm_count, self.object_count, self.last_episode_step], device=self.terminated.device)

    def evaluate(self, body_q, timestep):
        self.native.evaluate(body_q, timestep)
        n, c = self.native, self.config
        wp.launch(apply_recovery_termination, dim=n.world_count, inputs=[
            n.episode_step, n.palm_position_error, n.palm_orientation_error,
            n.object_position_error, n.object_orientation_error, n.pelvis_done,
            c.reset_freeze_steps, c.palm_position_threshold, c.palm_orientation_threshold,
            c.object_position_threshold, c.object_orientation_threshold,
            c.palm_dwell_steps, c.object_dwell_steps, c.palm_gross_position,
            c.palm_gross_orientation, c.object_gross_position, c.object_gross_orientation,
        ], outputs=[self.palm_count, self.object_count, self.last_episode_step,
                    n.palm_done, n.object_done, n.terminated, n.packed_diagnostics], device=n.terminated.device)


def install():
    """Install an idempotent reset hook affecting only opt-in termination instances.

The explicit mask hook is required: comparing episode ages alone cannot detect
all resets (e.g. repeated one-step episodes). Warp launches are capture-safe.
"""
    _require_runtime()
    from flash_chord.envs.base import BaseEnv
    if getattr(BaseEnv.reset, "_policy_refinement_hook", False):
        return
    original = BaseEnv.reset

    @functools.wraps(original)
    def reset(self, frame_id=0, reset_frame=None, reset_mask=None, finger_scale=None):
        original(self, frame_id=frame_id, reset_frame=reset_frame, reset_mask=reset_mask, finger_scale=finger_scale)
        if isinstance(self.termination, RefinementTermination):
            mask = self.reset_table.all_reset_mask if reset_mask is None else reset_mask
            self.termination.reset_counters(mask)

    reset._policy_refinement_hook = True
    BaseEnv.reset = reset

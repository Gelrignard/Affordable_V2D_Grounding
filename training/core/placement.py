"""Optional current-state supported placement shaping; no learned dynamics or history.

The support bonus is an additive part of raw object-position term7. Separate
load-transfer, settling and finger-unloading credit replaces the previous
all-factor readiness gate. Palm costs remain unchanged. All changes happen
before native n-step reward aggregation; use fresh replay.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import warp as wp


PLACEMENT_DIAGNOSTIC_NAMES = (
    'support_load_fraction', 'object_linear_speed', 'object_angular_speed',
    'placement_ready', 'method_window', 'placement_bonus',
    'placement_load_credit', 'placement_settle_credit', 'placement_release_credit',
    'placement_tilt_rad', 'placement_release_quality', 'placement_motion_quality',
    'placement_spatial_compatibility', 'placement_upright_compatibility',
)


@wp.kernel
def clear_support(force: wp.array(dtype=wp.vec3), count: wp.array(dtype=wp.int32)):
    w = wp.tid()
    force[w] = wp.vec3(0.0)
    count[w] = 0


@wp.kernel
def aggregate_support_contacts(
    count: wp.array(dtype=wp.int32), shape0: wp.array(dtype=wp.int32),
    shape1: wp.array(dtype=wp.int32), force: wp.array(dtype=wp.spatial_vector),
    shape_body: wp.array(dtype=wp.int32), shape_count: int,
    bodies_per_world: int, worlds: int, object_id: int, support_id: int,
    support_force: wp.array(dtype=wp.vec3), support_count: wp.array(dtype=wp.int32),
):
    i = wp.tid()
    if i < count[0]:
        s0 = shape0[i]
        s1 = shape1[i]
        if s0 >= 0 and s1 >= 0 and s0 < shape_count and s1 < shape_count:
            b0 = shape_body[s0]
            b1 = shape_body[s1]
            if b0 >= 0 and b1 >= 0:
                w0 = b0 // bodies_per_world
                w1 = b1 // bodies_per_world
                if w0 == w1 and w0 < worlds:
                    l0 = b0 % bodies_per_world
                    l1 = b1 % bodies_per_world
                    f = wp.spatial_top(force[i])  # Newton force acts on shape0.
                    if l0 == object_id and l1 == support_id:
                        wp.atomic_add(support_force, w0, f)
                        wp.atomic_add(support_count, w0, 1)
                    elif l1 == object_id and l0 == support_id:
                        wp.atomic_add(support_force, w0, -f)
                        wp.atomic_add(support_count, w0, 1)


@wp.kernel
def apply_supported_placement(
    body_q: wp.array(dtype=wp.transform), body_qd: wp.array(dtype=wp.spatial_vector),
    timestep: wp.array(dtype=wp.int32), support_force: wp.array(dtype=wp.vec3),
    hand_force: wp.array(dtype=wp.vec3), slots_per_world: int,
    centers: wp.array(dtype=wp.vec3), window: wp.array(dtype=wp.float32),
    reference_object_quat: wp.array(dtype=wp.quat),
    weights: wp.array(dtype=wp.float32), num_frames: int, bodies_per_world: int,
    object_id: int, object_com: wp.vec3, object_weight: float,
    support_center: wp.vec3, support_normal: wp.vec3, support_radius: float,
    frame_dt: float,
    terms: wp.array(dtype=wp.float32), score: wp.array(dtype=wp.float32),
    load_out: wp.array(dtype=wp.float32), linear_out: wp.array(dtype=wp.float32),
    angular_out: wp.array(dtype=wp.float32), ready_out: wp.array(dtype=wp.float32),
    window_out: wp.array(dtype=wp.float32), bonus_out: wp.array(dtype=wp.float32),
    load_credit_out: wp.array(dtype=wp.float32), settle_credit_out: wp.array(dtype=wp.float32),
    release_credit_out: wp.array(dtype=wp.float32), tilt_out: wp.array(dtype=wp.float32),
    release_quality_out: wp.array(dtype=wp.float32), motion_quality_out: wp.array(dtype=wp.float32),
    spatial_out: wp.array(dtype=wp.float32), upright_out: wp.array(dtype=wp.float32),
):
    w = wp.tid()
    frame = wp.clamp(timestep[w], 0, num_frames - 1)
    xf = body_q[w * bodies_per_world + object_id]
    velocity = body_qd[w * bodies_per_world + object_id]
    linear = wp.length(wp.spatial_top(velocity))
    angular = wp.length(wp.spatial_bottom(velocity))
    load = wp.max(wp.dot(support_force[w], support_normal) / object_weight, 0.0)
    delta = wp.transform_get_translation(xf) - centers[frame]
    distance = wp.max(wp.length(delta) - 0.03, 0.0)
    near = wp.exp(-distance * distance / (0.05 * 0.05))
    com_delta = wp.transform_point(xf, object_com) - support_center
    planar = com_delta - wp.dot(com_delta, support_normal) * support_normal
    margin = support_radius - wp.length(planar)
    interior = wp.clamp(margin / 0.01, 0.0, 1.0)
    spatial = near * interior
    # Inclination rejects stable tipping without adding another yaw target.
    reference_up_o = wp.quat_rotate_inv(wp.normalize(reference_object_quat[frame]), support_normal)
    actual_up_w = wp.quat_rotate(wp.normalize(wp.transform_get_rotation(xf)), reference_up_o)
    tilt = wp.acos(wp.clamp(wp.dot(actual_up_w, support_normal), -1.0, 1.0))
    upright = wp.clamp((0.7853981633974483 - tilt) / 0.5235987755982988, 0.0, 1.0)
    overload = wp.max(load - 1.0, 0.0)
    load_quality = wp.min(load, 1.0) / (1.0 + overload * overload)
    motion = 1.0 / (1.0 + linear * linear / (0.20 * 0.20) + angular * angular / (1.0 * 1.0))
    hand_load = float(0.0)
    for i in range(slots_per_world):
        hand_load += wp.length(hand_force[w * slots_per_world + i])
    release_quality = 1.0 / (1.0 + (hand_load / (0.25 * object_weight)) ** 2.0)
    eligible = window[frame] * spatial * upright
    load_credit = eligible * 0.20 * load_quality
    settle_credit = eligible * 0.10 * load_quality * motion
    release_credit = eligible * 0.20 * load_quality * motion * release_quality
    bonus = load_credit + settle_credit + release_credit
    load_out[w] = load
    linear_out[w] = linear
    angular_out[w] = angular
    # Compatibility diagnostic only: full completion quality, not a reward gate.
    ready_out[w] = spatial * upright * load_quality * motion * release_quality
    window_out[w] = window[frame]
    bonus_out[w] = bonus
    load_credit_out[w] = load_credit
    settle_credit_out[w] = settle_credit
    release_credit_out[w] = release_credit
    tilt_out[w] = tilt
    release_quality_out[w] = release_quality
    motion_quality_out[w] = motion
    spatial_out[w] = spatial
    upright_out[w] = upright
    # Zero-window performs no writes to terms/score, preserving the old path.
    if window[frame] > 0.0:
        base = w * 13
        terms[base + 7] += bonus
        total = float(0.0)
        for i in range(13):
            total += weights[i] * terms[base + i]
        score[w] = total * frame_dt


def table_from_native(reference, config, *, object_com_o=None):
    """Hash and coordinate-check input assets before constructing a phase table."""
    import trimesh
    from flash_chord.assets.registry import support_usda_for_reference
    from training.core.placement_geometry import parse_native_support, build_placement_table
    source = Path(config.placement_reference_path)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != config.placement_reference_sha256:
        raise ValueError("placement reference SHA256 mismatch")
    support_path = Path(config.placement_support_path)
    support_digest = hashlib.sha256(support_path.read_bytes()).hexdigest()
    if support_digest != config.placement_support_sha256:
        raise ValueError("placement support SHA256 mismatch")
    native_support = support_usda_for_reference(reference)
    if native_support is None or hashlib.sha256(Path(native_support).read_bytes()).hexdigest() != support_digest:
        raise ValueError("placement support is not the executed native reference support")
    p = np.asarray(reference.object_body_pos_w())[:, 0]
    q = np.asarray(reference.object_body_quat_w())[:, 0]
    with np.load(source, allow_pickle=False) as data:
        if data['object_position'].shape != p.shape or not np.allclose(data['object_position'], p, atol=1e-5, rtol=0):
            raise ValueError("placement portable/native object positions differ")
        pq = data['object_wxyz']; pq = pq / np.linalg.norm(pq, axis=-1, keepdims=True)
        nq = q / np.linalg.norm(q, axis=-1, keepdims=True)
        if pq.shape != nq.shape or not np.allclose(np.abs(np.sum(pq * nq, axis=-1)), 1, atol=1e-6, rtol=0):
            raise ValueError("placement portable/native object orientations differ")
        observed = data['object_observed_mask'].copy()
    mesh_path = Path(reference.object_mesh_paths()[0])
    mesh = trimesh.load(mesh_path, force='mesh', process=False)
    supports = parse_native_support(support_path, config.placement_support_sha256)
    if len(supports) != 1:
        raise ValueError("supported placement currently requires exactly one designated support cylinder")
    table = build_placement_table(object_positions_w=p, object_quaternions_xyzw=q[:, [1, 2, 3, 0]],
        contact_active=np.stack([reference.contact_active(side) for side in config.sides], axis=1),
        observed_mask=observed, object_vertices_o=mesh.vertices,
        object_com_o=np.zeros(3) if object_com_o is None else object_com_o,
        supports=supports, frame_dt=1.0 / reference.fps,
        source_hashes={'reference_npz': digest, 'support_usda': support_digest,
                       'mesh': hashlib.sha256(mesh_path.read_bytes()).hexdigest()})
    if table.audit['active_windows'] == 0:
        raise ValueError("placement has no valid release window: " + json.dumps(table.audit))
    if table.support['kind'] != 'cylinder':
        raise ValueError('runtime placement shaping currently supports a designated cylinder only')
    return table, table.support


class PlacementRuntime:
    """Instance-local contact hook; freshly zeroed measurements are not state history."""
    def __init__(self, native, model, config, reference):
        self.native, self.config = native, config
        self.worlds, self.bodies = native.world_count, native.bodies_per_world
        labels = list(model.body_label)[:self.bodies]
        candidates = [i for i, label in enumerate(labels) if str(label).split('/')[-1] == 'support']
        if len(candidates) != 1:
            raise ValueError('placement needs one executed support body identified by native label')
        self.support_id = candidates[0]
        if self.support_id == native.object_body_id:
            raise ValueError('placement object and support body overlap')
        self.model = model
        com = np.asarray(model.body_com.numpy())[native.object_body_id]
        masses = np.asarray(model.body_mass.numpy()).reshape(self.worlds, self.bodies)[:, native.object_body_id]
        if not np.allclose(masses, masses[0]) or not np.isfinite(masses[0]) or masses[0] <= 0:
            raise ValueError('placement requires equal positive object mass across worlds')
        gravity = np.asarray(model.gravity.numpy()) if model.gravity is not None else np.empty((0, 3))
        if gravity.size == 0 or not np.all(np.isfinite(gravity)) or not np.allclose(gravity, [0, 0, -9.81], atol=1e-4, rtol=0):
            raise ValueError('placement requires the native uniform downward gravity')
        self.object_com, self.object_weight = wp.vec3(*com), float(masses[0] * np.linalg.norm(gravity.reshape(-1, 3)[0]))
        if not np.allclose(native.command.object_scales.numpy(), 1, atol=1e-6, rtol=0):
            raise ValueError('placement requires unscaled object geometry')
        table, support = table_from_native(reference, config, object_com_o=com)
        self.audit = dict(table.audit, support_body_id=self.support_id, object_body_id=native.object_body_id,
                          object_mass_kg=float(masses[0]), runtime_contact_force=True,
                          method_identity='supported_load_transfer_v2',
                          raw_term_semantics='term7 += load + settle + finger-unloading credit; all other12rawterms unchanged')
        self.audit['diagnostic_semantics'] = {
            'placement_ready': 'spatial * upright * load_quality * motion_quality * release_quality; completion diagnostic only, not a reward gate',
            'support_load_fraction': 'nonnegative measured normal force / object weight, without upper clipping',
            'placement_tilt_rad': 'inclination of reference-normal body axis relative to the support normal; excludes yaw',
            'placement_release_quality': '1/(1+(sum of hand-slot force norms / (0.25mg))^2); measures finger unloading, not geometric clearance',
            'credit_components': 'each includes the fixed window and unchanged spatial/COM compatibility; sum equals placement_bonus',
        }
        self.audit['shaping_constants'] = {
            'maximum_raw_object_bonus': 0.5, 'component_weights': {'load': 0.20, 'settle': 0.10, 'release': 0.20},
            'linear_speed_scale_m_s': 0.20, 'angular_speed_scale_rad_s': 1.0,
            'tilt_full_credit_deg': 15.0, 'tilt_zero_credit_deg': 45.0,
            'support_load_full_weight_mg': 1.0, 'overload_scale_mg': 1.0, 'hand_load_scale_mg': 0.25,
            'position_deadzone_m': 0.03, 'position_std_m': 0.05,
            'support_edge_margin_m': 0.01, 'palm_cost_relief': False,
            'gravity_w_m_s2': gravity.reshape(-1, 3)[0].tolist(),
        }
        # Verify the actual model shape, not only the metadata describing it.
        shape_bodies = np.asarray(model.shape_body.numpy())
        support_shapes = np.flatnonzero(shape_bodies == self.support_id)
        if len(support_shapes) != 1:
            raise ValueError('placement requires exactly one executed support shape per world')
        shape = int(support_shapes[0])
        transforms = np.asarray(model.shape_transform.numpy())
        scales = np.asarray(model.shape_scale.numpy())
        if (not np.allclose(transforms[shape, :3], support['center_w'], atol=1e-5, rtol=0)
                or not np.allclose(np.abs(transforms[shape, 3:]), [0, 0, 0, 1], atol=1e-6, rtol=0)
                or not np.allclose(scales[shape], [support['radius'], support['height'] / 2, 0.0], atol=1e-5, rtol=0)):
            raise ValueError('placement support transform/dimensions differ from executed model')
        self.audit['runtime_support_transform_match'] = True
        self.audit['support_shape_id'] = shape
        self.support_center = wp.vec3(*support['top_center_w'])
        self.support_normal = wp.vec3(*support['normal_w'])
        self.support_radius = float(support['radius'])
        device = native.terms.device
        self.centers = wp.array(table.centers_w, dtype=wp.vec3, device=device)
        self.window = wp.array(table.phase_mask, dtype=wp.float32, device=device)
        self.force = wp.zeros(self.worlds, dtype=wp.vec3, device=device)
        self.count = wp.zeros(self.worlds, dtype=wp.int32, device=device)
        self.diagnostics = {'support_force_w_n': self.force}
        for name in PLACEMENT_DIAGNOSTIC_NAMES:
            self.diagnostics[name] = wp.zeros(self.worlds, dtype=wp.float32, device=device)
        tracker = native.contact
        if getattr(tracker, '_placement_runtime', None) is not None:
            raise ValueError('placement contact tracker is already bound')
        original = tracker.update
        def update(model, state, contacts):
            original(model, state, contacts)
            self.measure(model, contacts)
        tracker.update = update
        tracker._placement_runtime = self

    def measure(self, model, contacts):
        if getattr(contacts, 'force', None) is None:
            raise ValueError('placement requires actual solver contact force')
        wp.launch(clear_support, dim=self.worlds, inputs=[self.force, self.count], device=self.force.device)
        wp.launch(aggregate_support_contacts, dim=contacts.rigid_contact_max,
            inputs=[contacts.rigid_contact_count, contacts.rigid_contact_shape0, contacts.rigid_contact_shape1,
                    contacts.force, model.shape_body, model.shape_count, self.bodies, self.worlds,
                    self.native.object_body_id, self.support_id], outputs=[self.force, self.count], device=self.force.device)

    def evaluate(self, owner, state, timestep):
        n, d = self.native, self.diagnostics
        wp.launch(apply_supported_placement, dim=self.worlds, inputs=[
            state.body_q, state.body_qd, timestep, self.force, n.contact.contact_force_w,
            n.contact.layout.slots_per_world, self.centers, self.window, n.command.reference.body_quat_w,
            owner.weights, n.num_frames, self.bodies, n.object_body_id,
            self.object_com, self.object_weight, self.support_center, self.support_normal, self.support_radius,
            n.frame_dt, owner.terms, owner.score, *[d[name] for name in PLACEMENT_DIAGNOSTIC_NAMES]], device=self.force.device)


def preflight_placement_config(resolved_config):
    from copy import deepcopy
    from training.core.configuration import normalize_checkpoint_config
    values = json.loads(Path(resolved_config).read_text(encoding='utf-8-sig')) if isinstance(resolved_config, (str, Path)) else deepcopy(resolved_config)
    values = normalize_checkpoint_config(values)
    objective_values = values['env']['objective']
    if not objective_values.get('placement_enabled', False):
        return {'enabled': False, 'active_windows': 0, 'status': 'disabled'}
    scene = values['scene']
    if not scene.get('support', False):
        raise ValueError('placement requires the executed scene support')
    if scene.get('object_scale_min', 1.0) != 1.0 or scene.get('object_scale_max', 1.0) != 1.0:
        raise ValueError('placement requires unscaled object geometry')
    from hydra.utils import instantiate
    from flash_chord.data import load_reference, frame_window
    task = values['task']
    with wp.ScopedDevice('cpu'):
        config = instantiate(objective_values, _convert_='all')
        reference = load_reference(task['parquet'], control_fps=float(values['env']['sim']['fps']),
            motion_speed=float(task.get('motion_speed', 1)), source_frame_playback=bool(task.get('source_frame_playback', False)))
        reference = frame_window(reference, int(task.get('motion_start_frame', 0)), int(task.get('motion_end_frame', -1)))
        # The pinned single-object assets have explicit inertial COM; never use mesh centroid.
        import xml.etree.ElementTree as ET
        paths = reference.object_urdf_paths()
        tree = ET.parse(paths[0]); inertials = tree.findall('.//link/inertial')
        if len(inertials) != 1:
            raise ValueError('placement preflight requires one rigid-object inertial')
        origin = inertials[0].find('origin')
        com = np.zeros(3) if origin is None else np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
        table, _ = table_from_native(reference, config, object_com_o=com)
    return dict(table.audit, enabled=True, preflight_device='cpu', native_binding=True,
                reference_frames=int(reference.num_frames), runtime_contact_binding='validated during environment construction')


def main():
    import argparse
    parser = argparse.ArgumentParser(description='CPU-only input/support release preflight')
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    audit = preflight_placement_config(args.config)
    Path(args.output).write_text(json.dumps(audit, indent=2, allow_nan=False) + '\n')
    print(json.dumps(audit))


if __name__ == '__main__':
    main()

"""Synthetic CPU kernels: no simulator, model assets, JAX or GPU allocation."""
import math
import unittest
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import warp as wp

from training.overnight_refinement import (
    apply_recovery_termination, compose_refinement_objective,
    reset_refinement_counters, smooth_contact_blend,
    OvernightObjective, OvernightObjectiveConfig, OvernightTermination,
    OvernightTerminationConfig, _RUNTIME_IMPORT_ERROR, install,
)


def arr(values, dtype=wp.float32):
    return wp.array(values, dtype=dtype, device="cpu")


class ContactBlendTests(unittest.TestCase):
    def test_constant_and_short_clip(self):
        for value in (0, 1):
            np.testing.assert_allclose(smooth_contact_blend(np.full((2, 2), value), 10), value)

    def test_transition_is_bounded_smooth_and_hand_specific(self):
        labels = np.zeros((41, 2))
        labels[20:, 0] = 1
        blend = smooth_contact_blend(labels, 10)
        self.assertEqual(blend[0, 0], 0)
        self.assertEqual(blend[-1, 0], 1)
        self.assertTrue((np.diff(blend[:, 0]) >= 0).all())
        self.assertLess(np.diff(blend[:, 0]).max(), .1)
        np.testing.assert_array_equal(blend[:, 1], 0)
        np.testing.assert_array_equal(smooth_contact_blend(labels, 0), labels)


class ObjectiveKernelTests(unittest.TestCase):
    def run_objective(self, blend=1., translation=(0, 0, 0), rotation=0., use_relative=True,
                      hand_offset=(0, 0, 0), target_sign=1., tolerance=.03):
        # One rigid object and two palms; reference palms at +/- .1m on x.
        half = rotation / 2
        quat = np.array([0, 0, math.sin(half), math.cos(half)])
        rotation_matrix = np.array([[math.cos(rotation), -math.sin(rotation), 0],
                                    [math.sin(rotation), math.cos(rotation), 0], [0, 0, 1]])
        t = np.asarray(translation)
        reference_palms = np.array([[.1, 0, 0], [-.1, 0, 0]])
        live_palms = reference_palms @ rotation_matrix.T + t + hand_offset
        body = np.concatenate([np.r_[t, quat][None], np.column_stack([live_palms, np.tile(quat, (2, 1))])])
        native = np.arange(12, dtype=np.float32) / 12
        weights = np.arange(13, dtype=np.float32)
        terms, score = wp.zeros(13, device="cpu"), wp.zeros(1, device="cpu")
        wp.launch(compose_refinement_objective, dim=1, inputs=[
            arr(body, wp.transform), arr([0], wp.int32), arr(reference_palms, wp.vec3),
            arr([[0, 0, 0, 1]] * 2, wp.quat), arr([[0, 0, 0]], wp.vec3),
            arr([[0, 0, 0, 1]], wp.quat), arr([[0, 0, 0, target_sign]], wp.quat),
            arr([1, 2], wp.int32), arr([[0, 0, 0]] * 2, wp.vec3),
            arr([[0, 0, 0, 1]] * 2, wp.quat), arr([blend, blend]), arr(native), arr(weights),
            1, 3, 0, int(use_relative), tolerance, .2, .15, .8, .25, .6, .02,
        ], outputs=[terms, score], device="cpu")
        return terms.numpy(), score.numpy()[0], weights, native

    def test_contact_preserves_grasp_under_common_translation_rotation(self):
        terms, score, weights, native = self.run_objective(translation=(.3, -.2, .1), rotation=math.pi / 2)
        self.assertAlmostEqual(float(terms[9]), 1, places=6)
        self.assertAlmostEqual(float(terms[10]), .25, places=6)
        self.assertAlmostEqual(float(terms[12]), math.exp(-(math.pi / 2 / .6) ** 2), places=6)
        # Unrelated native terms, including body tracking, survive unchanged.
        np.testing.assert_array_equal(terms[:9], native[:9])
        self.assertEqual(terms[11], native[11])
        self.assertAlmostEqual(float(score), float(.02 * np.dot(weights, terms)), places=6)

    def test_no_contact_keeps_world_tracking(self):
        terms, _, _, _ = self.run_objective(blend=0, translation=(.3, 0, 0))
        self.assertAlmostEqual(float(terms[9]), math.exp(-2 * (.27 / .2) ** 2), places=6)
        disabled, _, _, _ = self.run_objective(blend=1, translation=(.3, 0, 0), use_relative=False)
        np.testing.assert_allclose(terms, disabled)

    def test_blended_cost_and_tolerance(self):
        terms, _, _, _ = self.run_objective(blend=.5, translation=(.3, 0, 0))
        self.assertAlmostEqual(float(terms[9]), math.exp(-(.27 / .2) ** 2), places=6)
        within, _, _, _ = self.run_objective(hand_offset=(.02, 0, 0))
        self.assertEqual(within[9], 1)
        beyond, _, _, _ = self.run_objective(hand_offset=(.08, 0, 0))
        self.assertAlmostEqual(float(beyond[9]), math.exp(-2 * (.05 / .15) ** 2), places=6)

    def test_quaternion_double_cover(self):
        a, _, _, _ = self.run_objective(rotation=.6)
        b, _, _, _ = self.run_objective(rotation=.6, target_sign=-2.)
        self.assertAlmostEqual(float(a[12]), float(b[12]), places=6)
        self.assertAlmostEqual(float(a[12]), math.exp(-1), places=5)


class TerminationKernelTests(unittest.TestCase):
    def setUp(self):
        self.n = 2
        self.age = arr([51, 51], wp.int32)
        self.palm_pos = arr([.3, 0])
        self.palm_ori = arr([0, 0])
        self.obj_pos = arr([0, .25])
        self.obj_ori = arr([0, 0])
        self.pelvis = arr([0, 0], wp.int32)
        self.pc = arr([0, 0], wp.int32)
        self.oc = arr([0, 0], wp.int32)
        self.last = arr([-1, -1], wp.int32)
        self.pd = arr([0, 0], wp.int32)
        self.od = arr([0, 0], wp.int32)
        self.done = arr([0, 0], wp.int32)
        self.packed = arr(np.zeros(18))

    def step(self, age):
        self.age.assign([age, age])
        wp.launch(apply_recovery_termination, dim=2, inputs=[
            self.age, self.palm_pos, self.palm_ori, self.obj_pos, self.obj_ori, self.pelvis,
            50, .25, 2.2, .2, 1.5, 10, 5, .6, 2.8, .4, 2.8,
        ], outputs=[self.pc, self.oc, self.last, self.pd, self.od, self.done, self.packed], device="cpu")

    def reset(self, mask):
        wp.launch(reset_refinement_counters, dim=2, inputs=[arr(mask, wp.int32)],
                  outputs=[self.pc, self.oc, self.last], device="cpu")

    def test_exact_dwell_and_masked_same_age_reset(self):
        for age in range(51, 55):
            self.step(age)
            np.testing.assert_array_equal(self.done.numpy(), [0, 0])
        self.step(55)
        np.testing.assert_array_equal(self.done.numpy(), [0, 1])
        self.reset([0, 1])  # Selective reset must not clear the other world's history.
        np.testing.assert_array_equal(self.pc.numpy(), [5, 0])
        np.testing.assert_array_equal(self.oc.numpy(), [0, 0])
        self.step(55)  # Same age as before reset, not detectable by age comparison.
        np.testing.assert_array_equal(self.done.numpy(), [0, 0])
        for age in range(56, 60):
            self.step(age)
        np.testing.assert_array_equal(self.done.numpy(), [1, 1])
        packed = self.packed.numpy().reshape(2, 9)
        np.testing.assert_array_equal(packed[:, 1], self.pd.numpy())
        np.testing.assert_array_equal(packed[:, 2], self.od.numpy())

    def test_interrupted_breach_and_age_rollback_clear_history(self):
        for age in range(51, 55):
            self.step(age)
        self.palm_pos.assign([0, 0])
        self.obj_pos.assign([0, 0])
        self.step(55)
        np.testing.assert_array_equal(self.pc.numpy(), [0, 0])
        np.testing.assert_array_equal(self.oc.numpy(), [0, 0])
        self.palm_pos.assign([.3, 0])
        self.obj_pos.assign([0, .25])
        self.step(56)
        self.step(51)
        np.testing.assert_array_equal(self.pc.numpy(), [1, 0])
        np.testing.assert_array_equal(self.oc.numpy(), [0, 1])

    def test_gross_cutoff_freeze_and_preserved_pelvis(self):
        self.palm_pos.assign([.61, 0])
        self.obj_pos.assign([0, .41])
        self.step(51)
        np.testing.assert_array_equal(self.done.numpy(), [1, 1])
        self.step(50)
        np.testing.assert_array_equal(self.done.numpy(), [0, 0])
        self.pelvis.assign([1, 0])
        self.step(50)
        np.testing.assert_array_equal(self.done.numpy(), [1, 0])

    def test_orientation_breaches_and_recovery(self):
        self.palm_pos.assign([0, 0])
        self.obj_pos.assign([0, 0])
        self.palm_ori.assign([2.3, 0])
        self.obj_ori.assign([0, 1.6])
        for age in range(51, 55):
            self.step(age)
        self.step(55)
        np.testing.assert_array_equal(self.done.numpy(), [0, 1])
        self.palm_ori.assign([2.9, 0])
        self.obj_ori.assign([0, 0])
        self.step(56)
        np.testing.assert_array_equal(self.done.numpy(), [1, 0])


class WrapperTests(unittest.TestCase):
    def test_thirteenth_raw_term_and_weight_updates(self):
        names = tuple(f"native_{i}" for i in range(12))
        seen = []
        native = SimpleNamespace(term_names=names, terms=arr(np.zeros(24)),
                                 world_count=2, num_frames=3,
                                 contact_active=arr(np.ones(6)),
                                 set_weights=lambda values: seen.append(values))
        config = SimpleNamespace(weights=tuple(range(12)), object_orientation_weight=0.,
                                 contact_blend_frames=1)
        wrapped = OvernightObjective(native, config)
        self.assertEqual(wrapped.terms.shape, (26,))
        self.assertEqual(wrapped.weights.shape, (13,))
        self.assertEqual(wrapped.term_names[-1], "object_orientation")
        self.assertEqual(wrapped.weights.numpy()[-1], 0)
        updated = dict(zip(wrapped.term_names, range(13)))
        wrapped.set_weights(updated)
        np.testing.assert_array_equal(wrapped.weights.numpy(), np.arange(13))
        self.assertEqual(seen, [{name: updated[name] for name in names}])
        with self.assertRaises(ValueError):
            wrapped.set_weights({name: 1. for name in names})

    def test_wrapper_reset_is_selective(self):
        wrapped = OvernightTermination(SimpleNamespace(world_count=2, terminated=arr([0, 0], wp.int32)), None)
        wrapped.palm_count.assign([8, 9])
        wrapped.object_count.assign([3, 4])
        wrapped.last_episode_step.assign([100, 100])
        wrapped.reset_counters(arr([1, 0], wp.int32))
        np.testing.assert_array_equal(wrapped.palm_count.numpy(), [0, 9])
        np.testing.assert_array_equal(wrapped.object_count.numpy(), [0, 4])
        np.testing.assert_array_equal(wrapped.last_episode_step.numpy(), [-1, 100])


@unittest.skipIf(_RUNTIME_IMPORT_ERROR is not None, "pinned simulator runtime is unavailable")
class NativeIntegrationTests(unittest.TestCase):
    def test_serializable_config_preserves_native_fields(self):
        from flash_chord.objectives.recon_body import ReconBodyObjectiveConfig
        from flash_chord.lifecycle.termination import ReconBodyTerminationConfig
        source = asdict(ReconBodyObjectiveConfig())
        objective = OvernightObjectiveConfig(**source, object_orientation_weight=0.)
        self.assertEqual(len(objective.weights), 12)  # Native builder gets exactly12.
        self.assertEqual(objective.body_joint_position_weight, 5)
        self.assertEqual(OvernightObjectiveConfig(**asdict(objective)), objective)
        termination = OvernightTerminationConfig(**asdict(ReconBodyTerminationConfig()))
        self.assertEqual(OvernightTerminationConfig(**asdict(termination)), termination)

    def test_reset_hook_calls_original_and_clears_selected_same_age(self):
        from flash_chord.envs.base import BaseEnv
        original = BaseEnv.reset
        calls = []

        def fake_original(env, **kwargs):
            calls.append(kwargs)

        try:
            BaseEnv.reset = fake_original
            install()
            installed = BaseEnv.reset
            install()
            self.assertIs(BaseEnv.reset, installed)
            termination = OvernightTermination(SimpleNamespace(world_count=2, terminated=arr([0, 0], wp.int32)), None)
            termination.palm_count.assign([7, 8])
            fake_env = SimpleNamespace(termination=termination,
                                       reset_table=SimpleNamespace(all_reset_mask=arr([1, 1], wp.int32)))
            selected = arr([0, 1], wp.int32)
            BaseEnv.reset(fake_env, reset_mask=selected, frame_id=4)
            self.assertIs(calls[0]["reset_mask"], selected)
            self.assertEqual(calls[0]["frame_id"], 4)
            np.testing.assert_array_equal(termination.palm_count.numpy(), [7, 0])
            BaseEnv.reset(fake_env)
            np.testing.assert_array_equal(termination.palm_count.numpy(), [0, 0])
        finally:
            BaseEnv.reset = original


if __name__ == "__main__":
    unittest.main()

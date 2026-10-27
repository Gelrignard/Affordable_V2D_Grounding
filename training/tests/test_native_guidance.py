"""Physical sign/frame tests, run in the existing native cloud environment."""

import unittest

import numpy as np

try:
    import warp as wp

    from training.native_guidance import accumulate_object_wrench, compose_dynamics_term
except ModuleNotFoundError:
    wp = None


@unittest.skipIf(wp is None, "native Newton runtime required")
class WrenchKernelTests(unittest.TestCase):
    def test_force_reaction_com_torque_and_support_separation(self):
        wp.init()
        device = "cpu"

        def arr(values, dtype):
            return wp.array(values, dtype=dtype, device=device)

        hand = wp.zeros(2, dtype=wp.spatial_vector, device=device)
        support = wp.zeros_like(hand)
        magnitude = wp.zeros(2, device=device)
        q = arr([[0, 0, 0, 0, 0, 0, 1]] * 4, wp.transform)
        wp.launch(
            accumulate_object_wrench,
            dim=2,
            device=device,
            inputs=[
                arr([2], wp.int32),
                arr([0, 1], wp.int32),
                arr([1, 4], wp.int32),
                arr([[0, 0, 0], [0.1, 0, 0]], wp.vec3),
                arr([[0.2, 0, 0], [0, 0, 0]], wp.vec3),
                arr([[0, 0, -5, 0.2, 0, 0], [0, 0, 4.81, 0, 0, 0]], wp.spatial_vector),
                arr([0, 1, 2, 3, -1], wp.int32),
                arr([0, -1], wp.int32),
                2,
                q,
                arr([[0, 0, 0], [0.1, 0, 0], [0, 0, 0], [0.1, 0, 0]], wp.vec3),
                2,
                1,
            ],
            outputs=[hand, support, magnitude],
        )
        np.testing.assert_allclose(hand.numpy()[0], [0, 0, 5, -0.2, -0.5, 0], atol=1e-6)
        np.testing.assert_allclose(support.numpy()[0], [0, 0, 4.81, 0, 0, 0], atol=1e-6)
        np.testing.assert_allclose(hand.numpy()[1], 0)
        np.testing.assert_allclose(support.numpy()[1], 0)
        np.testing.assert_allclose(magnitude.numpy(), [5, 0])

    def test_floor_bears_gravity_but_absent_physical_forces_do_not(self):
        device = "cpu"

        def arr(values, dtype=wp.float32):
            return wp.array(values, dtype=dtype, device=device)

        terms = wp.zeros(26, device=device)
        score = wp.zeros(2, device=device)
        diagnostics = wp.zeros((2, 6), device=device)
        wp.launch(
            compose_dynamics_term,
            dim=2,
            device=device,
            inputs=[
                arr(np.ones(24)),
                arr([0.5, 0.5]),
                arr([0, 0], wp.int32),
                arr([[0, 0, 0, 0, 0, 0, 1]] * 4, wp.transform),
                1,
                2,
                arr([[0, 0, 9.81]], wp.vec3),
                arr([[0, 0, 0]], wp.vec3),
                arr([1]),
                arr([0]),
                arr(np.zeros((2, 6)), wp.spatial_vector),
                arr([[0, 0, 9.81, 0, 0, 0], [0, 0, 0, 0, 0, 0]], wp.spatial_vector),
                arr([10, 0]),  # opposing grasp forces can cancel in the net wrench
                9.81,
                0.981,
                arr([1]),
                0.02,
            ],
            outputs=[terms, score, diagnostics],
        )
        np.testing.assert_allclose(terms.numpy().reshape(2, 13)[:, :12], 1)
        self.assertAlmostEqual(float(diagnostics.numpy()[0, 0]), 0, places=6)
        self.assertAlmostEqual(float(diagnostics.numpy()[1, 0]), 1, places=6)
        self.assertAlmostEqual(float(score.numpy()[0]), 0.52, places=6)
        self.assertLess(float(score.numpy()[1]), 0.52)
        self.assertEqual(float(diagnostics.numpy()[0, 1]), 1)


if __name__ == "__main__":
    unittest.main()

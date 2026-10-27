import unittest
from types import SimpleNamespace

import torch

from training.config import TrainConfig
from training.runtime import SonicEnv
from training.wrench import basis_directions


class RuntimeAdapterTests(unittest.TestCase):
    def environment(self):
        adapter = SonicEnv.__new__(SonicEnv)
        adapter.cfg = TrainConfig(device="cpu", worlds=2)
        adapter.worlds, adapter.frames, adapter.control_fps = 2, 4, 50
        adapter.device = torch.device("cpu")
        adapter.action_mask, adapter.radius = torch.ones(1), 1.0
        adapter.tensor = lambda value: value
        adapter.wp = SimpleNamespace(from_torch=lambda value: value)
        adapter.synchronize = lambda: None
        adapter.refresh_observation = lambda: None
        adapter.object_ids = torch.tensor([[0], [1]])
        adapter.basis = basis_directions(8)
        adapter.reference_confidence = adapter.object_confidence = torch.ones(4)
        adapter.contact_confidence = torch.zeros(4)
        adapter.target_support = torch.zeros(4, 8)
        adapter.target_object = torch.tensor([[0.0, 0, 0, 0, 0, 0, 1]]).repeat(4, 1)
        adapter.goals = torch.arange(4).float()[:, None]
        contact = SimpleNamespace(
            contact_active=torch.zeros(2, dtype=torch.int32),
            contact_pos_o=torch.zeros(2, 3),
            contact_force_direction_o=torch.zeros(2, 3),
        )
        physical = torch.zeros(2, 1)
        env = SimpleNamespace(
            timestep=torch.zeros(2, dtype=torch.int32),
            done=torch.zeros(2, dtype=torch.int32),
            truncation=torch.zeros(2, dtype=torch.int32),
            contact=contact,
            objective=SimpleNamespace(terms=torch.zeros(24), weights=torch.ones(12)),
            state_0=SimpleNamespace(body_q=adapter.target_object[:2].clone()),
        )
        adapter.env = env

        def step(raw):
            adapter.last_raw = raw.clone()
            physical.add_(1)
            env.done[0] = 1
            env.timestep[1] += 1

        def reset(mask):
            physical[mask] = 0
            env.timestep[mask] = 0
            return physical.clone(), adapter.goal().clone()

        env.step, adapter.reset = step, reset
        adapter.state = lambda: physical.clone()
        return adapter

    def test_terminal_state_copied_before_reset_and_executed_action_logged(self):
        env = self.environment()
        ns, ng, _, done, current, current_goal, info = env.step(
            torch.tensor([[2.0], [-2.0]])
        )
        torch.testing.assert_close(ns, torch.ones(2, 1))
        torch.testing.assert_close(ng, torch.ones(2, 1))
        self.assertEqual(done[:, 0].tolist(), [True, False])
        self.assertEqual(current[:, 0].tolist(), [0, 1])
        self.assertEqual(current_goal[:, 0].tolist(), [0, 1])
        torch.testing.assert_close(
            info["executed_action"], torch.tensor([[1.0], [-1.0]])
        )
        torch.testing.assert_close(env.last_raw, torch.tensor([3.0, -3.0]))

    def test_evaluation_failure_does_not_reset_or_stall_reference(self):
        env = self.environment()
        for _ in range(2):
            ns, ng, _, _done, current, current_goal, _ = env.step(
                torch.zeros(2, 1), reset=False
            )
        self.assertEqual(ns[:, 0].tolist(), [2, 2])
        self.assertEqual(ng[:, 0].tolist(), [2, 2])
        torch.testing.assert_close(current, ns)
        torch.testing.assert_close(current_goal, ng)


if __name__ == "__main__":
    unittest.main()

"""Small delayed-control system for CPU integration tests, never a V2D rollout."""

import torch


class ToyEnv:
    def __init__(self, worlds=4, device="cpu", seed=0):
        self.worlds, self.device = worlds, torch.device(device)
        self.rng = torch.Generator(device=self.device).manual_seed(seed)
        self.spec = {
            "state_dim": 4,
            "obs_dim": 4,
            "goal_dim": 2,
            "action_dim": 1,
            "action_mask": [1.0],
            "important_state_ranges": [[0, 2]],
            "interface": {"kind": "cpu_test_only", "version": 1},
        }
        self.identity = {
            "tier": 0,
            "episode_index": 0,
            "variant": "synthetic",
            "job_sha256": "synthetic_test_only",
        }
        self.state_value = torch.zeros(worlds, 4, device=self.device)
        self.target = torch.zeros(worlds, 1, device=self.device)
        self.reset(first=True)

    def goal(self):
        return torch.cat((self.target, torch.ones_like(self.target)), dim=-1)

    def reset(self, mask=None, first=False):
        if mask is None:
            mask = torch.ones(self.worlds, dtype=torch.bool, device=self.device)
        self.state_value[mask] = 0
        self.target[mask] = (
            0.2
            + torch.rand(int(mask.sum()), 1, generator=self.rng, device=self.device)
            * 0.6
        )
        return self.state_value.clone(), self.goal().clone()

    def step(self, action, reset=True):
        action = action.detach().clamp(-1, 1)
        s = self.state_value
        velocity = 0.9 * s[:, 1:2] + 0.1 * s[:, 2:3]
        position = s[:, 0:1] + 0.1 * velocity
        self.state_value = torch.cat(
            (position, velocity, action, s[:, 3:4] + 1 / 32), dim=-1
        )
        reward = -(position - self.target).square() - 0.01 * action.square()
        terminated = position[:, 0].abs() > 1.5
        done = (self.state_value[:, 3] >= 1) | terminated
        ns, ng = self.state_value.clone(), self.goal().clone()
        current = self.reset(done) if reset and done.any() else (ns, ng)
        return (
            ns,
            ng,
            reward,
            done[:, None],
            *current,
            {"executed_action": action, "terminated": terminated},
        )

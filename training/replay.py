"""Vector-step replay preserving terminal next states and episode boundaries."""

import torch


class Replay:
    def __init__(self, capacity, worlds, state_dim, action_dim, goal_dim, device="cpu"):
        self.rows = capacity // worlds
        self.worlds, self.cursor, self.count = worlds, 0, 0
        self.device = torch.device(device)
        self.data = {
            key: torch.empty(
                (self.rows, worlds, width), dtype=dtype, device=self.device
            )
            for key, width, dtype in (
                ("state", state_dim, torch.float32),
                ("next_state", state_dim, torch.float32),
                ("action", action_dim, torch.float32),
                ("goal", goal_dim, torch.float32),
                ("next_goal", goal_dim, torch.float32),
                ("reward", 1, torch.float32),
                ("done", 1, torch.bool),
            )
        }

    @property
    def size(self):
        return min(self.count, self.rows) * self.worlds

    def clear(self):
        """Forget trajectories at a curriculum reset without reallocating storage."""
        self.cursor = self.count = 0

    @property
    def bytes(self):
        return sum(t.numel() * t.element_size() for t in self.data.values())

    def add(self, **transition):
        if set(transition) != set(self.data):
            raise ValueError("incomplete transition")
        for key, target in self.data.items():
            source = (
                transition[key].detach().to(self.device).reshape_as(target[self.cursor])
            )
            if key != "done" and not torch.isfinite(source).all():
                raise ValueError(f"nonfinite executed transition: {key}")
            target[self.cursor].copy_(source)
        self.cursor = (self.cursor + 1) % self.rows
        self.count += 1

    def sample(self, batch, device):
        if not self.size:
            raise ValueError("empty replay")
        rows = torch.randint(min(self.count, self.rows), (batch,), device=self.device)
        worlds = torch.randint(self.worlds, (batch,), device=self.device)
        return {k: v[rows, worlds].to(device) for k, v in self.data.items()}

    def sequences(self, batch, horizon, device, validation=False):
        available = min(self.count, self.rows)
        # Disjoint world IDs provide held-out trajectories, rather than overlapping windows.
        split = max(1, self.worlds // 5)
        if available < horizon:
            raise ValueError("replay has not accumulated a complete model window")
        lower, upper = (0, split) if validation else (split, self.worlds)
        if upper <= lower:
            lower, upper = 0, self.worlds
        starts = torch.randint(available - horizon + 1, (batch,), device=self.device)
        oldest = self.cursor if self.count >= self.rows else 0
        rows = (
            oldest + starts[:, None] + torch.arange(horizon, device=self.device)[None]
        ) % self.rows
        worlds = torch.randint(lower, upper, (batch,), device=self.device)[
            :, None
        ].expand_as(rows)
        data = {k: v[rows, worlds].to(device) for k, v in self.data.items()}
        keep = torch.ones_like(data["done"], dtype=torch.float32)
        keep[:, 1:] = (~data["done"][:, :-1]).float().cumprod(dim=1)
        data["mask"] = keep
        return data

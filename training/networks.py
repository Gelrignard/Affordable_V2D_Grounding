"""Compact state-space models; SONIC is external and never optimized here."""

import math

import torch
from torch import nn
from torch.nn import functional as F


def mlp(input_dim, output_dim, hidden, layers=2):
    modules = []
    for _ in range(layers):
        modules += [nn.Linear(input_dim, hidden), nn.LayerNorm(hidden), nn.SiLU()]
        input_dim = hidden
    return nn.Sequential(*modules, nn.Linear(hidden, output_dim))


class Normalizer(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.register_buffer("mean", torch.zeros(width))
        self.register_buffer("std", torch.ones(width))
        self.register_buffer("fitted", torch.tensor(False))

    @torch.no_grad()
    def fit(self, data):
        self.mean.copy_(data.mean(dim=0))
        self.std.copy_(data.std(dim=0, unbiased=False).clamp_min(0.1))
        self.fitted.fill_(True)

    def forward(self, data):
        return ((data - self.mean) / self.std).clamp(-50, 50)


class Actor(nn.Module):
    def __init__(
        self,
        obs_dim,
        goal_dim,
        action_dim,
        hidden,
        action_mask=None,
        initial_std=0.1353352832366127,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.norm = Normalizer(obs_dim + goal_dim)
        self.net = mlp(obs_dim + goal_dim, action_dim * 2, hidden)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        with torch.no_grad():
            self.net[-1].bias[action_dim:].fill_(math.log(initial_std))
        self.register_buffer(
            "mask",
            torch.ones(action_dim) if action_mask is None else action_mask.float(),
        )

    def forward(self, state, goal, deterministic=False):
        mean, log_std = self.net(
            self.norm(torch.cat((state[..., : self.obs_dim], goal), dim=-1))
        ).chunk(2, dim=-1)
        log_std = log_std.clamp(-5, 0)
        dist = torch.distributions.Normal(mean, log_std.exp())
        raw = mean if deterministic else dist.rsample()
        action = raw.tanh()
        log_prob = dist.log_prob(raw) - 2 * (math.log(2) - raw - F.softplus(-2 * raw))
        log_prob = log_prob - self.mask.clamp_min(1e-8).log()
        log_prob = (log_prob * (self.mask != 0)).sum(dim=-1, keepdim=True)
        return action * self.mask, log_prob


class Critic(nn.Module):
    def __init__(self, state_dim, goal_dim, action_dim, hidden):
        super().__init__()
        self.norm = Normalizer(state_dim + goal_dim)
        self.q1 = mlp(state_dim + goal_dim + action_dim, 1, hidden)
        self.q2 = mlp(state_dim + goal_dim + action_dim, 1, hidden)

    def forward(self, state, goal, action):
        inputs = torch.cat(
            (self.norm(torch.cat((state, goal), dim=-1)), action), dim=-1
        )
        return self.q1(inputs), self.q2(inputs)


class WorldModel(nn.Module):
    def __init__(self, state_dim, goal_dim, action_dim, hidden):
        super().__init__()
        self.state_dim = state_dim
        self.norm = Normalizer(state_dim + goal_dim)
        self.net = mlp(state_dim + goal_dim + action_dim, state_dim + 2, hidden, 3)
        self.register_buffer("state_weights", torch.ones(state_dim))

    def forward(self, state, goal, action):
        output = self.net(
            torch.cat((self.norm(torch.cat((state, goal), dim=-1)), action), dim=-1)
        )
        next_state = (
            state + output[..., : self.state_dim] * self.norm.std[: self.state_dim]
        )
        # Reward is fit on executed transitions, not inferred from corrected action labels.
        reward = output[..., self.state_dim : self.state_dim + 1]
        termination_logits = output[..., -1:]
        return next_state, reward, termination_logits

    def sequence_loss(self, batch):
        state = batch["state"][:, 0]
        prediction_losses, reward_losses, terminal_losses = [], [], []
        for t in range(batch["action"].shape[1]):
            state, reward, terminal = self(
                state, batch["goal"][:, t], batch["action"][:, t]
            )
            error = (state - batch["next_state"][:, t]) / self.norm.std[
                : self.state_dim
            ]
            prediction_losses.append(
                (error.square() * self.state_weights).mean(dim=-1, keepdim=True)
            )
            reward_losses.append(
                F.smooth_l1_loss(reward, batch["reward"][:, t], reduction="none")
            )
            terminal_losses.append(
                F.binary_cross_entropy_with_logits(
                    terminal, batch["done"][:, t].float(), reduction="none"
                )
            )
        mask = batch["mask"]

        def reduce(values):
            return (torch.stack(values, dim=1) * mask).sum() / mask.sum().clamp_min(1)

        state_loss, reward_loss, terminal_loss = map(
            reduce, (prediction_losses, reward_losses, terminal_losses)
        )
        return state_loss + reward_loss + 0.1 * terminal_loss, {
            "model_state_mse": float(state_loss.detach()),
            "model_reward_loss": float(reward_loss.detach()),
            "model_terminal_loss": float(terminal_loss.detach()),
        }

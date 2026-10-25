"""Real-transition SAC with optional short differentiable model rollouts."""

import copy
from contextlib import contextmanager

import torch
from torch.nn import functional as F

from .networks import Actor, Critic, WorldModel


@contextmanager
def frozen(*modules):
    parameters = [p for module in modules for p in module.parameters()]
    previous = [p.requires_grad for p in parameters]
    try:
        for parameter in parameters:
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, flag in zip(parameters, previous):
            parameter.requires_grad_(flag)


class Learner:
    def __init__(self, config, spec, device=None):
        self.cfg, self.spec = config, spec
        self.device = torch.device(device or config.device)
        mask = torch.tensor(spec["action_mask"], device=self.device)
        self.actor = Actor(
            spec["obs_dim"], spec["goal_dim"], spec["action_dim"], config.hidden, mask
        ).to(self.device)
        self.critic = Critic(
            spec["state_dim"], spec["goal_dim"], spec["action_dim"], config.hidden
        ).to(self.device)
        self.target = copy.deepcopy(self.critic).requires_grad_(False)
        self.model = WorldModel(
            spec["state_dim"], spec["goal_dim"], spec["action_dim"], config.model_hidden
        ).to(self.device)
        for start, end in spec.get("important_state_ranges", []):
            self.model.state_weights[start:end] = 5.0
        self.actor_opt = torch.optim.Adam(
            self.actor.parameters(), lr=config.learning_rate
        )
        self.critic_opt = torch.optim.Adam(
            self.critic.parameters(), lr=config.learning_rate
        )
        self.model_opt = torch.optim.Adam(
            self.model.parameters(), lr=config.learning_rate
        )
        self.log_alpha = (
            torch.tensor(config.initial_alpha, device=self.device)
            .log()
            .requires_grad_()
        )
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=config.learning_rate)
        self.target_entropy = -float((mask != 0).sum())
        self.updates = 0
        self.model_updates = 0
        self.model_ready = False
        self.last_validation = {}

    def fit_normalizers(self, replay):
        batch = replay.sample(min(replay.size, 8192), self.device)
        if not self.actor.norm.fitted:
            self.actor.norm.fit(
                torch.cat(
                    (batch["state"][:, : self.spec["obs_dim"]], batch["goal"]), dim=-1
                )
            )
            self.critic.norm.fit(torch.cat((batch["state"], batch["goal"]), dim=-1))
            self.target.load_state_dict(self.critic.state_dict())
        if not self.model.norm.fitted:
            self.model.norm.fit(torch.cat((batch["state"], batch["goal"]), dim=-1))

    @torch.no_grad()
    def act(self, state, goal, deterministic=False):
        return self.actor(state, goal, deterministic)[0]

    def update_real(self, replay):
        c = self.cfg
        batch = replay.sample(c.batch_size, self.device)
        s, g, a = batch["state"], batch["goal"], batch["action"]
        alpha = self.log_alpha.detach().exp()
        with torch.no_grad():
            next_action, logp = self.actor(batch["next_state"], batch["next_goal"])
            tq1, tq2 = self.target(batch["next_state"], batch["next_goal"], next_action)
            # These are finite-horizon tasks: both failures and reference-end truncations end returns.
            target = batch["reward"] + c.gamma * (~batch["done"]).float() * (
                torch.minimum(tq1, tq2) - alpha * logp
            )
        q1, q2 = self.critic(s, g, a)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10)
        self.critic_opt.step()
        with frozen(self.critic):
            action, logp = self.actor(s, g)
            q1, q2 = self.critic(s, g, action)
            actor_loss = (alpha * logp - torch.minimum(q1, q2)).mean()
            self.actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10)
            self.actor_opt.step()
        alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
        self.alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_opt.step()
        self.log_alpha.data.clamp_(-10, 2)
        with torch.no_grad():
            for target_p, source_p in zip(
                self.target.parameters(), self.critic.parameters()
            ):
                target_p.lerp_(source_p, c.tau)
        self.updates += 1
        return {
            "critic_loss": float(critic_loss.detach()),
            "actor_loss": float(actor_loss.detach()),
            "alpha": float(alpha),
            "real_updates": self.updates,
        }

    def update_model(self, replay):
        batch = replay.sequences(
            self.cfg.batch_size, self.cfg.model_window, self.device
        )
        self.model_opt.zero_grad(set_to_none=True)
        loss, metrics = self.model.sequence_loss(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
        self.model_opt.step()
        self.model_updates += 1
        if self.model_updates % self.cfg.model_validation_interval == 0:
            with torch.no_grad():
                held = replay.sequences(
                    self.cfg.batch_size,
                    self.cfg.model_window,
                    self.device,
                    validation=True,
                )
                _, self.last_validation = self.model.sequence_loss(held)
            self.model_ready = (
                self.model_updates >= self.cfg.model_min_updates
                and self.last_validation["model_state_mse"]
                < self.cfg.model_validation_threshold
                and self.last_validation["model_reward_loss"]
                < self.cfg.model_validation_threshold
            )
        return {
            **metrics,
            "model_updates": self.model_updates,
            "imagination_enabled": self.model_ready,
            **{"validation_" + k: v for k, v in self.last_validation.items()},
        }

    def imagine(self, replay):
        if not self.model_ready:
            return {}
        c = self.cfg
        # Goals advance on logged, same-episode timelines. Never predict or optimize the desired goal.
        batch = replay.sequences(
            c.imagination_batch, c.imagination_horizon, self.device
        )
        state = batch["state"][:, 0].detach()
        survival = torch.ones_like(batch["reward"][:, 0])
        score = torch.zeros_like(survival)
        with frozen(self.model, self.target):
            for t in range(c.imagination_horizon):
                action, _ = self.actor(state, batch["goal"][:, t], deterministic=True)
                state, reward, terminal = self.model(state, batch["goal"][:, t], action)
                # Anchor to executed replay actions, including t=0, to limit
                # extrapolation. These are not expert/reference action labels.
                trust = (
                    (action - batch["action"][:, t]).square().mean(dim=-1, keepdim=True)
                )
                score = score + c.gamma**t * survival * batch["mask"][:, t] * (
                    reward - c.model_trust_weight * trust
                )
                survival = (
                    survival * (1 - terminal.sigmoid()) * (~batch["done"][:, t]).float()
                )
            last_goal = batch["next_goal"][:, -1]
            action, _ = self.actor(state, last_goal, deterministic=True)
            q1, q2 = self.target(state, last_goal, action)
            score = score + c.gamma**c.imagination_horizon * survival * torch.minimum(
                q1, q2
            )
            loss = -c.imagination_weight * score.mean()
            self.actor_opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10)
            self.actor_opt.step()
        return {"imagined_actor_loss": float(loss.detach())}

    def state_dict(self):
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target": self.target.state_dict(),
            "model": self.model.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "model_opt": self.model_opt.state_dict(),
            "alpha_opt": self.alpha_opt.state_dict(),
            "log_alpha": self.log_alpha.detach(),
            "updates": self.updates,
            "model_updates": self.model_updates,
            "last_validation": self.last_validation,
        }

    def load_state_dict(self, state):
        for name in (
            "actor",
            "critic",
            "target",
            "model",
            "actor_opt",
            "critic_opt",
            "model_opt",
            "alpha_opt",
        ):
            getattr(self, name).load_state_dict(state[name])
        with torch.no_grad():
            self.log_alpha.copy_(state["log_alpha"])
        self.updates, self.model_updates = state["updates"], state["model_updates"]
        self.last_validation = state["last_validation"]
        self.model_ready = (
            False  # Revalidate against fresh simulator data after resuming.
        )

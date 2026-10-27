import tempfile
import unittest
from pathlib import Path

import torch

from training.checkpoint import (
    restore_checkpoint,
    restore_model,
    save_checkpoint,
    save_model,
)
from training.config import TrainConfig
from training.learner import Learner
from training.replay import Replay
from training.runner import train
from training.toy import ToyEnv


def small_config(**values):
    defaults = {
        "worlds": 4,
        "device": "cpu",
        "replay_capacity": 256,
        "batch_size": 16,
        "hidden": 16,
        "model_hidden": 32,
        "model_window": 3,
        "imagination_horizon": 3,
        "imagination_batch": 8,
        "model_updates": 1,
        "model_min_updates": 2,
        "model_validation_interval": 1,
        "warmup_transitions": 24,
        "total_transitions": 96,
        "max_hours": 0.02,
        "collection_steps": 4,
        "log_interval": 100,
        "model_validation_threshold": 100.0,
    }
    return TrainConfig(**(defaults | values)).validate()


def collect(env, replay, steps=16):
    s, g = env.reset(first=True)
    for _ in range(steps):
        a = torch.randn(env.worlds, 1).clamp(-1, 1)
        ns, ng, r, d, cs, cg, info = env.step(a)
        replay.add(
            state=s,
            goal=g,
            action=info["executed_action"],
            next_state=ns,
            next_goal=ng,
            reward=r,
            done=d,
        )
        s, g = cs, cg


class LearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_replay_ring_world_and_episode_boundaries(self):
        replay = Replay(12, 2, 1, 1, 1)
        for row in range(12):
            s = torch.tensor([[row], [100 + row]], dtype=torch.float32)
            replay.add(
                state=s,
                next_state=s + 1,
                goal=s,
                next_goal=s + 1,
                action=s,
                reward=s,
                done=torch.tensor([[row % 3 == 0], [False]]),
            )
        for validation in (False, True):
            batch = replay.sequences(30, 4, "cpu", validation=validation)
            self.assertTrue(
                torch.all(batch["state"][:, 1:] - batch["state"][:, :-1] == 1)
            )
            self.assertTrue(torch.all(batch["state"] >= (0 if validation else 100)))
            if validation:
                for i in range(30):
                    terminal = torch.where(batch["done"][i, :, 0])[0]
                    if len(terminal):
                        self.assertTrue(
                            torch.all(batch["mask"][i, terminal[0] + 1 :] == 0)
                        )

    def test_model_fits_executed_dynamics_and_actor_gradient(self):
        torch.manual_seed(3)
        cfg, env = small_config(), ToyEnv()
        replay = Replay(256, 4, 4, 1, 2)
        collect(env, replay, 48)
        learner = Learner(cfg, env.spec)
        learner.fit_normalizers(replay)
        held = replay.sequences(48, 3, "cpu", validation=True)
        initial = float(learner.model.sequence_loss(held)[0].detach())
        for _ in range(100):
            learner.update_model(replay)
        final = float(learner.model.sequence_loss(held)[0].detach())
        self.assertLess(final, initial * 0.5)
        learner.update_real(replay)
        self.assertTrue(learner.model_ready)
        model_before = {k: v.clone() for k, v in learner.model.state_dict().items()}
        actor_before = [v.clone() for v in learner.actor.parameters()]
        learner.imagine(replay)
        self.assertTrue(
            any(
                not torch.equal(a, b)
                for a, b in zip(actor_before, learner.actor.parameters())
            )
        )
        self.assertTrue(
            all(
                torch.equal(v, learner.model.state_dict()[k])
                for k, v in model_before.items()
            )
        )

    def test_checkpoint_and_model_tier_isolation(self):
        cfg, env = small_config(), ToyEnv()
        learner = Learner(cfg, env.spec)
        with tempfile.TemporaryDirectory() as temp:
            path, model = Path(temp) / "policy.pt", Path(temp) / "model.pt"
            save_checkpoint(path, learner, env.identity, 32, 5)
            clone = Learner(cfg, env.spec)
            restore_checkpoint(path, clone, env.identity)
            s, g = env.reset()
            self.assertTrue(torch.equal(clone.act(s, g, True), learner.act(s, g, True)))
            with self.assertRaises(ValueError):
                restore_checkpoint(path, clone, env.identity | {"episode_index": 5})
            save_model(model, learner, env.identity)
            restore_model(model, clone, env.identity | {"episode_index": 6})
            with self.assertRaises(ValueError):
                restore_model(model, clone, env.identity | {"tier": 2})

    def test_runner_collects_real_transitions_and_resumes(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg = small_config()
            first = train(ToyEnv(), cfg, Path(temp) / "first")
            self.assertEqual(first["transitions"], cfg.total_transitions)
            self.assertGreater(first["real_updates"], 0)
            self.assertGreater(first["model_updates"], 0)
            second = train(
                ToyEnv(),
                cfg,
                Path(temp) / "resume",
                resume=Path(temp) / "first/latest.pt",
            )
            self.assertGreater(second["real_updates"], first["real_updates"])
            self.assertEqual(
                second["cumulative_transitions"], 2 * cfg.total_transitions
            )

    def test_sac_does_not_fit_world_model(self):
        with tempfile.TemporaryDirectory() as temp:
            report = train(ToyEnv(), small_config(algorithm="sac"), temp)
            self.assertEqual(report["model_updates"], 0)
            self.assertGreater(report["real_updates"], 0)

    def test_invalid_resource_settings(self):
        for values in (
            {"max_hours": float("nan")},
            {"warmup_transitions": 10000},
            {"lower_body_residual": 2},
        ):
            with self.assertRaises(ValueError):
                small_config(**values)


if __name__ == "__main__":
    unittest.main()

"""Read-only checkpoint diagnosis: bounded simulator rollouts, no optimizer steps.

Run from the training release with PYTHONPATH pointing at that release. Four
worlds compare action interventions for diagnosis, not a seeded success-rate
benchmark. A second short rollout checks WM errors on fresh assisted data.
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from training.checkpoint import read_checkpoint
from training.config import TrainConfig
from training.curriculum import stage_at
from training.networks import Actor, WorldModel
from training.runtime import SonicEnv


def main():
    parser = argparse.ArgumentParser()
    for name in ("checkpoint", "job", "runtime", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    ck = read_checkpoint(args.checkpoint)
    cfg = replace(TrainConfig(**ck["config"]), worlds=4)
    env = SonicEnv(args.job, args.runtime, cfg, out / "runtime")
    if env.spec != ck["spec"] or env.identity != ck["identity"]:
        raise ValueError("diagnostic environment must match the checkpoint")
    sp = env.spec
    actor = Actor(sp["obs_dim"], sp["goal_dim"], sp["action_dim"], cfg.hidden).to(
        env.device
    )
    actor.load_state_dict(ck["learner"]["actor"])
    model = WorldModel(
        sp["state_dim"], sp["goal_dim"], sp["action_dim"], cfg.model_hidden
    ).to(env.device)
    model.load_state_dict(ck["learner"]["model"])
    actor.eval()
    model.eval()
    finger_mask = torch.tensor(
        ["hand_" in n for n in env.action_names], device=env.device
    )
    labels = ["learned", "zero_residual", "residual_scaled_0.1", "fingers_only"]
    env.set_training_stage({"assistance_scale": 0.0, "first_frame_probability": 1.0})
    state, goal = env.reset(first=True)
    steps = int(
        np.ceil(
            (
                env.source_reference["timestamps"][-1]
                - env.source_reference["timestamps"][0]
            )
            * env.control_fps
        )
    )
    log = {
        k: []
        for k in (
            "root_z",
            "object",
            "error",
            "terminated",
            "contact",
            "action",
            "wrench",
        )
    }
    with torch.no_grad():
        for step in range(steps):
            action = actor(state, goal, deterministic=True)[0]
            action[1].zero_()
            action[2].mul_(0.1)
            action[3].mul_(finger_mask)
            ns, _ng, _reward, _done, state, goal, info = env.step(action, reset=False)
            q = env.tensor(env.env.state_0.joint_q).reshape(4, -1)
            con = env.env.contact
            force = env.tensor(con.contact_force_w).reshape(4, -1, 3).norm(dim=-1)
            active = env.tensor(con.contact_active).reshape(4, -1).bool()
            for key, value in (
                ("root_z", q[:, 2]),
                ("object", env.object_pose()[:, 0, :3]),
                ("error", info["object_position_error_m"]),
                ("terminated", info["terminated"]),
                ("contact", (active & (force > 0.01)).any(-1)),
                ("action", action),
                ("wrench", info["wrench_reward"]),
            ):
                log[key].append(value.cpu().numpy().copy())
            if not torch.isfinite(ns).all():
                raise RuntimeError("nonfinite diagnostic rollout")
            if step % 400 == 0:
                print(
                    json.dumps({"diagnostic_rollout_step": step, "total": steps}),
                    flush=True,
                )
    log = {k: np.stack(v) for k, v in log.items()}
    np.savez_compressed(out / "intervention_traces.npz", labels=labels, **log)
    interventions = []
    for i, label in enumerate(labels):
        failures = np.flatnonzero(log["terminated"][:, i])
        falls = np.flatnonzero(log["root_z"][:, i] < 0.35)
        interventions.append(
            {
                "label": label,
                "mean_object_position_error_m": float(log["error"][:, i].mean()),
                "first_termination_s": float((failures[0] + 1) / env.control_fps)
                if len(failures)
                else None,
                "first_root_below_0.35m_s": float((falls[0] + 1) / env.control_fps)
                if len(falls)
                else None,
                "root_min_z_m": float(log["root_z"][:, i].min()),
                "contact_fraction": float(log["contact"][:, i].mean()),
                "action_saturation_fraction": float(
                    (np.abs(log["action"][:, i]) > 0.95).mean()
                ),
                "mean_wrench_reward": float(log["wrench"][:, i].mean()),
                "object_final_xyz": log["object"][-1, i].tolist(),
            }
        )
    (out / "interventions.json").write_text(json.dumps(interventions, indent=2))
    print(json.dumps({"interventions": interventions}), flush=True)

    stage = stage_at(ck["transitions"], cfg.curriculum_transitions)
    env.set_training_stage(stage)
    state, goal = env.reset(first=False)
    data = {k: [] for k in ("state", "goal", "action", "next_state", "reward", "done")}
    with torch.no_grad():
        for step in range(512):
            action = actor(state, goal)[0]
            ns, _ng, r, done, cs, cg, info = env.step(action)
            for k, v in (
                ("state", state),
                ("goal", goal),
                ("action", info["executed_action"]),
                ("next_state", ns),
                ("reward", r),
                ("done", done),
            ):
                data[k].append(v.detach().clone())
            state, goal = cs, cg
    data = {k: torch.stack(v) for k, v in data.items()}
    starts = torch.arange(0, 505, 8, device=env.device)
    state = data["state"][starts]
    mask = torch.ones((*state.shape[:2], 1), device=env.device)
    block = env.blocks["object_state"]
    model_results = []
    with torch.no_grad():
        for t in range(8):
            target = data["next_state"][starts + t]
            state, r, _term = model(
                state, data["goal"][starts + t], data["action"][starts + t]
            )
            valid = mask[..., 0] > 0
            obj_error = (
                state[..., block[0] : block[0] + 3]
                - target[..., block[0] : block[0] + 3]
            ).norm(dim=-1)[valid]
            norm_error = ((state - target) / model.norm.std[: sp["state_dim"]]).square()
            normalized_loss = (norm_error * model.state_weights).mean(-1)[valid]
            reward_target = data["reward"][starts + t][..., 0][valid]
            reward_error = (r[..., 0][valid] - reward_target).abs()
            quat_norm = state[..., block[0] + 3 : block[0] + 7].norm(dim=-1)[valid]
            model_results.append(
                {
                    "horizon": t + 1,
                    "valid_samples": int(valid.sum()),
                    "object_position_mean_error_m": float(obj_error.mean()),
                    "object_position_p90_error_m": float(
                        torch.quantile(obj_error, 0.9)
                    ),
                    "weighted_normalized_state_mse": float(normalized_loss.mean()),
                    "reward_mae": float(reward_error.mean()),
                    "actual_reward_mean_abs": float(reward_target.abs().mean()),
                    "predicted_object_quaternion_norm_error": float(
                        (quat_norm - 1).abs().mean()
                    ),
                }
            )
            mask *= (~data["done"][starts + t]).float()
    summary = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_transitions": ck["transitions"],
        "identity": ck["identity"],
        "interventions": interventions,
        "wm_assistance_scale": stage["assistance_scale"],
        "wm_fresh_data": model_results,
        "scope": "Bounded diagnosis, no learning or official success-rate claim; interventions use different vector worlds, so random delays may differ.",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"diagnosis": summary}), flush=True)


if __name__ == "__main__":
    main()

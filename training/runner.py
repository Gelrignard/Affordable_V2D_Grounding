"""Bounded on-machine training and measured resource reports."""

import json
import time
from pathlib import Path

import torch

from .checkpoint import restore_checkpoint, restore_model, save_checkpoint, save_model
from .learner import Learner
from .replay import Replay


def train(env, config, output, resume=None, warm_model=None):
    if config.algorithm == "model_based" and env.worlds < 2:
        raise ValueError(
            "model training needs at least two worlds for held-out validation"
        )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "run.json").exists():
        raise FileExistsError(
            "choose a new run directory (resume reads a checkpoint into a new run)"
        )
    if resume and warm_model:
        raise ValueError("resume and warm-model are mutually exclusive")
    torch.manual_seed(config.seed)
    learner = Learner(config, env.spec)
    previous_transitions, previous_seconds = 0, 0.0
    if resume:
        previous = restore_checkpoint(resume, learner, env.identity)
        previous_transitions, previous_seconds = (
            previous["transitions"],
            previous["wall_seconds"],
        )
    elif warm_model:
        restore_model(warm_model, learner, env.identity)
    replay = Replay(
        config.replay_capacity,
        env.worlds,
        env.spec["state_dim"],
        env.spec["action_dim"],
        env.spec["goal_dim"],
        config.replay_device,
    )
    record = {
        "config": config.to_dict(),
        "spec": env.spec,
        "identity": env.identity,
        "replay_bytes": replay.bytes,
        "resume": str(resume) if resume else None,
        "warm_model": str(warm_model) if warm_model else None,
        "update_ratio_definition": "real minibatch examples per collected transition",
        "budget_definition": "additional transitions and wall hours in this invocation",
        "world_model": "closed_loop_sonic_plus_physics_trained_on_executed_transitions",
    }
    (output / "run.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    state, goal = env.reset(first=True)
    transitions, collections, update_credit, next_save = (
        0,
        0,
        0.0,
        config.checkpoint_interval,
    )
    started = time.monotonic()
    last_metrics, stop_reason = {}, "transition_budget"
    if learner.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(learner.device)
    minimum_rows = max(config.model_window, config.imagination_horizon)
    warmup = max(
        config.warmup_transitions, minimum_rows * env.worlds, config.batch_size
    )
    mask = torch.tensor(env.spec["action_mask"], device=learner.device)
    log_path = output / "metrics.jsonl"

    def elapsed():
        return time.monotonic() - started

    def save(path):
        save_checkpoint(
            path,
            learner,
            env.identity,
            previous_transitions + transitions,
            previous_seconds + elapsed(),
        )

    try:
        with log_path.open("w", encoding="utf-8", buffering=1) as stream:
            while transitions < config.total_transitions:
                if elapsed() >= config.max_hours * 3600:
                    stop_reason = "wall_budget"
                    break
                metrics, collected, rewards, failures = {}, 0, [], []
                for _ in range(config.collection_steps):
                    if (
                        transitions >= config.total_transitions
                        or elapsed() >= config.max_hours * 3600
                    ):
                        break
                    if replay.size < warmup:
                        # Small residual exploration around frozen SONIC, not uniform joint targets.
                        action = torch.randn(
                            env.worlds, env.spec["action_dim"], device=learner.device
                        )
                        action = action.mul(config.exploration_std).clamp(-1, 1) * mask
                    else:
                        action = learner.act(state, goal)
                    ns, ng, reward, done, current_s, current_g, info = env.step(action)
                    replay.add(
                        state=state,
                        goal=goal,
                        action=info["executed_action"],
                        next_state=ns,
                        next_goal=ng,
                        reward=reward,
                        done=done,
                    )
                    state, goal = current_s, current_g
                    transitions += env.worlds
                    collected += env.worlds
                    rewards.append(float(reward.mean()))
                    failures.append(float(info["terminated"].float().mean()))
                if not collected:
                    stop_reason = "wall_budget"
                    break
                collections += 1
                if replay.size >= warmup:
                    learner.fit_normalizers(replay)
                    update_credit += (
                        collected * config.updates_per_transition / config.batch_size
                    )
                    while update_credit >= 1 and elapsed() < config.max_hours * 3600:
                        metrics.update(learner.update_real(replay))
                        update_credit -= 1
                    if config.algorithm == "model_based":
                        for _ in range(config.model_updates):
                            if elapsed() >= config.max_hours * 3600:
                                break
                            metrics.update(learner.update_model(replay))
                        if elapsed() < config.max_hours * 3600:
                            metrics.update(learner.imagine(replay))
                metrics.update(
                    transitions=transitions,
                    cumulative_transitions=previous_transitions + transitions,
                    elapsed_s=elapsed(),
                    transitions_per_s=transitions / max(elapsed(), 1e-6),
                    mean_reward=sum(rewards) / len(rewards),
                    termination_fraction=sum(failures) / len(failures),
                    replay_size=replay.size,
                    model_ready=learner.model_ready,
                )
                if learner.device.type == "cuda":
                    free, total = torch.cuda.mem_get_info(learner.device)
                    metrics.update(
                        torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(
                            learner.device
                        ),
                        torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(
                            learner.device
                        ),
                        device_used_bytes=total - free,
                    )
                stream.write(json.dumps(metrics, allow_nan=False) + "\n")
                last_metrics = metrics
                if collections % config.log_interval == 0:
                    print(json.dumps(metrics, allow_nan=False), flush=True)
                if transitions >= next_save:
                    save(
                        output
                        / f"checkpoint_{previous_transitions + transitions:012d}.pt"
                    )
                    next_save += config.checkpoint_interval
    except KeyboardInterrupt:
        stop_reason = "interrupted"
    except Exception:
        save(output / "interrupted.pt")
        raise
    save(output / "latest.pt")
    if config.algorithm == "model_based" and learner.model_updates:
        save_model(output / "world_model.pt", learner, env.identity)
    report = {
        **last_metrics,
        "stop_reason": stop_reason,
        "replay_bytes": replay.bytes,
        "elapsed_s": elapsed(),
        "transitions": transitions,
        "checkpoint": str(output / "latest.pt"),
        "real_updates": learner.updates,
        "model_updates": learner.model_updates,
        "evaluated_in_simulator": False,
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return report

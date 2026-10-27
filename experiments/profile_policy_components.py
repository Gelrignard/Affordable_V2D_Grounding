"""Bounded wall-time profiling on a disposable copy of a saved learner.

Performs optimizer steps solely to measure steady-state training cost. Does not
write learner checkpoints or modify the paused training queue. Timed imagination
is deliberately enabled to measure its cost, not to certify model readiness.
"""

import argparse
import json
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from training.checkpoint import read_checkpoint
from training.config import TrainConfig
from training.curriculum import stage_at
from training.learner import Learner
from training.replay import Replay
from training.runtime import SonicEnv


def main():
    parser = argparse.ArgumentParser()
    for name in ("checkpoint", "job", "runtime", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--collections", type=int, default=32)
    parser.add_argument("--worlds", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--tf32", action="store_true")
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    ck = read_checkpoint(args.checkpoint)
    cfg = replace(TrainConfig(**ck["config"]), device="cuda:0")
    if args.worlds:
        cfg = replace(cfg, worlds=args.worlds)
    if args.batch_size:
        cfg = replace(cfg, batch_size=args.batch_size)
    if args.tf32:
        torch.set_float32_matmul_precision("high")
    env = SonicEnv(args.job, args.runtime, cfg, out / "runtime")
    env.set_training_stage(stage_at(ck["transitions"], cfg.curriculum_transitions))
    learner = Learner(cfg, env.spec)
    learner.load_state_dict(ck["learner"])
    replay = Replay(
        cfg.replay_capacity,
        cfg.worlds,
        env.spec["state_dim"],
        env.spec["action_dim"],
        env.spec["goal_dim"],
        cfg.replay_device,
    )
    state, goal = env.reset()
    samples = defaultdict(list)
    measured = False

    def timed(label, fn, *positional, **keyword):
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.profiler.record_function(label):
            result = fn(*positional, **keyword)
        torch.cuda.synchronize()
        if measured:
            samples[label].append(time.perf_counter() - started)
        return result

    def collect():
        nonlocal state, goal
        for _ in range(cfg.collection_steps):
            action = timed("actor_inference", learner.act, state, goal)
            ns, ng, r, done, cs, cg, info = timed("simulator_adapter", env.step, action)
            timed(
                "replay_write",
                replay.add,
                state=state,
                goal=goal,
                action=info["executed_action"],
                next_state=ns,
                next_goal=ng,
                reward=r,
                done=done,
            )
            state, goal = cs, cg

    for _ in range(16):
        collect()
    updates = int(
        cfg.collection_steps * cfg.worlds * cfg.updates_per_transition / cfg.batch_size
    )

    def cycle(wm=True, fitted_fastpath=False):
        collect()
        if not fitted_fastpath:
            timed("fit_normalizers", lambda: learner.fit_normalizers(replay))
        for _ in range(updates):
            timed("sac_update", lambda: learner.update_real(replay))
        if wm:
            for _ in range(cfg.model_updates):
                timed("wm_update", lambda: learner.update_model(replay))
            learner.model_ready = (
                True  # Timing only; never treated as accuracy evidence.
            )
            timed("imagined_actor_update", lambda: learner.imagine(replay))

    for _ in range(4):
        cycle()
    results = {}
    for label, use_wm, fastpath in [
        ("full", True, False),
        ("full_skip_fitted_normalizers", True, True),
        ("sac_only", False, True),
    ]:
        samples.clear()
        measured = True
        started = time.perf_counter()
        for _ in range(args.collections):
            cycle(use_wm, fastpath)
        elapsed = time.perf_counter() - started
        measured = False
        results[label] = {
            "wall_s": elapsed,
            "transitions_per_s": args.collections
            * cfg.collection_steps
            * cfg.worlds
            / elapsed,
            "component_seconds": {k: sum(v) for k, v in samples.items()},
            "component_mean_ms": {
                k: 1000 * float(np.mean(v)) for k, v in samples.items()
            },
        }
        print(json.dumps({label: results[label]}), flush=True)
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as prof:
        cycle()
    (out / "torch_profile_cpu.txt").write_text(
        prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=35)
    )
    (out / "torch_profile_cuda.txt").write_text(
        prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=35)
    )
    summary = {
        "config": cfg.to_dict(),
        "checkpoint_transitions": ck["transitions"],
        "threads": torch.get_num_threads(),
        "matmul_precision": torch.get_float32_matmul_precision(),
        "replay_bytes": replay.bytes,
        "peak_torch_reserved_bytes": torch.cuda.max_memory_reserved(),
        "results": results,
        "scope": "Disposable learner, fixed component ordering, short steady-state wall-time test; not a convergence or sample-efficiency comparison. No checkpoint saved.",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

"""Atomic checkpoints with episode, tier and controller compatibility checks."""

import hashlib
import json
from pathlib import Path

import torch


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def atomic_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def save_checkpoint(path, learner, identity, transitions, wall_seconds):
    atomic_save(
        {
            "schema": "v2d_policy_checkpoint_v1",
            "config": learner.cfg.to_dict(),
            "spec": learner.spec,
            "identity": identity,
            "learner": learner.state_dict(),
            "transitions": transitions,
            "wall_seconds": wall_seconds,
            "resume_semantics": "weights_optimizers_counters_fresh_replay",
        },
        path,
    )


def read_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("schema") != "v2d_policy_checkpoint_v1":
        raise ValueError("not a V2D policy checkpoint")
    return checkpoint


def restore_checkpoint(path, learner, identity):
    checkpoint = read_checkpoint(path, learner.device)
    if checkpoint["identity"] != identity or checkpoint["spec"] != learner.spec:
        raise ValueError(
            "checkpoint episode, tier, reference, physics or controller interface differs"
        )
    mutable = {
        "device",
        "worlds",
        "replay_device",
        "replay_capacity",
        "batch_size",
        "max_hours",
        "total_transitions",
        "checkpoint_interval",
        "log_interval",
        "capture_sim",
        "evaluation_interval",
    }
    differences = {
        k
        for k, v in checkpoint["config"].items()
        if k not in mutable and learner.cfg.to_dict()[k] != v
    }
    if differences:
        raise ValueError(f"resume changes learning semantics: {sorted(differences)}")
    learner.load_state_dict(checkpoint["learner"])
    return checkpoint


def model_scope(learner, identity):
    return {
        "tier": identity["tier"],
        "variant": identity["variant"],
        "spec_hash": fingerprint(learner.spec),
        "model_hidden": learner.cfg.model_hidden,
    }


def save_model(path, learner, identity):
    atomic_save(
        {
            "schema": "v2d_closed_loop_model_v1",
            "scope": model_scope(learner, identity),
            "source_identity": identity,
            "model": learner.model.state_dict(),
        },
        path,
    )


def restore_model(path, learner, identity):
    model = torch.load(path, map_location=learner.device, weights_only=True)
    if model.get("schema") != "v2d_closed_loop_model_v1" or model[
        "scope"
    ] != model_scope(learner, identity):
        raise ValueError(
            "world-model transfer requires the same tier, variant and controller/state schema"
        )
    learner.model.load_state_dict(model["model"])
    learner.model_ready = False

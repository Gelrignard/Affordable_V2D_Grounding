"""CPU-only native checkpoint and evaluation contract verification."""
import json
import math
from pathlib import Path
import struct
from training.core.bundles import sha256

def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default

def read_header(path):
    """Read metadata without importing a GPU framework or loading tensor bytes."""
    with Path(path).open("rb") as stream:
        raw = stream.read(8)
        if len(raw) != 8:
            raise ValueError("Truncated safetensors header")
        size = struct.unpack("<Q", raw)[0]
        if not 0 < size <= 32 * 1024 * 1024:
            raise ValueError("Invalid safetensors header length")
        payload = stream.read(size)
    header = json.loads(payload)
    metadata = header.get("__metadata__", {})
    if not isinstance(metadata, dict):
        raise ValueError("Checkpoint metadata must be a mapping")
    if metadata.get("format") != "learner_state" or metadata.get("version") != "1":
        raise ValueError("Expected the pinned native learner_state/version1 envelope")
    return {name.removeprefix("user."): value for name, value in metadata.items()
            if name.startswith("user.")}


def checkpoint_pair(path):
    path = Path(path).resolve()
    if path.name.startswith("policy_"):
        policy, state = path, path.with_name(path.name.replace("policy_", "state_", 1))
    elif path.name.startswith("state_"):
        state, policy = path, path.with_name(path.name.replace("state_", "policy_", 1))
    else:
        raise ValueError("Checkpoint must be a native policy_ or state_ safetensors file")
    metadata = read_header(state)
    actor_metadata = read_header(policy)
    for current, kind in ((metadata, "training_state"), (actor_metadata, "policy")):
        if current.get("algorithm") != "flash_sac" or current.get("checkpoint_kind") != kind:
            raise ValueError(f"Expected a native FlashSAC {kind} checkpoint")
    for name in ("checkpoint_set_id", "environment_steps", "policy_schema_json", "critic_schema_json"):
        if not metadata.get(name) or metadata[name] != actor_metadata.get(name):
            raise ValueError(f"Policy/state checkpoint pair differs in {name}")
    return state, policy, metadata


def completed_checkpoint(output, expected_config, *, required_steps=None, minimum_steps=0):
    status = read_json(output / "status.json", {})
    if status.get("state") != "complete" or status.get("exit_code") != 0 or status.get("forced_kill"):
        raise RuntimeError("Training did not finish cleanly")
    if read_json(output / "resolved_config.json") != expected_config:
        raise RuntimeError("Executed training config differs from the immutable planned config")
    latest = read_json(output / "latest_checkpoint.json", {})
    state, policy, metadata = checkpoint_pair(latest["state"])
    steps = int(metadata["environment_steps"])
    if Path(latest["policy"]).resolve() != policy or int(latest["environment_steps"]) != steps:
        raise RuntimeError("Latest marker and native checkpoint pair disagree")
    if steps <= minimum_steps or (required_steps is not None and steps != required_steps):
        raise RuntimeError(f"Training checkpoint has {steps} transitions; required {required_steps or 'additional learning'}")
    if steps > expected_config["training"]["total_environment_steps"]:
        raise RuntimeError("Checkpoint exceeds its planned transition cap")
    return state, policy, metadata


def evaluation_scores(report, policy, steps):
    expected = {"checkpoint_sha256": sha256(policy), "environment_steps": steps,
                "world_count": 16, "start_frame": 0, "reset_mode": "explicit", "settling_steps": 0}
    contract = {"frame_zero": True, "explicit_reset": True, "within_attempt_resets": 0,
                "voc_scale": 0.0, "reset_voc_scale": 0.0, "settling_steps": 0,
                "training_termination_disabled_during_rollout": True}
    if (any(report.get(k) != v for k, v in expected.items()) or report.get("non_finite_worlds")
            or any(report.get("evaluation_contract", {}).get(k) != v for k, v in contract.items())):
        raise RuntimeError("Evaluation checkpoint/cohort/unassisted rollout contract differs")
    scores = report["leaderboard"]["scores"]
    if any(not isinstance(scores.get(k), (float, int)) or not math.isfinite(scores[k])
           for k in ("add_auc", "mppe_cm", "spider_sr", "maniptrans_sr")):
        raise RuntimeError("Evaluation scores are not finite")
    return scores

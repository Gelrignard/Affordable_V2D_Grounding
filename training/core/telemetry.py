"""JSON and GPU telemetry helpers; importing this module never initializes CUDA."""
import datetime
import json
import math
import os
from pathlib import Path
import subprocess

def utc_now():
    return datetime.datetime.now(datetime.UTC).isoformat()


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False))
    temporary.replace(path)


def gpu_memory():
    line = subprocess.check_output(
        [
            "nvidia-smi",
            "--id=0",
            "--query-gpu=memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=10,
    ).strip()
    used, total, utilization = (int(value.strip()) for value in line.split(","))
    return {"used_mib": used, "total_mib": total, "utilization_percent": utilization}


def memory_limit_mib(total_mib, explicit=None):
    """Keep the established 4090 headroom, scaled to the actual device size."""
    if explicit is not None:
        if not isinstance(explicit, int) or explicit <= 0:
            raise ValueError("GPU memory limit must be a positive number of MiB")
        return explicit
    if not isinstance(total_mib, int) or total_mib <= 0:
        raise ValueError("Cannot determine GPU memory protection without device capacity")
    return max(1, int(total_mib * (23000 / 24576)))


def nonfinite_packet(value):
    """Losslessly label nonfinite scalars and record unambiguous JSON paths."""
    fields = []
    def visit(item, path):
        if isinstance(item, float) and not math.isfinite(item):
            raw = "NaN" if math.isnan(item) else ("+Infinity" if item > 0 else "-Infinity")
            fields.append({"path": path, "raw_value": raw})
            return {"__nonfinite_float__": raw}
        if isinstance(item, dict):
            return {key: visit(child, path + [key]) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [visit(child, path + [index]) for index, child in enumerate(item)]
        return item
    return visit(value, []), fields


def critical_learner_health(learner, transition, next_action):
    """Rare diagnostic scan; reduce on device, never copy the replay to the CPU.

    Actor/critic batch statistics and temperature state are included because the
    native gradient skip does not protect them. Historical replay storage and
    the intentionally contaminated logging accumulator are explicitly excluded.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np
    names = ("actor", "critic", "target_critic", "temperature", "reward_normalizer",
             "exploration", "loss_scale", "observation", "critic_context", "update_credit",
             "environment_steps", "global_update_step")
    tree = {name: getattr(learner, name) for name in names}
    tree.update(transition=transition, next_action=next_action, replay_pending=learner.replay.pending)
    leaves, _ = jax.tree_util.tree_flatten_with_path(tree)
    checks = [(jax.tree_util.keystr(path), jnp.all(jnp.isfinite(array)))
              for path, array in leaves if hasattr(array, "dtype")]
    results = np.asarray(jax.device_get(jnp.stack([check for _, check in checks])), dtype=bool)
    bad = [name for (name, _), passed in zip(checks, results, strict=True) if not passed]
    return {"critical_arrays_finite": not bad, "checked_array_count": len(checks),
            "nonfinite_array_paths": bad, "historical_replay_storage_checked": False,
            "logging_accumulator_checked": False,
            "environment_steps": int(jax.device_get(learner.environment_steps)),
            "global_update_step": int(jax.device_get(learner.global_update_step))}


def metric_decision(metrics, fields, health, consecutive):
    """A metric is evidence, not learner state; never interpolate either one."""
    known = {"sac/actor_loss_mean_per_update": "sac/actor_skipped_fraction",
             "sac/critic_loss_mean_per_update": "sac/critic_skipped_fraction"}
    if health.get("critical_arrays_finite") is not True or health.get("checked_array_count", 0) < 1:
        return "stop_invalid_or_unavailable_critical_state"
    confirmed_rejected_losses = True
    counters = {"transitions", "environment_steps", "global_update_step", "training_steps"}
    for field in fields:
        path = field["path"]
        if any(key in counters for key in path if isinstance(key, str)):
            return "stop_invalid_training_counter"
        if len(path) != 1 or path[0] not in known:
            confirmed_rejected_losses = False
            continue
        skipped = metrics.get(known[path[0]])
        if not isinstance(skipped, (float, int)) or not math.isfinite(skipped) or not 0 < skipped <= 1:
            confirmed_rejected_losses = False
    return "continue_rejected_loss" if confirmed_rejected_losses else "continue_diagnostic_nonfinite"


def preserve_metric_event(output, record, health_check, consecutive):
    """Persist the offending packet before any health check can itself fail."""
    safe, fields = nonfinite_packet(record)
    if not fields:
        return record, None
    directory = Path(output) / "nonfinite_metrics"
    directory.mkdir(exist_ok=True)
    packet_path = directory / f"step_{record['transitions']}_packet.json"
    with packet_path.open("x", encoding="utf-8") as stream:
        json.dump({"schema": "v2d_nonfinite_metric_packet_v1", "record": safe,
                   "nonfinite_fields": fields}, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        health = health_check()
    except Exception as exc:
        health = {"critical_arrays_finite": None, "error": f"{type(exc).__name__}: {exc}"}
    decision = metric_decision(record, fields, health, consecutive)
    event = {"packet": str(packet_path), "nonfinite_fields": fields, "health": health,
             "consecutive_nonfinite_windows": consecutive, "decision": decision,
             "replay_history_finiteness_not_established": True}
    write_json(directory / f"step_{record['transitions']}_decision.json", event)
    safe["telemetry_health"] = event
    return safe, event

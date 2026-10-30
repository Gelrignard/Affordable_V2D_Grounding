"""CPU-only explicit native recovery lineage and checkpoint health checks."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct


def finite_checkpoint(path):
    """Scan every stored floating tensor in bounded chunks, without JAX/CUDA."""
    import numpy as np
    path = Path(path)
    with path.open("rb") as stream:
        size = struct.unpack("<Q", stream.read(8))[0]
        if not 0 < size <= 32 * 1024 * 1024:
            raise ValueError("Invalid checkpoint header")
        header = json.loads(stream.read(size))
    checked, bad = 0, []
    float_types = {"F16": "<f2", "F32": "<f4", "F64": "<f8", "BF16": "<u2"}
    for name, item in header.items():
        if name == "__metadata__" or item["dtype"] not in float_types:
            continue
        begin, end = item["data_offsets"]
        dtype = np.dtype(float_types[item["dtype"]])
        if begin < 0 or end < begin or 8 + size + end > path.stat().st_size or (end - begin) % dtype.itemsize:
            raise ValueError("Invalid tensor offsets")
        if begin == end:
            continue
        values = np.memmap(path, mode="r", dtype=dtype, offset=8 + size + begin,
                           shape=((end - begin) // dtype.itemsize,))
        valid = True
        for offset in range(0, values.size, 1_048_576):
            block = values[offset:offset + 1_048_576]
            valid &= bool(np.all((block & 0x7f80) != 0x7f80) if item["dtype"] == "BF16"
                          else np.isfinite(block).all())
        del values
        checked += 1
        if not valid:
            bad.append(name)
    if not checked or bad:
        raise ValueError(f"Checkpoint floating tensors are invalid: {bad or 'none found'}")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"floating_tensors_checked": checked, "all_floating_tensors_finite": True,
            "sha256": digest}


def bind_native_recovery(args, output, job, *, api):
    """Bind a failed native run, keeping its request/status/preparation immutable."""
    old = Path(args.recovery_from).resolve()
    output = Path(output).resolve()
    if old == output or output.is_relative_to(old) or old.is_relative_to(output):
        raise ValueError("Recovery requires a separate, nonoverlapping output directory")
    destination = output / "recovery_lineage.json"
    if destination.exists():
        lineage = api.guard.read_json(destination)
        if (lineage["recovery_from"] != str(old)
                or Path(lineage["checkpoint"]["state"]).resolve() != Path(args.native_resume).resolve()):
            raise ValueError("Recovery source changed on resume")
        verify_recovery_history(output, api=api)
        return lineage
    previous = api.guard.read_json(old / "request.json")
    preparation = api.guard.read_json(old / "preparation.json")
    status = api.guard.read_json(old / "pipeline_status.json")
    if not previous or not preparation or status.get("state") != "failed" or status.get("failed_stage") != "bootstrap":
        raise ValueError("Explicit native recovery requires a failed bootstrap pipeline")
    if preparation.get("original_job", preparation["job"]) != job or previous["runtime_commit"] != api.runtime_identity(args.runtime):
        raise ValueError("Recovery changes input or native runtime")
    if (previous.get("recipe", "legacy") != args.recipe
            or previous["selection_rule"] != api.selection_rule(args.selection_policy)
            or previous.get("acquisition_guidance", "precision") != args.acquisition_guidance
            or previous.get("placement_guidance", "off") != args.placement_guidance):
        raise ValueError("Recovery must retain the original recipe and selection settings")
    record = api.checkpoint_record(args.native_resume)
    if not Path(record["state"]).is_relative_to(old / "bootstrap/checkpoints"):
        raise ValueError("Recovery checkpoint must belong to the failed native stage")
    _, _, metadata = api.checkpoint_pair(record["state"])
    saved = json.loads(metadata["resolved_config_json"])
    native_target = saved["training"]["total_environment_steps"]
    if not 0 < record["environment_steps"] < native_target:
        raise ValueError("Recovery requires a partial native checkpoint")
    expected = api.guard.read_json(preparation["native_config"])
    saved_without_reference = {k: v for k, v in saved.items() if k != "reference"}
    if saved_without_reference != {k: v for k, v in expected.items() if k != "reference"}:
        raise ValueError("Saved native configuration differs from the original prepared config")
    ledger_path = old / "ledger.json"
    ledger = api.guard.read_json(ledger_path)
    candidate = f"tier{job['manifest']['tier']}_episode_{job['manifest']['episode_index']:06d}"
    if not ledger or not ledger.get("jobs"):
        raise ValueError("Recovery requires the original charged ledger")
    api.guard.remaining_budget(ledger, candidate, 0)
    jobs = ledger["jobs"]
    if any(row.get("candidate") != candidate or row.get("name") != "bootstrap"
           or row.get("kind") != "training" or row.get("state") not in ("failed", "stopped", "complete") for row in jobs):
        raise ValueError("Only one original native acquisition attempt may precede recovery")
    if len(jobs) != 1:
        raise ValueError("Repeated or additional original attempts are not eligible for automatic recovery")
    files = {str(path): api.sha(path) for path in
             [old / "request.json", old / "preparation.json", old / "pipeline_status.json",
              Path(preparation["native_config"]), old / "bootstrap/status.json",
              old / "bootstrap/resolved_config.json"]}
    for key in ("runtime_motion", "runtime_support"):
        if preparation.get(key):
            files[preparation[key]] = api.sha(preparation[key])
    health = {kind: finite_checkpoint(record[kind]) for kind in ("state", "policy")}
    lineage = {"schema": "v2d_native_recovery_lineage_v1", "recovery_from": str(old),
               "ledger_path": str(ledger_path), "candidate": candidate,
               "historical_job_count": len(jobs), "historical_jobs_sha256": api.guard.canonical_sha256(jobs),
               "original_budget_policy": api.guard.ledger_budget_policy(ledger),
               "original_request": previous, "original_files": files,
               "checkpoint": record, "checkpoint_health": health,
               "native_target_steps": native_target, "world_count": saved["training"]["world_count"],
               "fresh_replay": True, "simulator_state_restored": False,
               "native_config_preserved": True, "automatic_retry": False}
    api.immutable_json(destination, lineage)
    return lineage


def verify_recovery_history(output, *, api=None, require_complete=False):
    """Collector-facing read-only check of immutable failure and appended jobs."""
    if api is None:
        from training.scripts import pipeline as api
    output = Path(output)
    lineage = api.guard.read_json(output / "recovery_lineage.json")
    if lineage.get("schema") != "v2d_native_recovery_lineage_v1":
        raise ValueError("Unsupported recovery lineage")
    request = api.guard.read_json(output / "request.json")
    if request and (request.get("recovery_lineage_sha256") != api.sha(output / "recovery_lineage.json")
                    or request.get("ledger_path") != lineage["ledger_path"]
                    or request.get("native_resume") != lineage["checkpoint"]["state"]):
        raise ValueError("Recovery request does not bind its lineage")
    claim = api.guard.read_json(Path(lineage["recovery_from"]) / "native_recovery_claim.json")
    if claim and (not request or claim.get("output") != str(output.resolve())
                  or claim.get("request_sha256") != api.sha(output / "request.json")
                  or claim.get("lineage_sha256") != api.sha(output / "recovery_lineage.json")):
        raise ValueError("Recovery belongs to another claimed attempt")
    for path, digest in lineage["original_files"].items():
        if api.sha(path) != digest:
            raise ValueError(f"Historical recovery evidence changed: {path}")
    api.verify_checkpoint(lineage["checkpoint"])
    ledger = api.guard.read_json(lineage["ledger_path"])
    api.guard.effective_budget_policy(ledger, lineage["candidate"], ledger_path=lineage["ledger_path"])
    count = lineage["historical_job_count"]
    if (api.guard.canonical_sha256(ledger["jobs"][:count]) != lineage["historical_jobs_sha256"]
            or api.guard.ledger_budget_policy(ledger) != lineage["original_budget_policy"]):
        raise ValueError("Original ledger policy or charged job prefix changed")
    appended = ledger["jobs"][count:]
    for row in appended:
        if (row.get("candidate") != lineage["candidate"] or not row.get("name", "").startswith("native_recovery_1__")
                or not Path(row["output"]).resolve().is_relative_to(output.resolve())):
            raise ValueError("Unrelated or repeated work was appended to the recovery ledger")
    if require_complete:
        if not appended or any(row.get("state") != "complete" or row.get("exit_code") != 0 for row in appended):
            raise ValueError("Recovery jobs did not all complete successfully")
        if api.guard.read_json(output / "pipeline_status.json").get("state") != "complete":
            raise ValueError("Recovery pipeline is incomplete")
    return {"lineage_sha256": api.sha(output / "recovery_lineage.json"),
            "historical_job_count": count, "appended_job_count": len(appended),
            "ledger_path": lineage["ledger_path"], "candidate": lineage["candidate"],
            "original_source": lineage["original_request"]["source"],
            "checkpoint": lineage["checkpoint"]}

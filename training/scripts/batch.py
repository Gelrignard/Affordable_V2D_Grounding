"""Run the fixed per-episode policy pipeline serially, with deterministic shards.

This command consumes already prepared portable jobs. It does not reconstruct
videos, allocate cloud instances, or submit predictions to a leaderboard.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def discover_jobs(root, tiers=(1, 2, 3), episodes=None):
    """Prefer corrected, otherwise original; order by task identity."""
    from training.core.input_selection import input_selection_record
    from training.core.input_repair import UnusableReference

    found = {}
    for path in sorted(Path(root).resolve().rglob("job.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("schema") != "v2d_policy_job_v1":
            if data.get("variant") in ("corrected", "original") or path.parent.name in ("corrected", "original"):
                raise ValueError(f"Unsupported portable job schema: {path}")
            continue
        if data.get("variant") not in ("corrected", "original"):
            raise ValueError(f"Unsupported portable job variant: {path}")
        tier, episode = data.get("tier"), data.get("episode_index")
        if type(tier) is not int or tier not in (1, 2, 3) or type(episode) is not int or episode < 0:
            raise ValueError(f"Invalid portable job identity: {path}")
        if tier not in tiers or (episodes is not None and episode not in episodes):
            continue
        identity = (tier, episode)
        variants = found.setdefault(identity, {})
        if data["variant"] in variants:
            raise ValueError(f"Duplicate portable {data['variant']} job for tier{tier}/episode{episode}: {path}")
        variants[data["variant"]] = path.parent
    result = []
    for (tier, episode), variants in sorted(found.items()):
        variant = "corrected" if "corrected" in variants else "original"
        fallback_reason = None
        try:
            record = input_selection_record(variants[variant])
        except UnusableReference as exc:
            if variant != "corrected" or "original" not in variants:
                raise
            fallback_reason = str(exc)
            print(f"WARNING tier{tier}/episode_{episode:06d}: corrected reference is unusable "
                  f"({exc}); checking original trajectory.", file=sys.stderr)
            variant = "original"
            record = input_selection_record(variants[variant])
        job = {"tier": tier, "episode_index": episode, "job": record["job"],
               "job_sha256": record["job_sha256"]}
        # Keep the legacy corrected-only plan byte structure for safe resumes.
        # Original fallback is new and is explicitly bound into its batch plan.
        if variant == "original":
            record["selection_reason"] = ("corrected_unusable_using_original" if fallback_reason is not None
                                          else "corrected_unavailable_using_original")
            if fallback_reason is not None:
                record["unusable_corrected_reference"] = fallback_reason
            job["input_selection"] = record
        result.append(job)
    return result


def make_plan(args):
    jobs = discover_jobs(args.jobs, args.tiers, args.episodes)
    if not jobs:
        raise ValueError("No matching corrected or original portable jobs")
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("Require 0 <= shard-index < shard-count")
    if args.require_complete:
        if args.episodes is None:
            raise ValueError("require-complete needs explicit episode IDs; arbitrary datasets have no fixed30-episode roster")
        expected = {(tier, episode) for tier in args.tiers
                    for episode in args.episodes}
        missing = sorted(expected - {(job["tier"], job["episode_index"]) for job in jobs})
        if missing:
            raise ValueError(f"Missing requested jobs: {missing}")
    selected = [job for index, job in enumerate(jobs) if index % args.shard_count == args.shard_index]
    if args.max_jobs is not None:
        selected = selected[:args.max_jobs]
    guidance = getattr(args, "acquisition_guidance", "precision")
    if guidance == "grasp-region" and any(job["tier"] == 1 for job in selected):
        raise ValueError("Use a separate Tier 2/3 shard for grasp-region guidance")
    placement = getattr(args, "placement_guidance", "off")
    if placement != "off" and (guidance != "precision" or any(job["tier"] == 1 for job in selected)):
        raise ValueError("Supported release requires a separate Tier 2/3 precision shard")
    root = Path(args.output).resolve()
    for job in selected:
        job["output"] = str(root / f"tier{job['tier']}" / f"episode_{job['episode_index']:06d}")
        if args.dataset_root:
            dataset = Path(args.dataset_root).resolve() / f"tier{job['tier']}"
            if not dataset.is_dir():
                raise ValueError(f"Released input tier directory missing: {dataset}")
            job["dataset_input"] = str(dataset)
    plan = {"schema": "v2d_policy_batch_v1", "jobs_root": str(Path(args.jobs).resolve()),
            "runtime": str(Path(args.runtime).resolve()), "output": str(root),
            "deadline_utc": args.deadline_utc, "gpu_hours": args.gpu_hours,
            "training_budget_seconds": args.training_budget_seconds,
            "episode_total_budget_seconds": args.episode_total_budget_seconds,
            "submission_kit": str(Path(args.submission_kit).resolve()) if args.submission_kit else None,
            "shard_index": args.shard_index, "shard_count": args.shard_count,
            "selection": "sorted(tier, episode), then index modulo shard_count", "jobs": selected}
    if guidance != "precision":
        plan["acquisition_guidance"] = guidance
    if placement != "off":
        plan["placement_guidance"] = placement
    recipe = getattr(args, "recipe", "legacy")
    selection_policy = getattr(args, "selection_policy", "success")
    if recipe != "legacy" and guidance != "precision":
        raise ValueError("Named recipes require precision acquisition")
    if recipe == "tier1-precision" and any(job["tier"] != 1 for job in selected):
        raise ValueError("tier1-precision requires a Tier 1 shard")
    if recipe == "tier23-load-transfer" and any(job["tier"] == 1 for job in selected):
        raise ValueError("tier23-load-transfer requires a Tier 2/3 shard")
    if recipe != "legacy":
        plan["recipe"] = recipe
    if selection_policy != "success":
        plan["selection_policy"] = selection_policy
    for field in ("native_max_seconds", "refinement_max_seconds", "evaluation_max_seconds", "strict_reproducibility"):
        if getattr(args, field, None):
            plan[field] = getattr(args, field)
    return plan


def charged_seconds(root):
    """Include failed/interrupted work, not only successful episode totals."""
    total = 0.0
    for path in Path(root).glob("tier*/episode_*/ledger.json"):
        ledger = json.loads(path.read_text(encoding="utf-8"))
        for job in ledger.get("jobs", []):
            value = float(job.get("elapsed_seconds", 0))
            if value < 0 or not math.isfinite(value):
                raise ValueError(f"Invalid charged GPU seconds: {path}")
            total += value
    return total


def episode_command(plan, job, deadline):
    command = [sys.executable, "-m", "training.scripts.pipeline", "--job", job["job"],
               "--runtime", plan["runtime"], "--output", job["output"]]
    for flag, value in (("--training-budget-seconds", plan["training_budget_seconds"]),
                        ("--total-budget-seconds", plan["episode_total_budget_seconds"]),
                        ("--deadline-utc", deadline)):
        if value is not None:
            command.extend([flag, str(value)])
    for field in ("native_max_seconds", "refinement_max_seconds", "evaluation_max_seconds"):
        if plan.get(field) is not None:
            command.extend(["--" + field.replace("_", "-"), str(plan[field])])
    if plan.get("strict_reproducibility"):
        command.append("--strict-reproducibility")
    if plan["submission_kit"]:
        command.extend(["--dataset-input", job["dataset_input"], "--submission-kit", plan["submission_kit"]])
    if plan.get("acquisition_guidance", "precision") != "precision":
        command.extend(["--acquisition-guidance", plan["acquisition_guidance"]])
    if plan.get("placement_guidance", "off") != "off":
        command.extend(["--placement-guidance", plan["placement_guidance"]])
    recipe = plan.get("recipe", "legacy")
    if recipe == "auto":
        recipe = "tier1-precision" if job["tier"] == 1 else "tier23-load-transfer"
    if recipe != "legacy":
        command.extend(["--recipe", recipe])
    selection_policy = plan.get("selection_policy", "success")
    if selection_policy == "auto":
        selection_policy = "tracking" if recipe == "tier23-load-transfer" else "success"
    if selection_policy != "success":
        command.extend(["--selection-policy", selection_policy])
    return command


def parse_deadline(value):
    if value is None:
        return math.inf
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("deadline-utc must include a UTC offset")
    return result.timestamp()


def batch_lock(path):
    from training.scripts.guard import exclusive_lock
    return exclusive_lock(path)


def run(args):
    if bool(args.dataset_root) != bool(args.submission_kit):
        raise ValueError("dataset-root and submission-kit must be supplied together for automatic export")
    for value in (args.gpu_hours, args.training_budget_seconds, args.episode_total_budget_seconds):
        if value is not None and (not math.isfinite(value) or value <= 0):
            raise ValueError("Explicit GPU budgets must be positive and finite")
    if any(x < 0 for x in (args.episodes or [])):
        raise ValueError("Episode IDs must be nonnegative")
    if args.max_jobs is not None and args.max_jobs <= 0:
        raise ValueError("max-jobs must be positive")
    deadline = parse_deadline(args.deadline_utc)
    plan = make_plan(args)
    from training.core.input_selection import input_selection_record, warn_input_selection
    selections = []
    for job in plan["jobs"]:
        record = input_selection_record(job["job"])
        if record["job_sha256"] != job["job_sha256"]:
            raise ValueError("Selected portable job changed during batch preparation")
        if "input_selection" in job:
            record["selection_reason"] = job["input_selection"]["selection_reason"]
            if "unusable_corrected_reference" in job["input_selection"]:
                record["unusable_corrected_reference"] = job["input_selection"]["unusable_corrected_reference"]
            if record != job["input_selection"]:
                raise ValueError("Original fallback evidence differs from batch plan")
        warn_input_selection(record)
        selections.append(record)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    # The same batch output is exclusively owned while children execute. This
    # lock is separate from each child's host/GPU and per-episode locks.
    root = Path(plan["output"])
    root.mkdir(parents=True, exist_ok=True)
    with batch_lock(root / "batch.lock"):
        path = root / "batch_plan.json"
        if path.exists():
            if json.loads(path.read_text(encoding="utf-8")) != plan:
                raise ValueError("Batch resume arguments/jobs differ from immutable batch_plan.json")
        else:
            write_json(path, plan)
        selection_path = root / "input_selection.json"
        selection_evidence = {"schema": "v2d_policy_batch_input_selection_v1",
                              "batch_plan_sha256": digest(path), "jobs": selections}
        if selection_path.exists():
            if json.loads(selection_path.read_text(encoding="utf-8")) != selection_evidence:
                raise ValueError("Batch input selection evidence changed")
        else:
            write_json(selection_path, selection_evidence)
        status = {"schema": "v2d_policy_batch_status_v1", "state": "running", "episodes": [],
                  "plan_sha256": digest(path), "active_episode": None}
        for job in plan["jobs"]:
            remaining = (plan["gpu_hours"] * 3600 if plan["gpu_hours"] is not None else math.inf) - charged_seconds(root)
            # Reservation is conservative: a new episode can consume its full
            # per-episode allowance without overspending the batch allowance.
            ledger = Path(job["output"]) / "ledger.json"
            prior = 0.0
            if ledger.exists():
                rows = json.loads(ledger.read_text())["jobs"]
                if any(row.get("state") == "running" for row in rows):
                    raise RuntimeError("Episode has a running canonical job; refusing overlapping batch work")
                prior = sum(float(j.get("elapsed_seconds", 0)) for j in rows)
            marker = Path(job["output"]) / "pipeline_status.json"
            prior_status = json.loads(marker.read_text(encoding="utf-8")) if marker.is_file() else {}
            completed_before = prior_status.get("state") == "complete"
            if prior_status.get("state") == "failed":
                status["episodes"].append({"tier": job["tier"], "episode_index": job["episode_index"],
                    "output": job["output"], "exit_code": 1, "state": "previous_terminal_failure_not_retried"})
                continue
            cap = plan["episode_total_budget_seconds"]
            reserve = 0 if completed_before or cap is None else max(0, cap - prior)
            if not completed_before and (remaining <= 0 or remaining < reserve or deadline <= time.time()):
                status.update(state="budget_exhausted", active_episode=None, charged_gpu_seconds=charged_seconds(root))
                write_json(root / "batch_status.json", status)
                return 2
            status["active_episode"] = {k: job[k] for k in ("tier", "episode_index")}
            write_json(root / "batch_status.json", status)
            episode_plan = dict(plan)
            if cap is None and math.isfinite(remaining):
                existing_request = Path(job["output"]) / "request.json"
                episode_plan["episode_total_budget_seconds"] = (
                    json.loads(existing_request.read_text())["total_budget_seconds"] if existing_request.exists()
                    else remaining)
            completed = subprocess.run(episode_command(episode_plan, job, plan["deadline_utc"]), check=False)
            status["episodes"].append({**status["active_episode"], "output": job["output"],
                                        "exit_code": completed.returncode})
            status["charged_gpu_seconds"] = charged_seconds(root)
            write_json(root / "batch_status.json", status)
            # A terminal episode fault does not retry or prevent other episodes.
            # A still-running canonical row is not a terminal failure.
            if completed.returncode:
                if ledger.exists() and any(row.get("state") == "running" for row in json.loads(ledger.read_text())["jobs"]):
                    status.update(state="needs_attention", active_episode=None)
                    write_json(root / "batch_status.json", status)
                    return completed.returncode
        failed = sum(row.get("exit_code", 0) != 0 for row in status["episodes"])
        status.update(state="complete_with_failures" if failed else "complete", failed_episode_count=failed,
                      active_episode=None, charged_gpu_seconds=charged_seconds(root))
        write_json(root / "batch_status.json", status)
    print(json.dumps({"state": status["state"], "output": str(root), "episodes": len(plan["jobs"]), "failed_episodes": failed}))
    return 1 if failed else 0


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("jobs", "runtime", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--deadline-utc", help="Optional calendar deadline")
    p.add_argument("--gpu-hours", type=float, help="Optional maximum charged GPU wall hours for this shard")
    p.add_argument("--training-budget-seconds", type=float)
    p.add_argument("--episode-total-budget-seconds", type=float)
    for name in ("native-max-seconds", "refinement-max-seconds", "evaluation-max-seconds"):
        p.add_argument("--" + name, type=float)
    p.add_argument("--strict-reproducibility", action="store_true")
    p.add_argument("--tiers", type=int, nargs="+", choices=(1, 2, 3), default=[1, 2, 3])
    p.add_argument("--episodes", type=int, nargs="+")
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--shard-count", type=int, default=1)
    p.add_argument("--max-jobs", type=int)
    p.add_argument("--require-complete", action="store_true", help="Require all requested identities before any GPU work")
    p.add_argument("--dataset-root", help="Released input dataset containing tier1/, tier2/, tier3/")
    p.add_argument("--submission-kit")
    p.add_argument("--acquisition-guidance", choices=("precision", "grasp-region"), default="precision",
                   help="Opt-in acquisition guidance for Tier 2/3 shards")
    p.add_argument("--placement-guidance", choices=("off", "supported-release"), default="off",
                   help="Opt-in support-aware release for Tier 2/3 shards")
    p.add_argument("--recipe", choices=("legacy", "auto", "tier1-precision", "tier23-load-transfer"), default="legacy",
                   help="legacy preserves resumes; auto uses Tier 1 precision or Tier 2/3 load transfer")
    p.add_argument("--selection-policy", choices=("auto", "success", "tracking"), default="success",
                   help="Fixed ranking; auto uses success for Tier 1 and tracking for Tier 2/3")
    p.add_argument("--dry-run", action="store_true", help="Print the complete deterministic roster without writing or training")
    return p


def main(argv=None):
    return run(parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())

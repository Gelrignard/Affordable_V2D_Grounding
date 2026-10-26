"""Read-only, standard-library status endpoint for an AutoDL pilot.

Safe to invoke through a forced-command SSH key: this module reads status files
and /proc, and never starts jobs, changes training configuration or powers off.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def tail(path, size=16384):
    try:
        with path.open("rb") as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - size))
            return stream.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def last_metric(path):
    for line in reversed(tail(path).splitlines()):
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict) and "transitions" in value:
            return value
    return {}


def process_alive(pid, proc=Path("/proc"), script="run_pilot.sh"):
    try:
        # Match the launch script too: a recycled PID is not our training queue.
        command = (proc / str(pid) / "cmdline").read_bytes()
        return script.encode() in command
    except OSError:
        return False


def diagnostic_snapshot(root, relative, *, now, alive=None):
    """Read an explicitly configured bounded native test queue independently."""
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("diagnostic status path escapes configured root")
    queue = read_json(path)
    if queue is None:
        return {"state": "status_unavailable", "path": str(path)}
    running = (
        process_alive(queue.get("pid"), script="run_native_chord_pair.py")
        if alive is None
        else alive
    )
    activity = [path.stat().st_mtime]
    runs = []
    for entry in queue.get("runs", []):
        variant = entry.get("variant")
        if variant not in ("original", "corrected"):
            continue
        folder = path.parent / variant
        training = read_json(folder / "status.json") or {}
        metric = last_metric(folder / "metrics.jsonl")
        evaluation = read_json(folder / "evaluation.json") or {}
        runs.append(
            {
                "variant": variant,
                "phase": entry.get("state"),
                "training_state": training.get("state"),
                "training_elapsed_s": training.get("elapsed_s"),
                "peak_gpu_used_mib": training.get("peak_gpu_used_mib"),
                "stop_reason": training.get("stop_reason"),
                "transitions": metric.get("transitions"),
                "transitions_per_second": metric.get("throughput/env_steps_per_second"),
                "assistance_scale": metric.get("curriculum/voc_scale"),
                "evaluation": evaluation.get("metrics"),
                "leaderboard": read_json(folder / "leaderboard.json"),
                "reference_tracking": evaluation.get("reference_tracking"),
                "video": entry.get("video"),
                "video_error": entry.get("video_error"),
            }
        )
        for name in ("status.json", "metrics.jsonl", "evaluation.json"):
            if (folder / name).exists():
                activity.append((folder / name).stat().st_mtime)
    state = queue.get("state")
    if state == "running" and not running:
        state = "stopped_without_exit_record"
    age = max(0, now - max(activity))
    if state == "running" and age > 600:
        state = "possibly_stalled"
    return {
        "state": state,
        "process_alive": running,
        "path": str(path),
        "activity_age_s": age,
        "elapsed_s": (
            max(0, now - datetime.fromisoformat(queue["started_utc"]).timestamp())
            if state in ("running", "possibly_stalled") and queue.get("started_utc")
            else queue.get("elapsed_s")
        ),
        "error": queue.get("error"),
        "runs": runs,
        "metric_reference": queue.get("metric_reference"),
        "transition_budget_per_policy": queue.get("transition_budget_per_policy"),
        "replay_capacity": queue.get("replay_capacity"),
        "planned_runs": queue.get("planned_runs"),
        "official_leaderboard_score": False,
    }


def candidate_snapshot(root, relative, *, now, alive=None):
    """Read bounded extension experiments separately from intentionally paused work."""
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("candidate status path escapes configured root")
    queue = read_json(path)
    if queue is None:
        return {"state": "status_unavailable", "path": str(path)}
    running = (
        process_alive(queue.get("pid"), script="run_chord_candidates.py")
        if alive is None
        else alive
    )
    activity = [path.stat().st_mtime]
    runs = []
    for entry in queue.get("runs", []):
        variant = entry.get("variant")
        if variant == "native_corrected":
            runs.append(
                {
                    key: entry.get(key)
                    for key in (
                        "variant",
                        "state",
                        "transitions",
                        "leaderboard",
                        "evaluation_seconds",
                    )
                }
            )
            continue
        if variant not in ("adaptive", "adaptive_wrench"):
            continue
        folder = path.parent / variant
        training = read_json(folder / "status.json") or {}
        extension = read_json(folder / "extension_status.json") or {}
        metric = last_metric(folder / "metrics.jsonl")
        outcome = extension.get("latest_unassisted", {})
        probe = extension.get("latest_probe", {})
        runs.append(
            {
                "variant": variant,
                "state": extension.get("state", entry.get("state")),
                "transitions": max(
                    metric.get("transitions") or 0,
                    extension.get("transitions") or 0,
                )
                or None,
                "throughput": metric.get("throughput/env_steps_per_second"),
                "assistance_scale": metric.get("curriculum/voc_scale"),
                "peak_gpu_used_mib": training.get("peak_gpu_used_mib"),
                "training_elapsed_s": extension.get("training_elapsed_s"),
                "evaluation_elapsed_s": extension.get("evaluation_elapsed_s"),
                "decision": extension.get("decision"),
                "leaderboard": read_json(folder / "leaderboard.json"),
                "latest_unassisted": outcome.get("leaderboard"),
                "reference_tracking": outcome.get("reference_tracking"),
                "physical_diagnostics": outcome.get("physical_diagnostics"),
                "probe_assistance": probe.get("probe_assistance"),
                "probe_physical_diagnostics": probe.get("physical_diagnostics"),
                "probe_reference_tracking": probe.get("reference_tracking"),
            }
        )
        for name in ("status.json", "metrics.jsonl", "extension_status.json"):
            file = folder / name
            if file.exists():
                activity.append(file.stat().st_mtime)
    state = queue.get("state")
    age = max(0, now - max(activity))
    if state == "running" and not running:
        state = "stopped_without_exit_record"
    elif state == "running" and age > 600:
        state = "possibly_stalled"
    plan = path.parent / "plan.json"
    elapsed = (
        max(0, now - plan.stat().st_mtime)
        if running and plan.exists()
        else queue.get("elapsed_s")
    )
    return {
        "state": state,
        "process_alive": running,
        "path": str(path),
        "activity_age_s": age,
        "elapsed_s": elapsed,
        "runs": runs,
        "error": queue.get("error"),
        "world_model": False,
        "scope": "corrected Tier 1 episode 5, seed 42; local input-reference metrics",
        "maximum_transitions_per_candidate": 60002304,
        "official_leaderboard_score": False,
    }


def bounded_test_snapshot(root, relative, *, now, script, alive=None):
    """Capacity/PPO queues have their own lifetimes; old paused work is unrelated."""
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("bounded test status path escapes configured root")
    queue = read_json(path)
    if queue is None:
        return {"state": "status_unavailable", "path": str(path)}
    running = process_alive(queue.get("pid"), script=script) if alive is None else alive
    activity = [path.stat().st_mtime]
    runs = []
    for entry in queue.get("runs", []):
        folder = Path(entry.get("path", "")).resolve()
        if not folder.is_relative_to(path.parent):
            continue
        progress = read_json(folder / "progress.json") or {}
        training = read_json(folder / "status.json") or {}
        metric = last_metric(folder / "metrics.jsonl")
        runs.append(
            {
                "variant": entry.get("variant"),
                "state": entry.get("state"),
                "progress": progress,
                "training": entry.get("training"),
                "peak_gpu_used_mib": training.get("peak_gpu_used_mib"),
                "stop_reason": training.get("stop_reason"),
                "learning_rate": metric.get("ppo/learning_rate"),
                "skipped_update_fraction": metric.get(
                    "ppo/skipped_update_fraction_per_iteration"
                ),
                "evaluations": entry.get("evaluations", []),
            }
        )
        for name in ("progress.json", "metrics.jsonl", "status.json"):
            item = folder / name
            if item.exists():
                activity.append(item.stat().st_mtime)
    age = max(0, now - max(activity))
    state = queue.get("state")
    hold = read_json(path.parent / "queue_hold.json")
    intentional_hold = bool(
        hold
        and hold.get("queue_pid") == queue.get("pid")
        and hold.get("reason") == "isolated_capacity_refinement"
        and process_alive(
            hold.get("refinement_pid"), script="refine_replay_capacity.py"
        )
    )
    if state in ("running", "waiting_for_capacity") and not running:
        state = "stopped_without_exit_record"
    elif state == "running" and intentional_hold:
        state = "waiting_for_capacity_refinement"
    elif state == "running" and age > 600:
        state = "possibly_stalled"
    return {
        "state": state,
        "path": str(path),
        "process_alive": running,
        "activity_age_s": age,
        "intentional_hold": hold if intentional_hold else None,
        "elapsed_s": queue.get("elapsed_s"),
        "active": queue.get("active"),
        "error": queue.get("error"),
        "runs": runs,
        "results": queue.get("results"),
        "largest_passing_default": queue.get("largest_passing_default"),
        "largest_passing_tested": queue.get("largest_passing_tested"),
        "failing_default_upper_bound": queue.get("failing_default_upper_bound"),
        "leaderboard": read_json(path.parent / "leaderboard.json"),
        "official_leaderboard_score": False,
    }


def snapshot(root, *, now=None, alive=None):
    root = Path(root)
    now = time.time() if now is None else now
    logs = root / "logs"
    monitor = read_json(root / "monitor_config.json") or {}
    prefix = monitor.get("log_prefix", "pilot")
    plan_path = root / monitor.get("plan", "project/outputs/policy_pilot_two/plan.json")
    plan = read_json(plan_path)
    if not plan or not plan.get("entries"):
        raise ValueError(f"missing or invalid pilot plan: {plan_path}")
    pid = read_json(logs / f"{prefix}_pid.txt")
    running = (
        process_alive(pid, script=monitor.get("script", "run_pilot.sh"))
        if alive is None
        else alive
    )
    exit_code = read_json(logs / f"{prefix}_exit_code.txt")
    pause = read_json(logs / f"{prefix}_pause.json")
    activity = []
    runs = []
    for entry in plan["entries"]:
        output = plan_path.parent / entry["output"]
        report = read_json(output / "report.json")
        evaluation = read_json(output / "evaluation/evaluation.json")
        metric = last_metric(output / "metrics.jsonl")
        phase = "pending"
        if (output / "planned_config.json").exists():
            phase = "initializing"
        if metric:
            phase = "training"
        if report:
            phase = "evaluating"
        if evaluation:
            phase = "complete"
        milestones = sorted((output / "milestones").glob("*/evaluation.json"))
        milestone = read_json(milestones[-1]) if milestones else None
        runs.append(
            {
                "output": entry["output"],
                "phase": phase,
                "transition_budget": plan.get("config", {}).get("total_transitions"),
                "latest_milestone": {
                    "path": str(milestones[-1]),
                    "evaluation": milestone,
                }
                if milestone
                else None,
                "metrics": {
                    key: (report or metric).get(key)
                    for key in (
                        "transitions",
                        "elapsed_s",
                        "transitions_per_s",
                        "mean_reward",
                        "termination_fraction",
                        "model_ready",
                        "model_updates",
                        "real_updates",
                        "validation_model_state_mse",
                        "stop_reason",
                        "curriculum_stage",
                        "assistance_scale",
                        "first_frame_probability",
                    )
                    if key in (report or metric)
                },
                "evaluation": {
                    key: evaluation.get(key)
                    for key in (
                        "valid_worlds",
                        "ever_terminated",
                        "control_steps",
                        "source_frames",
                        "mean_object_position_error_m_per_world",
                        "physical_diagnostics_per_world",
                        "metric_reference",
                        "official_leaderboard_score",
                        "assistance_scale",
                        "resets",
                        "leaderboard",
                    )
                }
                if evaluation
                else None,
            }
        )
        for file in (
            output / "metrics.jsonl",
            output / "report.json",
            output / "evaluation/evaluation.json",
        ):
            if file.exists():
                activity.append(file.stat().st_mtime)
    if (logs / f"{prefix}.log").exists():
        activity.append((logs / f"{prefix}.log").stat().st_mtime)
    age = max(0, now - max(activity)) if activity else None
    completed = sum(run["phase"] == "complete" for run in runs)
    paused_run = None
    if not running and pause and pause.get("state") == "paused_by_user":
        for run in runs:
            checkpoint = (plan_path.parent / run["output"] / "latest.pt").resolve()
            if (
                str(checkpoint) == str(Path(pause.get("checkpoint", "")).resolve())
                and checkpoint.is_file()
                and run["metrics"].get("stop_reason") == "interrupted"
                and run["metrics"].get("transitions") == pause.get("transitions")
            ):
                paused_run = run
                break
    if paused_run is not None:
        state = "paused_by_user"
        paused_run["phase"] = "paused"
    elif exit_code is not None:
        state = "complete" if exit_code == 0 and completed == len(runs) else "failed"
    elif not running:
        state = "stopped_without_exit_record"
    elif age is not None and age > 600:
        state = "possibly_stalled"
    else:
        state = "running"
    elapsed = None
    try:
        started_text = (logs / f"{prefix}_started_utc.txt").read_text().strip()
        started = datetime.fromisoformat(started_text)
        finished_text = (
            (logs / f"{prefix}_finished_utc.txt").read_text().strip()
            if exit_code is not None
            else None
        )
        end = (
            datetime.fromisoformat(finished_text).timestamp() if finished_text else now
        )
        elapsed = max(0, end - started.timestamp())
    except (OSError, ValueError):
        pass
    return {
        "checked_utc": datetime.fromtimestamp(now, UTC).isoformat(),
        "state": state,
        "pause": pause if paused_run is not None else None,
        "queue": prefix,
        "instance_power_policy": "keep_on_and_notify_user",
        "pid": pid,
        "queue_process_alive": running,
        "exit_code": exit_code,
        "completed_evaluations": completed,
        "planned_runs": len(runs),
        "activity_age_s": age,
        "queue_wall_seconds": elapsed,
        "estimated_queue_cost_cny_at_2_per_hour": elapsed * 2 / 3600
        if elapsed
        else None,
        "cost_excludes_prior_setup_and_powered_on_time_after_queue": True,
        "free_data_disk_bytes": shutil.disk_usage(root).free,
        "runs": runs,
        "research_checkpoint": read_json(root / "research/weights/manifest.json"),
        "research_sources": read_json(root / "research/repositories.json"),
        "diagnostics": diagnostic_snapshot(root, monitor["diagnostics"], now=now)
        if monitor.get("diagnostics")
        else None,
        "candidate_training": candidate_snapshot(root, monitor["candidates"], now=now)
        if monitor.get("candidates")
        else None,
        "replay_capacity_tests": bounded_test_snapshot(
            root, monitor["replay_capacity"], now=now, script="replay_capacity_sweep.py"
        )
        if monitor.get("replay_capacity")
        else None,
        "dex_ppo_tests": bounded_test_snapshot(
            root, monitor["dex_ppo"], now=now, script="run_dex_ppo_tests.py"
        )
        if monitor.get("dex_ppo")
        else None,
        "dex_ppo_full": bounded_test_snapshot(
            root, monitor["dex_ppo_full"], now=now, script="run_dex_ppo_full.py"
        )
        if monitor.get("dex_ppo_full")
        else None,
        "replay_capacity_refinement": bounded_test_snapshot(
            root,
            monitor["replay_refinement"],
            now=now,
            script="refine_replay_capacity.py",
        )
        if monitor.get("replay_refinement")
        else None,
        "log_tail_on_error": tail(logs / f"{prefix}.log", 6000)
        if state not in ("running", "complete", "paused_by_user")
        else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/root/autodl-tmp/v2d"))
    args = parser.parse_args()
    print(json.dumps(snapshot(args.root), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

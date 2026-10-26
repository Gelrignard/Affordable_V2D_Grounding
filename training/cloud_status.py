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

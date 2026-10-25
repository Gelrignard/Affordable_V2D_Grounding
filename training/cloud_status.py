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


def process_alive(pid, proc=Path("/proc")):
    try:
        # Match the launch script too: a recycled PID is not our training queue.
        command = (proc / str(pid) / "cmdline").read_bytes()
        return b"run_pilot.sh" in command
    except OSError:
        return False


def snapshot(root, *, now=None, alive=None):
    root = Path(root)
    now = time.time() if now is None else now
    logs = root / "logs"
    plan_path = root / "project/outputs/policy_pilot_two/plan.json"
    plan = read_json(plan_path)
    if not plan or not plan.get("entries"):
        raise ValueError(f"missing or invalid pilot plan: {plan_path}")
    pid = read_json(logs / "pilot_pid.txt")
    running = process_alive(pid) if alive is None else alive
    exit_code = read_json(logs / "pilot_exit_code.txt")
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
        runs.append(
            {
                "output": entry["output"],
                "phase": phase,
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
    if (logs / "pilot.log").exists():
        activity.append((logs / "pilot.log").stat().st_mtime)
    age = max(0, now - max(activity)) if activity else None
    completed = sum(run["phase"] == "complete" for run in runs)
    if exit_code is not None:
        state = "complete" if exit_code == 0 and completed == len(runs) else "failed"
    elif not running:
        state = "stopped_without_exit_record"
    elif age is not None and age > 600:
        state = "possibly_stalled"
    else:
        state = "running"
    elapsed = None
    try:
        started_text = (logs / "pilot_started_utc.txt").read_text().strip()
        started = datetime.fromisoformat(started_text)
        finished_text = (
            (logs / "pilot_finished_utc.txt").read_text().strip()
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
        "log_tail_on_error": tail(logs / "pilot.log", 6000)
        if state not in ("running", "complete")
        else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/root/autodl-tmp/v2d"))
    args = parser.parse_args()
    print(json.dumps(snapshot(args.root), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

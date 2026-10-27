"""Detached, bounded native-bootstrap then precision-refinement trial queue.

Every episode starts a fresh native learner. Evaluations and training share one
ledger; comparisons require agent diagnosis and never promote a checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))

from experiments.overnight_guard import (
    exclusive_lock, ledger_budget_policy, make_budget_policy, parse_deadline,
    read_json, remaining_budget, utc_now, write_json,
)
from experiments.overnight_policy import build_config, build_fresh_config, checkpoint_pair, transition_budget

EVALUATION_SECONDS = 600
OUTER_RESERVE_SECONDS = 60
BOOTSTRAP_STEPS = 349_999_104
REQUIRED_SOURCES = {
    "experiments/two_episode_queue.py", "experiments/overnight_policy.py",
    "experiments/overnight_guard.py", "experiments/evaluate_overnight_policy.py",
    "training/overnight_refinement.py",
}


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def immutable_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def under(path, parent):
    value, parent = Path(path).resolve(), Path(parent).resolve()
    if value == parent or not value.is_relative_to(parent):
        raise ValueError(f"Expected a child path inside {parent}: {value}")
    return value


def budget_policy(plan):
    return make_budget_policy(plan["total_budget_seconds"], plan["candidate_training_budget_seconds"],
                              plan["maximum_candidates"])


def verify_sources(plan):
    """Archive extraction is authenticated by its deployment sentinel and hashes."""
    if (SOURCE / "SOURCE_COMMIT").read_text().strip() != plan["source_commit"]:
        raise ValueError("Source archive commit sentinel differs from the plan")
    archive_hash = plan.get("source_archive_sha256", "")
    if len(archive_hash) != 64 or any(c not in "0123456789abcdef" for c in archive_hash):
        raise ValueError("Plan must record the deployment-verified source archive SHA256")
    hashes = plan["source_file_sha256"]
    if not REQUIRED_SOURCES.issubset(hashes):
        raise ValueError("Plan omits a required execution source hash")
    for name, expected in hashes.items():
        if file_hash(under(SOURCE / name, SOURCE)) != expected:
            raise ValueError(f"Source hash mismatch: {name}")


def verify_episode_inputs(episode):
    """Check portable job integrity and permit only asset-path rebasing in runtime input."""
    import pyarrow.parquet as pq

    job = Path(episode["job"]).resolve()
    metadata = read_json(job / "job.json")
    for name in ("episode_index", "tier", "sequence_id", "source_frames"):
        if not metadata or metadata.get(name) != episode[name]:
            raise ValueError(f"Job identity mismatch: {name}")
    if metadata.get("variant") != "corrected":
        raise ValueError("This trial requires the agreed corrected input variant")
    files = metadata["files"]
    if not {"motion.parquet", "assets/object.obj", "assets/object.urdf", "assets/support.usda"}.issubset(files):
        raise ValueError("Incomplete object/motion job manifest")
    for name, expected in files.items():
        if file_hash(under(job / name, job)) != expected:
            raise ValueError(f"Input manifest hash mismatch: {name}")
    config_path = Path(episode["bootstrap_config"]).resolve()
    if file_hash(config_path) != episode["bootstrap_config_sha256"]:
        raise ValueError("Bootstrap configuration hash mismatch")
    config = read_json(config_path)
    motion = Path(config["task"]["parquet"]).resolve()
    source_rows = pq.read_table(job / "motion.parquet").to_pylist()
    runtime_rows = pq.read_table(motion).to_pylist()
    if len(source_rows) != 1 or len(runtime_rows) != 1:
        raise ValueError("Expected one motion row per episode")
    source_row, runtime_row = source_rows[0], runtime_rows[0]
    for name in set(source_row) | set(runtime_row):
        if name not in ("object_mesh_paths", "object_urdf_paths") and source_row.get(name) != runtime_row.get(name):
            raise ValueError(f"Materialized input changes source trajectory: {name}")
    if (runtime_row.get("sequence_id") != episode["sequence_id"]
            or len(runtime_row["robot_joint_positions"]) != episode["source_frames"]
            or len(runtime_row["object_body_names"]) != 1):
        raise ValueError("Materialized sequence identity/frame/object count differs")
    for name, asset in (("object_mesh_paths", "assets/object.obj"), ("object_urdf_paths", "assets/object.urdf")):
        paths = runtime_row[name]
        if len(paths) != 1 or not Path(paths[0]).is_absolute() or file_hash(paths[0]) != files[asset]:
            raise ValueError(f"Materialized object asset differs: {name}")
    verified = {"job": str(job), "job_manifest_sha256": file_hash(job / "job.json"),
                "job_file_sha256": files, "bootstrap_config_sha256": file_hash(config_path),
                "runtime_motion": str(motion), "runtime_motion_sha256": file_hash(motion)}
    if "input_manifest" in episode or "input_manifest_sha256" in episode:
        manifest_path = Path(episode["input_manifest"])
        if file_hash(manifest_path) != episode["input_manifest_sha256"]:
            raise ValueError("Prepared input manifest hash mismatch")
        prepared = read_json(manifest_path)
        expected = {"job_manifest_sha256": verified["job_manifest_sha256"], "source_file_sha256": files,
                    "source_motion_sha256": files["motion.parquet"],
                    "runtime_motion_sha256": verified["runtime_motion_sha256"],
                    "sequence_id": episode["sequence_id"], "source_frames": episode["source_frames"],
                    "support": config["scene"]["support"], "source_frame_playback": True,
                    "estimated_force_guidance_used": False}
        if any(prepared.get(name) != value for name, value in expected.items()):
            raise ValueError("Prepared input manifest differs from the actual job/configuration")
        verified["input_manifest_sha256"] = file_hash(manifest_path)
    return verified


def validate_plan(plan):
    if plan.get("schema") != "v2d_two_episode_trial_v1":
        raise ValueError("Unsupported trial plan schema")
    policy = budget_policy(plan)
    if (policy["total_budget_seconds"] > 23400 or policy["candidate_training_budget_seconds"] > 10800
            or policy["maximum_candidates"] != 2):
        raise ValueError("Trial exceeds the authorized two-episode budget")
    if plan["bootstrap_transitions"] != BOOTSTRAP_STEPS or plan["refinement_transitions"] != 60_000_000:
        raise ValueError("Trial transition schedule differs from the agreed method")
    if not 60 <= plan["bootstrap_wall_seconds"] <= 9000 or not 60 <= plan["refinement_wall_seconds"] <= 1800:
        raise ValueError("Trial wall caps exceed the authorized phase limits")
    if parse_deadline(plan["deadline_utc"]) - time.time() < 660:
        raise ValueError("Insufficient calendar time for training plus evaluation")
    episodes = plan["episodes"]
    if len(episodes) != 2 or {(e["tier"], e["episode_index"]) for e in episodes} != {(1, 16), (1, 18)}:
        raise ValueError("Trial must contain exactly corrected Tier 1 episodes 16 and 18")
    root = Path(plan["root"]).resolve()
    reserved = []
    for episode in episodes:
        candidate = episode["candidate_id"]
        if candidate != f"episode_{episode['episode_index']:06d}":
            raise ValueError("Candidate ID must be stable across an episode's stages")
        for phase in ("bootstrap", "refinement"):
            name = f"{candidate}_{phase}"
            output = under(episode[phase + "_output"], root)
            reserved.extend([output, output.with_name(output.name + "_evaluation"),
                             root / "supervision" / name, root / "supervision" / (name + "_evaluation"),
                             root / (name + "_planned_config.json"), root / (name + "_manifest.json")])
        reserved.append(root / (candidate + "_comparison.json"))
    if len(set(reserved)) != len(reserved) or any(path.exists() for path in reserved):
        raise ValueError("Run paths overlap or already have artifacts; preserve them and use a new root")
    return policy


def training_caps(plan, ledger, candidate, requested_seconds, *, now=None):
    """Reserve evaluation and outer supervision time inside all applicable budgets."""
    now = time.time() if now is None else now
    outer = int(min(requested_seconds + OUTER_RESERVE_SECONDS,
                    remaining_budget(ledger, candidate, requested_seconds + OUTER_RESERVE_SECONDS),
                    remaining_budget(ledger, candidate, float("inf"), kind="evaluation") - EVALUATION_SECONDS,
                    parse_deadline(plan["deadline_utc"]) - now - EVALUATION_SECONDS))
    child = outer - OUTER_RESERVE_SECONDS
    if child < 60:
        raise RuntimeError("Insufficient training budget after reserving supervision and evaluation")
    return child, outer


def guarded(plan, candidate, name, kind, seconds, command):
    root = Path(plan["root"])
    invocation = [sys.executable, str(SOURCE / "experiments/overnight_guard.py"),
                  "--ledger", str(root / "ledger.json"), "--name", name, "--candidate", candidate,
                  "--kind", kind, "--max-seconds", str(seconds), "--deadline-utc", plan["deadline_utc"],
                  "--wait-queue", plan["wait_queue"], "--wait-seconds", "60", "--cwd", str(SOURCE),
                  "--output", str(root / "supervision" / name)]
    for name in ("total_budget_seconds", "candidate_training_budget_seconds", "maximum_candidates"):
        invocation.extend(["--" + name.replace("_", "-"), str(plan[name])])
    subprocess.run([*invocation, "--", *command], check=True)


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
    expected = {"checkpoint_sha256": file_hash(policy), "environment_steps": steps,
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


def run(plan_path):
    plan_path = Path(plan_path).resolve()
    plan = read_json(plan_path)
    root = Path(plan["root"]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    status_path = root / "queue_status.json"
    with exclusive_lock(root / "queue.lock"):
        if status_path.exists() or (root / "ledger.json").exists():
            raise RuntimeError("Queue already has status/ledger; automatic restart is refused")
        status = {"schema": "v2d_two_episode_queue_status_v1", "state": "verifying", "pid": os.getpid(),
                  "plan": str(plan_path), "plan_sha256": file_hash(plan_path), "started_utc": utc_now(),
                  "active_episode": None, "phase": "preflight", "completed": [], "automatic_promotion": False}
        write_json(status_path, status)

        def update(episode, phase):
            status.update(state="running", active_episode=episode["episode_index"], phase=phase, updated_utc=utc_now())
            write_json(status_path, status)

        try:
            policy = validate_plan(plan)
            verify_sources(plan)
            inputs = {e["candidate_id"]: verify_episode_inputs(e) for e in plan["episodes"]}
            bootstrap_configs = {e["candidate_id"]: build_fresh_config(
                read_json(e["bootstrap_config"]), output=Path(e["bootstrap_output"]).resolve(),
                end=BOOTSTRAP_STEPS, checkpoint_interval=50_000_000) for e in plan["episodes"]}
            immutable_json(root / "queue_preflight.json", {"plan_sha256": file_hash(plan_path),
                           "source_commit": plan["source_commit"], "source_archive_sha256": plan["source_archive_sha256"],
                           "source_file_sha256": plan["source_file_sha256"], "inputs": inputs, "verified_utc": utc_now()})
            write_json(root / "ledger.json", {"created_utc": utc_now(), "jobs": [], "budget_policy": policy})

            def reverify(episode):
                if file_hash(plan_path) != status["plan_sha256"]:
                    raise RuntimeError("Trial plan changed while queue was running")
                verify_sources(plan)
                if verify_episode_inputs(episode) != inputs[episode["candidate_id"]]:
                    raise RuntimeError("Trial inputs changed since preflight")
                if ledger_budget_policy(read_json(root / "ledger.json")) != policy:
                    raise RuntimeError("Ledger budget policy changed")

            def train(episode, phase, config, *, checkpoint=None, start=0, additional):
                reverify(episode)
                candidate, output = episode["candidate_id"], Path(episode[phase + "_output"]).resolve()
                name = candidate + "_" + phase
                child, outer = training_caps(plan, read_json(root / "ledger.json"), candidate, plan[phase + "_wall_seconds"])
                planned_path = root / (name + "_planned_config.json")
                immutable_json(planned_path, config)
                manifest = {"source_commit": plan["source_commit"], "source_file_sha256": plan["source_file_sha256"],
                            "input": inputs[candidate], "candidate": candidate, "phase": phase,
                            "effective_config": str(planned_path), "effective_config_sha256": file_hash(planned_path),
                            "source_steps": start, "additional_transitions": additional,
                            "training_seconds_cap": child, "outer_seconds_cap": outer,
                            "evaluation_seconds_cap": EVALUATION_SECONDS, "deadline_utc": plan["deadline_utc"],
                            "fresh_replay": True, "automatic_promotion": False, "prepared_utc": utc_now()}
                source_args = ["--fresh-config", episode["bootstrap_config"]]
                if checkpoint is not None:
                    state, actor, _ = checkpoint_pair(checkpoint)
                    manifest.update(parent_state=str(state), parent_state_sha256=file_hash(state),
                                    parent_policy=str(actor), parent_policy_sha256=file_hash(actor))
                    parent_evaluation = Path(episode["bootstrap_output"]).with_name(Path(episode["bootstrap_output"]).name + "_evaluation") / "evaluation.json"
                    manifest.update(parent_evaluation=str(parent_evaluation), parent_evaluation_sha256=file_hash(parent_evaluation))
                    source_args = ["--checkpoint", str(state)]
                immutable_json(root / (name + "_manifest.json"), manifest)
                update(episode, phase)
                guarded(plan, candidate, name, "training", outer, [sys.executable,
                        str(SOURCE / "experiments/overnight_policy.py"), "train", "--runtime", plan["runtime"],
                        *source_args, "--output", str(output), "--phase", "refine",
                        "--additional-transitions", str(additional), "--max-seconds", str(child),
                        "--deadline-utc", plan["deadline_utc"], "--checkpoint-interval", "50000000" if phase == "bootstrap" else "30000000"])
                pair = completed_checkpoint(output, config, required_steps=BOOTSTRAP_STEPS if phase == "bootstrap" else None,
                                            minimum_steps=start)
                if phase == "bootstrap":
                    initial = read_json(output / "initial_checkpoint.json", {})
                    _, _, initial_metadata = checkpoint_pair(initial["state"])
                    if int(initial_metadata["environment_steps"]) != 0:
                        raise RuntimeError("Fresh native initialization checkpoint is missing or not step zero")
                return pair

            def evaluate(episode, phase, actor, steps):
                reverify(episode)
                candidate = episode["candidate_id"]
                output = Path(episode[phase + "_output"]).resolve()
                destination = output.with_name(output.name + "_evaluation")
                available = min(remaining_budget(read_json(root / "ledger.json"), candidate, EVALUATION_SECONDS,
                                                 kind="evaluation"), parse_deadline(plan["deadline_utc"]) - time.time())
                if available < EVALUATION_SECONDS:
                    raise RuntimeError("Reserved evaluation budget is no longer available")
                update(episode, phase + "_evaluation")
                guarded(plan, candidate, candidate + "_" + phase + "_evaluation", "evaluation", EVALUATION_SECONDS,
                        [sys.executable, str(SOURCE / "experiments/evaluate_overnight_policy.py"),
                         "--runtime", plan["runtime"], "--checkpoint", str(actor), "--output", str(destination),
                         "--worlds", "16", "--tier", str(episode["tier"]), "--episode", str(episode["episode_index"]),
                         "--variant", candidate + "_" + phase])
                report = read_json(destination / "evaluation.json", {})
                if report.get("step_count") != episode["source_frames"]:
                    raise RuntimeError("Evaluation did not retain the complete source-frame sequence")
                return destination, report, evaluation_scores(report, actor, steps)

            for episode in plan["episodes"]:
                output = Path(episode["bootstrap_output"]).resolve()
                native_config = bootstrap_configs[episode["candidate_id"]]
                state, actor, metadata = train(episode, "bootstrap", native_config, additional=BOOTSTRAP_STEPS)
                parent_path, parent_report, parent_scores = evaluate(episode, "bootstrap", actor, BOOTSTRAP_STEPS)
                end, additional = transition_budget(BOOTSTRAP_STEPS, plan["refinement_transitions"])
                refine_config = build_config(json.loads(metadata["resolved_config_json"]), state=state,
                                             output=Path(episode["refinement_output"]).resolve(),
                                             start=BOOTSTRAP_STEPS, end=end, phase="refine", checkpoint_interval=30_000_000)
                _, actor, metadata = train(episode, "refinement", refine_config, checkpoint=state,
                                           start=BOOTSTRAP_STEPS, additional=additional)
                steps = int(metadata["environment_steps"])
                result_path, result_report, result_scores = evaluate(episode, "refinement", actor, steps)
                comparison_path = root / (episode["candidate_id"] + "_comparison.json")
                immutable_json(comparison_path, {"source_commit": plan["source_commit"],
                               "episode_index": episode["episode_index"], "tier": episode["tier"],
                               "parent": parent_scores, "candidate": result_scores,
                               "parent_evaluation": str(parent_path / "evaluation.json"),
                               "candidate_evaluation": str(result_path / "evaluation.json"),
                               "parent_phase_metrics": parent_report["phase_metrics"], "phase_metrics": result_report["phase_metrics"],
                               "refinement_environment_steps": steps, "refinement_transition_target": end,
                               "refinement_transition_cap_reached": steps == end,
                               "needs_agent_diagnosis": True, "automatic_promotion": False})
                status["completed"].append({"episode_index": episode["episode_index"],
                                            "comparison": str(comparison_path), "needs_agent_diagnosis": True})
                write_json(status_path, status)
            status.update(state="awaiting_agent_diagnosis", active_episode=None, phase="complete", completed_utc=utc_now())
            write_json(status_path, status)
        except BaseException as exc:
            status.update(state="needs_attention", error=f"{type(exc).__name__}: {exc}", updated_utc=utc_now())
            write_json(status_path, status)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    run(parser.parse_args().plan)


if __name__ == "__main__":
    main()

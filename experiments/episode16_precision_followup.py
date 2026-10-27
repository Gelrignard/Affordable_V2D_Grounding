"""One authorized unchanged 30M continuation, using the pinned trial helpers.

This orchestration file does not alter the learner, plan, budget, or input data.
It requires both finished screens and the episode16 midpoint comparison.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


def run(root: Path):
    plan_path = root / "plan_retry1.json"
    plan = json.loads(plan_path.read_text())
    source = root / "sources" / plan["source_commit"]
    sys.path.insert(0, str(source))
    from experiments.overnight_guard import exclusive_lock, pid_live, read_json, utc_now, write_json
    from experiments.overnight_policy import build_config, checkpoint_pair, transition_budget
    from experiments.two_episode_queue import (
        SOURCE, completed_checkpoint, evaluation_scores, file_hash, guarded,
        immutable_json, training_caps, verify_episode_inputs, verify_sources,
    )

    name = "episode_000016_precision_extension1"
    output = root / name
    status_path = root / (name + "_status.json")
    with exclusive_lock(root / "queue.lock"):
        queue = read_json(root / "queue_status.json")
        if (queue["state"] != "awaiting_agent_diagnosis" or queue["phase"] != "complete"
                or pid_live(queue["pid"])):
            raise RuntimeError("The original queue must have completed and exited")
        reserved = [output, status_path, root / (name + "_evaluation"),
                    root / (name + "_planned_config.json"), root / (name + "_manifest.json"),
                    root / "supervision" / name, root / "supervision" / (name + "_evaluation")]
        if any(path.exists() for path in reserved):
            raise RuntimeError("Follow-up already exists; automatic restart refused")
        verify_sources(plan)
        episode = next(e for e in plan["episodes"] if e["episode_index"] == 16)
        inputs = verify_episode_inputs(episode)
        if plan["root"] != str(root) or episode["candidate_id"] != "episode_000016":
            raise RuntimeError("Unexpected trial or candidate")

        state, actor, metadata = checkpoint_pair(
            root / "episode_000016_precision_screen/checkpoints/state_409997312.safetensors")
        start = int(metadata["environment_steps"])
        parent_path = root / "episode_000016_precision_screen_evaluation/evaluation.json"
        parent_scores = evaluation_scores(read_json(parent_path), actor, start)
        _, midpoint_actor, midpoint_metadata = checkpoint_pair(
            root / "episode_000016_precision_screen/checkpoints/state_390000640.safetensors")
        if start != 409997312 or int(midpoint_metadata["environment_steps"]) != 390000640:
            raise RuntimeError("Unexpected parent or midpoint training clock")
        midpoint_path = root / "episode_000016_precision_midpoint_evaluation/evaluation.json"
        midpoint_scores = evaluation_scores(read_json(midpoint_path), midpoint_actor,
                                            int(midpoint_metadata["environment_steps"]))
        if not (parent_scores["add_auc"] > midpoint_scores["add_auc"]
                and parent_scores["mppe_cm"] < midpoint_scores["mppe_cm"]):
            raise RuntimeError("Reconsider parent selection after midpoint evaluation")

        child, outer = training_caps(plan, read_json(root / "ledger.json"),
                                     episode["candidate_id"], 1500)
        end, additional = transition_budget(start, 30_000_000)
        config = build_config(json.loads(metadata["resolved_config_json"]), state=state,
                              output=output, start=start, end=end, phase="refine",
                              checkpoint_interval=30_000_000)
        if config["overnight"]["refinement_start_steps"] != 349999104:
            raise RuntimeError("Original refinement clock must remain unchanged")
        if Path(config["task"]["parquet"]).resolve() != Path(inputs["runtime_motion"]).resolve():
            raise RuntimeError("Checkpoint input differs from verified episode16 input")
        config_path = root / (name + "_planned_config.json")
        immutable_json(config_path, config)
        manifest = {
            "source_commit": plan["source_commit"], "source_file_sha256": plan["source_file_sha256"],
            "source_archive_sha256": plan["source_archive_sha256"],
            "orchestration_sha256": file_hash(__file__), "plan_sha256": file_hash(plan_path),
            "inputs": inputs, "parent_state": str(state), "parent_state_sha256": file_hash(state),
            "parent_policy": str(actor), "parent_policy_sha256": file_hash(actor),
            "parent_evaluation": str(parent_path), "parent_evaluation_sha256": file_hash(parent_path),
            "midpoint_evaluation_sha256": file_hash(midpoint_path),
            "planned_config_sha256": file_hash(config_path),
            "initial_steps": start, "additional_transition_cap": additional,
            "child_seconds": child, "outer_seconds": outer, "deadline_utc": plan["deadline_utc"],
            "decision": "Single unchanged continuation after endpoint beats saved midpoint; no sweep",
        }
        immutable_json(root / (name + "_manifest.json"), manifest)
        status = {"pid": os.getpid(), "state": "training", "started_utc": utc_now(), **manifest}
        write_json(status_path, status)
        try:
            guarded(plan, episode["candidate_id"], name, "training", outer, [
                sys.executable, str(SOURCE / "experiments/overnight_policy.py"), "train",
                "--runtime", plan["runtime"], "--checkpoint", str(state), "--output", str(output),
                "--phase", "refine", "--additional-transitions", str(additional),
                "--max-seconds", str(child), "--deadline-utc", plan["deadline_utc"],
                "--checkpoint-interval", "30000000"])
            _, actor, metadata = completed_checkpoint(output, config, minimum_steps=start)
            status.update(state="evaluating", updated_utc=utc_now())
            write_json(status_path, status)
            evaluation = root / (name + "_evaluation")
            guarded(plan, episode["candidate_id"], name + "_evaluation", "evaluation", 600, [
                sys.executable, str(SOURCE / "experiments/evaluate_overnight_policy.py"),
                "--runtime", plan["runtime"], "--checkpoint", str(actor), "--output", str(evaluation),
                "--worlds", "16", "--tier", "1", "--episode", "16", "--variant", name])
            report = read_json(evaluation / "evaluation.json")
            if report["step_count"] != 477:
                raise RuntimeError("Unexpected episode16 evaluation length")
            scores = evaluation_scores(report, actor, int(metadata["environment_steps"]))
            status.update(state="awaiting_agent_diagnosis", scores=scores, finished_utc=utc_now())
            write_json(status_path, status)
        except Exception as error:
            status.update(state="needs_attention", error=str(error), finished_utc=utc_now())
            write_json(status_path, status)
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    run(parser.parse_args().root.resolve())

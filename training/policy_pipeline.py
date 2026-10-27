"""Automatic, bounded native FlashSAC acquisition and candidate2 refinement.

Run with the pinned runtime's Python: ``python -m training.policy_pipeline
--job JOB --runtime RUNTIME --output OUTPUT``. ``--prepare-only`` (or
``--dry-run``) performs the same CPU preflight without launching GPU work.

The native 350M phase is policy acquisition, not a separately evaluated baseline.
Refinement runs once for up to 90M additional transitions. Native saves occur at
absolute 10M intervals; evaluate final, latest <=60M and latest <=40M additional
checkpoints (one 2048-transition collection tolerance), deduplicated, at most
three. Rank input-reference MP-SR, SP-SR, AUC,
then lower MPPE; ties prefer fewer transitions. This is checkpoint selection,
not a parameter sweep. Every candidate includes all 16 uninterrupted worlds.
Submission export must use fixed world 0, never a selected successful world.

Resume verifies completed stages and their hashes. It never relaunches a partial
or failed stage, silently changes a budget, or takes over a running worker.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET

from experiments import overnight_guard as guard
from experiments.overnight_policy import (
    OPTIMIZER_HORIZON, WORLDS, build_config, build_fresh_config,
    checkpoint_pair, transition_budget,
)
from experiments.two_episode_queue import completed_checkpoint, evaluation_scores
from training import RUNTIME_COMMIT

ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP_STEPS = 349_999_104
REFINEMENT_STEPS = 90_000_000
CHECKPOINT_INTERVAL = 10_000_000
EVALUATION_SECONDS = 300
EVALUATOR_CHILD_SECONDS = 240
SUPERVISION_RESERVE = 60
SELECTION_RULE = ["maniptrans_sr:descending", "spider_sr:descending", "add_auc:descending",
                  "mppe_cm:ascending", "environment_steps:ascending", "checkpoint_sha256:ascending"]


@contextlib.contextmanager
def output_lock(path):
    """CPU preparation also runs on Windows; actual GPU execution uses flock."""
    if os.name == "posix":
        with guard.exclusive_lock(path):
            yield
        return
    import msvcrt
    with Path(path).open("a+b") as handle:
        if not handle.seek(0, 2):
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise RuntimeError("Another worker owns this pipeline output") from exc
        try:
            yield
        finally:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def immutable_json(path, data):
    path = Path(path)
    if path.exists():
        if guard.read_json(path) != data:
            raise ValueError(f"Immutable record differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(data, indent=2, allow_nan=False) + "\n")


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def source_identity():
    """Output files may be untracked; execution code must match its recorded HEAD."""
    names = git(ROOT, "ls-files", "--", "training", "experiments").splitlines()
    names = [name for name in names if Path(name).suffix in {".py", ".json", ".yaml", ".yml", ".toml"}]
    untracked = git(ROOT, "ls-files", "--others", "--exclude-standard", "--", "training", "experiments").splitlines()
    if any(Path(name).suffix in {".py", ".json", ".yaml", ".yml", ".toml"} for name in untracked):
        raise ValueError("Untracked execution sources exist; commit or remove them before running")
    if "training/policy_pipeline.py" not in names:
        raise ValueError("Run the pipeline from a committed clone")
    if git(ROOT, "diff", "HEAD", "--name-only", "--", *names):
        raise ValueError("Tracked execution sources differ from HEAD; commit before running")
    return {"commit": git(ROOT, "rev-parse", "HEAD"), "files": {name: sha(ROOT / name) for name in names}}


def runtime_identity(runtime):
    if git(runtime, "rev-parse", "HEAD") != RUNTIME_COMMIT:
        raise ValueError(f"Runtime must be pinned to {RUNTIME_COMMIT}")
    if git(runtime, "status", "--porcelain", "--untracked-files=no"):
        raise ValueError("Pinned runtime has tracked modifications")
    return RUNTIME_COMMIT


def job_identity(job):
    job = Path(job).resolve()
    data = guard.read_json(job / "job.json")
    if (not data or data.get("schema") != "v2d_policy_job_v1"
            or type(data.get("tier")) is not int or data["tier"] not in (1, 2, 3)
            or type(data.get("episode_index")) is not int or not 0 <= data["episode_index"] < 30
            or data.get("runtime_commit") != RUNTIME_COMMIT
            or data.get("model_scope") != "single_rigid_object_g1_dex3"):
        raise ValueError("Expected a Tier 1/2/3 single-rigid-object G1/Dex3 portable job")
    required = {"motion.parquet", "reference.npz", "guidance.npz", "geometry.npz",
                "assets/object.obj", "assets/object.urdf", "assets/support.usda"}
    if not required.issubset(data.get("files", {})):
        raise ValueError("Portable job is missing required motion/assets")
    for name, digest in data["files"].items():
        path = (job / name).resolve()
        if not path.is_relative_to(job) or not path.is_file() or sha(path) != digest:
            raise ValueError(f"Portable input hash/path mismatch: {name}")
    return {"job": str(job), "job_sha256": sha(job / "job.json"), "manifest": data}


def validate_motion(row, job, runtime):
    """CPU-only schema, joint and finite trajectory validation before any allocation."""
    import numpy as np
    metadata = job["manifest"]
    frames = metadata["source_frames"]
    if type(frames) is not int or frames < 2 or not math.isfinite(metadata["source_fps"]) or metadata["source_fps"] <= 0:
        raise ValueError("Invalid source frame count/rate")
    sequence = metadata["sequence_id"]
    if (not isinstance(sequence, str) or not sequence or "/" in sequence or "\\" in sequence
            or sequence in (".", "..") or row.get("sequence_id") != sequence):
        raise ValueError("Invalid or mismatched sequence identity")
    if row.get("schema_version") != "motion_v1" or row.get("robot_name") != "g1" or len(row.get("object_body_names", [])) != 1:
        raise ValueError("Expected one motion_v1 G1 object")
    urdf = Path(runtime) / "robotic_grounding/flash_chord/src/flash_chord/assets/urdfs/g1/main_with_hand.urdf"
    names = {e.attrib["name"] for e in ET.parse(urdf).findall("joint") if e.attrib.get("type") not in ("fixed", "floating")}
    if len(row.get("robot_joint_names", [])) != 43 or set(row["robot_joint_names"]) != names:
        raise ValueError("Motion joint names do not match pinned G1/Dex3")
    canonical = hashlib.sha256(urdf.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    if metadata.get("urdf_canonical_sha256") != canonical:
        raise ValueError("Portable job and pinned robot URDF differ")
    for key, shape in [("robot_joint_positions", (frames, 43)), ("robot_root_position", (frames, 3)),
                       ("robot_root_wxyz", (frames, 4)), ("object_body_position", (frames, 1, 3)),
                       ("object_body_wxyz", (frames, 1, 4))]:
        value = np.asarray(row.get(key), dtype=float)
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"Invalid motion array: {key}")
        if key.endswith("wxyz") and not np.allclose(np.linalg.norm(value, axis=-1), 1, atol=1e-3):
            raise ValueError(f"Non-unit motion quaternion: {key}")
    if not math.isclose(float(row["fps"]), metadata["source_fps"], rel_tol=1e-6):
        raise ValueError("Source fps differs from job manifest")
    return urdf


def validate_parent(path, fresh, job):
    """Accept only exact existing task/assets and compatible native training state."""
    import pyarrow.parquet as pq
    state, policy, metadata = checkpoint_pair(path)
    saved = json.loads(metadata["resolved_config_json"])
    steps = int(metadata["environment_steps"])
    if steps < BOOTSTRAP_STEPS or steps % WORLDS:
        raise ValueError("Parent must have completed native acquisition and aligned counters")
    for key in ("sim", "scene", "collision", "embodiment", "action", "observation"):
        if saved.get(key) != fresh.get(key):
            raise ValueError(f"Parent changes native physical/policy contract: {key}")
    for key in ("network", "seed", "world_count"):
        if saved["training"].get(key) != fresh["training"].get(key):
            raise ValueError(f"Parent training contract differs: {key}")
    for key in ("motion_speed", "source_frame_playback", "motion_start_frame", "motion_end_frame"):
        if saved["task"].get(key) != fresh["task"].get(key):
            raise ValueError(f"Parent playback contract differs: {key}")
    original = Path(saved["task"]["parquet"])
    if not original.is_file():
        raise ValueError("Parent's original motion is unavailable; unverified path remapping is refused")
    old = pq.ParquetFile(original).read().to_pylist()
    new = pq.ParquetFile(Path(job["job"]) / "motion.parquet").read().to_pylist()
    if len(old) != 1 or len(new) != 1:
        raise ValueError("Parent motion must contain exactly one sequence")
    for key in set(old[0]) | set(new[0]):
        if key not in {"object_mesh_paths", "object_urdf_paths"} and old[0].get(key) != new[0].get(key):
            raise ValueError(f"Parent refers to a different trajectory: {key}")
    assets = {}
    for key, name in [("object_mesh_paths", "assets/object.obj"), ("object_urdf_paths", "assets/object.urdf")]:
        paths = old[0][key]
        if len(paths) != 1 or sha(paths[0]) != job["manifest"]["files"][name]:
            raise ValueError(f"Parent object asset differs: {key}")
        assets[str(Path(paths[0]).resolve())] = sha(paths[0])
    support = original.parents[3] / "reconstructed_stage" / (job["manifest"]["sequence_id"] + "_support.usda")
    if fresh["scene"]["support"] and (not support.is_file() or sha(support) != job["manifest"]["files"]["assets/support.usda"]):
        raise ValueError("Parent support geometry is unavailable or differs")
    if fresh["scene"]["support"]:
        assets[str(support.resolve())] = sha(support)
    return {"state": str(state), "policy": str(policy), "state_sha256": sha(state),
            "checkpoint_sha256": sha(policy), "environment_steps": steps,
            "resolved_config": saved, "original_motion": str(original), "original_motion_sha256": sha(original),
            "original_assets": assets}


def prepare(request):
    """Executed in a subprocess with CUDA hidden and JAX_PLATFORMS=cpu."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or os.environ.get("JAX_PLATFORMS") != "cpu":
        raise ValueError("Preparation must use the CPU-only subprocess")
    import pyarrow as pa
    import pyarrow.parquet as pq
    from pxr import Usd
    runtime, output = Path(request["runtime"]), Path(request["output"])
    sys.path.insert(0, str(runtime / "robotic_grounding/flash_chord/src"))
    from flash_chord.configuration import compose_config, resolved_dict, instantiate_typed
    from flash_chord.envs.rl import RLEnvConfig
    from flash_chord.training.flash_sac.config import TrainingConfig
    from omegaconf import OmegaConf
    job = job_identity(request["job"])
    rows = pq.ParquetFile(Path(job["job"]) / "motion.parquet").read().to_pylist()
    if len(rows) != 1:
        raise ValueError("Expected exactly one motion row")
    row = rows[0]
    urdf = validate_motion(row, job, runtime)
    export_inputs = export_preflight(request, job)
    prepared = output / "prepared"
    prepared.mkdir(exist_ok=False)
    sequence = row["sequence_id"]
    motion_dir = prepared / "input/retargeted" / f"sequence_id={sequence}" / "robot_name=g1"
    motion_dir.mkdir(parents=True)
    row["object_mesh_paths"] = [str(Path(job["job"]) / "assets/object.obj")]
    row["object_urdf_paths"] = [str(Path(job["job"]) / "assets/object.urdf")]
    motion = motion_dir / "motion.parquet"
    pq.write_table(pa.Table.from_pylist([row]), motion, compression="zstd")
    support = Path(job["job"]) / "assets/support.usda"
    stage = Usd.Stage.Open(str(support))
    if stage is None:
        raise ValueError("Cannot load support stage")
    has_support = any(p.GetTypeName() in ("Cube", "Cylinder") for p in stage.Traverse())
    destination = None
    if has_support:
        destination = prepared / "input/reconstructed_stage" / f"{sequence}_support.usda"
        destination.parent.mkdir()
        shutil.copy2(support, destination)
    cfg = resolved_dict(compose_config("train_flash_sac", [
        "experiment=g1_recon_body_flash_sac", f"embodiment.config.asset_sha256={sha(urdf)}",
        f"task.parquet={json.dumps(str(motion))}", f"scene.support={str(has_support).lower()}",
        "scene.world_count=2048", "training.replay.batch_size=2048", "training.replay.capacity=4000000",
        f"training.optimization.learning_rate_schedule_updates={OPTIMIZER_HORIZON}",
        "logging.mode=offline", "logging.log_interval=25", "logging.upload_checkpoints=false",
        "logging.upload_training_state=false"]))
    cfg = build_fresh_config(cfg, output=output / "bootstrap", end=BOOTSTRAP_STEPS, checkpoint_interval=50_000_000)
    instantiate_typed(OmegaConf.create(cfg["env"]), RLEnvConfig)
    instantiate_typed(OmegaConf.create(cfg["training"]), TrainingConfig)
    config_path = prepared / "native_config.json"
    immutable_json(config_path, cfg)
    parent = validate_parent(request["parent_state"], cfg, job) if request["parent_state"] else None
    record = {"schema": "v2d_policy_preparation_v1", "job": job, "native_config": str(config_path),
              "native_config_sha256": sha(config_path), "runtime_motion": str(motion),
              "runtime_motion_sha256": sha(motion), "support": has_support, "parent": parent,
              "runtime_support": str(destination) if destination else None,
              "runtime_support_sha256": sha(destination) if destination else None,
              "export_inputs": export_inputs}
    immutable_json(output / "preparation.json", record)


def export_preflight(request, job):
    if not request["dataset_input"]:
        return None
    from experiments.export_policy_rollout import validate_export_inputs
    report = validate_export_inputs(job["job"], request["dataset_input"],
                                    job["manifest"]["episode_index"], request["submission_kit"])
    if report.get("state") != "verified" or not report.get("file_sha256"):
        raise ValueError("Export preflight did not verify its input files")
    return report


def checkpoint_record(path):
    state, policy, metadata = checkpoint_pair(path)
    return {"state": str(state), "policy": str(policy), "state_sha256": sha(state),
            "checkpoint_sha256": sha(policy), "environment_steps": int(metadata["environment_steps"])}


def verify_checkpoint(record):
    current = checkpoint_record(record["state"])
    if any(current[key] != record[key] for key in current):
        raise ValueError("Recorded checkpoint pair changed")
    return current


def choose_candidates(records, start, final):
    """Fixed checkpoint subset, determined before examining any evaluation score."""
    by_steps = {record["environment_steps"]: record for record in records}
    if len(by_steps) != len(records):
        raise ValueError("Ambiguous duplicate checkpoint counters")
    if final not in by_steps or final <= start:
        raise ValueError("Final checkpoint missing or no refinement progress")
    selected = [by_steps[final]]
    for additional in (60_000_000, 40_000_000):
        eligible = [step for step in by_steps if start < step <= min(start + additional + WORLDS, final)]
        if eligible:
            record = by_steps[max(eligible)]
            if record not in selected:
                selected.append(record)
    return selected


def rank_key(candidate):
    scores = candidate["metrics"]
    names = ("maniptrans_sr", "spider_sr", "add_auc", "mppe_cm")
    if any(isinstance(scores.get(key), bool) or not isinstance(scores.get(key), (float, int))
           or not math.isfinite(scores[key]) for key in names):
        raise ValueError("Cannot rank nonfinite/missing metrics")
    if any(not 0 <= scores[key] <= 1 for key in names[:3]) or scores["mppe_cm"] < 0:
        raise ValueError("Metrics are outside their valid ranges")
    return (-scores["maniptrans_sr"], -scores["spider_sr"], -scores["add_auc"], scores["mppe_cm"],
            candidate["environment_steps"], candidate["checkpoint_sha256"])


def refinement_checkpoints(output, config):
    records = []
    for path in sorted((Path(output) / "refinement/checkpoints").glob("state_*.safetensors")):
        _, _, metadata = checkpoint_pair(path)
        if json.loads(metadata["resolved_config_json"]) != config:
            raise ValueError("Refinement checkpoint has a different effective configuration")
        records.append(checkpoint_record(path))
    return records


def phase_caps(request, ledger, candidate, *, bootstrap, now=None):
    now = time.time() if now is None else now
    reserve = 3 * EVALUATION_SECONDS
    available = min(guard.remaining_budget(ledger, candidate, float("inf")),
                    guard.remaining_budget(ledger, candidate, float("inf"), kind="evaluation") - reserve,
                    guard.parse_deadline(request["deadline_utc"]) - now - reserve)
    child = int(min(9000 if bootstrap else 7200, available - SUPERVISION_RESERVE))
    if child < 60:
        raise RuntimeError("Insufficient training budget after reserving evaluation and supervision")
    return child, child + SUPERVISION_RESERVE


def run_guard(request, candidate, name, kind, seconds, command):
    output = Path(request["output"])
    # This is a local readiness adapter, not an assertion about any historical
    # queue. The host-wide flock is held, and legacy guard independently checks
    # GPU idleness twice immediately before each child starts.
    readiness = output / "readiness"
    immutable_json(readiness / "queue_status.json", {"state": "complete", "pid": None,
                   "role": "local_readiness_sentinel_not_a_prior_training_queue"})
    subprocess.run([sys.executable, str(ROOT / "experiments/overnight_guard.py"),
                    "--ledger", str(output / "ledger.json"), "--candidate", candidate,
                    "--name", name, "--kind", kind, "--max-seconds", str(seconds),
                    "--total-budget-seconds", str(request["total_budget_seconds"]),
                    "--candidate-training-budget-seconds", str(request["training_budget_seconds"]),
                    "--maximum-candidates", "1", "--deadline-utc", request["deadline_utc"],
                    "--wait-queue", str(readiness), "--wait-seconds", "0", "--cwd", str(ROOT),
                    "--output", str(output / "supervision" / name), "--", *command], check=True)


def reverify(request, preparation):
    if source_identity() != request["source"] or runtime_identity(request["runtime"]) != request["runtime_commit"]:
        raise ValueError("Source/runtime changed since preparation")
    if job_identity(request["job"]) != preparation["job"]:
        raise ValueError("Input changed since preparation")
    for path, key in [(preparation["native_config"], "native_config_sha256"),
                      (preparation["runtime_motion"], "runtime_motion_sha256")]:
        if sha(path) != preparation[key]:
            raise ValueError("Prepared configuration/motion changed")
    if preparation.get("runtime_support") and sha(preparation["runtime_support"]) != preparation["runtime_support_sha256"]:
        raise ValueError("Prepared support geometry changed")
    export_inputs = preparation.get("export_inputs")
    if request["dataset_input"] and not export_inputs:
        raise ValueError("Export inputs were not verified before training")
    if export_inputs:
        for path, expected in export_inputs["file_sha256"].items():
            if sha(path) != expected:
                raise ValueError("Released export input or official kit changed")
    if preparation["parent"]:
        verify_checkpoint(preparation["parent"])
        if sha(preparation["parent"]["original_motion"]) != preparation["parent"]["original_motion_sha256"]:
            raise ValueError("Parent motion changed")
        for path, expected in preparation["parent"]["original_assets"].items():
            if sha(path) != expected:
                raise ValueError("Parent asset changed")


def train_stage(request, preparation, candidate, stage, config, *, parent=None):
    output = Path(request["output"])
    stage_output = output / stage
    completion = output / (stage + "_complete.json")
    planned = output / (stage + "_planned_config.json")
    reverify(request, preparation)
    immutable_json(planned, config)
    if completion.exists():
        record = guard.read_json(completion)
        verify_stage(output, record)
        return record
    if stage_output.exists():
        # Recover only a clean, completed guard command; never start it again.
        jobs = guard.read_json(output / "ledger.json")["jobs"]
        if not any(j["name"] == stage and j["state"] == "complete" for j in jobs):
            raise RuntimeError(f"Stage {stage} has partial/unverified output; duplicate launch refused")
    else:
        child, outer = phase_caps(request, guard.read_json(output / "ledger.json"), candidate, bootstrap=stage == "bootstrap")
        source = ["--fresh-config", preparation["native_config"]] if parent is None else ["--checkpoint", parent["state"]]
        start = 0 if parent is None else parent["environment_steps"]
        command = [sys.executable, str(ROOT / "experiments/overnight_policy.py"), "train", "--runtime", request["runtime"],
                   *source, "--output", str(stage_output), "--phase", "refine", "--additional-transitions",
                   str(config["training"]["total_environment_steps"] - start), "--max-seconds", str(child),
                   "--deadline-utc", request["deadline_utc"], "--checkpoint-interval",
                   str(50_000_000 if parent is None else CHECKPOINT_INTERVAL)]
        immutable_json(output / (stage + "_manifest.json"), {"source": request["source"], "job": preparation["job"],
                       "planned_config_sha256": sha(planned), "parent": parent, "child_seconds": child, "outer_seconds": outer})
        run_guard(request, candidate, stage, "training", outer, command)
    state, _, _ = completed_checkpoint(stage_output, config, required_steps=BOOTSTRAP_STEPS if parent is None else None,
                                       minimum_steps=0 if parent is None else parent["environment_steps"])
    initial_record = None
    if parent is None:
        initial = guard.read_json(stage_output / "initial_checkpoint.json", {})
        _, _, metadata = checkpoint_pair(initial["state"])
        if int(metadata["environment_steps"]) != 0:
            raise RuntimeError("Native acquisition did not start from the recorded step-zero state")
        initial_record = checkpoint_record(initial["state"])
    record = {**checkpoint_record(state), "planned_config_sha256": sha(planned), "stage": stage,
              "initial_checkpoint": initial_record, "parent": parent}
    immutable_json(completion, record)
    return record


def verify_stage(output, record):
    verify_checkpoint(record)
    stage = record["stage"]
    planned = Path(output) / (stage + "_planned_config.json")
    if sha(planned) != record["planned_config_sha256"]:
        raise ValueError("Completed stage configuration changed")
    parent = record["parent"]
    if parent:
        verify_checkpoint(parent)
    if record["initial_checkpoint"]:
        verify_checkpoint(record["initial_checkpoint"])
    state, _, _ = completed_checkpoint(Path(output) / stage, guard.read_json(planned),
                                      required_steps=BOOTSTRAP_STEPS if stage == "bootstrap" else None,
                                      minimum_steps=0 if parent is None else parent["environment_steps"])
    current = checkpoint_record(state)
    if any(current[key] != record[key] for key in current):
        raise ValueError("Completed stage latest checkpoint changed")


def evaluate_candidate(request, preparation, candidate, checkpoint):
    output = Path(request["output"])
    name = f"evaluation_{checkpoint['environment_steps']}"
    destination = output / name
    completion = output / (name + "_complete.json")
    reverify(request, preparation)
    verify_checkpoint(checkpoint)
    if completion.exists():
        record = guard.read_json(completion)
        if record["evaluation_sha256"] != sha(destination / "evaluation.json"):
            raise ValueError("Completed evaluation changed")
    elif not destination.exists():
        available = min(guard.remaining_budget(guard.read_json(output / "ledger.json"), candidate, EVALUATION_SECONDS, kind="evaluation"),
                        guard.parse_deadline(request["deadline_utc"]) - time.time())
        if available < EVALUATION_SECONDS:
            raise RuntimeError("Reserved evaluation budget is unavailable")
        command = [sys.executable, "-m", "training.policy_pipeline", "--_evaluate", request["runtime"],
                   checkpoint["policy"], str(destination), str(preparation["job"]["manifest"]["tier"]),
                   str(preparation["job"]["manifest"]["episode_index"])]
        run_guard(request, candidate, name, "evaluation", EVALUATION_SECONDS, command)
    else:
        jobs = guard.read_json(output / "ledger.json")["jobs"]
        if not any(j["name"] == name and j["state"] == "complete" for j in jobs):
            raise RuntimeError("Partial evaluation exists; duplicate launch refused")
    report = guard.read_json(destination / "evaluation.json")
    scores = evaluation_scores(report, Path(checkpoint["policy"]), checkpoint["environment_steps"])
    tier = preparation["job"]["manifest"]["tier"]
    if (report["step_count"] != preparation["job"]["manifest"]["source_frames"]
            or report.get("metric_reference") != f"tier{tier}_input_reference"
            or report.get("official_leaderboard_score") is not False):
        raise ValueError("Evaluation reference/frame coverage differs")
    record = {**checkpoint, "evaluation_directory": str(destination), "evaluation_sha256": sha(destination / "evaluation.json"),
              "rollout_arrays_sha256": sha(destination / "rollout_arrays.npz"), "metrics": scores}
    immutable_json(completion, record)
    return record


def finish_export(selected, request):
    """CPU-only fixed-world export; selection never consumes clean-reference data."""
    if not request["dataset_input"]:
        return None
    output = Path(request["output"])
    completion = output / "export_complete.json"
    if completion.exists():
        record = guard.read_json(completion)
    else:
        directory = output / "exports" / f"tier{selected['tier']}" / f"episode_{selected['episode_index']:06d}"
        manifest_path = directory / "export_manifest.json"
        if manifest_path.exists():
            # A crash between the exporter's final manifest and our completion
            # marker is recoverable without regenerating or replacing anything.
            manifest = guard.read_json(manifest_path)
            artifact = (directory / manifest["artifact"]["filename"]).resolve()
            if not artifact.is_relative_to(directory.resolve()):
                raise ValueError("Export artifact escapes its output directory")
            record = {"parquet": str(artifact), "parquet_sha256": manifest["artifact"]["sha256"],
                      "manifest": str(manifest_path), "manifest_sha256": sha(manifest_path),
                      "selected_manifest_sha256": manifest["selected_policy_manifest"]["sha256"],
                      "episode": manifest["episode"], "tier": manifest["tier"],
                      "source_world_id": manifest["source_world_id"], "state": manifest["state"]}
        else:
            from experiments.export_policy_rollout import export_selected
            record = export_selected(output / "selected_checkpoint.json", request["dataset_input"],
                                     request["submission_kit"], output / "exports")
        verify_export(record, selected, output / "selected_checkpoint.json")
        immutable_json(completion, record)
    verify_export(record, selected, output / "selected_checkpoint.json")
    return record


def verify_export(record, selected, selected_path):
    if (not record or record.get("state") != "complete" or record.get("source_world_id") != 0
            or record.get("tier") != selected["tier"] or record.get("episode") != selected["episode_index"]
            or record.get("selected_manifest_sha256") != sha(selected_path)):
        raise ValueError("Fixed-world export completion does not match selection")
    for key in ("parquet", "manifest"):
        if sha(record[key]) != record[key + "_sha256"]:
            raise ValueError("Export evidence changed")
    manifest = guard.read_json(record["manifest"])
    if (manifest.get("state") != "complete" or manifest.get("source_world_id") != 0
            or manifest.get("clean_reference_used") is not False
            or manifest.get("achieved_alignment") is not False or manifest.get("exported_world_count") != 1
            or manifest.get("tier") != selected["tier"] or manifest.get("episode") != selected["episode_index"]
            or manifest.get("selected_policy_manifest", {}).get("sha256") != sha(selected_path)
            or manifest.get("validation", {}).get("passed") is not True
            or manifest.get("artifact", {}).get("sha256") != record["parquet_sha256"]
            or manifest.get("inputs", {}).get("policy_checkpoint", {}).get("sha256") != selected["checkpoint_sha256"]):
        raise ValueError("Export manifest violates selected policy or fixed-world contract")


def verify_selected(selected):
    verify_checkpoint(selected)
    for record in selected["candidates"]:
        verify_checkpoint(record)
        directory = Path(record["evaluation_directory"])
        if sha(directory / "evaluation.json") != record["evaluation_sha256"] or sha(directory / "rollout_arrays.npz") != record["rollout_arrays_sha256"]:
            raise ValueError("Selected evaluation evidence changed")
    if min(selected["candidates"], key=rank_key)["checkpoint_sha256"] != selected["checkpoint_sha256"]:
        raise ValueError("Selection does not match preregistered ranking")


def write_progress(output, stage):
    guard.write_json(Path(output) / "pipeline_status.json", {"state": "running", "pid": os.getpid(),
                     "stage": stage, "request_sha256": sha(Path(output) / "request.json"), "updated_utc": guard.utc_now()})


def require_gpu_platform():
    if os.name != "posix":
        raise RuntimeError("GPU execution requires Linux; CPU preparation is portable")


def run(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / "pipeline_status.json"
    with output_lock(output / "pipeline.lock"):
        stage = "request"
        try:
            previous = guard.read_json(output / "request.json")
            deadline = args.deadline_utc or (previous["deadline_utc"] if previous else
                         (dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=args.total_budget_seconds)).isoformat())
            request = {"schema": "v2d_policy_pipeline_request_v1", "job": str(Path(args.job).resolve()),
                   "runtime": str(Path(args.runtime).resolve()), "output": str(output),
                   "parent_state": str(Path(args.parent_state).resolve()) if args.parent_state else None,
                   "training_budget_seconds": args.training_budget_seconds, "total_budget_seconds": args.total_budget_seconds,
                   "deadline_utc": deadline, "source": source_identity(), "runtime_commit": runtime_identity(args.runtime),
                   "dataset_input": str(Path(args.dataset_input).resolve()) if args.dataset_input else None,
                   "submission_kit": str(Path(args.submission_kit).resolve()) if args.submission_kit else None,
                   "selection_rule": SELECTION_RULE, "refinement_transition_cap": REFINEMENT_STEPS,
                   "candidate_subset": "final, latest<=60M additional, latest<=40M additional; deduplicated",
                   "candidate_boundary_tolerance_transitions": WORLDS}
            policy = guard.make_budget_policy(args.total_budget_seconds, args.training_budget_seconds, 1)
            immutable_json(output / "request.json", request)
            stage = "prepare"
            if not (output / "preparation.json").exists():
                env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "JAX_PLATFORMS": "cpu"}
                subprocess.run([sys.executable, "-m", "training.policy_pipeline", "--_prepare-request", str(output / "request.json")],
                               cwd=ROOT, env=env, check=True)
            preparation = guard.read_json(output / "preparation.json")
            reverify(request, preparation)
            existing = guard.read_json(status_path, {})
            if existing.get("state") == "complete":
                selected = guard.read_json(output / "selected_checkpoint.json")
                if sha(output / "selected_checkpoint.json") != existing["selected_manifest_sha256"]:
                    raise ValueError("Selected manifest changed")
                verify_selected(selected)
                for name in ("bootstrap", "refinement"):
                    record = guard.read_json(output / (name + "_complete.json"))
                    if record:
                        verify_stage(output, record)
                if request["dataset_input"]:
                    verify_export(existing.get("export"), selected, output / "selected_checkpoint.json")
                return selected
            if args.prepare_only:
                guard.write_json(status_path, {"state": "prepared", "updated_utc": guard.utc_now(), "request_sha256": sha(output / "request.json")})
                return preparation
            require_gpu_platform()
            candidate = f"tier{preparation['job']['manifest']['tier']}_episode_{preparation['job']['manifest']['episode_index']:06d}"
            ledger_path = output / "ledger.json"
            ledger = guard.read_json(ledger_path, {"created_utc": guard.utc_now(), "jobs": []})
            guard.ensure_budget_policy(ledger, policy, ledger_exists=ledger_path.exists())
            guard.remaining_budget(ledger, candidate, 0)  # Running records fail closed, including stale ones.
            guard.write_json(ledger_path, ledger)
            with guard.exclusive_lock(Path(tempfile.gettempdir()) / "v2d-policy-gpu.lock"):
                if guard.gpu_pids():
                    raise RuntimeError("GPU is in use; no worker launched")
                write_progress(output, stage)
                parent = preparation["parent"]
                if parent is None:
                    stage = "bootstrap"
                    write_progress(output, stage)
                    parent = train_stage(request, preparation, candidate, stage, guard.read_json(preparation["native_config"]))
                stage = "refinement"
                write_progress(output, stage)
                _, _, metadata = checkpoint_pair(parent["state"])
                start = parent["environment_steps"]
                end, _ = transition_budget(start, REFINEMENT_STEPS)
                config = build_config(json.loads(metadata["resolved_config_json"]), state=Path(parent["state"]),
                                      output=output / stage, start=start, end=end, phase="refine", checkpoint_interval=CHECKPOINT_INTERVAL)
                final = train_stage(request, preparation, candidate, stage, config, parent=parent)
                stage = "checkpoint_selection"
                write_progress(output, stage)
                records = refinement_checkpoints(output, config)
                candidates = choose_candidates(records, start, final["environment_steps"])
                immutable_json(output / "candidate_plan.json", {"selection_rule": SELECTION_RULE, "candidates": candidates,
                               "refinement_start_steps": start, "clean_reference_used": False,
                               "boundary_tolerance_transitions": WORLDS, "boundary_capped_at_final": True})
                stage = "evaluation"
                write_progress(output, stage)
                evaluated = [evaluate_candidate(request, preparation, candidate, record) for record in candidates]
                winner = min(evaluated, key=rank_key)
                selected = {"schema": "v2d_selected_policy_v1", **winner,
                            "tier": preparation["job"]["manifest"]["tier"], "episode_index": preparation["job"]["manifest"]["episode_index"],
                            "job": request["job"], "runtime": request["runtime"], "source_commit": request["source"]["commit"],
                            "request_sha256": sha(output / "request.json"), "input_job_sha256": preparation["job"]["job_sha256"],
                            "candidates": evaluated, "selection_rule": SELECTION_RULE, "clean_reference_used": False,
                            "world0_export_required": True, "refinement_transition_cap_reached": final["environment_steps"] == end}
                immutable_json(output / "selected_checkpoint.json", selected)
                stage = "export"
                write_progress(output, stage)
                export = finish_export(selected, request)
                guard.write_json(status_path, {"state": "complete", "stage": "complete", "updated_utc": guard.utc_now(),
                                               "selected_manifest_sha256": sha(output / "selected_checkpoint.json"), "export": export})
                return selected
        except BaseException as exc:
            guard.write_json(status_path, {"state": "failed", "failed_stage": stage, "updated_utc": guard.utc_now(),
                                           "error": f"{type(exc).__name__}: {exc}", "automatic_retry": False})
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--job")
    parser.add_argument("--runtime")
    parser.add_argument("--output")
    parser.add_argument("--parent-state")
    parser.add_argument("--training-budget-seconds", type=float, default=10800)
    parser.add_argument("--total-budget-seconds", type=float, default=11700)
    parser.add_argument("--deadline-utc")
    parser.add_argument("--prepare-only", "--dry-run", action="store_true", dest="prepare_only")
    parser.add_argument("--dataset-input")
    parser.add_argument("--submission-kit")
    parser.add_argument("--_prepare-request", help=argparse.SUPPRESS)
    parser.add_argument("--_evaluate", nargs=5, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args._prepare_request:
        prepare(guard.read_json(args._prepare_request))
        return
    if args._evaluate:
        runtime, checkpoint, output, tier, episode = args._evaluate
        subprocess.run([sys.executable, str(ROOT / "experiments/evaluate_overnight_policy.py"),
                        "--runtime", runtime, "--checkpoint", checkpoint, "--output", output,
                        "--tier", tier, "--episode", episode, "--worlds", "16", "--variant", "automatic_precision"],
                       check=True, timeout=EVALUATOR_CHILD_SECONDS)
        return
    if not all((args.job, args.runtime, args.output)):
        parser.error("--job, --runtime and --output are required")
    if bool(args.dataset_input) != bool(args.submission_kit):
        parser.error("--dataset-input and --submission-kit must be supplied together")
    guard.make_budget_policy(args.total_budget_seconds, args.training_budget_seconds, 1)
    if args.training_budget_seconds > args.total_budget_seconds:
        parser.error("Training budget cannot exceed total budget")
    if args.deadline_utc:
        guard.parse_deadline(args.deadline_utc)
    result = run(args)
    print(json.dumps({"output": str(Path(args.output).resolve()), "prepared_only": args.prepare_only,
                      "selected_policy": result.get("policy")}, indent=2))


if __name__ == "__main__":
    main()

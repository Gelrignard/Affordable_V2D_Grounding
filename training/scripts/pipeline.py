"""Native FlashSAC acquisition and precision refinement, with optional limits.

Run with the pinned runtime's Python: ``python -m training.scripts.pipeline
--job JOB --runtime RUNTIME --output OUTPUT``. ``--prepare-only`` (or
``--dry-run``) performs the same CPU preflight without launching GPU work.

The native 350M phase is policy acquisition, not a separately evaluated baseline.
Tier 1 refinement runs once for up to 90M additional transitions. Tier 2/3 can
use 60M precision followed by 30M supported load transfer, on their own inputs.
The second phase restores the first phase's optimizer clocks and original reward
ramp, and refills fresh replay. Native saves occur at
absolute 10M intervals; evaluate final, latest <=60M and latest <=40M additional
checkpoints (one 2048-transition collection tolerance), deduplicated, at most
three. Success ranking uses MP-SR, SP-SR, AUC, then lower MPPE; tracking ranking
uses AUC, lower MPPE, MP-SR, then SP-SR. Ties prefer fewer transitions. This is checkpoint selection,
not a parameter sweep. Every candidate includes all 16 uninterrupted worlds.
Submission export must use fixed world 0, never a selected successful world.

Resume verifies completed stages and their hashes. Explicit --native-resume
starts a separate recovery with saved native state and the historical ledger;
ordinary resume never relaunches failed work or takes over a running worker.
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

from training.scripts import guard
from training.scripts.train import (
    OPTIMIZER_HORIZON, WORLDS, build_config, build_fresh_config,
    checkpoint_pair, transition_budget,
)
from training.core.checkpoints import completed_checkpoint, evaluation_scores
from training.core.configuration import DEFAULTS, REPLAY_CAPACITY, build_native_resume_config
from training.core.recovery import bind_native_recovery, verify_recovery_history
from training import RUNTIME_COMMIT

ROOT = Path(__file__).resolve().parents[2]
BOOTSTRAP_STEPS = DEFAULTS["bootstrap_steps"]
REFINEMENT_STEPS = DEFAULTS["refinement_steps"]
CHECKPOINT_INTERVAL = DEFAULTS["checkpoint_interval"]
EVALUATION_SECONDS = DEFAULTS["evaluation_seconds"]
EVALUATOR_CHILD_SECONDS = DEFAULTS["evaluator_child_seconds"]
SUPERVISION_RESERVE = DEFAULTS["supervision_reserve"]
SELECTION_RULE = ["maniptrans_sr:descending", "spider_sr:descending", "add_auc:descending",
                  "mppe_cm:ascending", "environment_steps:ascending", "checkpoint_sha256:ascending"]
TRACKING_SELECTION_RULE = ["add_auc:descending", "mppe_cm:ascending", "maniptrans_sr:descending",
                          "spider_sr:descending", "environment_steps:ascending", "checkpoint_sha256:ascending"]
RECIPES = ("legacy", "tier1-precision", "tier23-load-transfer")


def selection_rule(policy):
    if policy not in ("success", "tracking"):
        raise ValueError("Unknown checkpoint selection policy")
    return list(TRACKING_SELECTION_RULE if policy == "tracking" else SELECTION_RULE)


def validate_recipe(recipe, tier, acquisition="precision", placement="off"):
    if recipe not in RECIPES:
        raise ValueError("Unknown training recipe")
    if recipe == "legacy":
        return placement
    if acquisition != "precision":
        raise ValueError("Named recipes require precision acquisition")
    if recipe == "tier1-precision":
        if tier != 1 or placement != "off":
            raise ValueError("Tier 1 precision requires Tier 1 input and placement off")
        return "off"
    if tier not in (2, 3):
        raise ValueError("Tier 2/3 load transfer requires that tier's own input")
    return "supported-release"


def validate_recipe_parent(recipe, parent):
    if recipe != "legacy" and parent:
        saved = parent["resolved_config"]
        if (parent["environment_steps"] != BOOTSTRAP_STEPS or saved.get("refinement")
                or saved.get("overnight") or saved.get("env", {}).get("objective", {}).get("_target_")
                != "flash_chord.objectives.recon_body.ReconBodyObjectiveConfig"):
            raise ValueError("Named recipes must start from this episode's native 350M parent")


def refinement_schedule(request, preparation):
    """Fixed phases: applicability is determined on CPU before GPU allocation."""
    if request.get("recipe") != "tier23-load-transfer":
        return [("refinement", REFINEMENT_STEPS, request.get("placement_guidance", "off"))]
    audit = guard.read_json(preparation["placement_preflight"]["audit"])
    applicable = audit.get("applicable", True)
    if not isinstance(applicable, bool):
        raise ValueError("Invalid placement applicability record")
    return [("refinement", 60_000_000, "off"),
            ("load_transfer", 30_000_000, "supported-release" if applicable else "off")]


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
    return subprocess.check_output(["git", "-C", str(root), *args], text=True,
                                   stderr=subprocess.DEVNULL).strip()


_SOURCE_CACHE = {}


def source_identity(strict=False):
    """Hash public execution files; a clean Git checkout is an optional contract."""
    suffixes = {".py", ".json", ".yaml", ".yml", ".toml"}
    paths = [ROOT / "training/__init__.py", ROOT / "training/__main__.py"]
    for directory in ("core", "scripts", "evaluate", "configs"):
        paths.extend(path for path in (ROOT / "training" / directory).rglob("*")
                     if path.is_file() and path.suffix in suffixes)
    paths.extend(path for path in (ROOT / "data_tools").rglob("*")
                 if path.is_file() and path.suffix in suffixes)
    names = sorted(path.relative_to(ROOT).as_posix() for path in paths)
    signature = tuple((name, (ROOT / name).stat().st_size, (ROOT / name).stat().st_mtime_ns) for name in names)
    try:
        current_head = git(ROOT, "rev-parse", "HEAD")
    except (subprocess.CalledProcessError, FileNotFoundError):
        current_head = None
    key = (str(ROOT), strict, current_head)
    cached = _SOURCE_CACHE.get(key)
    if cached and cached[0] == signature:
        return cached[1]
    commit, dirty = None, None
    try:
        commit = git(ROOT, "rev-parse", "HEAD")
        tracked = set(git(ROOT, "ls-files", "--", "training", "data_tools").splitlines())
        dirty = bool(git(ROOT, "diff", "HEAD", "--name-only", "--", *names)) or not set(names).issubset(tracked)
    except (subprocess.CalledProcessError, FileNotFoundError):
        if strict:
            raise ValueError("Strict reproducibility requires a Git checkout") from None
    if strict and dirty:
        raise ValueError("Strict reproducibility requires clean, committed execution files")
    result = {"commit": commit, "files": {name: sha(ROOT / name) for name in names},
              "dirty": dirty, "strict_reproducibility": strict}
    _SOURCE_CACHE[key] = (signature, result)
    return result


def preserve_source_snapshot(output, source):
    """Preserve exact code even for a dirty checkout or a source archive."""
    root = Path(output) / "source_snapshot"
    for name, digest in source.get("files", {}).items():
        path = root / name
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, path)
        if sha(path) != digest:
            raise ValueError("Execution source changed while its snapshot was created")
    immutable_json(root / "manifest.json", source)


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
            or type(data.get("episode_index")) is not int or data["episode_index"] < 0
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


def prepare_effective_job(job, row, destination, urdf):
    """Derive one coherent repaired job; finite portable inputs stay untouched."""
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    from scipy.spatial.transform import Rotation
    from training.core.bundles import arrays, validate_reference
    from training.core.input_repair import (repair_motion_row, repair_reference,
                                            merge_portable_counterparts, PORTABLE_PAIRS)
    from training.core.kinematics import URDFKinematics
    from training.core.prepare import regenerate_repaired_fk

    original = arrays(Path(job["job"]) / "reference.npz")
    reference_input, motion_input, counterpart_frames = merge_portable_counterparts(original, row)
    reference, reference_repairs = repair_reference(reference_input)
    repaired_row, motion_repairs = repair_motion_row(motion_input)
    if reference_repairs is None and motion_repairs is None and not counterpart_frames:
        return job, row, None
    reference, repaired_row = dict(reference), dict(repaired_row)
    reference_repairs = dict(reference_repairs or {})
    motion_fields = (motion_repairs or {}).get("repaired_source_frame_indices", {})
    if list(original["joint_names"]) != row["robot_joint_names"]:
        raise ValueError("Cannot repair portable motion/reference with different joint order")
    # Preserve any available finite counterpart. Only interpolate where both
    # representations lack the source value; disagreement is not missing data.
    for ref_key, row_key, object_axis in PORTABLE_PAIRS:
        a = np.asarray(original[ref_key], dtype=float)
        b = np.asarray(row[row_key], dtype=float)
        if object_axis:
            b = b[:, 0]
        if a.shape != b.shape:
            raise ValueError(f"Portable motion/reference shape mismatch: {ref_key}")
        good_a, good_b = np.isfinite(a), np.isfinite(b)
        if ref_key.endswith("wxyz"):
            good_a = np.broadcast_to((good_a.all(-1) & (np.linalg.norm(a, axis=-1) > 1e-12))[:, None], a.shape)
            good_b = np.broadcast_to((good_b.all(-1) & (np.linalg.norm(b, axis=-1) > 1e-12))[:, None], b.shape)
        if not np.allclose(a[good_a & good_b], b[good_a & good_b], atol=1e-7, rtol=0):
            raise ValueError(f"Already-finite portable motion/reference values disagree: {ref_key}")
        value = np.asarray(reference[ref_key], dtype=float).copy()
        value[good_b] = b[good_b]
        value[good_a] = a[good_a]
        reference[ref_key] = value.astype(original[ref_key].dtype, copy=False)
        repaired_row[row_key] = (value[:, None] if object_axis else value).tolist()
        changed = np.flatnonzero(~(good_a & good_b).reshape(len(a), -1).all(-1)).tolist()
        if changed:
            reference_repairs[ref_key] = sorted(set(reference_repairs.get(ref_key, [])) | set(changed))
    object_changed = sorted(set(reference_repairs.get("object_position", [])) |
                            set(reference_repairs.get("object_wxyz", [])))
    if object_changed:
        reference["object_observed_mask"] = original["object_observed_mask"].copy()
        reference["object_observed_mask"][object_changed] = False
    if motion_fields.get("ee_pose_w"):
        reference_repairs["ee_pose_w"] = sorted(set(reference_repairs.get("ee_pose_w", [])) |
                                                set(motion_fields["ee_pose_w"]))
    reference, fk_frames = regenerate_repaired_fk(reference, reference_repairs, URDFKinematics(urdf))
    if fk_frames:
        indices = [reference["ee_link_names"].tolist().index(name) for name in row["ee_link_names"]]
        palms = np.asarray(repaired_row["ee_pose_w"], dtype=float)
        palms[fk_frames] = reference["ee_pose_w"][fk_frames][:, indices]
        repaired_row["ee_pose_w"] = palms.tolist()
    # Derived contact locations follow the repaired object; masks and local
    # patches are copied unchanged and do not become contact certification.
    guide = arrays(Path(job["job"]) / "guidance.npz")
    if object_changed and "contact_body_names" in guide:
        rotation = Rotation.from_quat(reference["object_wxyz"][:, [1, 2, 3, 0]]).as_matrix()
        for side_index, side in enumerate(("left", "right")):
            selected = [i for i, name in enumerate(guide["contact_body_names"]) if str(name).startswith(side + "_")]
            if not selected:
                continue
            for key, local, translate in (("hand_object_contact_positions", "object_local_points", True),
                                          ("hand_object_contact_normals", "inward_normals_local", False)):
                value = np.asarray(repaired_row[key][side_index], dtype=float)
                points = np.einsum("tij,tkj->tki", rotation, guide[local][:, selected])
                if translate:
                    points += reference["object_position"][:, None]
                value[object_changed] = points[object_changed]
                repaired_row[key] = list(repaired_row[key])
                repaired_row[key][side_index] = value.tolist()
    validate_reference(reference)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    for name in job["manifest"]["files"]:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(job["job"]) / name, target)
    repaired_row["object_mesh_paths"] = [str(destination / "assets/object.obj")]
    repaired_row["object_urdf_paths"] = [str(destination / "assets/object.urdf")]
    pq.write_table(pa.Table.from_pylist([repaired_row]), destination / "motion.parquet", compression="zstd")
    np.savez_compressed(destination / "reference.npz", **reference)
    report = {"schema": "v2d_derived_portable_input_repair_v1", "original_job": job,
              "reference_repaired_frames": reference_repairs, "motion_repairs": motion_repairs,
              "paired_representation_missing_frames": counterpart_frames,
              "regenerated_fk_frames": fk_frames, "original_input_unchanged": True,
              "source_frames_unchanged": True, "contact_masks_unchanged": True}
    immutable_json(destination / "input_repair.json", report)
    manifest = {**job["manifest"], "input_repairs": reference_repairs,
                "derived_from_job": {"path": job["job"], "sha256": job["job_sha256"]},
                "files": {path.relative_to(destination).as_posix(): sha(path)
                          for path in destination.rglob("*") if path.is_file()}}
    immutable_json(destination / "job.json", manifest)
    return job_identity(destination), repaired_row, report


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
    if request.get("recovery_lineage_sha256"):
        verify_recovery_history(output)
        lineage = guard.read_json(output / "recovery_lineage.json")
        old = guard.read_json(Path(lineage["recovery_from"]) / "preparation.json")
        if job != old.get("original_job", old["job"]):
            raise ValueError("Recovery job changed")
        parent = lineage["checkpoint"]
        _, _, metadata = checkpoint_pair(parent["state"])
        cfg = build_native_resume_config(json.loads(metadata["resolved_config_json"]),
            state=Path(parent["state"]), output=output / "bootstrap_resume",
            start=parent["environment_steps"], end=lineage["native_target_steps"], checkpoint_interval=50_000_000)
        instantiate_typed(OmegaConf.create(cfg["env"]), RLEnvConfig)
        instantiate_typed(OmegaConf.create(cfg["training"]), TrainingConfig)
        config_path = output / "prepared/native_resume_config.json"
        immutable_json(config_path, cfg)
        record = {**old, "native_config": str(config_path), "native_config_sha256": sha(config_path),
                  "native_resume": parent, "parent": None,
                  "export_inputs": export_preflight(request, old["job"]),
                  "recovery_lineage_sha256": request["recovery_lineage_sha256"]}
        immutable_json(output / "preparation.json", record)
        return
    if request.get("acquisition_guidance", "precision") == "grasp-region" and job["manifest"]["tier"] == 1:
        raise ValueError("Grasp-region guidance is for Tier 2/3; Tier 1 uses precision")
    rows = pq.ParquetFile(Path(job["job"]) / "motion.parquet").read().to_pylist()
    if len(rows) != 1:
        raise ValueError("Expected exactly one motion row")
    row = rows[0]
    original_job = job
    urdf_path = runtime / "robotic_grounding/flash_chord/src/flash_chord/assets/urdfs/g1/main_with_hand.urdf"
    job, row, repair_report = prepare_effective_job(job, row, output / "repaired_input", urdf_path)
    urdf = validate_motion(row, job, runtime)
    export_inputs = export_preflight(request, job)
    prepared = output / "prepared"
    prepared.mkdir(exist_ok=False)
    if repair_report is not None:
        immutable_json(prepared / "input_repair.json", repair_report)
        print("WARNING: nonfinite portable motion samples were repaired in the prepared copy; "
              "see prepared/input_repair.json. Original inputs remain unchanged.", file=sys.stderr)
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
        "scene.world_count=2048", "training.replay.batch_size=2048", f"training.replay.capacity={REPLAY_CAPACITY}",
        f"training.optimization.learning_rate_schedule_updates={OPTIMIZER_HORIZON}",
        "logging.mode=offline", "logging.log_interval=25", "logging.upload_checkpoints=false",
        "logging.upload_training_state=false"]))
    cfg = build_fresh_config(cfg, output=output / "bootstrap", end=BOOTSTRAP_STEPS, checkpoint_interval=50_000_000)
    instantiate_typed(OmegaConf.create(cfg["env"]), RLEnvConfig)
    instantiate_typed(OmegaConf.create(cfg["training"]), TrainingConfig)
    config_path = prepared / "native_config.json"
    immutable_json(config_path, cfg)
    parent = validate_parent(request["parent_state"], cfg, job) if request["parent_state"] else None
    validate_recipe_parent(request.get("recipe", "legacy"), parent)
    record = {"schema": "v2d_policy_preparation_v1", "job": job, "native_config": str(config_path),
              "native_config_sha256": sha(config_path), "runtime_motion": str(motion),
              "runtime_motion_sha256": sha(motion), "support": has_support, "parent": parent,
              "runtime_support": str(destination) if destination else None,
              "runtime_support_sha256": sha(destination) if destination else None,
              "export_inputs": export_inputs}
    if repair_report is not None:
        record["input_repair"] = str(prepared / "input_repair.json")
        record["input_repair_sha256"] = sha(prepared / "input_repair.json")
        record["original_job"] = original_job
    if request.get("acquisition_guidance") == "grasp-region":
        from training.core.acquisition import preflight_acquisition_config
        start = parent["environment_steps"] if parent else BOOTSTRAP_STEPS
        preview = build_config(parent["resolved_config"] if parent else cfg,
                               state=Path(parent["state"]) if parent else prepared / "future_state.safetensors",
                               output=output / "refinement", start=start, end=start + REFINEMENT_STEPS,
                               phase="refine", checkpoint_interval=CHECKPOINT_INTERVAL,
                               acquisition_guidance="grasp-region",
                               acquisition_reference=Path(job["job"]) / "reference.npz")
        preview_path = prepared / "acquisition_config.json"
        audit_path = prepared / "acquisition_guidance.json"
        immutable_json(preview_path, preview)
        immutable_json(audit_path, preflight_acquisition_config(preview))
        record["acquisition_preflight"] = {
            "config": str(preview_path), "config_sha256": sha(preview_path),
            "audit": str(audit_path), "audit_sha256": sha(audit_path)}
    if request.get("placement_guidance") == "supported-release":
        from training.core.placement import PlacementNotApplicable, preflight_placement_config
        if request.get("acquisition_guidance", "precision") != "precision":
            raise ValueError("Supported release requires precision acquisition")
        if job["manifest"]["tier"] == 1:
            raise ValueError("Supported-release guidance is opt-in for Tier 2/3 only")
        start = parent["environment_steps"] if parent else BOOTSTRAP_STEPS
        preview = build_config(parent["resolved_config"] if parent else cfg,
                               state=Path(parent["state"]) if parent else prepared / "future_state.safetensors",
                               output=output / "refinement", start=start, end=start + REFINEMENT_STEPS,
                               phase="refine", checkpoint_interval=CHECKPOINT_INTERVAL,
                               acquisition_guidance="precision", placement_guidance="supported-release",
                               placement_reference=Path(job["job"]) / "reference.npz",
                               placement_support=Path(job["job"]) / "assets/support.usda")
        preview_path = prepared / "placement_config.json"
        audit_path = prepared / "placement_guidance.json"
        immutable_json(preview_path, preview)
        try:
            placement_audit = preflight_placement_config(preview)
        except PlacementNotApplicable as exc:
            if request.get("recipe") != "tier23-load-transfer":
                raise
            placement_audit = {"applicable": False, "reason": exc.reason,
                               "details": exc.audit, "fallback": "precision_30M",
                               "warning": str(exc)}
            print(f"WARNING: load transfer is not applicable ({exc}); "
                  "the final 30M phase will use precision guidance.", file=sys.stderr)
        immutable_json(audit_path, placement_audit)
        record["placement_preflight"] = {
            "config": str(preview_path), "config_sha256": sha(preview_path),
            "audit": str(audit_path), "audit_sha256": sha(audit_path)}
    immutable_json(output / "preparation.json", record)


def export_preflight(request, job):
    if not request["dataset_input"]:
        return None
    from training.evaluate.export import validate_export_inputs
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


def choose_candidates(records, start, final, worlds=WORLDS):
    """Fixed checkpoint subset, determined before examining any evaluation score."""
    by_steps = {record["environment_steps"]: record for record in records}
    if len(by_steps) != len(records):
        raise ValueError("Ambiguous duplicate checkpoint counters")
    if final not in by_steps or final <= start:
        raise ValueError("Final checkpoint missing or no refinement progress")
    selected = [by_steps[final]]
    for additional in (60_000_000, 40_000_000):
        eligible = [step for step in by_steps if start < step <= min(start + additional + worlds, final)]
        if eligible:
            record = by_steps[max(eligible)]
            if record not in selected:
                selected.append(record)
    return selected


def rank_key(candidate, policy="success"):
    scores = candidate["metrics"]
    names = ("maniptrans_sr", "spider_sr", "add_auc", "mppe_cm")
    if any(isinstance(scores.get(key), bool) or not isinstance(scores.get(key), (float, int))
           or not math.isfinite(scores[key]) for key in names):
        raise ValueError("Cannot rank nonfinite/missing metrics")
    if any(not 0 <= scores[key] <= 1 for key in names[:3]) or scores["mppe_cm"] < 0:
        raise ValueError("Metrics are outside their valid ranges")
    selection_rule(policy)
    if policy == "tracking":
        return (-scores["add_auc"], scores["mppe_cm"], -scores["maniptrans_sr"], -scores["spider_sr"],
                candidate["environment_steps"], candidate["checkpoint_sha256"])
    return (-scores["maniptrans_sr"], -scores["spider_sr"], -scores["add_auc"], scores["mppe_cm"],
            candidate["environment_steps"], candidate["checkpoint_sha256"])


def refinement_checkpoints(output, config, stage="refinement"):
    records = []
    for path in sorted((Path(output) / stage / "checkpoints").glob("state_*.safetensors")):
        _, _, metadata = checkpoint_pair(path)
        if json.loads(metadata["resolved_config_json"]) != config:
            raise ValueError("Refinement checkpoint has a different effective configuration")
        records.append(checkpoint_record(path))
    return records


def phase_caps(request, ledger, candidate, *, bootstrap, now=None):
    now = time.time() if now is None else now
    reserve = 3 * (request.get("evaluation_max_seconds") or 0)
    available = min(guard.remaining_budget(ledger, candidate, float("inf")),
                    guard.remaining_budget(ledger, candidate, float("inf"), kind="evaluation") - reserve,
                    guard.parse_deadline(request["deadline_utc"]) - now - reserve)
    explicit = request.get("native_max_seconds" if bootstrap else "refinement_max_seconds")
    value = min(explicit if explicit is not None else math.inf, available - SUPERVISION_RESERVE)
    if not math.isfinite(value):
        return None, None
    child = int(value)
    if child <= 30:
        raise RuntimeError("Insufficient training budget after reserving evaluation and supervision")
    return child, child + SUPERVISION_RESERVE


def ledger_path(request):
    return Path(request.get("ledger_path", str(Path(request["output"]) / "ledger.json")))


def ledger_job_name(request, stage):
    return "native_recovery_1__" + stage if request.get("recovery_lineage_sha256") else stage


def run_guard(request, candidate, name, kind, seconds, command):
    output = Path(request["output"])
    guard_command = [sys.executable, str(ROOT / "training/scripts/guard.py"),
                    "--ledger", str(ledger_path(request)), "--candidate", candidate,
                    "--name", ledger_job_name(request, name), "--kind", kind,
                    "--idle-only", "--cwd", str(ROOT),
                    "--output", str(output / "supervision" / name)]
    for flag, value in (("--max-seconds", seconds), ("--total-budget-seconds", request["total_budget_seconds"]),
                        ("--candidate-training-budget-seconds", request["training_budget_seconds"]),
                        ("--deadline-utc", request["deadline_utc"])):
        if value is not None:
            guard_command.extend([flag, str(value)])
    subprocess.run([*guard_command, "--", *command], check=True)


def reverify(request, preparation):
    if source_identity(request.get("strict_reproducibility", False)) != request["source"] or runtime_identity(request["runtime"]) != request["runtime_commit"]:
        raise ValueError("Source/runtime changed since preparation")
    if job_identity(request["job"]) != preparation.get("original_job", preparation["job"]):
        raise ValueError("Input changed since preparation")
    if preparation.get("original_job") and job_identity(preparation["job"]["job"]) != preparation["job"]:
        raise ValueError("Derived repaired input changed since preparation")
    if request.get("recovery_lineage_sha256"):
        if (sha(Path(request["output"]) / "recovery_lineage.json") != request["recovery_lineage_sha256"]
                or preparation.get("recovery_lineage_sha256") != request["recovery_lineage_sha256"]):
            raise ValueError("Recovery lineage changed")
        verify_recovery_history(request["output"])
    for path, key in [(preparation["native_config"], "native_config_sha256"),
                      (preparation["runtime_motion"], "runtime_motion_sha256")]:
        if sha(path) != preparation[key]:
            raise ValueError("Prepared configuration/motion changed")
    if preparation.get("runtime_support") and sha(preparation["runtime_support"]) != preparation["runtime_support_sha256"]:
        raise ValueError("Prepared support geometry changed")
    if preparation.get("input_repair") and sha(preparation["input_repair"]) != preparation["input_repair_sha256"]:
        raise ValueError("Prepared repair provenance changed")
    acquisition = preparation.get("acquisition_preflight")
    if request.get("acquisition_guidance") == "grasp-region" and not acquisition:
        raise ValueError("Acquisition regions were not verified before training")
    if acquisition:
        for key in ("config", "audit"):
            if sha(acquisition[key]) != acquisition[key + "_sha256"]:
                raise ValueError("Acquisition preflight changed")
    placement = preparation.get("placement_preflight")
    if request.get("placement_guidance") == "supported-release" and not placement:
        raise ValueError("Placement support was not verified before training")
    if placement:
        for key in ("config", "audit"):
            if sha(placement[key]) != placement[key + "_sha256"]:
                raise ValueError("Placement preflight changed")
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
        jobs = guard.read_json(ledger_path(request))["jobs"]
        if not any(j["name"] == ledger_job_name(request, stage) and j["state"] == "complete" for j in jobs):
            raise RuntimeError(f"Stage {stage} has partial/unverified output; duplicate launch refused")
    else:
        native_resume = stage == "bootstrap_resume"
        child, outer = phase_caps(request, guard.read_json(ledger_path(request)), candidate,
                                  bootstrap=stage in ("bootstrap", "bootstrap_resume"))
        source = (["--native-resume", parent["state"]] if native_resume else
                  (["--fresh-config", preparation["native_config"]] if parent is None else ["--checkpoint", parent["state"]]))
        start = 0 if parent is None else parent["environment_steps"]
        command = [sys.executable, str(ROOT / "training/scripts/train.py"), "train", "--runtime", request["runtime"],
                   *source, "--output", str(stage_output), "--phase", "refine", "--additional-transitions",
                   str(config["training"]["total_environment_steps"] - start), "--checkpoint-interval",
                   str(50_000_000 if parent is None or native_resume else CHECKPOINT_INTERVAL)]
        if child is not None:
            command.extend(["--max-seconds", str(child)])
        if request["deadline_utc"] is not None:
            command.extend(["--deadline-utc", request["deadline_utc"]])
        if parent is not None and not native_resume:
            objective = config["env"]["objective"]
            acquisition = "grasp-region" if objective.get("acquisition_enabled", False) else "precision"
            placement = "supported-release" if objective.get("placement_enabled", False) else "off"
            command.extend(["--acquisition-guidance", acquisition, "--placement-guidance", placement])
            if acquisition == "grasp-region":
                command.extend(["--acquisition-reference", str(Path(preparation["job"]["job"]) / "reference.npz")])
            if placement == "supported-release":
                command.extend(["--placement-reference", str(Path(preparation["job"]["job"]) / "reference.npz"),
                                "--placement-support", str(Path(preparation["job"]["job"]) / "assets/support.usda")])
        immutable_json(output / (stage + "_manifest.json"), {"source": request["source"], "job": preparation["job"],
                       "planned_config_sha256": sha(planned), "parent": parent, "child_seconds": child, "outer_seconds": outer})
        run_guard(request, candidate, stage, "training", outer, command)
    state, _, _ = completed_checkpoint(stage_output, config, required_steps=config["training"]["total_environment_steps"] if stage in ("bootstrap", "bootstrap_resume") else None,
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
    config = guard.read_json(planned)
    state, _, _ = completed_checkpoint(Path(output) / stage, config,
                                      required_steps=config["training"]["total_environment_steps"] if stage in ("bootstrap", "bootstrap_resume") else None,
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
        available = min(guard.remaining_budget(guard.read_json(ledger_path(request)), candidate,
                                               request.get("evaluation_max_seconds"), kind="evaluation"),
                        guard.parse_deadline(request["deadline_utc"]) - time.time())
        if available <= 0:
            raise RuntimeError("Evaluation budget is unavailable")
        command = [sys.executable, "-m", "training.scripts.pipeline", "--_evaluate", request["runtime"],
                   checkpoint["policy"], str(destination), str(preparation["job"]["manifest"]["tier"]),
                   str(preparation["job"]["manifest"]["episode_index"])]
        run_guard(request, candidate, name, "evaluation", available if math.isfinite(available) else None, command)
    else:
        jobs = guard.read_json(ledger_path(request))["jobs"]
        if not any(j["name"] == ledger_job_name(request, name) and j["state"] == "complete" for j in jobs):
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
            from training.evaluate.export import export_selected
            record = export_selected(output / "selected_checkpoint.json", request["dataset_input"],
                                     request["submission_kit"], output / "exports")
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
    policy = "tracking" if selected.get("selection_rule") == TRACKING_SELECTION_RULE else "success"
    if selected.get("selection_rule") != selection_rule(policy):
        raise ValueError("Unknown saved checkpoint selection rule")
    if min(selected["candidates"], key=lambda c: rank_key(c, policy))["checkpoint_sha256"] != selected["checkpoint_sha256"]:
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
    with output_lock(output / "pipeline.lock"), contextlib.ExitStack() as recovery_locks:
        stage = "request"
        try:
            recipe = getattr(args, "recipe", "legacy")
            ranking = getattr(args, "selection_policy", "success")
            placement = validate_recipe(recipe, job_identity(args.job)["manifest"]["tier"],
                                        getattr(args, "acquisition_guidance", "precision"),
                                        getattr(args, "placement_guidance", "off"))
            previous = guard.read_json(output / "request.json")
            deadline = args.deadline_utc or (previous["deadline_utc"] if previous else None)
            strict = getattr(args, "strict_reproducibility", False)
            request = {"schema": "v2d_policy_pipeline_request_v1", "job": str(Path(args.job).resolve()),
                   "runtime": str(Path(args.runtime).resolve()), "output": str(output),
                   "parent_state": str(Path(args.parent_state).resolve()) if args.parent_state else None,
                   "training_budget_seconds": args.training_budget_seconds, "total_budget_seconds": args.total_budget_seconds,
                   "deadline_utc": deadline, "source": source_identity(strict), "runtime_commit": runtime_identity(args.runtime),
                   "dataset_input": str(Path(args.dataset_input).resolve()) if args.dataset_input else None,
                   "submission_kit": str(Path(args.submission_kit).resolve()) if args.submission_kit else None,
                   "selection_rule": selection_rule(ranking), "refinement_transition_cap": REFINEMENT_STEPS,
                   "candidate_subset": "final, latest<=60M additional, latest<=40M additional; deduplicated",
                   "candidate_boundary_tolerance_transitions": WORLDS}
            if getattr(args, "acquisition_guidance", "precision") != "precision":
                request["acquisition_guidance"] = args.acquisition_guidance
            if placement != "off":
                request["placement_guidance"] = placement
            if recipe != "legacy":
                request["recipe"] = recipe
            if strict:
                request["strict_reproducibility"] = True
            for field in ("native_max_seconds", "refinement_max_seconds", "evaluation_max_seconds"):
                if getattr(args, field, None) is not None:
                    request[field] = getattr(args, field)
            if getattr(args, "native_resume", None):
                recovery_locks.enter_context(output_lock(Path(args.recovery_from).resolve() / "pipeline.lock"))
                lineage = bind_native_recovery(args, output, job_identity(args.job), api=sys.modules[__name__])
                request.update(native_resume=lineage["checkpoint"]["state"],
                               recovery_from=lineage["recovery_from"], ledger_path=lineage["ledger_path"],
                               recovery_lineage_sha256=sha(output / "recovery_lineage.json"),
                               candidate_boundary_tolerance_transitions=lineage["world_count"])
            policy = guard.make_budget_policy(args.total_budget_seconds, args.training_budget_seconds)
            immutable_json(output / "request.json", request)
            preserve_source_snapshot(output, request["source"])
            stage = "prepare"
            if not (output / "preparation.json").exists():
                env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "JAX_PLATFORMS": "cpu", "WP_DEVICE": "cpu"}
                subprocess.run([sys.executable, "-m", "training.scripts.pipeline", "--_prepare-request", str(output / "request.json")],
                               cwd=ROOT, env=env, check=True)
            preparation = guard.read_json(output / "preparation.json")
            reverify(request, preparation)
            existing = guard.read_json(status_path, {})
            if existing.get("state") == "complete":
                selected = guard.read_json(output / "selected_checkpoint.json")
                if sha(output / "selected_checkpoint.json") != existing["selected_manifest_sha256"]:
                    raise ValueError("Selected manifest changed")
                verify_selected(selected)
                required_stages = {"refinement"}
                if preparation["parent"] is None:
                    required_stages.add("bootstrap_resume" if preparation.get("native_resume") else "bootstrap")
                if recipe == "tier23-load-transfer":
                    required_stages.add("load_transfer")
                for name in ("bootstrap", "bootstrap_resume", "refinement", "load_transfer"):
                    record = guard.read_json(output / (name + "_complete.json"))
                    if name in required_stages and not record:
                        raise ValueError(f"Completed pipeline is missing its {name} stage record")
                    if record:
                        verify_stage(output, record)
                if request["dataset_input"]:
                    verify_export(existing.get("export"), selected, output / "selected_checkpoint.json")
                if request.get("recovery_lineage_sha256"):
                    verify_recovery_history(output, require_complete=True)
                return selected
            if args.prepare_only:
                guard.write_json(status_path, {"state": "prepared", "updated_utc": guard.utc_now(), "request_sha256": sha(output / "request.json")})
                return preparation
            require_gpu_platform()
            candidate = f"tier{preparation['job']['manifest']['tier']}_episode_{preparation['job']['manifest']['episode_index']:06d}"
            canonical_ledger = ledger_path(request)
            ledger = guard.read_json(canonical_ledger, {"created_utc": guard.utc_now(), "jobs": []})
            guard.ensure_budget_policy(ledger, policy, ledger_exists=canonical_ledger.exists())
            guard.remaining_budget(ledger, candidate, 0)  # Running records fail closed, including stale ones.
            if request.get("recovery_lineage_sha256"):
                immutable_json(Path(request["recovery_from"]) / "native_recovery_claim.json",
                    {"schema": "v2d_single_native_recovery_claim_v1", "output": str(output),
                     "request_sha256": sha(output / "request.json"),
                     "lineage_sha256": request["recovery_lineage_sha256"]})
            else:
                guard.write_json(canonical_ledger, ledger)
            with guard.exclusive_lock(Path(tempfile.gettempdir()) / "v2d-policy-gpu.lock"):
                if guard.gpu_pids():
                    raise RuntimeError("GPU is in use; no worker launched")
                write_progress(output, stage)
                parent = preparation["parent"]
                if parent is None:
                    stage = "bootstrap_resume" if preparation.get("native_resume") else "bootstrap"
                    write_progress(output, stage)
                    parent = train_stage(request, preparation, candidate, stage, guard.read_json(preparation["native_config"]),
                                         parent=preparation.get("native_resume"))
                start = parent["environment_steps"]
                records = []
                schedule = refinement_schedule(request, preparation)
                stage_parent = parent
                for index, (stage, additional, stage_placement) in enumerate(schedule):
                    write_progress(output, stage)
                    _, _, metadata = checkpoint_pair(stage_parent["state"])
                    stage_start = stage_parent["environment_steps"]
                    saved_config = json.loads(metadata["resolved_config_json"])
                    end, _ = transition_budget(stage_start, additional, saved_config["training"]["world_count"])
                    config = build_config(json.loads(metadata["resolved_config_json"]), state=Path(stage_parent["state"]),
                                          output=output / stage, start=stage_start, end=end, phase="refine", checkpoint_interval=CHECKPOINT_INTERVAL,
                                          acquisition_guidance=request.get("acquisition_guidance", "precision"),
                                          acquisition_reference=str(Path(preparation["job"]["job"]) / "reference.npz"),
                                          placement_guidance=stage_placement,
                                          placement_reference=str(Path(preparation["job"]["job"]) / "reference.npz"),
                                          placement_support=str(Path(preparation["job"]["job"]) / "assets/support.usda"))
                    final = train_stage(request, preparation, candidate, stage, config, parent=stage_parent)
                    if index < len(schedule) - 1 and final["environment_steps"] != end:
                        raise RuntimeError("Precision stage stopped short of 60M; load transfer was not started")
                    records.extend(record for record in refinement_checkpoints(output, config, stage)
                                   if stage_start < record["environment_steps"] <= final["environment_steps"])
                    stage_parent = final
                stage = "checkpoint_selection"
                write_progress(output, stage)
                candidates = choose_candidates(records, start, final["environment_steps"],
                                               request["candidate_boundary_tolerance_transitions"])
                immutable_json(output / "candidate_plan.json", {"selection_rule": request["selection_rule"], "candidates": candidates,
                               "refinement_start_steps": start, "clean_reference_used": False,
                               "boundary_tolerance_transitions": request["candidate_boundary_tolerance_transitions"], "boundary_capped_at_final": True})
                stage = "evaluation"
                write_progress(output, stage)
                evaluated = [evaluate_candidate(request, preparation, candidate, record) for record in candidates]
                winner = min(evaluated, key=lambda c: rank_key(c, ranking))
                selected = {"schema": "v2d_selected_policy_v1", **winner,
                            "tier": preparation["job"]["manifest"]["tier"], "episode_index": preparation["job"]["manifest"]["episode_index"],
                            "job": preparation["job"]["job"], "runtime": request["runtime"], "source_commit": request["source"]["commit"],
                            "request_sha256": sha(output / "request.json"), "input_job_sha256": preparation["job"]["job_sha256"],
                            "candidates": evaluated, "selection_rule": request["selection_rule"], "clean_reference_used": False,
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
    parser.add_argument("--native-resume", help="Explicit partial-native checkpoint recovery, preserving saved curriculum and clocks")
    parser.add_argument("--recovery-from", help="Failed original pipeline output; keeps its canonical ledger and charged history")
    parser.add_argument("--native-max-seconds", type=float, help="Optional native stage wall limit")
    parser.add_argument("--refinement-max-seconds", type=float, help="Optional refinement stage wall limit")
    parser.add_argument("--evaluation-max-seconds", type=float, help="Optional per-evaluation wall limit")
    parser.add_argument("--strict-reproducibility", action="store_true", help="Require clean committed execution sources")
    parser.add_argument("--recipe", choices=RECIPES, default="legacy",
                        help="Tier 1: precision90M; Tier 2/3: precision60M then load-transfer30M")
    parser.add_argument("--selection-policy", choices=("success", "tracking"), default="success",
                        help="success: MP/SP/AUC/MPPE; tracking: AUC/MPPE/MP/SP")
    parser.add_argument("--acquisition-guidance", choices=("precision", "grasp-region"), default="precision",
                        help="Opt-in object-attached acquisition regions for Tier 2/3")
    parser.add_argument("--placement-guidance", choices=("off", "supported-release"), default="off",
                        help="Opt-in support-aware placement and release for Tier 2/3")
    parser.add_argument("--training-budget-seconds", type=float, help="Optional cumulative training wall limit")
    parser.add_argument("--total-budget-seconds", type=float, help="Optional cumulative training/evaluation wall limit")
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
        subprocess.run([sys.executable, str(ROOT / "training/evaluate/policy.py"),
                        "--runtime", runtime, "--checkpoint", checkpoint, "--output", output,
                        "--tier", tier, "--episode", episode, "--worlds", "16", "--variant", "automatic_precision"],
                       check=True)
        return
    if not all((args.job, args.runtime, args.output)):
        parser.error("--job, --runtime and --output are required")
    if bool(args.dataset_input) != bool(args.submission_kit):
        parser.error("--dataset-input and --submission-kit must be supplied together")
    if bool(args.native_resume) != bool(args.recovery_from) or (args.native_resume and args.parent_state):
        parser.error("--native-resume and --recovery-from are required together and exclude --parent-state")
    for name in ("native_max_seconds", "refinement_max_seconds", "evaluation_max_seconds"):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value <= 0):
            parser.error(f"{name} must be a positive finite value when supplied")
    guard.make_budget_policy(args.total_budget_seconds, args.training_budget_seconds)
    if args.training_budget_seconds is not None and args.total_budget_seconds is not None and args.training_budget_seconds > args.total_budget_seconds:
        parser.error("Training budget cannot exceed total budget")
    if args.deadline_utc:
        guard.parse_deadline(args.deadline_utc)
    result = run(args)
    print(json.dumps({"output": str(Path(args.output).resolve()), "prepared_only": args.prepare_only,
                      "selected_policy": result.get("policy")}, indent=2))


if __name__ == "__main__":
    main()

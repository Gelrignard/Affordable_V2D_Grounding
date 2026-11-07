#!/usr/bin/env python3
"""Verify or replay Sunfish's frozen Track 2 submissions without changing provenance.

The official verifier remains unchanged. Two standard manifests preserve the two
producing commits. replay_plan.json supplies explicit runtime and input paths for
every episode; no checkpoint, world, source path, or policy is selected here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from functools import lru_cache


LABELS = {"tier1": "track_2_tier1", "tier2": "track_2_tier2", "tier3": "track_2_tier3"}
MANIFEST_FILES = ("submission.json", "submission.tier3.json")
EVALUATION_MODULES = ("__init__.py", "harness.py", "metrics.py", "rollout.py")
UNIFORM = ("eval_script_sha256", "eval_code_sha256", "packer_script_sha256",
           "uploader_script_sha256", "code_commit_url")
POSE = ("pos_x", "pos_y", "pos_z", "quat_x", "quat_y", "quat_z", "quat_w")
ROW = re.compile(r"^track_2/e(\d{6})/w000/f\d{6}/o0$")
COMMIT = re.compile(r"^https://[\w.-]*github[\w.-]*/[\w.-]+/[\w.-]+/commit/([0-9a-fA-F]{40})/?$")
JOB_REQUIRED = {"reference.npz", "motion.parquet", "assets/object.obj",
                "assets/object.urdf", "assets/support.usda"}


class ReplayError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise ReplayError(message)


def read_json(path):
    path = Path(path)
    require(path.is_file(), f"Required JSON missing: {path}")
    with path.open(encoding="utf-8-sig") as handle:
        return json.load(handle)


def sha256(path):
    path = Path(path)
    require(path.is_file(), f"Required file missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def path_value(base, value, field, *, resolve_symlinks=True):
    require(isinstance(value, str) and bool(value.strip()), f"{field} must name an explicit path")
    path = Path(value)
    path = path if path.is_absolute() else Path(base) / path
    return path.resolve() if resolve_symlinks else path.absolute()


def evaluation_python(root, recorded):
    override = os.environ.get("EVAL_PYTHON")
    value = override or recorded
    if override and os.sep not in override:
        value = shutil.which(override)
        require(value is not None, f"EVAL_PYTHON command not found on PATH: {override}")
    python = path_value(root, value, "EVAL_PYTHON" if override else "python", resolve_symlinks=False)
    require(python.is_file() and os.access(python, os.X_OK), f"Python interpreter is not executable: {python}")
    return python


def package_hash(package):
    digest = hashlib.sha256()
    for name in sorted(EVALUATION_MODULES):
        path = Path(package) / name
        require(path.is_file(), f"Evaluation module missing: {path}")
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def read_csv_contract(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"row_id", *POSE, *UNIFORM, "checkpoint_sha256"}
        require(required.issubset(reader.fieldnames or []), f"Required CSV columns missing: {path}")
        rows = list(reader)
    require(bool(rows), f"Empty submitted CSV: {path}")
    ids, checkpoints = [], {}
    for row in rows:
        match = ROW.fullmatch(row["row_id"])
        require(match is not None, f"Unexpected row_id in {path}: {row['row_id']}")
        episode = int(match.group(1))
        digest = row["checkpoint_sha256"]
        require(re.fullmatch(r"[0-9a-f]{64}", digest) is not None,
                f"Invalid checkpoint hash on episode {episode}: {path}")
        require(episode not in checkpoints or checkpoints[episode] == digest,
                f"Multiple checkpoint hashes on episode {episode}: {path}")
        checkpoints[episode] = digest
        pose = [float(row[column]) for column in POSE]
        require(all(math.isfinite(value) for value in pose), f"Nonfinite pose in {path}")
        require(sum(value * value for value in pose[3:]) > 1e-16, f"Zero quaternion in {path}")
        ids.append(row["row_id"])
    require(len(ids) == len(set(ids)), f"Duplicate row IDs in {path}")
    require(set(checkpoints) == set(range(30)), f"Expected all episodes 0..29 in {path}")
    provenance = {}
    for column in UNIFORM:
        values = {row[column] for row in rows}
        require(len(values) == 1 and bool(next(iter(values))), f"Nonuniform or blank {column}: {path}")
        provenance[column] = next(iter(values))
    return {"ids": ids, "checkpoints": checkpoints, "provenance": provenance}


def manifests(root):
    result, owners = {}, {}
    for filename in MANIFEST_FILES:
        path = root / filename
        data = read_json(path)
        require(data.get("schema") == "v2d_container_v1", f"Unsupported manifest schema: {path}")
        require(isinstance(data.get("team"), str) and data["team"].strip(), f"Missing team name: {path}")
        for entry in data.get("submissions", []):
            label = entry.get("label")
            require(label in LABELS and entry.get("track") == LABELS[label], f"Unknown label/track in {path}")
            require(label not in owners, f"Duplicate label across manifests: {label}")
            owners[label] = path
        result[path] = data
    require(set(owners) == set(LABELS), "The two manifests must cover tier1, tier2, and tier3 exactly once")
    require(owners["tier1"].name == owners["tier2"].name == "submission.json",
            "Tier1 and Tier2 must use submission.json")
    require(owners["tier3"].name == "submission.tier3.json", "Tier3 must use submission.tier3.json")
    return result, owners


def check_checkout(code, commit_url):
    match = COMMIT.fullmatch(commit_url)
    require(match is not None, f"Expected a full GitHub commit URL: {commit_url}")
    require((code / ".git").exists(), f"Missing .git in producing checkout: {code}")
    env = dict(os.environ, GIT_OPTIONAL_LOCKS="0")
    command = ["git", "-c", "safe.directory=*", "-C", str(code)]
    head = subprocess.check_output([*command, "rev-parse", "HEAD"], text=True, env=env).strip()
    dirty = subprocess.check_output([*command, "status", "--porcelain", "--untracked-files=no"],
                                    text=True, env=env).strip()
    require(head == match.group(1).lower(), f"Checkout HEAD {head} differs from {commit_url}")
    require(not dirty, f"Producing checkout has modified tracked files: {code}\n{dirty}")


def label_context(root, all_manifests, owners, label):
    manifest_path = owners[label]
    manifest = all_manifests[manifest_path]
    entry = next(entry for entry in manifest["submissions"] if entry["label"] == label)
    base = manifest_path.parent
    code = path_value(base, manifest.get("code", {}).get("path"), "code.path")
    commit_url = manifest.get("code", {}).get("commit_url", "")
    check_checkout(code, commit_url)
    kit = path_value(base, entry.get("kit") or manifest.get("kit"), "kit")
    submitted = path_value(base, entry.get("file"), "file")
    contract = read_csv_contract(submitted)
    require(contract["provenance"]["code_commit_url"] == commit_url,
            f"Manifest commit differs from frozen {label} CSV")
    with (kit / "data/track_2_sample_submission.csv").open(newline="", encoding="utf-8-sig") as handle:
        expected = [row["row_id"] for row in csv.DictReader(handle)]
    require(contract["ids"] == expected, f"Frozen {label} rows differ from the official sample")
    evaluator = entry.get("evaluator", {})
    script = path_value(base, evaluator.get("script"), "evaluator.script")
    package = path_value(base, evaluator.get("evaluation_package"), "evaluator.evaluation_package")
    require(script == (code / "training/evaluate/policy.py").resolve(),
            f"{label} must use its producing checkout's training/evaluate/policy.py")
    for column, path in (("eval_script_sha256", script),
                         ("packer_script_sha256", kit / "tools/pack_submission.py"),
                         ("uploader_script_sha256", kit / "submit/v2d_submit.py")):
        require(sha256(path) == contract["provenance"][column], f"{column} differs from frozen {label} CSV: {path}")
    require(package_hash(package) == contract["provenance"]["eval_code_sha256"],
            f"Manifest evaluation package differs from frozen {label} CSV: {package}")
    checkpoints = entry.get("checkpoints", {})
    require(set(checkpoints) == {str(episode) for episode in range(30)},
            f"{label} manifest must name every checkpoint 0..29")
    frame_references = (path_value(base, entry["frame_references"], "frame_references")
                        if entry.get("frame_references") else None)
    if frame_references is not None:
        require(frame_references.is_dir(), f"Frame reference directory missing: {frame_references}")
    return dict(label=label, manifest_path=manifest_path, entry=entry, code=code,
                kit=kit, submitted=submitted, submitted_hash=sha256(submitted),
                contract=contract, evaluator=script, frame_references=frame_references,
                checkpoints={int(key): path_value(base, value, "checkpoint") for key, value in checkpoints.items()})


def read_plan(root):
    plan = read_json(root / "replay_plan.json")
    require(plan.get("schema") == "sunfish_replay_v1", "Unsupported replay_plan.json schema")
    require(set(plan.get("labels", {})) == set(LABELS), "Replay plan must explicitly contain all three tier labels")
    for label, value in plan["labels"].items():
        require(value.get("track") == LABELS[label], f"Wrong track in replay plan: {label}")
        require(isinstance(value.get("execution_code"), str) and value["execution_code"].strip(),
                f"Replay plan must explicitly name execution_code for {label}")
        require(set(value.get("episodes", {})) == {str(episode) for episode in range(30)},
                f"Replay plan must explicitly contain episodes 0..29 for {label}")
    return plan


def execution_context(root, context, label_plan, all_manifests):
    """Use the requested full source checkout after checking the shared policy core."""
    execution_code = path_value(root, label_plan["execution_code"], "execution_code")
    sources = {path_value(path.parent, data.get("code", {}).get("path"), "code.path"):
               data.get("code", {}).get("commit_url", "") for path, data in all_manifests.items()}
    require(execution_code in sources, "execution_code must be one of the manifest's recorded clean checkouts")
    check_checkout(execution_code, sources[execution_code])
    if execution_code != context["code"]:
        folders = ("training", "data_tools", "retargeting", "correction")
        def tracked(code):
            result = subprocess.check_output(["git", "-c", "safe.directory=*", "-C", str(code),
                                              "ls-files", "-z", "--", *folders])
            return {name.decode("utf-8") for name in result.split(b"\0") if name}
        files = tracked(context["code"])
        require(files == tracked(execution_code), "Executing checkout has a different policy/export dependency file set")
        for name in files:
            produced = (context["code"] / name).read_bytes().replace(b"\r\n", b"\n")
            executing = (execution_code / name).read_bytes().replace(b"\r\n", b"\n")
            require(produced == executing, f"Executing policy/export dependency differs from producing source: {name}")
    script = execution_code / "training/evaluate/policy.py"
    require(sha256(script) == context["contract"]["provenance"]["eval_script_sha256"],
            "Executing evaluator bytes differ from the frozen CSV hash")
    context["execution_code"] = execution_code
    context["execution_commit_url"] = sources[execution_code]
    context["evaluator"] = script
    return context


def selected_episodes(value):
    if value is None:
        return list(range(30))
    try:
        episodes = [int(part.strip()) for part in value.split(",")]
    except ValueError as error:
        raise ReplayError("--episodes must be comma-separated integers 0..29") from error
    require(bool(episodes) and len(episodes) == len(set(episodes)) and all(0 <= e < 30 for e in episodes),
            "--episodes must contain distinct integers 0..29")
    return sorted(episodes)


def validate_job(path, tier, episode):
    data = read_json(path / "job.json")
    require(data.get("schema") == "v2d_policy_job_v1" and data.get("tier") == tier
            and data.get("episode_index") == episode, f"Portable job identity mismatch: {path}")
    files = data.get("files", {})
    require(JOB_REQUIRED.issubset(files), f"Portable job lacks required assets: {path}")
    for name, digest in files.items():
        candidate = (path / name).resolve()
        require(candidate.is_relative_to(path), f"Portable job asset escapes its directory: {name}")
        require(sha256(candidate) == digest, f"Portable job asset hash mismatch: {candidate}")
    return data


def validate_dataset_input(path, job, tier, episode):
    """Catch incompatible exporter inputs before starting a GPU rollout."""
    if tier in (1, 2):
        require((path / "meta").is_dir(),
                f"Tier{tier} exporter input requires the original LeRobot dataset, not frame-reference NPZs: {path}")
        manifest = read_json(path / "download_manifest.json")
        require(manifest.get("repository") == "nvidia/video_to_data_challenge"
                and manifest.get("tier") == tier and bool(manifest.get("revision")),
                f"Released input provenance does not match Tier{tier}: {path}")
        matches = list((path / "data").rglob(f"episode_{episode:06d}.parquet"))
        require(len(matches) == 1, f"Expected one released input Parquet for episode {episode}: {path}")
        source = matches[0]
        digest = sha256(source)
        require(manifest.get("files", {}).get(source.relative_to(path).as_posix(), {}).get("sha256") == digest
                and job.get("source_sha256") == digest,
                f"Released input Parquet differs from download or portable-job provenance: {source}")
    else:
        manifest = read_json(path / "input_frame_manifest.json")
        require(manifest.get("schema") == "v2d_track2_input_frames_v1"
                and manifest.get("tier") == 3 and manifest.get("coordinate_frame") == "released_track2_world",
                f"Prepared Tier3 input-frame manifest is invalid: {path}")
        record = manifest.get("episodes", {}).get(str(episode), {})
        source = path / f"episode_{episode:06d}.npz"
        require(record.get("sequence_id") == job.get("sequence_id")
                and record.get("source_sha256") == job.get("source_sha256")
                and manifest.get("files", {}).get(source.name) == sha256(source),
                f"Prepared Tier3 input differs from portable-job provenance: {source}")


def validate_checkpoint_robot_asset(checkpoint, runtime):
    """Check the saved G1 asset identity without importing a GPU dependency."""
    with checkpoint.open("rb") as handle:
        length_bytes = handle.read(8)
        require(len(length_bytes) == 8, f"Incomplete safetensors header: {checkpoint}")
        length = struct.unpack("<Q", length_bytes)[0]
        require(0 < length <= 16 * 1024 * 1024, f"Unexpected safetensors header size: {checkpoint}")
        header = json.loads(handle.read(length))
    metadata = header.get("__metadata__", {})
    config = json.loads(metadata["user.resolved_config_json"])
    embodiment = config["embodiment"]["config"]
    require(embodiment.get("asset_id") == "g1_dex3_upstream_main",
            f"Unsupported checkpoint robot asset identity: {checkpoint}")
    asset_path = embodiment.get("urdf_path")
    require(asset_path is None, f"Explicit checkpoint robot URDF path needs a reviewed replay mapping: {asset_path}")
    urdf = runtime / "robotic_grounding/flash_chord/src/flash_chord/assets/urdfs/g1/main_with_hand.urdf"
    require(sha256(urdf) == embodiment.get("asset_sha256"),
            f"Runtime G1 URDF bytes differ from the checkpoint's saved asset hash: {checkpoint}; {urdf}")


def alias_paths(root, item):
    require(isinstance(item, dict) and set(item) == {"alias", "target"},
            "Each path_aliases item must contain exactly alias and target")
    alias = Path(item["alias"])
    require(alias.is_absolute() and ".." not in alias.parts, f"Alias must be an absolute container path: {alias}")
    protected = (root, Path("/opt"), Path("/data"), Path("/output"), Path("/usr"), Path("/bin"),
                 Path("/sbin"), Path("/etc"), Path("/dev"), Path("/proc"), Path("/sys"))
    require(alias != Path(alias.anchor) and not any(alias == p or alias.is_relative_to(p) for p in protected),
            f"Alias cannot replace protected container paths: {alias}")
    target = path_value(root, item["target"], "alias.target")
    require(target.is_relative_to(root) and target.exists(), f"Alias target must exist under {root}: {target}")
    return alias, target


def episode_plan(root, context, raw, episode):
    fields = {"python", "runtime", "job", "dataset_input", "source_root", "worlds", "path_aliases"}
    require(isinstance(raw, dict) and fields.issubset(raw),
            f"Incomplete replay plan for {context['label']} episode {episode}; required: {sorted(fields)}")
    worlds = raw["worlds"]
    require(type(worlds) is int and 1 <= worlds <= 64,
            f"Recorded worlds must be explicitly set to 1..64: {context['label']} episode {episode}")
    result = {field: path_value(root, raw[field], field) for field in ("runtime", "job", "dataset_input")}
    # Keep the venv launch path for both dependency discovery and evaluation.
    result["python"] = evaluation_python(root, raw["python"])
    require(result["runtime"].is_dir(), f"Runtime missing: {result['runtime']}")
    require(result["dataset_input"].is_dir(), f"Episode input reference missing: {result['dataset_input']}")
    source_root = None if raw["source_root"] is None else path_value(root, raw["source_root"], "source_root")
    if source_root is not None:
        require(source_root.is_dir(), f"source_root missing: {source_root}")
    runtime_package = result["runtime"] / "robotic_grounding/flash_chord/src/flash_chord/evaluation"
    require(package_hash(runtime_package) == context["contract"]["provenance"]["eval_code_sha256"],
            f"Episode runtime hash differs from frozen {context['label']} CSV: {runtime_package}")
    tier = int(context["label"][-1])
    job = validate_job(result["job"], tier, episode)
    validate_dataset_input(result["dataset_input"], job, tier, episode)
    checkpoint = context["checkpoints"][episode]
    require(sha256(checkpoint) == context["contract"]["checkpoints"][episode],
            f"Checkpoint differs from frozen CSV: {checkpoint}")
    validate_checkpoint_robot_asset(checkpoint, result["runtime"])
    require(isinstance(raw["path_aliases"], list), "path_aliases must be an explicit list (possibly empty)")
    aliases = [alias_paths(root, alias) for alias in raw["path_aliases"]]
    return dict(result, source_root=source_root, worlds=worlds, aliases=aliases, checkpoint=checkpoint)


def prepare_aliases(plans, *, create):
    aliases = {}
    for plan in plans.values():
        for alias, target in plan["aliases"]:
            require(alias not in aliases or aliases[alias] == target, f"Conflicting path alias: {alias}")
            aliases[alias] = target
    for alias, target in aliases.items():
        if alias.exists() or alias.is_symlink():
            require(alias.resolve() == target, f"Existing path alias points elsewhere; will not replace it: {alias}")
        elif create:
            alias.parent.mkdir(parents=True, exist_ok=True)
            alias.symlink_to(target, target_is_directory=target.is_dir())


@lru_cache(maxsize=8)
def python_nvidia_libraries(python, pythonpath=None):
    code = """
import importlib.util
import os
from pathlib import Path
import sysconfig

# Namespace paths include packages exposed through .pth files or user sites.
spec = importlib.util.find_spec('nvidia')
roots = list(spec.submodule_search_locations or []) if spec is not None else []
roots.extend(str(Path(sysconfig.get_path(name)) / 'nvidia') for name in ('purelib', 'platlib'))
libraries = []
for root in dict.fromkeys(roots):
    for path in sorted(Path(root).glob('*/lib')):
        if path.is_dir() and str(path) not in libraries:
            libraries.append(str(path))
print(os.pathsep.join(libraries))
"""
    env = None if pythonpath is None else dict(os.environ, PYTHONPATH=pythonpath)
    return subprocess.check_output([str(python), "-c", code], text=True, env=env).strip()


def environment(context, plan):
    env = dict(os.environ)
    env.update(PYTHONDONTWRITEBYTECODE="1", XLA_PYTHON_CLIENT_PREALLOCATE="false", MUJOCO_GL="egl",
               HYDRA_FULL_ERROR="1", PYTHONPATH=os.pathsep.join((str(context["execution_code"]),
               str(plan["runtime"] / "robotic_grounding/flash_chord/src"))))
    libraries = python_nvidia_libraries(str(plan["python"]), env["PYTHONPATH"])
    env["LD_LIBRARY_PATH"] = os.pathsep.join(value for value in (libraries, env.get("LD_LIBRARY_PATH", "")) if value)
    uid = str(os.getuid()) if hasattr(os, "getuid") else str(os.getpid())
    cache = Path(tempfile.gettempdir()) / ("sunfish-cache-" + uid)
    for key, name in (("XDG_CACHE_HOME", "xdg"), ("JAX_COMPILATION_CACHE_DIR", "jax"),
                      ("CUDA_CACHE_PATH", "cuda"), ("TRITON_CACHE_DIR", "triton"), ("NUMBA_CACHE_DIR", "numba")):
        path = cache / name
        path.mkdir(parents=True, exist_ok=True)
        require(os.access(path, os.W_OK), f"Runtime cache is not writable: {path}")
        env[key] = str(path)
    return env


def execute(command, *, cwd=None, env=None, log=None):
    command = [str(value) for value in command]
    print("Running: " + json.dumps(command), flush=True)
    if log is None:
        subprocess.run(command, cwd=cwd, env=env, check=True)
    else:
        with Path(log).open("x", encoding="utf-8") as handle:
            completed = subprocess.run(command, cwd=cwd, env=env, stdout=handle, stderr=subprocess.STDOUT)
        require(completed.returncode == 0, f"Command failed with exit {completed.returncode}; see {log}")


def validate_export_provenance(report, context, episode):
    require(report.get("schema") == "v2d_policy_rollout_export_v1"
            and report.get("state") == "complete" and report.get("episode") == episode
            and report.get("tier") == int(context["label"][-1])
            and report.get("source_world_id") == 0 and report.get("exported_world_count") == 1,
            f"Invalid fixed-world export manifest for {context['label']} episode {episode}")
    provenance = report.get("recorded_evaluator_provenance", {})
    for column in ("eval_script_sha256", "eval_code_sha256"):
        require(provenance.get(column) == context["contract"]["provenance"][column],
                f"Reproduced export {column} differs from frozen CSV")
    require(provenance.get("checkpoint_sha256") == context["contract"]["checkpoints"][episode],
            "Reproduced export checkpoint hash differs from frozen CSV")


def frozen_inputs_unchanged(context, episodes):
    require(sha256(context["submitted"]) == context["submitted_hash"], "Frozen submitted CSV changed during replay")
    for episode in episodes:
        require(sha256(context["checkpoints"][episode]) == context["contract"]["checkpoints"][episode],
                f"Frozen checkpoint changed during replay: episode {episode}")
    check_checkout(context["code"], context["contract"]["provenance"]["code_commit_url"])
    if context.get("execution_code") != context["code"]:
        check_checkout(context["execution_code"], context["execution_commit_url"])


def verify(root, all_manifests, owners, args):
    verifier = root / "verifier_kit/tools/verify_container.py"
    require(verifier.is_file(), f"Official verifier missing: {verifier}")
    require(args.dataset_root.is_dir(), f"Mounted challenge dataset missing: {args.dataset_root}")
    reproduced = {}
    for value in args.reproduced:
        require("=" in value, "--reproduced requires LABEL=FILE")
        label, path = value.split("=", 1)
        require(label in owners and label not in reproduced, f"Unknown/duplicate reproduced label: {label}")
        reproduced[label] = Path(path).resolve()
    if args.report_dir:
        args.report_dir.mkdir(parents=True, exist_ok=True)
    failed = []
    for manifest in all_manifests:
        command = [sys.executable, verifier, "--manifest", manifest]
        if args.dataset_root:
            command += ["--dataset-root", args.dataset_root.resolve()]
        if args.kaggle:
            command += ["--kaggle"]
        for label, path in reproduced.items():
            if owners[label] == manifest:
                command += ["--reproduced", f"{label}={path}"]
        if args.report_dir:
            command += ["--report", args.report_dir / (manifest.stem + ".verification.json")]
        try:
            execute(command)
        except subprocess.CalledProcessError:
            failed.append(manifest.name)
    require(not failed, "Official verification failed for: " + ", ".join(failed))


def check_plan(root, all_manifests, owners, args):
    plan = read_plan(root)
    for label in ([args.label] if args.label else LABELS):
        context = label_context(root, all_manifests, owners, label)
        context = execution_context(root, context, plan["labels"][label], all_manifests)
        selected = {episode: episode_plan(root, context, plan["labels"][label]["episodes"][str(episode)], episode)
                    for episode in range(30)}
        prepare_aliases(selected, create=False)
        print(f"PASS {label}: all 30 explicit plans, jobs, checkpoints, evaluator and runtime hashes checked", flush=True)


def reproduce(root, all_manifests, owners, args):
    context = label_context(root, all_manifests, owners, args.label)
    plan = read_plan(root)
    context = execution_context(root, context, plan["labels"][args.label], all_manifests)
    episodes = selected_episodes(args.episodes)
    selected = {episode: episode_plan(root, context, plan["labels"][args.label]["episodes"][str(episode)], episode)
                for episode in episodes}
    dataset_root = args.dataset_root.resolve()
    require(dataset_root.is_dir(), f"Mounted challenge dataset missing: {dataset_root}")
    output = args.output.resolve()
    require(not output.is_relative_to(root) and not output.is_relative_to(dataset_root),
            "Output must be outside immutable /submission and mounted challenge dataset")
    run_root = output / args.label
    require(not run_root.exists(), f"Output already exists; use a fresh --output directory: {run_root}")
    prepare_aliases(selected, create=False)
    run_root.mkdir(parents=True)
    (run_root / "logs").mkdir()
    collected = run_root / "rollouts"
    collected.mkdir()
    receipt = {"schema": "sunfish_replay_receipt_v1", "started_utc": datetime.now(timezone.utc).isoformat(),
               "label": args.label, "episodes": episodes, "frozen_csv_sha256": context["submitted_hash"],
               "replay_plan_sha256": sha256(root / "replay_plan.json"), "completed_episodes": [], "state": "running"}
    receipt_path = run_root / "replay_receipt.json"
    try:
        prepare_aliases(selected, create=True)
        for episode, episode_data in selected.items():
            stem = f"episode_{episode:06d}"
            evaluation = run_root / "evaluations" / stem
            export = run_root / "exports" / stem
            env = environment(context, episode_data)
            command = [episode_data["python"], context["evaluator"], "--checkpoint", episode_data["checkpoint"],
                       "--runtime", episode_data["runtime"], "--output", evaluation, "--worlds", str(episode_data["worlds"]),
                       "--tier", args.label[-1], "--episode", str(episode)]
            if episode_data["source_root"] is not None:
                command += ["--source-root", episode_data["source_root"]]
            execute(command, cwd=episode_data["runtime"] / "robotic_grounding/flash_chord", env=env,
                    log=run_root / "logs" / (stem + ".evaluate.log"))
            command = [episode_data["python"], "-m", "training.evaluate.export", "--kit", context["kit"],
                       "--output-dir", export, "--dataset-input", episode_data["dataset_input"],
                       "--arrays", evaluation / "rollout_arrays.npz", "--native-trajectory", evaluation / "object_rollouts.parquet",
                       "--job", episode_data["job"], "--episode", str(episode), "--checkpoint", episode_data["checkpoint"],
                       "--world-id", "0"]
            execute(command, cwd=context["execution_code"], env=env, log=run_root / "logs" / (stem + ".export.log"))
            export_report = read_json(export / "export_manifest.json")
            validate_export_provenance(export_report, context, episode)
            artifact = export / (stem + ".parquet")
            require(export_report.get("artifact", {}).get("filename") == artifact.name
                    and export_report["artifact"].get("sha256") == sha256(artifact), "Export artifact identity mismatch")
            shutil.copyfile(artifact, collected / artifact.name)
            receipt["completed_episodes"].append(episode)
            frozen_inputs_unchanged(context, [episode])
            print(f"PASS {args.label} episode {episode}: fixed world 0 exported with frozen provenance", flush=True)
        if episodes == list(range(30)):
            packed = run_root / context["submitted"].name
            command = [sys.executable, context["kit"] / "tools/pack_submission.py", "--track", LABELS[args.label],
                       "--results", collected, "--dataset-root", dataset_root, "--code-commit",
                       context["contract"]["provenance"]["code_commit_url"], "--eval-script", context["evaluator"], "-o", packed]
            if context["frame_references"] is not None:
                command += ["--frame-references", context["frame_references"]]
            execute(command, cwd=context["kit"], log=run_root / "logs/pack.log")
            recreated = read_csv_contract(packed)
            require(recreated == context["contract"], "Reproduced rows or provenance differ from frozen submission")
            execute([sys.executable, root / "verifier_kit/tools/verify_container.py", "--manifest", context["manifest_path"],
                     "--dataset-root", dataset_root, "--reproduced", f"{args.label}={packed}",
                     "--report", run_root / "verification.json"])
            receipt["reproduced_file"] = str(packed)
            receipt["reproduced_sha256"] = sha256(packed)
            receipt["state"] = "complete"
        else:
            receipt["state"] = "partial_episodes_complete"
            print("Requested episode subset completed; no complete submission was packed.", flush=True)
        frozen_inputs_unchanged(context, episodes)
    except BaseException as error:
        receipt["state"] = "failed"
        receipt["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        receipt["finished_utc"] = datetime.now(timezone.utc).isoformat()
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(f"Replay receipt: {receipt_path}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission-root", type=Path, default=Path("/submission"))
    commands = parser.add_subparsers(dest="command", required=True)
    verify_parser = commands.add_parser("verify", help="Run both unchanged official manifest verifications")
    verify_parser.add_argument("--dataset-root", type=Path, default=Path("/data"))
    verify_parser.add_argument("--reproduced", action="append", default=[], metavar="LABEL=FILE")
    verify_parser.add_argument("--report-dir", type=Path)
    verify_parser.add_argument("--kaggle", action="store_true", help="Optional network comparison against Kaggle copies")
    plan_parser = commands.add_parser("check-plan", help="CPU-only checks of all frozen inputs and explicit plans")
    plan_parser.add_argument("--label", choices=LABELS)
    replay_parser = commands.add_parser("reproduce", help="Replay policies, export fixed world 0, pack and verify")
    replay_parser.add_argument("--label", required=True, choices=LABELS)
    replay_parser.add_argument("--dataset-root", required=True, type=Path)
    replay_parser.add_argument("--output", required=True, type=Path)
    replay_parser.add_argument("--episodes", help="Optional comma-separated subset; only a full 0..29 run is packed")
    args = parser.parse_args(argv)
    root = args.submission_root.resolve()
    try:
        all_manifests, owners = manifests(root)
        {"verify": verify, "check-plan": check_plan, "reproduce": reproduce}[args.command](root, all_manifests, owners, args)
    except (ReplayError, OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

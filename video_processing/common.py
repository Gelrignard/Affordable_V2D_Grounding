"""Small shared contracts; importing this module never loads a GPU model."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

UPSTREAM_COMMIT = "193382a184308f6800331f3db829f76dc4294cfe"
NATIVE_RELATIVE = "reconstruction/modules/v2d_cari4d/lib/cari4d/prep/run_foundationpose_mhr_export.py"
NATIVE_SHA256 = "b37a5575b83a3269c7f8d8482cf7f61600c9ca1c1430c922e012fb010ac4dde8"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def identity(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": digest(path), "bytes": path.stat().st_size}


def adapter_identity():
    root = Path(__file__).resolve().parent
    return {path.relative_to(root).as_posix(): digest(path) for path in sorted(root.rglob("*.py"))}


def verify_upstream(root):
    root = Path(root).resolve(strict=True)
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()
    if git("rev-parse", "HEAD") != UPSTREAM_COMMIT:
        raise ValueError(f"Use the pinned video_to_data commit {UPSTREAM_COMMIT}")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise ValueError("Use an unmodified pinned upstream checkout; compatibility is applied in memory")
    if digest(root / NATIVE_RELATIVE) != NATIVE_SHA256:
        raise ValueError("FoundationPose source differs from the audited API")
    return root


def runtime_environment(root, weights):
    root, weights = Path(root).resolve(), Path(weights).resolve()
    modules = root / "reconstruction/modules"
    paths = [Path(__file__).resolve().parents[1], modules / "v2d_cari4d/lib/cari4d",
             modules / "v2d_sam3d_body/lib", modules / "v2d_foundation_pose/lib/FoundationPose"]
    env = os.environ.copy()
    env.update(PYTHONPATH=os.pathsep.join([*(str(p) for p in paths), env.get("PYTHONPATH", "")]),
               SAM3D_BODY_ROOT=str(paths[2]), MHR_ASSETS_ROOT=str(weights / "sam3d_body"),
               FOUNDATIONPOSE_WEIGHTS_DIR=str(weights / "foundationpose/nvlabs_pytorch"),
               HF_HOME=str(weights / "hf_home"), TORCH_HOME=str(weights / "sam3d_body/torch_home"),
               HF_HUB_OFFLINE="1", OPENCV_IO_ENABLE_OPENEXR="1", PYOPENGL_PLATFORM="egl",
               HDF5_USE_FILE_LOCKING="FALSE", PYTHONUNBUFFERED="1")
    return env


def activate_runtime(root, weights):
    env = runtime_environment(root, weights)
    os.environ.update(env)
    for path in reversed(env["PYTHONPATH"].split(os.pathsep)):
        if path and path not in sys.path:
            sys.path.insert(0, path)


def fresh_directory(path):
    path = Path(path).resolve()
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Preserve existing results; choose an empty output directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path

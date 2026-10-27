"""Print a Linux native-policy installation plan; never install or initialize CUDA.

First fetch the pinned runtime/assets with ``python -m training.bootstrap``.
Then, on the destination Linux host, for example::

    python -m training.setup_policy_runtime --runtime vendor/flash-runtime \
        --venv .native-policy-env --profile cuda12 --driver-major 570

Review and run the printed commands in order. CUDA 12 reproduces the measured
GPU package versions in a fresh Python 3.12 environment, without the old host's
Torch/base-environment overlays. CUDA 13 uses NVIDIA's frozen uv.lock directly.
The CUDA 12 constraints pin direct/core dependencies, not every transitive
package; a fresh installation still needs its own bounded native GPU smoke run.
The CPU import check printed here does not establish GPU/SONIC compatibility.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import tomllib
from urllib.parse import urlsplit

from training import RUNTIME_COMMIT


ORT_CUDA12_INDEX = "https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/"
ORT_CUDA12_WHEEL_SHA256 = "6e33c999981b746684fea80f3cc3d42dd651a76431a7bb4b03368a58e2db5b88"
CONSTRAINTS = Path(__file__).with_name("requirements-native-cuda12.txt")


def _name(requirement):
    return re.split(r"[\[<>=!~; ]", requirement, maxsplit=1)[0].lower().replace("_", "-")


def _tfp_wheel(lock):
    package = next(p for p in lock["package"] if p["name"] == "tfp-nightly")
    if package["version"] != "0.26.0.dev20260705":
        raise ValueError("unexpected TFP version in pinned runtime lock")
    wheel = next(w for w in package["wheels"] if w["url"].endswith("none-any.whl"))
    url = urlsplit(wheel["url"])
    if url.scheme != "https" or url.hostname != "files.pythonhosted.org" or url.username:
        raise ValueError("TFP wheel must use the public upstream PyPI host")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", wheel["hash"]):
        raise ValueError("missing TFP wheel SHA256")
    return wheel["url"] + "#" + wheel["hash"].replace(":", "=", 1)


def build_plan(runtime, venv, profile, driver_major):
    """Read Git/TOML only. Return commands; deliberately never execute them."""
    runtime, venv = Path(runtime).resolve(), Path(venv).resolve()
    if profile not in ("cuda12", "cuda13"):
        raise ValueError("profile must be cuda12 or cuda13")
    minimum = 570 if profile == "cuda12" else 580
    if driver_major < minimum:
        raise ValueError(f"{profile} plan requires driver major >= {minimum}")
    if venv.exists():
        raise ValueError("choose a new environment path; existing environments are not overwritten")
    head = subprocess.check_output(["git", "-C", str(runtime), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(runtime), "status", "--porcelain", "--untracked-files=no"], text=True)
    if head != RUNTIME_COMMIT or dirty.strip():
        raise ValueError("runtime must be clean at the documented pin; use training.bootstrap")
    project = runtime / "robotic_grounding/flash_chord"
    metadata = tomllib.loads((project / "pyproject.toml").read_text(encoding="utf-8"))
    lock_path = project / "uv.lock"
    lock = tomllib.loads(lock_path.read_text(encoding="utf-8"))
    base = metadata["project"]["dependencies"]
    train = metadata["project"]["optional-dependencies"]["train"]
    if "jax[cuda13]==0.10.2" not in train or "onnxruntime-gpu==1.28.0" not in base:
        raise ValueError("runtime dependency contract changed")
    python = str(venv / "bin/python")
    commands = []
    notes = ["Linux x86_64, Python 3.12; create a fresh environment.",
             "Fetch the pinned G1 and SONIC LFS assets using training.bootstrap --download-assets first.",
             "No GPU check or training runs while this plan is generated."]
    if profile == "cuda13":
        commands.append(["env", "UV_PROJECT_ENVIRONMENT=" + str(venv), "uv", "sync", "--project", str(project),
                         "--frozen", "--extra", "train", "--no-dev", "--python", "3.12"])
        notes.append("Install uv separately; this profile uses the official lock without CUDA 12 substitutions.")
    else:
        commands.append(["python3.12", "-m", "venv", str(venv)])
        # Do not ask pip for the runtime train extra: that explicitly selects CUDA 13.
        requirements = [r for r in base + train if _name(r) not in ("jax", "onnxruntime-gpu")]
        # ORT is installed with --no-deps from its CUDA 12 feed, so resolve its
        # ordinary Python dependencies in the main resolver transaction.
        requirements += ["jax[cuda12]==0.10.2", _tfp_wheel(lock), "flatbuffers==25.12.19",
                         "packaging==23.2", "protobuf==6.31.0", "sympy==1.14.0",
                         "nvidia-cuda-cccl-cu12==12.9.27", "nvidia-curand-cu12==10.3.9.55"]
        commands.append([python, "-m", "pip", "install", "--index-url", "https://pypi.org/simple",
                         "--constraint", str(CONSTRAINTS.resolve()), *requirements])
        commands.append([python, "-m", "pip", "install", "--no-deps", "--index-url", ORT_CUDA12_INDEX,
                         "onnxruntime-gpu==1.28.0"])
        commands.append([python, "-m", "pip", "install", "--no-deps", str(project)])
        notes += ["CUDA 12 core constraints were read from the effective working environment on 2026-10-27; unrelated base packages are excluded.",
                  "TFP wheel is pinned by the upstream lock URL and SHA256; ORT must use the separate Microsoft feed.",
                  "Expected CPython 3.12 Linux x86_64 ORT wheel SHA256: " + ORT_CUDA12_WHEEL_SHA256,
                  "Do not install legacy Torch requirements into this native JAX environment."]
    commands.append([python, "-m", "pip", "check"])
    commands.append(["env", "CUDA_VISIBLE_DEVICES=", "JAX_PLATFORMS=cpu", python, "-c",
                     "import jax, flax, optax, distrax, onnxruntime, warp, flash_chord; print('Native imports OK; GPU execution remains untested')"])
    library_code = "import pathlib,sysconfig; print(':'.join(str(p) for p in sorted((pathlib.Path(sysconfig.get_paths()['purelib'])/'nvidia').glob('*/lib'))))"
    activation = ["export PATH=" + shlex.quote(str(venv / "bin")) + ':"$PATH"',
                  "export LD_LIBRARY_PATH=\"$(" + shlex.join([python, "-c", library_code]) + ')${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}\"']
    return {"schema": "v2d_native_runtime_setup_plan_v1", "runtime_commit": head,
            "profile": profile, "driver_major": driver_major, "python": "3.12", "venv": str(venv),
            "runtime_lock_sha256": hashlib.sha256(lock_path.read_bytes()).hexdigest(),
            "cuda12_constraints_sha256": hashlib.sha256(CONSTRAINTS.read_bytes()).hexdigest() if profile == "cuda12" else None,
            "commands": commands, "activation": activation, "notes": notes}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--venv", required=True)
    parser.add_argument("--profile", choices=["cuda12", "cuda13"], required=True)
    parser.add_argument("--driver-major", required=True, type=int)
    parser.add_argument("--json", action="store_true", help="print a structured plan instead of shell commands")
    args = parser.parse_args()
    plan = build_plan(args.runtime, args.venv, args.profile, args.driver_major)
    if args.json:
        print(json.dumps(plan, indent=2))
    else:
        for note in plan["notes"]:
            print("# " + note)
        for command in plan["commands"]:
            print(shlex.join(command))
        print("# Activate before running the native pipeline from this repository root:")
        print("\n".join(plan["activation"]))


if __name__ == "__main__":
    main()

"""Fetch only the pinned simulator, G1 assets and SONIC when explicitly invoked."""

import argparse
import os
import subprocess
from pathlib import Path

from training import RUNTIME_COMMIT


def bootstrap(destination, download_assets=False):
    path = Path(destination).resolve()
    env = dict(os.environ, GIT_LFS_SKIP_SMUDGE="1")

    def git(*args):
        subprocess.run(["git", "-C", str(path), *args], check=True, env=env)

    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "--no-checkout",
                "https://github.com/nvidia-isaac/video_to_data.git",
                str(path),
            ],
            check=True,
            env=env,
        )
        git("config", "core.autocrlf", "false")
        git("sparse-checkout", "init", "--cone")
        git("sparse-checkout", "set", "robotic_grounding/flash_chord")
        git("checkout", "--detach", RUNTIME_COMMIT)
    else:
        head = subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
        ).strip()
        status = subprocess.check_output(
            ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"],
            text=True,
        )
        if head != RUNTIME_COMMIT or status.strip():
            raise ValueError(
                "existing runtime must already be clean and pinned; choose a new directory"
            )
    if download_assets:
        git("lfs", "version")
        git(
            "lfs",
            "pull",
            "--include=robotic_grounding/flash_chord/src/flash_chord/assets/meshes/g1/**,"
            "robotic_grounding/flash_chord/src/flash_chord/assets/policies/sonic/*.onnx",
            "--exclude=",
        )
    print(
        f"Pinned runtime: {path}\nFollow docs/training/POLICY_HANDOFF.md for the native environment; no training started."
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--download-assets", action="store_true")
    args = parser.parse_args(argv)
    bootstrap(args.runtime, args.download_assets)


if __name__ == "__main__":
    main()

"""Load imperfect references without converting rejected rollouts into experts."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def inside(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"path escapes bundle: {relative}")
    return path


def arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


@dataclass
class EpisodeBundle:
    root: Path
    manifest: dict
    reference: dict
    guidance: dict
    variant: str

    @classmethod
    def load(cls, root, tier, episode, variant="corrected", verify=True):
        if tier not in (1, 2, 3) or variant not in ("original", "corrected"):
            raise ValueError("invalid tier or reference variant")
        root = Path(root).resolve()
        index = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if index.get("schema") != "v2d_rl_pilot_v1":
            raise ValueError("expected v2d_rl_pilot_v1 bundle")
        entries = [
            e
            for e in index["entries"]
            if e["tier"] == tier and e["episode_index"] == episode
        ]
        if len(entries) != 1:
            raise ValueError(f"expected exactly one tier {tier}, episode {episode}")
        entry = entries[0]
        folder = inside(root, entry["path"])
        if verify and sha256(folder / "manifest.json") != entry["manifest_sha256"]:
            raise ValueError("episode manifest hash mismatch")
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        if (manifest["tier"], manifest["episode_index"]) != (tier, episode) or manifest[
            "source"
        ]["tier"] != tier:
            raise ValueError("tier/episode provenance mismatch")
        if variant == "corrected" and not manifest.get("corrected_reference_available"):
            raise ValueError(
                "corrected reference unavailable; choose original explicitly"
            )
        if verify:
            for name, record in manifest["files"].items():
                if sha256(inside(folder, name)) != record["sha256"]:
                    raise ValueError(f"bundle hash mismatch: {name}")
        ref = arrays(folder / f"{variant}_reference.npz")
        guide = (
            arrays(folder / "contact_guidance.npz")
            if (folder / "contact_guidance.npz").exists()
            else {}
        )
        validate_reference(ref)
        if guide:
            if not np.array_equal(guide["timestamps"], ref["timestamps"]):
                raise ValueError("guidance and motion timelines differ")
            for key in (
                "force_ff",
                "robot_local_points",
                "object_local_points",
                "inward_normals_local",
            ):
                if not np.isfinite(guide[key]).all():
                    raise ValueError(f"nonfinite guidance: {key}")
        return cls(folder, manifest, ref, guide, variant)


def validate_reference(ref):
    time = ref["timestamps"]
    n = len(time)
    if n < 2 or not np.isfinite(time).all() or np.any(np.diff(time) <= 0):
        raise ValueError("timestamps must be finite and strictly increasing")
    frames = ref["source_frame_indices"]
    if frames.shape != (n,) or np.any(np.diff(frames) != 1):
        raise ValueError("reference must retain consecutive source frames")
    names = ref["joint_names"].tolist()
    if len(names) != len(set(names)) or len(names) != 43:
        raise ValueError("expected 43 unique named G1/Dex3 joints")
    shapes = {
        "joint_positions": (n, 43),
        "joint_velocity": (n, 43),
        "root_position": (n, 3),
        "root_wxyz": (n, 4),
        "object_position": (n, 3),
        "object_wxyz": (n, 4),
        "object_observed_mask": (n,),
        "world_from_source_cv": (4, 4),
    }
    for key, shape in shapes.items():
        if ref[key].shape != shape or not np.isfinite(ref[key]).all():
            raise ValueError(f"invalid {key}: expected finite {shape}")
    for key in ("root_wxyz", "object_wxyz"):
        if not np.allclose(np.linalg.norm(ref[key], axis=-1), 1, atol=1e-4):
            raise ValueError(f"{key} must be unit wxyz quaternions")
    if (
        str(ref["quaternion_order"]) != "wxyz"
        or str(ref["coordinate_frame"]) != "world_z_up"
    ):
        raise ValueError("unsupported quaternion or world convention")


def inventory(root, tiers=(1, 2), verify=True):
    index = json.loads((Path(root) / "manifest.json").read_text())
    result = []
    for entry in index["entries"]:
        if entry["tier"] in tiers:
            bundle = EpisodeBundle.load(
                root, entry["tier"], entry["episode_index"], "original", verify
            )
            result.append(
                {
                    **entry,
                    "joint_count": len(bundle.reference["joint_names"]),
                    "missing_object_frames": int(
                        (~bundle.reference["object_observed_mask"]).sum()
                    ),
                    "usage": "soft_reference_not_expert_actions",
                }
            )
    return result

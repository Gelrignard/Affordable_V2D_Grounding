"""Select an available portable reference without promoting its quality flags.

Contact and correction-acceptance diagnostics are warnings, not eligibility
gates. Missing correction permits the original; corrupt or unsupported inputs
remain errors and are never silently replaced.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys

from training import RUNTIME_COMMIT


def _sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_portable_input(path):
    """Verify the portable artifact contract; native scene checks run later."""
    root = Path(path).resolve()
    data = json.loads((root / "job.json").read_text(encoding="utf-8"))
    if (data.get("schema") != "v2d_policy_job_v1" or data.get("obsolete")
            or type(data.get("tier")) is not int or data["tier"] not in (1, 2, 3)
            or type(data.get("episode_index")) is not int or not 0 <= data["episode_index"] < 30
            or data.get("variant") not in ("corrected", "original")
            or data.get("runtime_commit") != RUNTIME_COMMIT
            or data.get("model_scope") != "single_rigid_object_g1_dex3"):
        raise ValueError(f"Unsupported portable job: {root}")
    required = {"motion.parquet", "reference.npz", "guidance.npz", "geometry.npz",
                "assets/object.obj", "assets/object.urdf", "assets/support.usda"}
    files = data.get("files", {})
    if not isinstance(files, dict) or not required.issubset(files):
        raise ValueError(f"Portable job is missing required motion/assets: {root}")
    for name, expected in files.items():
        target = (root / name).resolve()
        if not target.is_relative_to(root) or not target.is_file() or _sha(target) != expected:
            raise ValueError(f"Portable input hash/path mismatch: {root / name}")
    if (type(data.get("source_frames")) is not int or data["source_frames"] < 2
            or not isinstance(data.get("source_fps"), (int, float))
            or not math.isfinite(data["source_fps"]) or data["source_fps"] <= 0):
        raise ValueError(f"Invalid portable trajectory length/rate: {root}")
    from .bundles import arrays, validate_reference
    reference = arrays(root / "reference.npz")
    validate_reference(reference)
    if len(reference["timestamps"]) != data["source_frames"]:
        raise ValueError(f"Portable reference frame count differs: {root}")
    return data


def input_selection_record(path):
    """Return hash-bound selection/quality evidence without changing input bytes."""
    import numpy as np

    root = Path(path).resolve()
    data = validate_portable_input(root)
    with np.load(root / "guidance.npz", allow_pickle=False) as guide:
        if "contact_active" in guide:
            mask = guide["contact_active"]
            if (mask.ndim != 2 or mask.shape[0] != data["source_frames"]
                    or not np.isfinite(mask).all() or not np.isin(mask, [0, 1]).all()):
                raise ValueError(f"Invalid contact reference mask: {root}")
            contact_frames = int(np.any(mask.astype(bool), axis=1).sum())
            contact_state = "active" if contact_frames else "no_active_contacts"
        else:
            contact_frames, contact_state = 0, "not_provided"
    warnings = []
    if data["variant"] == "original":
        warnings.append("original_reference")
    if not contact_frames:
        warnings.append("no_contact_reference")
    if data.get("reference_kind") == "diagnostic_pre_spline_candidate":
        warnings.append("diagnostic_correction_candidate")
    elif data["variant"] == "corrected" and data.get("physical_accepted") is False:
        warnings.append("physical_acceptance_false")
    return {"schema": "v2d_policy_input_selection_v1", "job": str(root),
            "job_sha256": _sha(root / "job.json"), "tier": data["tier"],
            "episode_index": data["episode_index"], "variant": data["variant"],
            "reference_kind": data.get("reference_kind"),
            "physical_accepted": data.get("physical_accepted"),
            "planning_accepted": data.get("planning_accepted"),
            "source_manifest_sha256": data.get("source_manifest_sha256"),
            "source_sha256": data.get("source_sha256"),
            "contact_reference": contact_state, "contact_frames": contact_frames,
            "source_frames": data["source_frames"], "warnings": warnings,
            "selection_reason": "produced_corrected_reference" if data["variant"] == "corrected"
                                else "original_reference_selected"}


def select_portable_input(path):
    """Accept a job directory or one episode directory with variant children.

    An explicitly named original job stays original. Automatic episode selection
    prefers corrected even without contacts or physical/planning acceptance.
    """
    root = Path(path).resolve()
    if (root / "job.json").is_file():
        return input_selection_record(root)
    for variant in ("corrected", "original"):
        candidate = root / variant
        if (candidate / "job.json").is_file():
            record = input_selection_record(candidate)
            if record["variant"] != variant:
                raise ValueError(f"Portable variant directory/metadata mismatch: {candidate}")
            if variant == "original":
                record["selection_reason"] = "corrected_unavailable_using_original"
            return record
    raise ValueError(f"No prepared corrected or original portable job: {root}")


def select_bundle_variant(path, tier, episode):
    """Choose only from an intact bundle's explicitly available references."""
    from .bundles import EpisodeBundle, inside

    root = Path(path).resolve()
    index = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if index.get("schema") != "v2d_rl_pilot_v1":
        raise ValueError("Expected v2d_rl_pilot_v1 bundle")
    entries = [entry for entry in index["entries"]
               if entry["tier"] == tier and entry["episode_index"] == episode]
    if len(entries) != 1:
        raise ValueError(f"Expected exactly one tier {tier}, episode {episode}")
    entry = entries[0]
    manifest_path = inside(root, entry["path"]) / "manifest.json"
    if _sha(manifest_path) != entry["manifest_sha256"]:
        raise ValueError("Episode manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    available = manifest.get("corrected_reference_available")
    if type(available) is not bool:
        raise ValueError("Bundle must declare corrected_reference_available")
    if manifest.get("status") == "failed":
        raise ValueError("Correction stage failed; automatic selection does not mask stage errors")
    variant = "corrected" if available else "original"
    # This validates the complete hash chain and finite reference arrays. In
    # particular, a missing/corrupt declared correction is not a fallback case.
    EpisodeBundle.load(root, tier, episode, variant)
    return variant


def warn_input_selection(record, stream=None):
    """Print deterministic warnings separately from machine-readable stdout."""
    if not record["warnings"]:
        return
    details = []
    if "original_reference" in record["warnings"]:
        details.append("using the original trajectory"
                       + (" because no corrected job was produced" if
                          record["selection_reason"] == "corrected_unavailable_using_original" else ""))
    if "no_contact_reference" in record["warnings"]:
        details.append("no active contact reference; using the available trajectory")
    if "diagnostic_correction_candidate" in record["warnings"]:
        details.append("using the produced diagnostic correction candidate; physical/planning acceptance is not established")
    elif "physical_acceptance_false" in record["warnings"]:
        details.append("the correction is not physically accepted")
    print(f"WARNING tier{record['tier']}/episode_{record['episode_index']:06d}: "
          + "; ".join(details) + ".", file=stream or sys.stderr)

"""Strict repair inputs: reconstructed poses, RGB evidence, masks and native depth."""
from pathlib import Path
import pickle

import numpy as np

from video_processing.common import digest, read_json
from video_processing.processing.output.native import _rigid

DEFAULTS = {"min_iou": .20, "min_precision": .45, "min_recall": .35,
            "min_visible_pixels": 50, "min_depth_pixels": 8,
            "depth_absolute_tolerance_m": .20, "depth_relative_tolerance": .06,
            "max_track_distance": 256, "max_unverified_steps": 2, "tracking_iteration": 5,
            "translation_transition_weight": .10, "rotation_transition_weight": .025}
ORACLE_FIELDS = ("first_usable_frame_gt_rotation_oracle", "foundationpose_selection_uses_gt_pose",
                 "gt_translation_oracle", "gt_object_pose_available")


def validate_track(track, c2w=None):
    count = len(track["frames"])
    if count < 1 or [int(f) for f in track["frames"]] != list(range(count)) or track["camera_id"] != 0:
        raise ValueError("Require a complete zero-based monocular timeline")
    if any(track["metadata"].get(k, False) for k in ORACLE_FIELDS):
        raise ValueError("Ground-truth poses are forbidden in this repair")
    poses = _rigid(np.asarray(track["fp_poses_cam"])[:, 0], "camera poses", count)
    world = _rigid(track["obj_pose_world"], "world poses", count)
    c2w = world[0] @ np.linalg.inv(poses[0]) if c2w is None else _rigid(c2w, "camera-to-native-world")
    if not np.allclose(world, c2w[None] @ poses, atol=2e-5):
        raise ValueError("Native camera/world calibration is inconsistent")
    if not np.allclose(world[:, :3, :3], track["obj_rot_init"]) or not np.allclose(world[:, :3, 3], track["obj_t_init"]):
        raise ValueError("Native object pose fields disagree")
    return c2w


def validate_plan(plan, count, baseline_sha, plan_dir):
    allowed = {"schema", "baseline_sha256", "anchors", "intervals", "settings"}
    if set(plan) - allowed or plan.get("schema") != "v2d.reviewed_anchor_repair.v1":
        raise ValueError("Invalid repair schema or unexpected input; no external 3D pose input is supported")
    if plan.get("baseline_sha256") != baseline_sha:
        raise ValueError("The reviewed plan must bind the exact reconstructed baseline SHA256")
    anchors = plan.get("anchors", [])
    ids = [a.get("id") for a in anchors]
    if not anchors or len(set(ids)) != len(ids) or any(not isinstance(a, str) or not a for a in ids):
        raise ValueError("Provide uniquely named reviewed reconstruction anchors")
    auxiliary = {}
    for a in anchors:
        if set(a) != {"id", "frame", "review_image", "review_image_sha256", "reviewer", "note"}:
            raise ValueError("Anchor requires only id, frame, exact review image/hash, reviewer and note")
        if type(a["frame"]) is not int or not 0 <= a["frame"] < count:
            raise ValueError("Anchor frame is outside the timeline")
        if any(not isinstance(a[k], str) or not a[k].strip() for k in ("reviewer", "note")):
            raise ValueError("Record who reviewed the RGB and why this reconstructed pose is usable")
        path = (Path(plan_dir) / a["review_image"]).resolve(strict=True)
        if digest(path) != a["review_image_sha256"]:
            raise ValueError("Anchor review image changed")
        auxiliary[str(path)] = a["review_image_sha256"]
    indices, previous = [], -1
    for interval in plan.get("intervals", []):
        if set(interval) != {"start", "end", "anchor_ids"}:
            raise ValueError("Intervals need inclusive start/end and anchor_ids")
        start, end = interval["start"], interval["end"]
        if type(start) is not int or type(end) is not int or not previous < start <= end < count:
            raise ValueError("Intervals must be ordered, disjoint and within the timeline")
        if not interval["anchor_ids"] or any(a not in ids for a in interval["anchor_ids"]):
            raise ValueError("Every interval needs known reviewed anchors")
        previous = end
        indices.extend(range(start, end + 1))
    if not indices:
        raise ValueError("Provide an explicit repair interval")
    settings = {**DEFAULTS, **plan.get("settings", {})}
    if set(settings) != set(DEFAULTS):
        raise ValueError("Unknown repair setting")
    for key in ("min_iou", "min_precision", "min_recall"):
        if not np.isfinite(settings[key]) or not .1 <= settings[key] <= 1:
            raise ValueError("Invalid silhouette threshold")
    for key, lo, hi in (("min_visible_pixels", 10, 5000), ("min_depth_pixels", 4, 5000),
                        ("max_track_distance", 1, 1500), ("max_unverified_steps", 0, 3), ("tracking_iteration", 1, 10)):
        if type(settings[key]) is not int or not lo <= settings[key] <= hi:
            raise ValueError(f"Invalid {key}")
    for key in ("depth_absolute_tolerance_m", "depth_relative_tolerance"):
        if not np.isfinite(settings[key]) or not 0 < settings[key] <= .5:
            raise ValueError("Invalid depth gate")
    for key in ("translation_transition_weight", "rotation_transition_weight"):
        if not np.isfinite(settings[key]) or not 0 <= settings[key] <= .5:
            raise ValueError("Invalid temporal ranking weight")
    return indices, settings, auxiliary


def bind_inputs(args):
    native_root = args.native_root.resolve(strict=True)
    if not args.sequence or Path(args.sequence).name != args.sequence or args.sequence in (".", ".."):
        raise ValueError("Sequence must be a filename component")
    export = native_root / "export" / args.sequence
    if (export / "poses.npy").exists():
        raise ValueError("Ground-truth object pose files must not be present in the native export")
    baseline = native_root / "foundationpose" / f"{args.sequence}.pkl"
    # Native files are pickle: only open artifacts produced by your trusted local run.
    with baseline.open("rb") as stream:
        track = pickle.load(stream)
    validate_track(track)
    binding = read_json(native_root / "input_identity.json")
    files = {baseline, native_root / "depth/moge2/aligned.h5", export / "wild_export.json", export / "edex",
             Path(track["metadata"]["mesh"])}
    files.update(export.rglob("*.h5"))
    for key in ("video", "masks", "mesh"):
        entry = binding[key]
        path = Path(entry["path"])
        if digest(path) != entry["sha256"]:
            raise ValueError(f"Reconstruction input changed: {key}")
        files.add(path)
    hashes = {str(path.resolve(strict=True)): digest(path) for path in sorted(files)}
    return baseline, track, export, hashes


def validate_reviewed_candidate(candidate, review_path, baseline):
    candidate, baseline = Path(candidate).resolve(), Path(baseline).resolve()
    report = read_json(candidate.parent / "report.json")
    review = read_json(review_path)
    if report.get("candidate_sha256") != digest(candidate) or report.get("baseline_sha256") != digest(baseline):
        raise ValueError("Candidate or its exact source baseline changed")
    if report.get("numeric_screen_pass") is not True:
        raise ValueError("Candidate has unresolved numeric screening failures")
    for path, expected in report["input_sha256"].items():
        if digest(path) != expected:
            raise ValueError(f"Repair source evidence changed: {path}")
    if (review.get("schema") != "v2d.repair_visual_review.v1" or review.get("decision") != "accept"
            or review.get("candidate_sha256") != digest(candidate)):
        raise ValueError("Supply an actual visual acceptance bound to this candidate")
    frames = review.get("viewed_frames", [])
    if not frames or any(type(f) is not int or not 0 <= f < report["frame_count"] for f in frames):
        raise ValueError("List the actual RGB overlay frames viewed")
    if not set(report["required_review_frames"]).issubset(frames):
        raise ValueError("View every generated boundary, anchor, sample and unresolved-run representative before accepting")
    for path, expected in report["review_images_sha256"].items():
        if digest(path) != expected:
            raise ValueError("Candidate review image changed")
    if not review.get("reviewer") or not review.get("notes"):
        raise ValueError("Review must name the reviewer and specific observations/residuals")
    return report

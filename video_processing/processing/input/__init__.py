"""Explicit local inputs and image-coordinate prompts."""
import math
from pathlib import Path


def validate_prompts(data, frame_count):
    prompts = data.get("prompts", [])
    if not prompts or {p.get("object_id") for p in prompts} != {0, 1}:
        raise ValueError("Provide human object_id=0 and object object_id=1 prompts")
    for p in prompts:
        if type(p.get("frame_index")) is not int or not 0 <= p["frame_index"] < frame_count:
            raise ValueError("Prompt frame_index is outside the video")
        if sum(bool(p.get(k)) for k in ("points", "box")) != 1 or p.get("mask_path"):
            raise ValueError("Use exactly one image-coordinate points or box prompt")
        if p.get("points"):
            if len(p["points"]) != len(p.get("point_labels", [])) or any(type(v) is not int or v not in (0, 1) for v in p["point_labels"]):
                raise ValueError("Each point requires a binary point_label")
            coordinates = [v for point in p["points"] for v in (point["x"], point["y"])]
        else:
            box = p["box"]
            coordinates = [box[k] for k in ("x0", "y0", "x1", "y1")]
            if coordinates[2] <= coordinates[0] or coordinates[3] <= coordinates[1]:
                raise ValueError("Prompt box must have positive extent")
        if any(not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in coordinates):
            raise ValueError("Prompt coordinates must be finite nonnegative RGB pixels")


def reconstruction_inputs(args):
    suffix = ".0.color.mp4"
    if not args.video.name.endswith(suffix) or args.video.name == suffix:
        raise ValueError("Pinned CARI4D requires video filename SEQUENCE.0.color.mp4")
    if args.frames < 1 or not math.isfinite(args.fps) or args.fps <= 0:
        raise ValueError("Positive exact frame count and constant video fps are required")
    for path in (args.video, args.masks, args.mesh):
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    if bool(args.repair_candidate) != bool(args.repair_review):
        raise ValueError("A repair candidate and its actual visual review must be supplied together")
    return args.video.name[:-len(suffix)]

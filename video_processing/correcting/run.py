"""Repair object tracking from reviewed RGB-supported reconstruction anchors."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from video_processing.common import NATIVE_RELATIVE, activate_runtime, adapter_identity, digest, fresh_directory, read_json, verify_upstream, write_json


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for flag in ("native-root", "plan", "output", "upstream-root", "weights-root"):
        result.add_argument("--" + flag, required=True, type=Path)
    result.add_argument("--sequence", required=True)
    result.add_argument("--dry-run", action="store_true", help="Validate exact inputs and reviewed scope without loading GPU models")
    return result


def json_safe(value):
    import numpy as np
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def run(args):
    import numpy as np
    from video_processing.correcting.input import bind_inputs, validate_plan, validate_track
    root = verify_upstream(args.upstream_root)
    baseline, track, export, hashes = bind_inputs(args)
    plan = read_json(args.plan)
    indices, settings, auxiliary = validate_plan(plan, len(track["frames"]), digest(baseline), args.plan.parent)
    hashes.update(auxiliary)
    hashes[str(args.plan.resolve())] = digest(args.plan)
    if args.dry_run:
        return {"validated": True, "target_frames": len(indices), "anchor_count": len(plan["anchors"]),
                "gpu_or_candidate_fit_tested": False, "tier1_trajectory_used": False}
    output = fresh_directory(args.output)
    write_json(output / "plan.json", plan)
    activate_runtime(root, args.weights_root)
    native_path = root / NATIVE_RELATIVE
    namespace = {"__name__": "native_reviewed_anchor_backend", "__file__": str(native_path)}
    exec(compile(native_path.read_bytes(), str(native_path), "exec"), namespace)
    from video_processing.correcting.core.backend import NativeBackend
    from video_processing.correcting.core.algorithm import serial_candidate
    from video_processing.correcting.core.repair import generate
    from video_processing.correcting.output.native import merge_selected
    from video_processing.correcting.output.review import render_review, representative_frames
    backend = NativeBackend(namespace, export, args.native_root / "depth/moge2/aligned.h5", output, settings)
    validate_track(track, backend.c2w)
    pools, selected, unresolved, events, reasons, uncertainty = generate(backend, plan, track, indices, settings)
    merged = merge_selected(track, selected, indices, unresolved, backend.c2w, namespace)
    validate_track(merged, backend.c2w)
    review_frames = sorted({i for interval in plan["intervals"] for i in range(interval["start"], interval["end"] + 1, 30)}
                          | {i for interval in plan["intervals"] for i in (interval["start"], interval["end"])}
                          | {a["frame"] for a in plan["anchors"]} | representative_frames(unresolved))
    render_review(backend, track, merged, review_frames, output / "review")
    for path, expected in hashes.items():
        if digest(path) != expected:
            raise ValueError(f"Repair input changed during execution: {path}")
    candidate_path = output / "candidate.pkl"
    with candidate_path.open("wb") as stream:
        pickle.dump(merged, stream, protocol=pickle.HIGHEST_PROTOCOL)
    report = {"schema": "v2d.reviewed_anchor_candidate.v1", "candidate_sha256": digest(candidate_path),
              "adapter_sha256": adapter_identity(),
              "baseline_sha256": digest(baseline), "input_sha256": hashes,
              "frame_count": len(track["frames"]), "target_frames": indices,
              "numeric_screen_pass": not reasons, "numeric_screen_reasons": reasons,
              "uncertainty": uncertainty, "unresolved_frames": unresolved,
              "required_review_frames": review_frames, "visually_reviewed": False, "training_ready": False,
              "review_sampling": "Every 30th target frame, interval endpoints, anchors, and first/middle/last of each unresolved run; not every video frame",
              "review_images_sha256": {str((output / "review" / f"frame_{f:06d}.jpg").resolve()):
                                        digest(output / "review" / f"frame_{f:06d}.jpg") for f in review_frames},
              "tier1_trajectory_used": False, "interpolation_applied": False, "native_contact_refinement": False,
              "events": events, "tracking_calls": backend.tracking_calls,
              "selected": {str(f): serial_candidate(c) for f, c in selected.items()},
              "candidates": {str(f): [serial_candidate(c) for c in values] for f, values in pools.items()}}
    write_json(output / "report.json", json_safe(report))
    return {"candidate": str(candidate_path), "numeric_screen_pass": not reasons,
            "numeric_screen_reasons": reasons, "unresolved_frames": unresolved,
            "review_directory": str(output / "review"), "visually_reviewed": False}


def main(argv=None):
    result = run(parser().parse_args(argv))
    print(json.dumps(result, indent=2, allow_nan=False))
    return 2 if result.get("numeric_screen_pass") is False else 0


if __name__ == "__main__":
    raise SystemExit(main())

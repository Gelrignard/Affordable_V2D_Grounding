"""Local precontact reconstruction, SAM2 masks, and calibrated SOMA conversion."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from video_processing.common import UPSTREAM_COMMIT, adapter_identity, digest, identity, read_json, runtime_environment, verify_upstream, write_json
from video_processing.processing.input import reconstruction_inputs, validate_prompts


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    def runtime(p):
        p.add_argument("--upstream-root", required=True, type=Path)
        p.add_argument("--weights-root", required=True, type=Path)
        p.add_argument("--dry-run", action="store_true", help="Validate inputs and print the execution plan without loading GPU models")
    p = commands.add_parser("reconstruct", help="Run CARI4D through FoundationPose or CoCoNet")
    runtime(p)
    for flag in ("video", "masks", "mesh", "output"):
        p.add_argument("--" + flag, required=True, type=Path)
    p.add_argument("--frames", required=True, type=int)
    p.add_argument("--fps", required=True, type=float)
    p.add_argument("--stop-after", choices=("tracking", "coconet"), default="coconet")
    p.add_argument("--repair-candidate", type=Path)
    p.add_argument("--repair-review", type=Path)
    p = commands.add_parser("masks", help="Run SAM2 with image-coordinate prompts and pack native HDF5 masks")
    runtime(p)
    for flag in ("video", "prompts", "output"):
        p.add_argument("--" + flag, required=True, type=Path)
    p.add_argument("--frames", required=True, type=int)
    p = commands.add_parser("convert", help="Convert compact MHR to 77-joint canonical SOMA using explicit calibration")
    runtime(p)
    for flag in ("native", "mesh", "calibration", "output"):
        p.add_argument("--" + flag, required=True, type=Path)
    p.add_argument("--batch-size", type=int, default=16)
    p = commands.add_parser("inspect", help="Render reconstructed baseline poses over RGB before choosing an anchor")
    runtime(p)
    for flag in ("native-root", "output"):
        p.add_argument("--" + flag, required=True, type=Path)
    p.add_argument("--sequence", required=True)
    p.add_argument("--frames", required=True, help="Comma-separated zero-based frames to view")
    return result


def reconstruct(args):
    sequence = reconstruction_inputs(args)
    root = verify_upstream(args.upstream_root)
    binding = {"schema": "v2d.video_processing.inputs.v1", "upstream_commit": UPSTREAM_COMMIT,
               "video": identity(args.video), "masks": identity(args.masks), "mesh": identity(args.mesh),
               "frames": args.frames, "fps": args.fps, "adapter_sha256": adapter_identity()}
    native_root = args.output.resolve() / sequence
    result_root = native_root
    if args.repair_candidate:
        from video_processing.processing.core.native import review_candidate
        review_candidate(args.repair_candidate, args.repair_review, native_root / "foundationpose" / f"{sequence}.pkl")
        if args.stop_after != "coconet":
            raise ValueError("A reviewed candidate is consumed at CoCoNet, after tracking")
        result_root = native_root / "repaired" / digest(args.repair_candidate)[:16]
    request = {key: str(getattr(args, key).resolve()) for key in ("video", "masks", "mesh", "weights_root", "output")}
    request.update(upstream_root=str(root), frames=args.frames, fps=args.fps, stop_after=args.stop_after,
                   result_root=str(result_root), identity_sha256=hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest(),
                   repair_candidate=str(args.repair_candidate.resolve()) if args.repair_candidate else None,
                   repair_review=str(args.repair_review.resolve()) if args.repair_review else None)
    marker = native_root / "input_identity.json"
    if marker.exists() and read_json(marker) != binding:
        raise ValueError("Inputs changed; preserve this run and choose a fresh output directory")
    if native_root.exists() and any(native_root.iterdir()) and not marker.exists():
        raise ValueError("Existing output has no verified input identity; choose a fresh output directory")
    if args.dry_run:
        return {"request": request, "inputs": binding, "gpu_tested": False, "native_contact_refinement": False}
    write_json(marker, binding)
    request_path = result_root / f"request_{args.stop_after}.json"
    write_json(request_path, request)
    subprocess.run([sys.executable, "-m", "video_processing.processing.core.native", "--request", str(request_path)],
                   check=True, env=runtime_environment(root, args.weights_root))
    if args.stop_after == "coconet":
        from video_processing.processing.output.native import export_artifact
        target = result_root / "native_coconet.npz"
        bundle = result_root / "inference/coconet.pth"
        if target.exists():
            report = read_json(target.with_suffix(".json"))
            if report["artifact"]["sha256"] != digest(target) or report["source_bundle"]["sha256"] != digest(bundle):
                raise ValueError("Existing native export changed; preserve and investigate")
        else:
            export_artifact(bundle, target, fps=args.fps, expected_frames=args.frames, sequence_id=sequence,
                            source_mesh=args.mesh.resolve(), aligned_mesh=native_root / "export" / sequence / "object_mesh/output_aligned.glb",
                            export_metadata=native_root / "export" / sequence / "wild_export.json")
    return {"completed_through": args.stop_after, "result_root": str(result_root),
            "native_contact_refinement": False, "visually_reviewed": False, "training_ready": False}


def masks(args):
    root = verify_upstream(args.upstream_root)
    if not args.video.is_file() or args.frames < 1:
        raise ValueError("Provide a local video and positive frame count")
    validate_prompts(read_json(args.prompts), args.frames)
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Preserve previous masks; choose an empty output directory")
    masks_dir = output / "sam2_masks"
    commands = [[sys.executable, "-m", "video_processing.processing.core.sam2", "--video_path", str(args.video.resolve()),
                 "--prompts_path", str(args.prompts.resolve()), "--masks_dir", str(masks_dir),
                 "--weights_dir", str(args.weights_root.resolve() / "sam2")],
                [sys.executable, "-m", "v2d.cari4d.lib.pack_masks", "--video_path", str(args.video.resolve()),
                 "--human_masks_path", str(masks_dir / "0"), "--object_masks_path", str(masks_dir / "1"),
                 "--output_path", str(output / "masks.h5")]]
    if args.dry_run:
        return {"commands": commands, "gpu_tested": False}
    output.mkdir(parents=True, exist_ok=True)
    for command in commands:
        subprocess.run(command, check=True, env=runtime_environment(root, args.weights_root))
    report = {"video": identity(args.video), "prompts": identity(args.prompts), "masks": identity(output / "masks.h5"),
              "expected_frames": args.frames, "upstream_commit": UPSTREAM_COMMIT,
              "adapter_sha256": adapter_identity(), "visually_reviewed": False}
    write_json(output / "report.json", report)
    return report


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "convert":
        from video_processing.processing.output.soma import convert
        result = convert(args)
    elif args.command == "inspect":
        result = inspect(args)
    else:
        result = reconstruct(args) if args.command == "reconstruct" else masks(args)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


def inspect(args):
    from video_processing.common import NATIVE_RELATIVE, activate_runtime, fresh_directory
    from video_processing.correcting.input import DEFAULTS, bind_inputs, validate_track
    root = verify_upstream(args.upstream_root)
    baseline, track, export, hashes = bind_inputs(args)
    frames = sorted(set(int(v) for v in args.frames.split(",")))
    if not frames or any(not 0 <= f < len(track["frames"]) for f in frames):
        raise ValueError("Inspection frames must be within the native timeline")
    if args.dry_run:
        return {"frames": frames, "baseline_sha256": digest(baseline), "gpu_tested": False}
    output = fresh_directory(args.output)
    activate_runtime(root, args.weights_root)
    source = root / NATIVE_RELATIVE
    namespace = {"__name__": "native_anchor_inspection", "__file__": str(source)}
    exec(compile(source.read_bytes(), str(source), "exec"), namespace)
    from video_processing.correcting.core.backend import NativeBackend
    from video_processing.correcting.output.review import render_review
    backend = NativeBackend(namespace, export, args.native_root / "depth/moge2/aligned.h5", output, DEFAULTS)
    validate_track(track, backend.c2w)
    render_review(backend, track, track, frames, output / "review")
    report = {"baseline_sha256": digest(baseline), "frames": frames, "visually_reviewed": False,
              "images": [identity(output / "review" / f"frame_{f:06d}.jpg") for f in frames], "input_sha256": hashes}
    write_json(output / "report.json", report)
    return report


if __name__ == "__main__":
    raise SystemExit(main())

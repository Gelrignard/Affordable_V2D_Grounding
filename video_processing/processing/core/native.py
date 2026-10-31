"""Execute upstream stages 01–06, intercepting the boundary before refinement."""
import argparse
from pathlib import Path
import sys

from video_processing.common import activate_runtime, digest, read_json, verify_upstream, write_json


class BoundaryReached(Exception):
    pass


def review_candidate(candidate, review_path, baseline):
    from video_processing.correcting.input import validate_reviewed_candidate
    return validate_reviewed_candidate(candidate, review_path, baseline)


def execute(request):
    root = verify_upstream(request["upstream_root"])
    activate_runtime(root, request["weights_root"])
    from v2d.cari4d.lib import run_inference as native
    expected = root / "reconstruction/modules/v2d_cari4d/lib/run_inference.py"
    if Path(native.__file__).resolve() != expected.resolve():
        raise ValueError("Installed v2d.cari4d must point to the supplied pinned checkout")
    native.SAM3D_SOURCE_ROOT = root / "reconstruction/modules/v2d_sam3d_body/lib"
    original = native._stage_run
    stop = "06_coconet" if request["stop_after"] == "tracking" else "07_refinement"
    stages = []
    def stage(name, command, inputs, outputs, marker_root, env, overwrite):
        if name == stop:
            raise BoundaryReached()
        command, inputs, outputs = list(command), list(inputs), list(outputs)
        if name == "05_object_pose":
            command = [command[0], "-m", "video_processing.processing.core.foundationpose", "--source", command[1], *command[2:]]
        if name == "06_coconet" and request.get("repair_candidate"):
            candidate = Path(request["repair_candidate"])
            baseline = Path(command[command.index("--foundationpose-file") + 1])
            review_candidate(candidate, Path(request["repair_review"]), baseline)
            branch = Path(request["result_root"])
            replacements = {"--foundationpose-file": candidate,
                            "--output": branch / "inference/coconet.pth",
                            "--input-cache": branch / "inference/materialized_inputs.h5"}
            inputs = [candidate if Path(p) == baseline else p for p in inputs]
            outputs = [replacements["--output"], replacements["--input-cache"]]
            for key, value in replacements.items():
                command[command.index(key) + 1] = str(value)
            for path in outputs:
                path.parent.mkdir(parents=True, exist_ok=True)
            marker_root = branch / ".stages"
            marker_root.mkdir(parents=True, exist_ok=True)
        result = original(name, command, inputs, outputs, marker_root, env, overwrite)
        stages.append(result)
        return result
    native._stage_run = stage
    try:
        native.run_inference(request["video"], request["masks"], request["mesh"],
                             request["weights_root"], request["output"], download_models=False,
                             expected_frames=request["frames"], moge_batch_size=2,
                             sam3d_chunk_size=4, sam3d_refit_batch_size=128,
                             depth_alignment_render_batch_size=2, coconet_render_batch_size=8)
        raise RuntimeError("Pinned runner did not reach the required precontact boundary")
    except BoundaryReached:
        write_json(Path(request["result_root"]) / "execution.json",
                   {"completed_through": request["stop_after"], "native_contact_refinement": False,
                    "stages": stages, "request_sha256": request["identity_sha256"]})
    finally:
        native._stage_run = original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    execute(read_json(parser.parse_args().request))


if __name__ == "__main__":
    main()

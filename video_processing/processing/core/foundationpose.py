"""Run pinned stage 05 with the audited missing-centroid serialization fix."""
import argparse
from pathlib import Path
import sys

from video_processing.common import NATIVE_SHA256, digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    args, remaining = parser.parse_known_args()
    if digest(args.source) != NATIVE_SHA256:
        raise ValueError("Unsupported FoundationPose source")
    source = args.source.read_text(encoding="utf-8")
    needle = "    result.update(selection_arrays)\n    validate_foundationpose_selection_metadata"
    replacement = ("    result.update(selection_arrays)\n"
                   "    from video_processing.processing.core.fp_diagnostics import normalize_missing_centroids\n"
                   "    normalize_missing_centroids(result)\n    validate_foundationpose_selection_metadata")
    if source.count(needle) != 1:
        raise ValueError("Pinned diagnostic finalization site changed")
    sys.argv = [str(args.source), *remaining]
    exec(compile(source.replace(needle, replacement), str(args.source), "exec"),
         {"__name__": "__main__", "__file__": str(args.source)})


if __name__ == "__main__":
    main()

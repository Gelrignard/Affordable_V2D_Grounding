"""Authenticate Track 2 scoring row IDs without decoding reference poses.

The official packer uses the released Tier 1 visibility grid for every Track 2
input tier. This module reads only frame/episode/visibility columns and schema
identity metadata. Policy training and coordinate conversion retain their own
input tier. The source Parquets remain read-only and available for verification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from data_tools.fetch_sample import REPO, REVISION


SIDECAR = "official_scoring_frames.json"
SCHEMA = "v2d_official_scoring_frames_v1"
COLUMNS = ("frame_index", "episode_index", "observation.object.visible")


def _sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _identity(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": _sha(path), "bytes": path.stat().st_size}


def _verify_identity(record):
    if not isinstance(record, dict) or _identity(record["path"]) != record:
        raise ValueError("Official scoring-grid source hash/size/path differs")
    return Path(record["path"])


def _source_manifest(path):
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if (manifest.get("repository") != REPO or manifest.get("revision") != REVISION
            or manifest.get("tier") != 1):
        raise ValueError("Scoring-grid metadata requires the pinned official Tier 1 release")
    return manifest


def _episode_record(path, manifest_path, manifest):
    import pyarrow as pa
    import pyarrow.parquet as pq

    path, manifest_path = Path(path).resolve(), Path(manifest_path).resolve()
    relative = path.relative_to(manifest_path.parent).as_posix()
    match = re.fullmatch(r"data/[^/]+/episode_(\d{6})\.parquet", relative)
    if match is None:
        raise ValueError("Scoring-grid source must be a released episode Parquet")
    episode = int(match[1])
    identity = _identity(path)
    source = manifest.get("files", {}).get(relative, {})
    url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/track_2/tier_1_multiview_caption/{relative}"
    if (source.get("sha256") != identity["sha256"] or source.get("bytes") != identity["bytes"]
            or source.get("url") != url):
        raise ValueError("Official scoring-grid Parquet differs from download provenance")
    # Deliberately never request observation.object.pose or any human pose column.
    table = pq.read_table(path, columns=list(COLUMNS))
    if (any(table[name].null_count for name in COLUMNS)
            or not pa.types.is_integer(table["frame_index"].type)
            or not pa.types.is_integer(table["episode_index"].type)
            or not pa.types.is_boolean(table["observation.object.visible"].type)):
        raise ValueError("Official scoring-grid frame/episode/visibility columns are malformed")
    frames = np.asarray(table["frame_index"], dtype=np.int64)
    episodes = np.asarray(table["episode_index"], dtype=np.int64)
    visible = np.asarray(table["observation.object.visible"], dtype=bool)
    metadata = {k.decode(): v.decode() for k, v in (table.schema.metadata or {}).items()}
    sequence = metadata.get("source_sequence_id")
    names = json.loads(metadata.get("objects", "[]"))
    if (not len(frames) or not np.array_equal(frames, np.arange(len(frames)))
            or not np.all(episodes == episode) or not visible.any()
            or not isinstance(sequence, str) or not sequence
            or not isinstance(names, list) or len(names) != 1
            or not isinstance(names[0], str) or not names[0]):
        raise ValueError("Official scoring-grid episode/frame/object identity is invalid")
    return {"episode_index": episode, "source_sequence_id": sequence, "object_names": names,
            "source_frame_count": len(frames), "scored_source_frame_indices": frames[visible].tolist(),
            "source_parquet": identity}


def create(reference_root, dataset_input):
    """Write a new sidecar for all available source episodes; never choose rows manually."""
    reference_root, dataset_input = Path(reference_root).resolve(), Path(dataset_input).resolve()
    if not dataset_input.is_dir():
        raise ValueError("Dataset-input directory must already exist")
    destination = dataset_input / SIDECAR
    if destination.exists():
        raise FileExistsError("Scoring-grid sidecar already exists; preserve the original")
    manifest_path = reference_root / "download_manifest.json"
    manifest_identity = _identity(manifest_path)
    manifest = _source_manifest(manifest_path)
    records = {}
    for path in sorted((reference_root / "data").rglob("episode_*.parquet")):
        record = _episode_record(path, manifest_path, manifest)
        key = str(record["episode_index"])
        if key in records:
            raise ValueError("Duplicate official scoring-grid episode")
        records[key] = record
    if not records or _identity(manifest_path) != manifest_identity:
        raise ValueError("No scoring-grid episodes or source manifest changed during creation")
    result = {"schema": SCHEMA, "repository": REPO, "revision": REVISION,
              "reference_tier": 1, "columns_read": list(COLUMNS), "clean_pose_values_read": False,
              "source_download_manifest": manifest_identity, "episodes": records}
    with destination.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return {"sidecar": _identity(destination), "episodes": sorted(map(int, records)),
            "scored_frames": sum(len(v["scored_source_frame_indices"]) for v in records.values()),
            "clean_pose_values_read": False}


def attach(data, episode):
    """Attach a verified optional grid to Tier 2/3 inputs; Tier 1 is unchanged."""
    sidecar = Path(data.root) / SIDECAR
    if data.tier == 1 or not sidecar.exists():
        return data
    sidecar_identity = _identity(sidecar)
    document = json.loads(sidecar.read_text(encoding="utf-8"))
    if (document.get("schema") != SCHEMA or document.get("repository") != REPO
            or document.get("revision") != REVISION or document.get("reference_tier") != 1
            or document.get("columns_read") != list(COLUMNS)
            or document.get("clean_pose_values_read") is not False):
        raise ValueError("Official scoring-grid sidecar contract differs")
    manifest_path = _verify_identity(document["source_download_manifest"])
    manifest = _source_manifest(manifest_path)
    record = document.get("episodes", {}).get(str(episode))
    if not isinstance(record, dict):
        raise ValueError(f"Official scoring-grid sidecar lacks episode {episode}")
    source = _verify_identity(record["source_parquet"])
    expected = _episode_record(source, manifest_path, manifest)
    if record != expected or expected["episode_index"] != episode:
        raise ValueError("Scoring rows differ from official frame/visibility metadata")
    if (expected["source_sequence_id"] != data.sequence
            or expected["object_names"] != data.reference.object_names
            or expected["source_frame_count"] != len(data.frames)
            or not np.array_equal(data.frames, np.arange(len(data.frames)))
            or data.provenance.get("revision", REVISION) != REVISION):
        raise ValueError("Official scoring-grid identity differs from the policy input")
    if _identity(sidecar) != sidecar_identity:
        raise ValueError("Official scoring-grid sidecar changed during verification")
    data.official_scoring_frames = np.asarray(expected["scored_source_frame_indices"], dtype=np.int64)
    data.provenance["official_scoring_frames"] = {
        **sidecar_identity, "episode_index": episode,
        "source_frame_count": expected["source_frame_count"],
        "scored_source_frame_indices": expected["scored_source_frame_indices"],
        "source_file_sha256": {str(manifest_path): document["source_download_manifest"]["sha256"],
                               str(source): expected["source_parquet"]["sha256"]},
        "columns_read": list(COLUMNS), "clean_pose_values_read": False}
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True,
                        help="Pinned released Tier 1 cache, used only for frame/visibility metadata")
    parser.add_argument("--dataset-input", type=Path, required=True,
                        help="Tier 2/3 input directory that receives the new metadata sidecar")
    args = parser.parse_args()
    print(json.dumps(create(args.reference_root, args.dataset_input), indent=2))


if __name__ == "__main__":
    main()

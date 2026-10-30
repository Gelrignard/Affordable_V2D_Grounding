"""Decode released Tier 1/2 SOMA parameters to the retargeter's canonical NPZ.

Run with .soma-env (Torch/SOMA), not the Pinocchio retargeting environment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import warnings
from pathlib import Path
from time import perf_counter

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation
import torch
from soma import SOMALayer

from correction.src.dynamics.object_wrench import contiguous_runs
from data_tools.observation_completion import complete_object_poses, complete_samples


def level_rotation(normal):
    normal = normal / np.linalg.norm(normal)
    cross = np.cross(normal, [0., 0., 1.])
    sine = np.linalg.norm(cross)
    if sine < 1e-12:
        return np.eye(3)
    return Rotation.from_rotvec(cross / sine * np.arctan2(sine, normal[2])).as_matrix()


def decode(data, output, episode, assets, *, tier=1, all_frames=False):
    torch.set_num_threads(4)
    started = perf_counter()
    records = [json.loads(s) for s in (data / "meta/episodes_metadata.jsonl").read_text().splitlines()]
    metadata = next(item for item in records if item["episode_index"] == episode)
    source = data / f"data/chunk-000/episode_{episode:06d}.parquet"
    table = pq.ParquetFile(source).read()
    manifest = json.loads((data / 'download_manifest.json').read_text())
    if manifest['tier'] != tier:
        raise ValueError('Dataset tier does not match requested decoding provenance')
    variant = (table.schema.metadata or {}).get(b'trajectory_variant', b'').decode()
    if tier == 2 and variant != 'tier_2_synthetic_noise':
        raise ValueError('Tier 2 must come from its own noised parquet, not clean Tier 1')
    info = json.loads((data / "meta/info.json").read_text())
    names = [name.removesuffix(".rx") for name in info["features"]["observation.human.pose"]["names"][::3]]
    arrays = {key: np.asarray(table[key].to_pylist(), dtype=np.float32) for key in (
        "observation.human.pose", "observation.human.translation", "observation.human.identity_coeffs",
        "observation.human.scale_params", "observation.human.bone_length_flexibles", "observation.object.pose")}
    times = np.asarray(table['timestamp'], dtype=float)
    human_repairs = {}
    for key in arrays:
        if not key.startswith('observation.human.'):
            continue
        value = arrays[key]
        if key == 'observation.human.pose':
            repaired, valid = complete_samples(times, value.reshape(len(times), -1, 3), rotation='rotvec')
            repaired = repaired.reshape(value.shape)
        else:
            repaired, valid = complete_samples(times, value)
        missing = np.flatnonzero(~valid.reshape(len(times), -1).all(axis=1))
        if len(missing):
            arrays[key] = repaired
            human_repairs[key] = missing.tolist()
    if human_repairs:
        warnings.warn(f'Filled missing same-tier human source samples: {human_repairs}', RuntimeWarning)
    visible = np.asarray(table["observation.object.visible"], dtype=bool)
    completed, observed = complete_object_poses(np.asarray(table['timestamp'], dtype=float),
                                               arrays['observation.object.pose'], visible)
    selected = np.arange(len(table)) if all_frames else max(contiguous_runs(observed), key=len)
    # The legacy segment mode still refuses disjoint observations; full-timeline
    # mode retains completed poses together with their false observation masks.
    if not all_frames and len(selected) != observed.sum():
        raise ValueError("this first adapter requires one contiguous visible segment")
    shape_keys = ("observation.human.identity_coeffs", "observation.human.scale_params", "observation.human.bone_length_flexibles")
    for key in shape_keys[:2]:
        if not np.allclose(arrays[key], arrays[key][0], atol=1e-6, rtol=0):
            warnings.warn(f'{key} varies; decoding the supplied parameters per frame', RuntimeWarning)
    print("Initializing the pinned SOMA/MHR model on CPU", flush=True)
    layer = SOMALayer(data_root=str(assets.resolve()), identity_model_type="mhr", device="cpu",
                      enable_procedural_transforms=False, correctives_model_path=None)
    if list(layer.public_joint_names)[1:] != names:
        raise ValueError("dataset and model joint orders disagree")
    with torch.no_grad():
        frames = []
        pose = arrays["observation.human.pose"].reshape(-1, 77, 3)
        translation = arrays["observation.human.translation"]
        for start in range(0, len(selected), 64):
            ids = selected[start:start+64]
            # Flexible bone parameters are genuinely time-varying in Tier 1.
            # Decode them per frame instead of silently freezing their values.
            layer.prepare_identity(torch.from_numpy(arrays[shape_keys[0]][ids]),
                                   torch.from_numpy(arrays[shape_keys[1]][ids]), repose_to_bind_pose=False,
                                   kwargs={"bone_length_flexibles": torch.from_numpy(arrays[shape_keys[2]][ids])})
            result = layer.pose(torch.from_numpy(pose[ids]), transl=torch.from_numpy(translation[ids]),
                                fk_only=True, apply_correctives=False)
            frames.append(result.transforms[:, 1:].cpu().numpy())
            print(f"Decoded {min(start+64, len(selected))}/{len(selected)} frames", flush=True)
        transforms = np.concatenate(frames).astype(np.float64)
    # Exactly the upstream coordinate chain: CV->SOMA, first-frame source
    # anchor, configured SOMA->G1 rotation, and one constant plane leveling.
    cv_to_soma = np.diag([1., -1., -1.])
    soma_to_g1 = np.array([[0., 0., -1.], [-1., 0., 0.], [0., 1., 0.]])
    root_inv = Rotation.from_rotvec(pose[0, 0]).as_matrix().T
    rotation_pre = soma_to_g1 @ root_inv
    translation_pre = -rotation_pre @ translation[0]
    plane = np.asarray(json.loads((data / metadata["ground_plane"]).read_text())["plane"])
    plane /= np.linalg.norm(plane[:3])
    normal_pre = rotation_pre @ cv_to_soma @ plane[:3]
    offset_pre = plane[3] - normal_pre @ translation_pre
    if normal_pre[2] < 0:
        normal_pre, offset_pre = -normal_pre, -offset_pre
    leveling = level_rotation(normal_pre)
    rotation_world = leveling @ rotation_pre
    translation_world = leveling @ translation_pre + [0., 0., offset_pre]
    positions = transforms[:, :, :3, 3] @ rotation_world.T + translation_world
    rotations = Rotation.from_matrix((rotation_world @ transforms[:, :, :3, :3]).reshape(-1, 3, 3)).as_quat(scalar_first=True).reshape(len(selected), 77, 4)
    object_cv = completed[selected]
    world_from_cv = rotation_world @ cv_to_soma
    object_positions = object_cv[:, :3] @ world_from_cv.T + translation_world
    object_rotations = Rotation.from_matrix(world_from_cv @ Rotation.from_quat(object_cv[:, 3:], scalar_first=True).as_matrix()).as_quat(scalar_first=True)
    transform = np.eye(4)
    transform[:3, :3], transform[:3, 3] = world_from_cv, translation_world
    reference_soma = np.eye(4)
    reference_soma[:3, :3], reference_soma[:3, 3] = rotation_world, translation_world
    report = {"source": str(source.resolve()), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "tier": tier, "episode_index": episode, "dataset_revision": manifest['revision'],
              "soma_package": "py-soma-x==0.2.1", "soma_asset_revision": "466879a83d57eabf3d875ded2d869f2075f90348",
              "coordinate_frame": "world_z_up", "selected_source_indices": selected.tolist(),
              "world_from_source_cv": transform.tolist(), "world_from_source_soma": reference_soma.tolist(),
              "ground_plane_source_cv": plane.tolist(), "support_surface": str((data / metadata["support_surface"]).resolve()),
              "source_fps": 30., "timestamp_source": "original_challenge_parquet",
              "body_decode": "MHR identity/scale and time-varying flexible bones applied per frame; official SOMA CPU forward kinematics; no vertex correctives",
              "dropped_invisible_frames": [] if all_frames else np.flatnonzero(~observed).tolist(),
              "object_observed_mask": observed[selected].tolist(),
              "object_visibility_mask": visible[selected].tolist(),
              "object_imputed_frames": np.flatnonzero(~observed[selected]).tolist(),
              "human_imputed_source_frames": human_repairs,
              "observation_completion": 'Linear position/SLERP on same-tier visible observations; nearest endpoint hold; imputed object frames have zero observation confidence; human repairs are recorded separately',
              "trajectory_variant": variant,
              "object_motion_edited": bool(np.any(~observed[selected])), "elapsed_seconds": perf_counter()-started}
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, joint_names=np.array(names), positions=positions, rotations_wxyz=rotations,
                        timestamps=np.asarray(table["timestamp"], dtype=float)[selected],
                        object_positions=object_positions, object_wxyz=object_rotations,
                        object_mesh=str((data / metadata["mesh"]).resolve()), sequence_id=metadata["sequence_id"],
                        object_observed_mask=observed[selected], raw_object_pose_source_cv=arrays['observation.object.pose'][selected],
                        metadata_json=json.dumps(report), coordinate_frame="world_z_up", units="metres_seconds")
    output.with_suffix(".json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"saved": str(output.resolve()), "frames": len(selected), "dropped": report["dropped_invisible_frames"],
                      "head_z_range": [float(positions[:, names.index('Head'), 2].min()), float(positions[:, names.index('Head'), 2].max())],
                      "object_z_range": [float(object_positions[:, 2].min()), float(object_positions[:, 2].max())],
                      "seconds": report["elapsed_seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("correction/data/tier1"))
    parser.add_argument("--assets", type=Path, default=Path("correction/data/soma_assets"))
    parser.add_argument("--episode", type=int, default=11)
    parser.add_argument("--tier", type=int, choices=(1, 2), default=1)
    parser.add_argument("--all-frames", action='store_true', help='Retain full timeline; interpolate missing object poses with explicit masks')
    parser.add_argument("--output", type=Path, default=Path("correction/outputs/tier1_book/human_motion.npz"))
    args = parser.parse_args()
    decode(args.data, args.output, args.episode, args.assets, tier=args.tier, all_frames=args.all_frames)

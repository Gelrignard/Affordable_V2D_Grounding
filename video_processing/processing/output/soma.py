"""Convert compact native motion using an explicit rigid camera/world calibration."""
from pathlib import Path
import shutil
import json

from video_processing.common import activate_runtime, adapter_identity, digest, fresh_directory, identity, read_json, verify_upstream, write_json


def load_calibration(path):
    import numpy as np
    from .native import _rigid
    data = read_json(path)
    allowed = {"schema", "world_from_camera", "source", "notes", "benchmark_frame_indices_used"}
    if set(data) - allowed or data.get("schema") != "v2d.camera_world_calibration.v1":
        raise ValueError("Invalid calibration schema or unexpected field")
    if not isinstance(data.get("source"), str) or not data["source"].strip():
        raise ValueError("Calibration must name its measured/manual provenance")
    if data.get("benchmark_frame_indices_used", []) not in ([], [0]):
        raise ValueError("Optional benchmark alignment may use frame zero only")
    transform = _rigid(np.asarray(data.get("world_from_camera")), "world_from_camera")
    return transform, data


def convert(args):
    import numpy as np
    transform, calibration = load_calibration(args.calibration)
    if not 1 <= args.batch_size <= 32:
        raise ValueError("Conversion batch size must be in [1, 32]")
    metadata = read_json(args.native.with_suffix(".json"))
    if metadata.get("artifact", {}).get("sha256") != digest(args.native):
        raise ValueError("Native NPZ does not match its export report")
    if metadata.get("source_mesh", {}).get("sha256") != digest(args.mesh):
        raise ValueError("Use the exact original mesh from this reconstruction")
    if metadata.get("mesh_transform_status") != "upstream_transform_and_current_local_mesh_identities_matched":
        raise ValueError("Native export has not verified the original-to-aligned mesh relationship")
    if args.dry_run:
        return {"validated": True, "gpu_tested": False, "calibration": calibration,
                "output": str(args.output / "human_motion.npz")}
    root = verify_upstream(args.upstream_root)
    activate_runtime(root, args.weights_root)
    import torch
    from scipy.spatial.transform import Rotation
    from .mhr_soma import Converter, ASSET_REVISION
    from .geometry import apply_human_transform, object_source_mesh_poses
    output = fresh_directory(args.output)
    with np.load(args.native, allow_pickle=False) as stream:
        arrays = {k: stream[k] for k in stream.files}
    count = len(arrays["timestamps"])
    converter, chunks = Converter(args.weights_root.resolve()), []
    for start in range(0, count, args.batch_size):
        end = min(count, start + args.batch_size)
        params = {k.removeprefix("predicted_"): torch.from_numpy(v[start:end].copy())
                  for k, v in arrays.items() if k.startswith("predicted_mhr_")}
        chunks.append(converter.convert(params))
        print(f"SOMA_CONVERSION_PROGRESS {end}/{count}", flush=True)
    values = {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0]}
    names = converter.names
    if len(names) != 77:
        raise ValueError("Expected the audited 77-joint SOMA output")
    positions, quaternions = apply_human_transform(values["positions"], values["rotations_wxyz"], transform)
    poses = object_source_mesh_poses(arrays["predicted_object_pose_camera"],
                                    arrays["source_object_mesh_to_pose_body_transform"], transform)
    object_q = Rotation.from_matrix(poses[:, :3, :3]).as_quat()[:, [3, 0, 1, 2]]
    mesh_rel = "mesh/" + args.mesh.name
    (output / "mesh").mkdir()
    shutil.copyfile(args.mesh, output / mesh_rel)
    report = {"schema": "v2d.canonical_soma_conversion.v2", "source": identity(args.native),
              "adapter_sha256": adapter_identity(),
              "calibration": calibration, "calibration_sha256": digest(args.calibration),
              "world_from_camera": transform.tolist(), "asset_revision": ASSET_REVISION,
              "frame_count": count, "joint_count": len(names), "coordinate_frame": "world_z_up",
              "units": "metres_seconds", "retarget_ready": True, "training_ready": False,
              "native_contact_refinement": False, "trajectory_repair_uses_tier1": False,
              "source_mesh_sha256": digest(args.mesh),
              "fit_error_mean_m": float(values["fit_error_mean_m"].mean()),
              "fit_error_max_m": float(values["fit_error_max_m"].max()),
              "quality_note": "Format-valid reconstruction; fit residual is not ground-truth accuracy or training validation."}
    np.savez_compressed(output / "camera_soma_motion.npz", **values, joint_names=names,
                        timestamps=arrays["timestamps"], coordinate_frame="camera_cv")
    np.savez_compressed(output / "human_motion.npz", joint_names=names, positions=positions,
                        rotations_wxyz=quaternions, timestamps=arrays["timestamps"],
                        object_positions=poses[:, :3, 3], object_wxyz=object_q,
                        object_mesh=mesh_rel, sequence_id=arrays["sequence_id"],
                        coordinate_frame="world_z_up", units="metres_seconds", metadata_json=json.dumps(report),
                        hand_contact_logits=arrays.get("hand_contact_logits", np.empty((count, 0))))
    report["human_motion_sha256"] = digest(output / "human_motion.npz")
    write_json(output / "report.json", report)
    return report

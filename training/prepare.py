"""Materialize the portable pilot as motion_v1 plus explicit guidance/provenance."""

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import trimesh
from scipy.spatial.transform import Rotation

from . import RUNTIME_COMMIT
from .bundles import EpisodeBundle, sha256
from .kinematics import URDFKinematics


def prepare_episode(bundle_root, tier, episode, variant, urdf, output):
    bundle = EpisodeBundle.load(bundle_root, tier, episode, variant)
    ref, guide, manifest = bundle.reference, bundle.guidance, bundle.manifest
    canonical_sha = hashlib.sha256(
        Path(urdf).read_bytes().replace(b"\r\n", b"\n")
    ).hexdigest()
    newline_variants = {
        "ab13c2fb8f51ee92f33e015e264b9826b92bc4bae0f757aa2947f6b5e66972a6",
        "8c7f768dc8da6c969d8a4b4efb3dcbe3d9d2b05fca5104df12205a69879d19e9",
    }
    if sha256(urdf) != manifest["asset_paths"]["urdf_sha256"] and not (
        canonical_sha in newline_variants
        and manifest["asset_paths"]["urdf_sha256"] in newline_variants
    ):
        raise ValueError(
            "URDF differs from the named-joint reference; resolve robot parity first"
        )
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    assets = output / "assets"
    assets.mkdir()
    for key, name in (
        ("object_mesh", "object.glb"),
        ("support_surface", "support.usda"),
    ):
        shutil.copy2(bundle.root / manifest["asset_paths"][key], assets / name)
    mesh = trimesh.load(assets / "object.glb", force="mesh", process=False)
    vertices = np.asarray(mesh.vertices)
    if not len(vertices) or not np.isfinite(vertices).all():
        raise ValueError("object mesh has no finite geometry")
    # Match the pinned soma_to_g1.py::_build_object_urdf defaults, independently
    # of the corrector's prototype metadata. Keep the mesh's object-local frame.
    (assets / "object.obj").write_text(
        trimesh.exchange.obj.export_obj(
            mesh, include_normals=False, include_color=False, include_texture=False
        ),
        encoding="utf-8",
    )
    (assets / "object.urdf").write_text(
        """<?xml version="1.0"?>
<robot name="retarget_object"><link name="object">
  <inertial><origin xyz="0 0 0" rpy="0 0 0"/><mass value="1.0"/>
    <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
  </inertial>
  <visual><geometry><mesh filename="object.obj"/></geometry></visual>
  <collision><geometry><mesh filename="object.obj"/></geometry></collision>
</link></robot>
""",
        encoding="utf-8",
    )
    radius = max(float(np.linalg.norm(vertices, axis=-1).max()), 1e-3)
    n = len(ref["timestamps"])
    dt = (ref["timestamps"][-1] - ref["timestamps"][0]) / (n - 1)
    if not np.allclose(np.diff(ref["timestamps"]), dt, atol=4e-6, rtol=1e-4):
        raise ValueError(
            "nonuniform input requires explicit resampling before motion_v1 export"
        )
    kin = URDFKinematics(urdf)
    palms = [f"{side}_hand_palm_link" for side in ("left", "right")]
    poses = kin.poses(
        ref["joint_positions"],
        ref["root_position"],
        ref["root_wxyz"],
        ref["joint_names"],
        palms,
    )
    stored = ref["ee_pose_w"][
        :, [ref["ee_link_names"].tolist().index(p) for p in palms]
    ]
    position_error = float(
        np.max(np.linalg.norm(poses[..., :3] - stored[..., :3], axis=-1))
    )
    orientation_error = float(
        np.max(1 - np.abs(np.sum(poses[..., 3:] * stored[..., 3:], axis=-1)))
    )
    if position_error > 1e-4 or orientation_error > 1e-5:
        raise ValueError(
            f"FK parity failed: position={position_error}, quaternion={orientation_error}"
        )
    active, contact_pos, contact_normals, part_ids = [], [], [], []
    rotation = Rotation.from_quat(ref["object_wxyz"][:, [1, 2, 3, 0]]).as_matrix()
    for side in ("left", "right"):
        selected = [
            i
            for i, name in enumerate(guide.get("contact_body_names", []))
            if str(name).startswith(side + "_")
        ]
        if selected:
            mask = guide["contact_active"][:, selected].astype(bool)
            local = guide["object_local_points"][:, selected]
            normal = guide["inward_normals_local"][:, selected]
            pos = (
                np.einsum("tij,tcj->tci", rotation, local)
                + ref["object_position"][:, None]
            )
            normals = np.einsum("tij,tcj->tci", rotation, normal)
            # motion_v1 uses one-based body IDs; zero denotes no contact.
            ids = np.where(mask, 1, 0)
        else:
            mask, pos, normals, ids = (
                np.zeros((n, 0), bool),
                np.zeros((n, 0, 3)),
                np.zeros((n, 0, 3)),
                np.zeros((n, 0), int),
            )
        active.append(mask.any(axis=-1).astype(float).tolist())
        contact_pos.append(pos.tolist() if selected else [])
        contact_normals.append(normals.tolist() if selected else [])
        part_ids.append(ids.tolist() if selected else [])
    row = {
        "schema_version": "motion_v1",
        "motion_kind": "single_robot",
        "robot_name": "g1",
        "sequence_id": manifest["sequence_id"],
        "source_dataset": "v2d_rl_pilot",
        "raw_motion_file": "reference.npz",
        "fps": 1 / dt,
        "coord_frame": "world",
        "robot_joint_names": ref["joint_names"].tolist(),
        "robot_joint_positions": ref["joint_positions"].tolist(),
        "robot_root_position": ref["root_position"].tolist(),
        "robot_root_wxyz": ref["root_wxyz"].tolist(),
        "ee_link_names": palms,
        "ee_pose_w": poses.tolist(),
        "hand_sides": ["left", "right"],
        "hand_frame_names": [],
        "hand_frames_w": [],
        "object_name": "object",
        "object_body_names": ["object"],
        "object_mesh_paths": [str(assets / "object.obj")],
        "object_urdf_paths": [str(assets / "object.urdf")],
        "object_mesh_radius": [radius],
        "object_articulation": [],
        "object_body_position": ref["object_position"][:, None].tolist(),
        "object_body_wxyz": ref["object_wxyz"][:, None].tolist(),
        "hand_contact_active": active,
        "hand_object_contact_positions": contact_pos,
        "hand_object_contact_normals": contact_normals,
        "hand_object_contact_part_ids": part_ids,
    }
    pq.write_table(
        pa.Table.from_pylist([row]), output / "motion.parquet", compression="zstd"
    )
    np.savez_compressed(output / "reference.npz", **ref)
    np.savez_compressed(output / "guidance.npz", **guide)
    # Deterministic surface descriptor; physics parameters come from the runtime, not the prototype.
    selected_vertices = (
        vertices[np.linspace(0, len(vertices) - 1, 32).astype(int)] / radius
    )
    np.savez_compressed(
        output / "geometry.npz",
        points=selected_vertices.astype(np.float32),
        radius=radius,
    )
    job = {
        "schema": "v2d_policy_job_v1",
        "tier": tier,
        "episode_index": episode,
        "variant": variant,
        "source_manifest_sha256": sha256(bundle.root / "manifest.json"),
        "source_sha256": manifest["source"]["source_sha256"],
        "runtime_commit": RUNTIME_COMMIT,
        "sequence_id": manifest["sequence_id"],
        "source_frames": n,
        "source_fps": 1 / dt,
        "reference_kind": manifest["reference_kind"]
        if variant == "corrected"
        else "contact_aware_original",
        "physical_accepted": manifest["physical_accepted"],
        "usage": "soft_reference_not_expert_actions",
        "urdf_sha256": sha256(urdf),
        "urdf_canonical_sha256": canonical_sha,
        "fk_max_position_error_m": position_error,
        "physics_source": "pinned_soma_to_g1_object_template_and_flash_chord_scene_builder",
        "object_physics": {
            "mass_kg": 1.0,
            "com_local_m": [0.0, 0.0, 0.0],
            "inertia_diagonal_kg_m2": [0.01, 0.01, 0.01],
        },
        "model_scope": "single_rigid_object_g1_dex3",
        "files": {
            p.relative_to(output).as_posix(): sha256(p)
            for p in output.rglob("*")
            if p.is_file()
        },
    }
    (output / "job.json").write_text(json.dumps(job, indent=2) + "\n", encoding="utf-8")
    return job


def load_job(folder, verify=True):
    from .bundles import inside

    folder = Path(folder).resolve()
    job = json.loads((folder / "job.json").read_text())
    if job.get("obsolete"):
        raise ValueError(
            "obsolete preparation output; use the validated training_jobs_v1 jobs or prepare again"
        )
    if (
        job.get("schema") != "v2d_policy_job_v1"
        or job.get("runtime_commit") != RUNTIME_COMMIT
    ):
        raise ValueError("incompatible prepared job")
    if verify:
        for name, digest in job["files"].items():
            if sha256(inside(folder, name)) != digest:
                raise ValueError(f"prepared job changed: {name}")
    return job

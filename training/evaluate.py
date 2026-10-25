"""Full physical rollouts and submission-kit-readable trajectory export.

This is our evaluator for our PyTorch policy. It does not claim to be the
organizer's FlashSAC checkpoint loader or an official leaderboard score.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from scipy.spatial.transform import Rotation, Slerp

from .bundles import sha256
from .checkpoint import read_checkpoint
from .networks import Actor


def resample_poses(times, poses, target_times):
    """Interpolate actual recordings at source timestamps, without changing time."""
    times, target_times = np.asarray(times), np.asarray(target_times)
    if (
        np.any(np.diff(times) <= 0)
        or target_times[0] < times[0] - 1e-7
        or target_times[-1] > times[-1] + 1e-7
    ):
        raise ValueError("source timestamps must be covered by the physical rollout")
    target_times = target_times.clip(times[0], times[-1])
    poses = np.asarray(poses)
    output = np.full((len(target_times), *poses.shape[1:]), np.nan, dtype=np.float64)
    valid = np.isfinite(poses).all(axis=(0, 2, 3)) & (
        np.linalg.norm(poses[..., 3:], axis=-1) > 1e-6
    ).all(axis=(0, 2))
    for world in range(poses.shape[1]):
        if not valid[world]:
            continue  # Keep the failed world in the export, explicitly invalid.
        for body in range(poses.shape[2]):
            track = poses[:, world, body]
            output[:, world, body, :3] = np.stack(
                [np.interp(target_times, times, track[:, i]) for i in range(3)], axis=-1
            )
            output[:, world, body, 3:] = Slerp(times, Rotation.from_quat(track[:, 3:]))(
                target_times
            ).as_quat()
    return output, valid


def write_trajectory(
    path, achieved, reference, times, source_frames, episode, fps, valid, provenance
):
    """Exact column/convention contract consumed by v2dlb.submission_parquet."""
    achieved, reference = np.asarray(achieved), np.asarray(reference)
    t, worlds, bodies, width = achieved.shape
    if (
        width != 7
        or reference.shape != (t, bodies, 7)
        or len(times) != t
        or len(source_frames) != t
    ):
        raise ValueError("inconsistent trajectory shape")
    if np.any(np.diff(source_frames) != 1) or np.any(np.diff(times) <= 0):
        raise ValueError(
            "export requires all consecutive source frames in timestamp order"
        )
    if not np.isfinite(reference).all():
        raise ValueError("nonfinite reference pose")
    if len(valid) != worlds:
        raise ValueError("one validity flag per world is required")
    valid = np.asarray(valid, bool) & np.isfinite(achieved).all(axis=(0, 2, 3))
    # The submission kit requires finite placeholders for invalid worlds.
    # The untouched physical trace, including NaNs, is retained separately.
    achieved = achieved.copy()
    achieved[:, ~valid] = 0
    grid = np.indices((t, worlds, bodies))
    shape = (t, worlds, bodies)
    columns = {
        "environment_id": pa.array(grid[1].ravel(), type=pa.int32()),
        "step": pa.array(grid[0].ravel(), type=pa.int32()),
        "reference_frame": pa.array(
            np.broadcast_to(np.asarray(source_frames)[:, None, None], shape).ravel(),
            type=pa.int32(),
        ),
        "time_s": pa.array(
            np.broadcast_to(np.asarray(times)[:, None, None], shape).ravel(),
            type=pa.float64(),
        ),
        "object_id": pa.array(grid[2].ravel(), type=pa.int32()),
        "object_body_id": pa.array(grid[2].ravel(), type=pa.int32()),
        "object_body_name": pa.array(
            ["object"] * (t * worlds * bodies)
        ).dictionary_encode(),
        "environment_valid": pa.array(
            np.broadcast_to(valid[None, :, None], shape).ravel()
        ),
    }
    suffixes = (
        "position_x",
        "position_y",
        "position_z",
        "quaternion_x",
        "quaternion_y",
        "quaternion_z",
        "quaternion_w",
    )
    for prefix, poses in (
        ("achieved", achieved),
        ("reference", np.broadcast_to(reference[:, None], achieved.shape)),
    ):
        for i, suffix in enumerate(suffixes):
            columns[f"{prefix}_{suffix}"] = pa.array(
                poses[..., i].ravel(), type=pa.float32()
            )
    metadata = {
        "schema": "flash_chord_object_trajectories_v1",
        "episode_index": episode,
        "position_unit": "meter",
        "quaternion_order": "xyzw",
        "row_order": "step,environment_id,object_body_id",
        "control_fps": fps,
        "start_frame": int(source_frames[0]),
        "step_count": t,
        "world_count": worlds,
        "object_body_count": bodies,
        "pose_width": 7,
        **provenance,
    }
    table = pa.table(columns).replace_schema_metadata(
        {("flash_chord." + k).encode(): str(v).encode() for k, v in metadata.items()}
    )
    pq.write_table(table, path, compression="zstd")


def evaluation_code_hash():
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


@torch.no_grad()
def evaluate(env, checkpoint_path, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "evaluation.json").exists() or (
        output / "object_trajectories.parquet"
    ).exists():
        raise FileExistsError("choose a new evaluation directory")
    checkpoint = read_checkpoint(checkpoint_path, env.device)
    if checkpoint["identity"] != env.identity or checkpoint["spec"] != env.spec:
        raise ValueError(
            "checkpoint does not match evaluation episode/reference/controller"
        )
    spec = env.spec
    actor = Actor(
        spec["obs_dim"],
        spec["goal_dim"],
        spec["action_dim"],
        checkpoint["config"]["hidden"],
    ).to(env.device)
    actor.load_state_dict(checkpoint["learner"]["actor"])
    actor.eval()
    state, goal = env.reset(first=True)
    source = env.source_reference
    times = source["timestamps"] - source["timestamps"][0]
    # ceil ensures the last source timestamp is covered even when runtime resampling rounded down.
    steps = int(np.ceil(times[-1] * env.control_fps - 1e-8))
    poses = [env.object_pose().cpu().numpy()]
    joints = [
        env.tensor(env.env.state_0.joint_q).reshape(env.worlds, -1).cpu().numpy().copy()
    ]

    def contact_snapshot():
        contact = env.env.contact
        return (
            env.tensor(contact.contact_force_w)
            .reshape(env.worlds, -1, 3)
            .cpu()
            .numpy()
            .copy(),
            env.tensor(contact.contact_active)
            .reshape(env.worlds, -1)
            .cpu()
            .numpy()
            .copy(),
        )

    initial_force, initial_contact = contact_snapshot()
    contact_forces, contact_active = [initial_force], [initial_contact]
    actions, terminations = [], []
    for step in range(steps):
        action, _ = actor(state, goal, deterministic=True)
        ns, ng, _, _, _, _, info = env.step(action, reset=False)
        state, goal = ns, ng
        poses.append(env.object_pose().cpu().numpy())
        joints.append(
            env.tensor(env.env.state_0.joint_q)
            .reshape(env.worlds, -1)
            .cpu()
            .numpy()
            .copy()
        )
        actions.append(info["executed_action"].cpu().numpy())
        terminations.append(info["terminated"].cpu().numpy())
        force, active = contact_snapshot()
        contact_forces.append(force)
        contact_active.append(active)
        if step % 100 == 0:
            print(
                f"evaluation: {step + 1}/{steps} physical steps, no resets", flush=True
            )
        if not torch.isfinite(state).all():
            # A broken world must not be hidden by ending the file early.
            remaining = steps - step - 1
            poses.extend([np.full_like(poses[-1], np.nan) for _ in range(remaining)])
            joints.extend([np.full_like(joints[-1], np.nan) for _ in range(remaining)])
            actions.extend(
                [np.full_like(actions[-1], np.nan) for _ in range(remaining)]
            )
            terminations.extend(
                [np.ones_like(terminations[-1]) for _ in range(remaining)]
            )
            contact_forces.extend(
                [np.full_like(force, np.nan) for _ in range(remaining)]
            )
            contact_active.extend([np.zeros_like(active) for _ in range(remaining)])
            break
    control_times = np.arange(steps + 1) / env.control_fps
    poses = np.stack(poses)
    achieved, valid = resample_poses(control_times, poses, times)
    reference = np.concatenate(
        (source["object_position"], source["object_wxyz"][:, [1, 2, 3, 0]]), axis=-1
    )[:, None]
    provenance = {
        "eval_script_sha256": sha256(__file__),
        "eval_code_sha256": evaluation_code_hash(),
        "checkpoint_sha256": sha256(checkpoint_path),
        "evaluator": "v2d_training_pytorch_v1",
        "runtime_commit": env.identity["runtime_commit"],
        "tier": env.identity["tier"],
        "assistance_scale": 0,
        "within_episode_resets": 0,
        "sampling": "actual_50hz_rollout_resampled_to_source_times",
    }
    write_trajectory(
        output / "object_trajectories.parquet",
        achieved,
        reference,
        times,
        source["source_frame_indices"],
        env.job["episode_index"],
        env.job["source_fps"],
        valid,
        provenance,
    )
    np.savez_compressed(
        output / "control_trace.npz",
        times=control_times,
        object_pose_xyzw=poses,
        joint_q=np.stack(joints),
        normalized_action=np.stack(actions),
        terminated=np.stack(terminations),
        measured_robot_on_object_force_w=np.stack(contact_forces),
        measured_contact_active=np.stack(contact_active),
        action_names=np.asarray(env.action_names),
        source_times=times,
    )
    finite_error = np.linalg.norm(
        achieved[..., :3] - reference[:, None, :, :3], axis=-1
    )
    per_world_error = [
        float(finite_error[:, w].mean()) if valid[w] else None
        for w in range(env.worlds)
    ]
    report = {
        "identity": env.identity,
        "provenance": provenance,
        "control_steps": steps,
        "source_frames": len(times),
        "valid_worlds": valid.tolist(),
        "ever_terminated": np.stack(terminations).any(axis=0).tolist(),
        "mean_object_position_error_m_per_world": per_world_error,
        "metric_reference": f"tier{env.job['tier']}_input_reference",
        "official_leaderboard_score": False,
        "resets": 0,
        "assistance_scale": 0,
    }
    (output / "evaluation.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return report

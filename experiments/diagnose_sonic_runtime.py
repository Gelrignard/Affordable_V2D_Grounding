"""Bounded zero-residual reference/controller diagnostics; no learning.

Native cases use pinned FlashCHORD construction and reset logic. To inspect
physical behavior after a tracking failure, the reference clock continues on
an explicit schedule without resetting the physical state. This is not a
native policy evaluation or a task-success benchmark.
"""

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from training.bundles import sha256
from training.config import TrainConfig
from training.runtime import (
    SonicEnv,
    materialize_motion,
    runtime_source,
    support_geometry,
    synchronize_reference_frames,
)


def native_env(args, out):
    source = runtime_source(args.runtime)
    sys.path.insert(0, str(source))
    import warp as wp
    from flash_chord.data import load_reference
    from flash_chord.envs.rl import RLEnv
    from flash_chord.scene.builder import build_scene
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    wp.init()
    wp.set_device("cuda:0")
    with initialize_config_dir(
        config_dir=str(source / "flash_chord/configs"), version_base="1.3"
    ):
        cfg = compose(
            config_name="train_flash_sac",
            overrides=["experiment=g1_recon_body_flash_sac"],
        )
    cfg.env.auto_reset = False
    cfg.env.capture_terminal_state = False
    cfg.env.voc_scale = 0.0
    cfg.reset.always_reset_to_first_frame = True
    cfg.reset.seed = args.seed
    cfg.action.body_delay.seed = args.seed
    urdf = source / "flash_chord/assets/urdfs/g1/main_with_hand.urdf"
    cfg.embodiment.config.asset_sha256 = sha256(urdf)
    motion = materialize_motion(args.job, out / "motion.parquet")
    ref = load_reference(
        motion,
        control_fps=float(cfg.sim.fps),
        source_frame_playback=args.timing == "source",
    )
    if args.reference_part in ("frozen_legs", "frozen_locomotion"):
        joints = ref.robot_joint_pos().copy()
        leg_ids = [
            i
            for i, name in enumerate(ref.robot_joint_names())
            if any(word in name for word in ("hip_", "knee_", "ankle_"))
        ]
        joints[:, leg_ids] = joints[0, leg_ids]
        ref = replace(ref, _robot_joint_pos=joints)
    if args.reference_part in ("frozen_root_orientation", "frozen_locomotion"):
        quat = ref.robot_root_quat_w().copy()
        quat[:] = quat[0]
        ref = replace(ref, _robot_root_quat_w=quat)
    if args.timing == "resampled" or args.reference_part != "full":
        ref = synchronize_reference_frames(ref, urdf)
    support = Path(args.job) / "assets/support.usda"
    scene = build_scene(
        instantiate(cfg.embodiment, _convert_="all"),
        ref,
        world_count=args.worlds,
        support_usda=support if support_geometry(support)[:, 0].any() else None,
        collision=instantiate(cfg.collision, _convert_="all"),
        decompose_objects=True,
        contact_friction=1.0,
        object_free_joint_damping=0.01,
    )
    env = RLEnv(
        scene, ref, config=instantiate(cfg.env, _convert_="all"), device="cuda:0"
    )
    env.set_voc_scale(0.0)
    if args.sonic_carrier:
        from training.controller_reference import install_sonic_carrier

        carrier_path = materialize_motion(args.sonic_carrier, out / "carrier.parquet")
        carrier = load_reference(
            carrier_path,
            control_fps=float(cfg.sim.fps),
            source_frame_playback=args.timing == "source",
        )
        install_sonic_carrier(env, scene, ref, carrier)
    for session in (env.action.encoder_session, env.action.decoder_session):
        assert session.provider == "CUDAExecutionProvider"
    env.reset() if args.settle else env.reset(frame_id=0)
    env.capture_step()
    env.reset() if args.settle else env.reset(frame_id=0)
    (out / "config.json").write_text(
        json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2)
    )
    return env, ref, wp, float(cfg.sim.fps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--backend", choices=["adapter", "native"], default="native")
    parser.add_argument(
        "--timing", choices=["resampled", "source"], default="resampled"
    )
    parser.add_argument("--settle", action="store_true")
    parser.add_argument("--hold", action="store_true")
    parser.add_argument(
        "--sonic-carrier",
        help="Separate body reference for SONIC only; retain corrected task and finger references",
    )
    parser.add_argument(
        "--reference-part",
        choices=["full", "frozen_legs", "frozen_root_orientation", "frozen_locomotion"],
        default="full",
    )
    parser.add_argument("--seconds", type=float, default=16.0)
    parser.add_argument("--worlds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    started = time.perf_counter()
    adapter = None
    if args.backend == "adapter":
        if (
            args.settle
            or args.hold
            or args.timing != "resampled"
            or args.reference_part != "full"
        ):
            raise ValueError("adapter case must preserve its existing semantics")
        if (
            args.sonic_carrier
            and Path(args.sonic_carrier).resolve()
            != (Path(args.job).parent / "original").resolve()
        ):
            raise ValueError("adapter carrier must be the same episode's original job")
        adapter = SonicEnv(
            args.job,
            args.runtime,
            TrainConfig(
                worlds=args.worlds,
                seed=args.seed,
                sonic_reference_mode="original_body" if args.sonic_carrier else "task",
            ),
            out / "runtime",
        )
        env, ref, wp, fps = (
            adapter.env,
            adapter.reference,
            adapter.wp,
            adapter.control_fps,
        )
        adapter.reset(first=True)
    else:
        env, ref, wp, fps = native_env(args, out)
    wp.synchronize()
    setup_seconds = time.perf_counter() - started
    tensor = wp.to_torch
    action = torch.zeros((args.worlds, env.action.action_dim), device="cuda")
    ids = tensor(env.command.body_ids_w).long().reshape(args.worlds, -1)[:, 0]
    labels = list(env.model.body_label)
    initial_q = (
        tensor(env.state_0.joint_q).reshape(args.worlds, -1).clone().cpu().numpy()
    )
    initial_body = tensor(env.state_0.body_q).clone().cpu().numpy()
    log = {
        k: []
        for k in (
            "root_z",
            "root_error",
            "object_error",
            "contact",
            "terminated",
            "frame",
            "joint_q",
        )
    }
    steps = int(args.seconds * fps)
    ref_root = torch.as_tensor(ref.robot_root_pos_w(), device="cuda")
    ref_obj = torch.as_tensor(ref.object_body_pos_w()[:, 0], device="cuda")
    started = time.perf_counter()
    with torch.no_grad():
        for step in range(steps):
            if adapter:
                frame = tensor(env.timestep).long().clone()
                adapter.step(action, reset=False)
            else:
                index = (
                    0
                    if args.hold
                    else min(
                        max(0, step - (50 if args.settle else 0)), ref.num_frames - 1
                    )
                )
                frame = torch.full(
                    (args.worlds,), index, device="cuda", dtype=torch.long
                )
                tensor(env.timestep).copy_(frame.to(torch.int32))
                # Continue the configured reset-age schedule after task failure,
                # as well as the reference clock, without teleporting the robot.
                tensor(env.reset_policy.steps_since_reset).fill_(
                    step if args.settle else step + 51
                )
                wp.synchronize()
                env.step(wp.from_torch(action.flatten()))
            wp.synchronize()
            q = tensor(env.state_0.joint_q).reshape(args.worlds, -1)
            body = tensor(env.state_0.body_q)
            con = env.contact
            force = tensor(con.contact_force_w).reshape(args.worlds, -1, 3).norm(dim=-1)
            contact = tensor(con.contact_active).reshape(args.worlds, -1).bool()
            values = {
                "root_z": q[:, 2],
                "root_error": (q[:, :3] - ref_root[frame]).norm(dim=-1),
                "object_error": (body[ids, :3] - ref_obj[frame]).norm(dim=-1),
                "contact": (contact & (force > 0.01)).any(-1),
                "terminated": tensor(env.done).bool(),
                "frame": frame,
                "joint_q": q,
            }
            for k, v in values.items():
                log[k].append(v.cpu().numpy().copy())
    elapsed = time.perf_counter() - started
    log = {k: np.stack(v) for k, v in log.items()}
    np.savez_compressed(
        out / "trace.npz",
        initial_q=initial_q,
        initial_body=initial_body,
        body_labels=labels,
        **log,
    )
    results = []
    for w in range(args.worlds):
        fall = np.flatnonzero(log["root_z"][:, w] < 0.35)
        fail = np.flatnonzero(log["terminated"][:, w])
        results.append(
            {
                "world": w,
                "first_root_below_0.35m_s": float((fall[0] + 1) / fps)
                if len(fall)
                else None,
                "first_tracking_failure_s": float((fail[0] + 1) / fps)
                if len(fail)
                else None,
                "min_root_z_m": float(log["root_z"][:, w].min()),
                "final_root_z_m": float(log["root_z"][-1, w]),
                "mean_root_error_m": float(log["root_error"][:, w].mean()),
                "mean_object_error_m": float(log["object_error"][:, w].mean()),
                "contact_fraction": float(log["contact"][:, w].mean()),
            }
        )
    summary = {
        "arguments": vars(args),
        "setup_seconds": setup_seconds,
        "rollout_wall_seconds": elapsed,
        "results": results,
        "reference_frames": ref.num_frames,
        "reference_root_z_range": [
            float(ref_root[:, 2].min()),
            float(ref_root[:, 2].max()),
        ],
        "note": "Zero residual, no learner; forced continuing reference clock after failures. Native settle includes reset-time object assistance for at most 1 second. Not an official success benchmark.",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()

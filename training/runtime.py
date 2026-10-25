"""Adapter to pinned FlashCHORD physics and frozen SONIC; no FlashSAC learner.

Imports GPU dependencies only when constructing an environment. The adapter owns
reset ordering so replay always receives the physical terminal next state.
"""

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from . import RUNTIME_COMMIT
from .bundles import arrays, sha256
from .prepare import load_job
from .wrench import basis_directions, capability_reward, wrench_support


def runtime_source(path):
    root = Path(path).resolve()
    source = root / "robotic_grounding/flash_chord/src"
    if not (source / "flash_chord").is_dir():
        raise FileNotFoundError("--runtime must name the pinned video_to_data checkout")
    commit = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    )
    if commit != RUNTIME_COMMIT or dirty.strip():
        raise ValueError(f"runtime must be clean at {RUNTIME_COMMIT}")
    return source


def materialize_motion(job_dir, destination):
    """Rebase paths after cloud transfer without changing the hashed input job."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    folder = Path(job_dir).resolve()
    row = pq.read_table(folder / "motion.parquet").to_pylist()[0]
    row["object_mesh_paths"] = [str(folder / "assets/object.obj")]
    row["object_urdf_paths"] = [str(folder / "assets/object.urdf")]
    pq.write_table(pa.Table.from_pylist([row]), destination, compression="zstd")
    return Path(destination)


def canonical_urdf_sha(path):
    # Git's Windows checkout changes line endings, not the robot model.
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def canonical_history(history, head, initialized):
    worlds, length, width = history.shape
    index = (
        head.long()[:, None] + 1 + torch.arange(length, device=history.device)[None]
    ) % length
    ordered = history.gather(1, index[..., None].expand(worlds, length, width))
    return ordered * initialized[:, None, None]


def support_geometry(path, max_shapes=8):
    """Describe the same local Cylinder/Cube primitives the pinned builder uses.

    Empty support files in ground-contact episodes mean ground only. We do not
    pass them to the upstream loader, which requires at least one support shape.
    """
    from pxr import Usd

    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise ValueError(f"cannot open support surface {path}")
    shapes = []
    for prim in stage.Traverse():
        kind = prim.GetTypeName()
        if kind not in ("Cube", "Cylinder"):
            continue
        position = prim.GetAttribute("xformOp:translate").Get()
        position = tuple(position) if position is not None else (0, 0, 0)
        if kind == "Cylinder":
            dimensions = (
                float(prim.GetAttribute("radius").Get()),
                float(prim.GetAttribute("height").Get()),
                1.0,
            )
        else:
            size, scale = (
                prim.GetAttribute("size").Get(),
                prim.GetAttribute("xformOp:scale").Get(),
            )
            scale = tuple(scale) if scale is not None else (1, 1, 1)
            dimensions = tuple(
                float(size if size is not None else 2) * abs(float(x)) for x in scale
            )
        if (
            not np.isfinite(position).all()
            or not np.isfinite(dimensions).all()
            or min(dimensions) <= 0
        ):
            raise ValueError("invalid support dimensions")
        shapes.append([1, float(kind == "Cylinder"), *position, *dimensions])
    if len(shapes) > max_shapes:
        raise ValueError(f"support descriptor supports at most {max_shapes} primitives")
    descriptor = np.zeros((max_shapes, 8), np.float32)
    if shapes:
        descriptor[: len(shapes)] = shapes
    return descriptor


class SonicEnv:
    def __init__(self, job_dir, runtime, config, scratch):
        self.cfg, self.job_dir = config, Path(job_dir).resolve()
        self.job = load_job(job_dir)
        self.device = torch.device(config.device)
        self.worlds = config.worlds
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError(
                "G1/SONIC requires CUDA; use the CPU tests for local validation"
            )
        torch.cuda.set_device(self.device)
        source = runtime_source(runtime)
        sys.path.insert(0, str(source))
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        import warp as wp
        from flash_chord.data import load_reference
        from flash_chord.envs.rl import RLEnv
        from flash_chord.scene.builder import build_scene
        from hydra import compose, initialize_config_dir
        from hydra.utils import instantiate
        from omegaconf import OmegaConf

        self.wp = wp
        wp.init()
        wp.set_device(str(self.device))
        urdf = source / "flash_chord/assets/urdfs/g1/main_with_hand.urdf"
        expected = self.job.get(
            "urdf_canonical_sha256",
            "8c7f768dc8da6c969d8a4b4efb3dcbe3d9d2b05fca5104df12205a69879d19e9",
        )
        if canonical_urdf_sha(urdf) != expected:
            raise ValueError("runtime robot differs from prepared reference kinematics")
        scratch = Path(scratch)
        scratch.mkdir(parents=True, exist_ok=True)
        motion = materialize_motion(self.job_dir, scratch / "runtime_motion.parquet")
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
        cfg.reset.reset_freeze_steps = 0
        cfg.reset.voc_decay_steps = 0
        cfg.reset.reset_voc_scale = 0.0
        cfg.reset.shoulder_spread = 0.0
        cfg.reset.seed = config.seed
        cfg.termination.reset_freeze_steps = 0
        # Permit departures from an imperfect posture. Keep fall/object-loss termination.
        cfg.termination.palm_position_threshold = 0.6
        cfg.termination.palm_orientation_threshold = 3.14
        cfg.termination.object_position_threshold = 0.4
        cfg.termination.object_orientation_threshold = 3.14
        cfg.objective.body_joint_position_weight = config.reference_weight
        cfg.objective.palm_position_weight = config.reference_weight
        cfg.objective.palm_orientation_weight = config.reference_weight
        cfg.objective.force_closure_weight = 0.0
        cfg.objective.num_wrench_basis = config.wrench_basis
        cfg.action.body_delay.seed = config.seed
        cfg.action.enable_cuda_graph = config.capture_sim
        # Upstream verifies raw bytes. Normalize only the expected hash for CRLF checkouts.
        cfg.embodiment.config.asset_sha256 = sha256(urdf)
        self.resolved_config = OmegaConf.to_container(cfg.env, resolve=True)
        self.control_fps = float(cfg.sim.fps)
        self.reference = load_reference(
            motion, control_fps=self.control_fps, source_frame_playback=False
        )
        self.source_reference = arrays(self.job_dir / "reference.npz")
        self.guidance = arrays(self.job_dir / "guidance.npz")
        self.geometry = arrays(self.job_dir / "geometry.npz")
        self.radius = float(self.geometry["radius"])
        support_path = self.job_dir / "assets/support.usda"
        self.support_descriptor = support_geometry(support_path)
        self.scene = build_scene(
            instantiate(cfg.embodiment, _convert_="all"),
            self.reference,
            world_count=self.worlds,
            support_usda=support_path if self.support_descriptor[:, 0].any() else None,
            collision=instantiate(cfg.collision, _convert_="all"),
            decompose_objects=True,
            contact_friction=1.0,
            object_free_joint_damping=0.01,
        )
        self.env = RLEnv(
            self.scene,
            self.reference,
            config=instantiate(cfg.env, _convert_="all"),
            device=str(self.device),
        )
        self.env.set_voc_scale(0.0)
        self.frames = self.reference.num_frames
        self.object_ids = (
            self.tensor(self.env.command.body_ids_w).long().reshape(self.worlds, -1)
        )
        if self.object_ids.shape[1] != 1:
            raise ValueError(
                "this implementation supports the pilot's single rigid object"
            )
        action = self.env.action
        self.action_names = list(action.residual_joint_names) + list(
            action.finger_joint_names
        )
        self.action_mask = torch.tensor(
            [
                config.lower_body_residual
                if any(word in name for word in ("hip_", "knee_", "ankle_"))
                else 1.0
                for name in self.action_names
            ],
            device=self.device,
        )
        self.basis = basis_directions(config.wrench_basis).to(self.device)
        self._make_goals()
        self.rng = torch.Generator(device=self.device).manual_seed(config.seed)
        self.reset(first=True)
        if config.capture_sim:
            self.env.capture_step()
            self.reset(first=True)
        state = self.state()
        signature = {
            "runtime_commit": RUNTIME_COMMIT,
            "state_blocks": self.blocks,
            "goal_blocks": self.goal_blocks,
            "action_names": self.action_names,
            "action_mask": self.action_mask.tolist(),
            "action_bound": config.action_bound,
            "residual_scale": action.residual_scale,
            "finger_residual_scale": action.finger_residual_scale,
            "env": self.resolved_config,
            "robot_canonical_sha256": canonical_urdf_sha(urdf),
            "controller": "frozen_sonic",
            "contact_force_scale_n": 100.0,
            "control_fps": self.control_fps,
            "version": 1,
            "reward": {
                k: getattr(config, k)
                for k in (
                    "reference_weight",
                    "wrench_weight",
                    "orientation_weight",
                    "friction",
                    "wrench_basis",
                )
            },
        }
        # Runtime random seeds differ between training/evaluation, but the dynamics interface does not.
        signature["env"]["reset"]["seed"] = 0
        signature["env"]["action"]["body_delay"]["seed"] = 0
        self.spec = {
            "state_dim": state.shape[-1],
            "obs_dim": self.obs_dim,
            "goal_dim": self.goals.shape[-1],
            "action_dim": len(self.action_names),
            "action_mask": self.action_mask.tolist(),
            "important_state_ranges": [
                self.blocks[name]
                for name in ("physical_q", "physical_qd", "object_state")
            ],
            "interface": signature,
        }
        self.identity = {
            "job_sha256": sha256(self.job_dir / "job.json"),
            "tier": self.job["tier"],
            "episode_index": self.job["episode_index"],
            "variant": self.job["variant"],
            "runtime_commit": RUNTIME_COMMIT,
            "physics": self.physics,
        }

    def tensor(self, array):
        return self.wp.to_torch(array)

    def synchronize(self):
        self.wp.synchronize_device(self.env.device)
        torch.cuda.synchronize(self.device)

    def _make_goals(self):
        action = self.env.action
        latent = []
        # Encode the exact future-reference window once; SONIC remains frozen.
        for start in range(0, self.frames, self.worlds):
            indices = torch.arange(
                start, start + self.worlds, device=self.device
            ).clamp_max(self.frames - 1)
            self.tensor(action.encoder_input).copy_(
                self.tensor(action.encoder_reference)[indices]
            )
            self.synchronize()
            action.encoder_session.run()
            self.synchronize()
            latent.append(self.tensor(action.latent).clone())
        latent = torch.cat(latent)[: self.frames]
        t_source = (
            self.source_reference["timestamps"] - self.source_reference["timestamps"][0]
        )
        t_control = np.minimum(np.arange(self.frames) / self.control_fps, t_source[-1])
        right = np.searchsorted(t_source, t_control).clip(0, len(t_source) - 1)
        left = np.maximum(right - 1, 0)
        nearest = np.where(
            np.abs(t_source[right] - t_control) < np.abs(t_source[left] - t_control),
            right,
            left,
        )
        to = lambda a: torch.as_tensor(
            np.asarray(a), dtype=torch.float32, device=self.device
        )
        g = self.guidance
        active = to(g["contact_active"][nearest]).bool()
        self.target_support = wrench_support(
            to(g["object_local_points"][nearest]),
            to(g["inward_normals_local"][nearest]),
            active,
            self.basis,
            self.radius,
            self.cfg.friction,
        )
        observed = to(self.source_reference["object_observed_mask"][nearest])
        # Accepted static/wrench checks raise confidence, never certify dynamic feasibility.
        self.contact_confidence = (
            active.any(dim=-1).float()
            * observed
            * (
                0.25
                + 0.25
                * to(
                    g.get("phase_has_accepted_static_seed", np.zeros(len(t_source)))[
                        nearest
                    ]
                )
                + 0.25
                * to(
                    g.get("phase_has_accepted_nominal_wrench", np.zeros(len(t_source)))[
                        nearest
                    ]
                )
            )
        )
        self.object_confidence = 0.25 + 0.75 * observed
        self.reference_confidence = 0.25 + 0.75 * observed
        self.target_object = torch.cat(
            (
                to(self.reference.object_body_pos_w()[:, 0]),
                to(self.reference.object_body_quat_w()[:, 0, [1, 2, 3, 0]]),
            ),
            dim=-1,
        )
        ids = self.object_ids[0]
        mass = self.tensor(self.scene.model.body_mass)[ids].reshape(-1)
        inertia = self.tensor(self.scene.model.body_inertia)[ids].reshape(-1)
        self.physics = {
            "object_mass": mass.cpu().tolist(),
            "object_inertia": inertia.cpu().tolist(),
            "contact_friction": 1.0,
            "object_free_joint_damping": 0.01,
            "assistance_scale": 0.0,
        }
        descriptor = torch.cat(
            (
                to(self.geometry["points"].reshape(-1)),
                to([self.radius, 1.0, 0.01]),
                mass,
                inertia,
                to(self.support_descriptor.reshape(-1)),
            )
        )
        self.goal_blocks, pieces, offset = {}, [], 0
        for name, value in (
            ("phase", to(t_control / max(t_source[-1], 1e-6))[:, None]),
            ("object_pose_xyzw", self.target_object),
            ("reference_joints", to(self.reference.robot_joint_pos())),
            (
                "reference_root_pose_wxyz",
                torch.cat(
                    (
                        to(self.reference.robot_root_pos_w()),
                        to(self.reference.robot_root_quat_w()),
                    ),
                    dim=-1,
                ),
            ),
            ("sonic_reference_latent", latent),
            ("contact_capability", self.target_support),
            (
                "confidence",
                torch.stack((self.object_confidence, self.contact_confidence), dim=-1),
            ),
            ("object_geometry_physics", descriptor[None].expand(self.frames, -1)),
        ):
            self.goal_blocks[name] = [offset, offset + value.shape[-1]]
            offset += value.shape[-1]
            pieces.append(value)
        self.goals = torch.cat(pieces, dim=-1)

    def goal(self):
        return self.goals[
            self.tensor(self.env.timestep).long().clamp(0, self.frames - 1)
        ]

    def object_pose(self):
        return self.tensor(self.env.state_0.body_q)[self.object_ids].clone()

    def state(self):
        self.synchronize()
        e, a, w = self.env, self.env.action, self.worlds
        contact = e.contact
        active = self.tensor(contact.contact_active).reshape(w, -1).float()
        force = (
            self.tensor(contact.contact_force_w).reshape(w, -1, 3) * active[..., None]
        ) / 100.0
        points = (
            self.tensor(contact.contact_pos_o).reshape(w, -1, 3) * active[..., None]
        )
        normals = (
            self.tensor(contact.contact_force_direction_o).reshape(w, -1, 3)
            * active[..., None]
        )
        pieces = [
            ("policy_observation", self.tensor(e.observation).reshape(w, -1)),
            (
                "live_contact",
                torch.cat((active[..., None], points, normals, force), dim=-1).reshape(
                    w, -1
                ),
            ),
        ]
        self.obs_dim = sum(value.shape[-1] for _, value in pieces)
        history = canonical_history(
            self.tensor(a.history),
            self.tensor(a.history_head),
            self.tensor(a.history_initialized),
        )
        pieces.extend(
            [
                (
                    "physical_q",
                    self.tensor(e.state_0.joint_q).reshape(w, -1)[
                        :, : self.scene.layout.num_joint_q
                    ],
                ),
                (
                    "physical_qd",
                    self.tensor(e.state_0.joint_qd).reshape(w, -1)[
                        :, : self.scene.layout.num_joint_dof
                    ],
                ),
                (
                    "object_state",
                    torch.cat(
                        (
                            self.object_pose(),
                            self.tensor(e.state_0.body_qd)[self.object_ids],
                        ),
                        dim=-1,
                    ).reshape(w, -1),
                ),
                ("sonic_history", history.reshape(w, -1)),
                (
                    "sonic_initialized",
                    self.tensor(a.history_initialized).float()[:, None],
                ),
                ("sonic_last_action", self.tensor(a.last_sonic_action).reshape(w, -1)),
                ("action_history", self.tensor(a.actor_action_history).reshape(w, -1)),
                (
                    "reference_offset",
                    self.tensor(e.reference_joint_q_offset).reshape(w, -1)[
                        :, : self.scene.layout.num_joint_q
                    ],
                ),
                (
                    "episode_context",
                    torch.stack(
                        (
                            self.tensor(e.episode_start_frame).float() / self.frames,
                            self.tensor(e.episode_step).float() / self.frames,
                        ),
                        dim=-1,
                    ),
                ),
            ]
        )
        for i, delay in enumerate(a.delays):
            cursor = self.tensor(delay.cursor).long()[0]
            order = (
                cursor + torch.arange(delay.capacity, device=self.device)
            ) % delay.capacity
            hist = (
                self.tensor(delay.history)
                .reshape(delay.capacity, w, -1)[order]
                .permute(1, 0, 2)
            )
            pushes = self.tensor(delay.num_pushes).clamp(max=delay.capacity)
            # Unfilled/reset history is stale memory and is never consumed by the controller.
            valid = (
                torch.arange(delay.capacity, device=self.device)[None]
                >= delay.capacity - pushes[:, None]
            )
            pieces.append(
                (
                    f"delay_{i}",
                    torch.cat(
                        (
                            (hist * valid[..., None]).reshape(w, -1),
                            self.tensor(delay.delay_steps).float()[:, None],
                            pushes.float()[:, None],
                        ),
                        dim=-1,
                    ),
                )
            )
        self.blocks, offset = {}, 0
        for name, value in pieces:
            self.blocks[name] = [offset, offset + value.shape[-1]]
            offset += value.shape[-1]
        return torch.cat([value for _, value in pieces], dim=-1).clone()

    def reset(self, mask=None, first=False):
        if mask is None:
            mask = torch.ones(self.worlds, dtype=torch.bool, device=self.device)
        if first:
            frames = torch.zeros(self.worlds, dtype=torch.int32, device=self.device)
        else:
            frames = torch.randint(
                max(self.frames - 1, 1),
                (self.worlds,),
                generator=self.rng,
                device=self.device,
                dtype=torch.int32,
            )
            frames[
                torch.rand(self.worlds, generator=self.rng, device=self.device)
                < self.cfg.first_frame_probability
            ] = 0
        mask_i = mask.to(torch.int32).contiguous()
        self.synchronize()
        self.env.reset(
            reset_frame=self.wp.from_torch(frames),
            reset_mask=self.wp.from_torch(mask_i),
        )
        self.synchronize()
        # Contact forces belong to the previous transition, not the teleported initial state.
        for name in (
            "contact_active",
            "contact_pos_w",
            "contact_force_w",
            "contact_pos_b",
            "contact_force_direction_b",
            "contact_pos_o",
            "contact_force_direction_o",
        ):
            self.tensor(getattr(self.env.contact, name)).reshape(self.worlds, -1)[
                mask
            ] = 0
        self.refresh_observation()
        return self.state(), self.goal()

    def refresh_observation(self):
        self.synchronize()
        self.env.command.gather(self.env.timestep)
        self.env.observation_strategy.compute(self.env.state_0)
        self.synchronize()

    def step(self, normalized_action, reset=True):
        e = self.env
        pre = self.tensor(e.timestep).long().clamp(0, self.frames - 1).clone()
        action = (
            normalized_action.detach()
            .clamp(-self.action_mask, self.action_mask)
            .contiguous()
        )
        if not torch.isfinite(action).all():
            raise ValueError("nonfinite policy action")
        raw = (action * self.cfg.action_bound).flatten().contiguous()
        self.synchronize()
        e.step(self.wp.from_torch(raw))
        self.synchronize()
        terminated = self.tensor(e.done).bool().clone()
        truncated = self.tensor(e.truncation).bool().clone() | (
            pre + 1 >= self.frames - 1
        )
        terms = self.tensor(e.objective.terms).reshape(self.worlds, -1)
        weights = self.tensor(e.objective.weights)
        shaped = terms * weights[None]
        shaped[:, [6, 9, 10]] *= self.reference_confidence[pre, None]
        shaped[:, 7] *= self.object_confidence[pre]
        obj = self.object_pose()[:, 0]
        q = torch.nn.functional.normalize(obj[:, 3:], dim=-1)
        angle = (
            2 * (q * self.target_object[pre, 3:]).sum(dim=-1).abs().clamp(0, 1).acos()
        )
        contact = e.contact
        active = self.tensor(contact.contact_active).reshape(self.worlds, -1).bool()
        support = wrench_support(
            self.tensor(contact.contact_pos_o).reshape(self.worlds, -1, 3),
            self.tensor(contact.contact_force_direction_o).reshape(self.worlds, -1, 3),
            active,
            self.basis,
            self.radius,
            self.cfg.friction,
        )
        wrench = capability_reward(
            support, self.target_support[pre], self.contact_confidence[pre]
        )
        measured = (
            self.tensor(contact.contact_force_direction_o)
            .reshape(self.worlds, -1, 3)
            .norm(dim=-1)
            > 1e-6
        )
        wrench *= (active & measured).any(dim=-1).float()
        orientation = self.object_confidence[pre] * torch.exp(-angle.square() / 0.4**2)
        reward = (
            shaped.sum(dim=-1)
            + self.cfg.wrench_weight * wrench
            + self.cfg.orientation_weight * orientation
        ) / self.control_fps
        # Continue the reference clock after failure during evaluation; never reset a rollout.
        self.tensor(e.timestep).copy_(
            (pre + 1).clamp_max(self.frames - 1).to(torch.int32)
        )
        self.refresh_observation()
        next_state, next_goal = self.state(), self.goal().clone()
        done = terminated | truncated
        info = {
            "terminated": terminated,
            "truncated": truncated,
            "object_position_error_m": (obj[:, :3] - self.target_object[pre, :3]).norm(
                dim=-1
            ),
            "object_angle_rad": angle,
            "wrench_reward": wrench,
            "executed_action": action,
        }
        if reset and done.any():
            current_state, current_goal = self.reset(done)
        else:
            current_state, current_goal = next_state, next_goal
        return (
            next_state,
            next_goal,
            reward[:, None],
            done[:, None],
            current_state,
            current_goal,
            info,
        )

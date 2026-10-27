"""Bounded, uninterrupted native SAC evaluation and CPU phase analysis.

Every evaluated world starts explicitly at local frame zero, with both training
and reset VOC disabled. Training failure conditions are disabled during physics;
common native thresholds are applied afterwards to the complete recorded trace.
No trajectory alignment, failed-world filtering, or within-attempt reset is used.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from training.leaderboard_metrics import pose_metrics


COMMON_LIMITS = {
    "pelvis_position_m": 0.70,
    "pelvis_orientation_rad": 1.50,
    "palm_position_m": 0.15,
    "palm_orientation_rad": 1.50,
    "object_position_m": 0.10,
    "object_orientation_rad": 1.50,
}


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def first_bad_frame(mask):
    """One first local frame per world; None means no violation, not frame zero."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("Expected [frame, world] violation mask")
    return [int(np.flatnonzero(column)[0]) if column.any() else None for column in mask.T]


def episode_phase_windows(frames, episode=5):
    """Diagnostic windows; thirds are temporal partitions, not inferred task phases."""
    windows = {"full_sequence": (0, frames)}
    if episode == 5:
        windows["handover_730_850"] = (730, min(851, frames))
    else:
        first, second = frames // 3, 2 * frames // 3
        windows.update({
            "early_third": (0, first),
            "middle_third": (first, second),
            "late_third": (second, frames),
        })
    windows["final_200"] = (max(0, frames - 200), frames)
    return windows


def phase_analysis(achieved, reference, vertices, object_ids, *, invalid_worlds=(), episode=5):
    """Pure NumPy analysis; frame bounds below are zero-based and end-exclusive."""
    achieved = np.asarray(achieved)
    reference = np.asarray(reference)
    # Validate shape and preserve the existing challenge-metric implementation.
    full = pose_metrics(achieved, reference, vertices, object_ids)
    invalid = sorted(set(full["invalid_worlds"]) | {int(w) for w in invalid_worlds})
    frames = achieved.shape[0]
    windows = episode_phase_windows(frames, episode)
    report = {
        "schema": "v2d_overnight_phases_v1",
        "frame_convention": "zero_based_local_reference_frames; intervals_end_exclusive",
        "reference_alignment": "none",
        "invalid_worlds": invalid,
        "world_count": int(achieved.shape[1]),
        "frame_count": int(frames),
        "phases": {},
    }
    if invalid:
        # Native rollout replaces non-finite poses with zero; the explicit invalid
        # cohort flag is authoritative and must survive serialization/reanalysis.
        for name, (start, stop) in windows.items():
            report["phases"][name] = {
                "frame_start": start, "frame_stop": stop, "available": False,
                "reason": "Invalid rollout cohort; no worlds excluded from scoring",
            }
        return report
    q = achieved[..., 3:].astype(np.float64)
    qr = reference[:, None, :, 3:].astype(np.float64)
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    qr /= np.linalg.norm(qr, axis=-1, keepdims=True)
    dot = np.clip(np.abs((q * qr).sum(axis=-1)), 0.0, 1.0)
    angle = 2 * np.arctan2(np.sqrt(np.maximum(0.0, 1.0 - dot * dot)), dot)
    position = np.linalg.norm(achieved[..., :3] - reference[:, None, :, :3], axis=-1)
    bad = ((position > 0.10) | (angle > 1.50)).any(axis=-1)
    report["first_object_10cm_or_1p5rad_frame_by_world"] = first_bad_frame(bad)
    report["object_threshold_diagnostic"] = {
        "position_m": 0.10, "orientation_rad": 1.50,
        "definition": "any_object_body; any_frame; diagnostic_only_not_SP_SR_or_MP_SR",
    }
    for name, (start, stop) in windows.items():
        if start >= stop:
            report["phases"][name] = {
                "frame_start": start, "frame_stop": stop, "available": False,
                "reason": "Reference does not reach the requested window",
            }
            continue
        p, a = position[start:stop], angle[start:stop]
        scores = full if name == "full_sequence" else pose_metrics(
            achieved[start:stop], reference[start:stop], vertices, object_ids
        )
        report["phases"][name] = {
            "frame_start": start, "frame_stop": stop, "available": True,
            "scores": scores["scores"],
            "position_mean_cm": float(p.mean() * 100),
            "position_p95_cm": float(np.quantile(p, 0.95) * 100),
            "position_max_cm": float(p.max() * 100),
            "orientation_mean_rad": float(a.mean()),
            "orientation_p95_rad": float(np.quantile(a, 0.95)),
            "position_mean_cm_by_world": (p.mean(axis=(0, 2)) * 100).tolist(),
            "orientation_mean_rad_by_world": a.mean(axis=(0, 2)).tolist(),
            "every_frame_object_threshold_pass_fraction": float((~bad[start:stop].any(axis=0)).mean()),
        }
    return report


def common_tracking_analysis(errors, names, *, freeze_steps, invalid_worlds=()):
    """A fixed diagnostic across checkpoints, independent of training cutoffs/dwell."""
    errors = np.asarray(errors)
    if errors.ndim != 3 or errors.shape[-1] != len(names):
        raise ValueError("Tracking errors must be [frame, world, named_error]")
    violations = np.zeros(errors.shape[:2], dtype=bool)
    first, fraction = {}, {}
    frame = np.arange(errors.shape[0])[:, None]
    for i, name in enumerate(names):
        if name not in COMMON_LIMITS:
            continue
        active = np.ones_like(violations) if name.startswith("pelvis_") else frame > freeze_steps
        bad = active & ((errors[..., i] > COMMON_LIMITS[name]) | ~np.isfinite(errors[..., i]))
        violations |= bad
        first[name] = first_bad_frame(bad)
        fraction[name] = float(bad.any(axis=0).mean())
    for world in invalid_worlds:
        violations[:, int(world)] = True
    return {
        "thresholds": COMMON_LIMITS,
        "activation": {"pelvis": "all_frames", "palm_and_object": f"frame > {freeze_steps}"},
        "semantics": "posthoc_original_native_cutoffs; not_modified_training_dwell_or_official_task_success",
        "success_fraction": float((~violations.any(axis=0)).mean()),
        "first_failure_frame_by_world": first_bad_frame(violations),
        "first_failure_frame_by_term": first,
        "violation_fraction_by_term": fraction,
    }


def quat_multiply(a, b):
    av, aw, bv, bw = a[..., :3], a[..., 3:], b[..., :3], b[..., 3:]
    return np.concatenate((aw * bv + bw * av + np.cross(av, bv), aw * bw - (av * bv).sum(axis=-1, keepdims=True)), axis=-1)


def quat_rotate(q, v):
    t = 2 * np.cross(q[..., :3], v)
    return v + q[..., 3:] * t + np.cross(q[..., :3], t)


class Trace:
    """Bounded CPU copies after completed steps; no simulator or action mutation."""

    def __init__(self, env, steps, stride, max_samples):
        self.env, self.steps = env, steps
        self.stride = max(stride, math.ceil(max(1, steps - 1) / (max_samples - 1)))
        self.data = {}
        self.index = 0
        native = env.objective
        while hasattr(native, "native"):
            native = native.native
        self.native = native
        self.available = {"palms": hasattr(native, "wrist_frames"), "contact_forces": env.contact is not None,
                          "raw_policy_actions": hasattr(env.action, "raw_action")}
        self.palm_cache = None
        if self.available["palms"]:
            mapping = native.wrist_frames
            self.palm_cache = (
                mapping.body_ids.numpy().astype(int), mapping.body_to_frame_pos.numpy(),
                mapping.body_to_frame_quat.numpy(),
                native.reference.wrist_pos_w.numpy().reshape(native.num_frames, native.num_hands, 3),
                native.reference.wrist_quat_w.numpy().reshape(native.num_frames, native.num_hands, 4),
                native.contact_active.numpy().reshape(native.num_frames, native.num_hands, -1),
            )
        self.sides = list(env.contact.layout.sides) if env.contact is not None else []

    def append(self, key, value):
        self.data.setdefault(key, []).append(np.asarray(value).copy())

    def after_step(self):
        env, frame = self.env, self.index
        self.index += 1
        actual_frame = env.timestep.numpy()
        expected = min(frame + 1, self.steps - 1)
        if np.any(actual_frame != expected):
            raise RuntimeError(f"Rollout reset/froze at step {frame}: reference counters {actual_frame.tolist()}, expected {expected}")
        scale = env.reset_policy.applied_voc_scale.numpy()
        if not np.isfinite(scale).all() or np.any(np.abs(scale) > 1e-7):
            raise RuntimeError("Unassisted evaluation observed nonzero reset/training VOC scale")
        if frame % self.stride and frame != self.steps - 1:
            return
        self.append("reference_frame", frame)
        self.append("applied_voc_scale", scale)
        if self.palm_cache is not None:
            ids, local_p, local_q, ref_p, ref_q, active = self.palm_cache
            body = env.state_0.body_q.numpy().reshape(env.world_count, -1, 7)[:, ids]
            palm = np.concatenate((body[..., :3] + quat_rotate(body[..., 3:], local_p), quat_multiply(body[..., 3:], local_q)), axis=-1)
            self.append("palm_pose_w_xyzw", palm)
            self.append("reference_palm_pose_w_xyzw", np.concatenate((ref_p[frame], ref_q[frame]), axis=-1))
            self.append("reference_contact_active", active[frame])
        if self.available["contact_forces"]:
            contact = env.contact
            slots = np.asarray(contact.layout.slot_hand_ids)
            force = contact.contact_force_w.numpy().reshape(env.world_count, -1, 3)
            active = contact.contact_active.numpy().reshape(env.world_count, -1)
            self.append("hand_contact_force_sum_norm_n", np.stack([
                np.linalg.norm(force[:, slots == hand], axis=-1).sum(axis=-1)
                for hand in range(contact.layout.num_hands)], axis=-1))
            self.append("hand_contact_active", np.stack([
                active[:, slots == hand].any(axis=-1)
                for hand in range(contact.layout.num_hands)], axis=-1))
        if self.available["raw_policy_actions"]:
            self.append("policy_action", env.action.raw_action.numpy().reshape(env.world_count, -1))

    def write(self, out, network):
        arrays = {name: np.asarray(values) for name, values in self.data.items()}
        np.savez_compressed(out / "diagnostics.npz", **arrays)
        summary = {
            "available": self.available, "stride": self.stride, "sample_count": len(arrays["reference_frame"]),
            "hand_order": self.sides,
            "contact_force_semantics": "sum_of_per_contact_slot_force_norms; solved_last_physics_substep_not_control_interval_impulse",
            "palm_reference_semantics": "original_world_reference; not_soft_object_relative_training_target",
            "actuator_torque_clamp_fraction": None,
            "clamp_note": "Actuator saturation is not directly recorded; action magnitudes are not mislabeled as torque clipping.",
        }
        actions = arrays.get("policy_action")
        if actions is not None and np.isfinite(actions).all():
            scale = float(network.get("action_scale", 1.0))
            summary["policy_action_abs_p99"] = float(np.quantile(np.abs(actions), 0.99))
            summary["policy_action_abs_max"] = float(np.abs(actions).max())
            summary["policy_action_transform"] = network.get("action_transform", "unspecified")
            summary["near_tanh_bound_fraction"] = float((np.abs(actions) >= 0.99 * scale).mean()) if network.get("action_transform") == "tanh" else None
        forces = arrays.get("hand_contact_force_sum_norm_n")
        if forces is not None and np.isfinite(forces).all():
            summary["hand_contact_force_mean_n"] = forces.mean(axis=(0, 1)).tolist()
            summary["hand_contact_force_p95_n"] = np.quantile(forces, 0.95, axis=(0, 1)).tolist()
        return summary


def run(args):
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    if args.runtime:
        sys.path.insert(0, str(Path(args.runtime).resolve() / "robotic_grounding/flash_chord/src"))
    from flash_chord.evaluation import harness
    from flash_chord.lifecycle.termination import ReconBodyTerminationConfig
    from training.leaderboard import leaderboard_row, leaderboard_scores, write_report

    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    captured = {}
    original_composition = harness.apply_evaluation_composition
    original_record = harness.record_object_pose_rollout
    allowed = {field.name for field in dataclasses.fields(ReconBodyTerminationConfig)}

    def composition(config, **kwargs):
        captured["saved_environment_config"] = copy.deepcopy(config["env"])
        captured["network"] = copy.deepcopy(config["training"]["network"])
        if harness.checkpoint_training_algorithm(config) != "flash_sac":
            raise ValueError("Overnight evaluator expects a native FlashSAC checkpoint")
        original_composition(config, **kwargs)
        if "reset_voc_scale" in config["env"]["reset"]:
            config["env"]["reset"]["reset_voc_scale"] = 0.0
        # Custom training dwell/gross-failure logic must not freeze the physical
        # evaluation. Retain native diagnostic errors, with every cutoff disabled.
        for location in (config["env"], config):
            block = location.get("termination")
            if not isinstance(block, dict):
                continue
            if not harness._is_recon_body_termination(block):
                raise ValueError("Expected ReconBody termination diagnostics")
            native = {key: value for key, value in block.items() if key in allowed}
            native["_target_"] = "flash_chord.lifecycle.termination.ReconBodyTerminationConfig"
            location["termination"] = native
        captured["evaluation_environment_config"] = copy.deepcopy(config["env"])
        return config

    def record(env, jax_env, *pos, **kwargs):
        trace = Trace(env, kwargs["step_count"], args.diagnostic_stride, args.max_diagnostic_samples)
        if np.any(env.timestep.numpy() != 0) or env.config.auto_reset:
            raise RuntimeError("Evaluation must begin at frame zero with auto-reset disabled")
        if env.config.voc.max_force != 0 or env.config.voc.max_torque != 0:
            raise RuntimeError("Evaluation object force/torque limits must be exactly zero")

        class MeasuredStep:
            def step(self, action):
                result = jax_env.step(action)
                trace.after_step()
                return result

        result = original_record(env, MeasuredStep(), *pos, **kwargs)
        captured["rollout"], captured["trace"] = result, trace
        return result

    harness.apply_evaluation_composition = composition
    harness.record_object_pose_rollout = record
    try:
        outcome = harness.evaluate_checkpoint(
            args.checkpoint, world_count=args.worlds, start_frame=0, reset_mode="explicit",
            object_trajectories_output=out / "object_rollouts.parquet", eval_script=__file__,
            source_root=args.source_root,
        )
    finally:
        harness.apply_evaluation_composition = original_composition
        harness.record_object_pose_rollout = original_record
    harness.write_evaluation_report(out / "evaluation.json", outcome)
    data = json.loads((out / "evaluation.json").read_text(encoding="utf-8"))
    rollout = captured["rollout"]
    np.savez_compressed(
        out / "rollout_arrays.npz", achieved_pose_w=rollout.achieved_pose_w,
        reference_pose_w=rollout.reference_pose_w, object_vertices_o=rollout.object_vertices_o,
        body_object_ids=rollout.body_object_ids, tracking_error=rollout.tracking_error,
        tracking_error_names=np.asarray(rollout.tracking_error_names),
        invalid_worlds=np.asarray(rollout.non_finite_worlds, dtype=np.int32),
    )
    phases = phase_analysis(rollout.achieved_pose_w, rollout.reference_pose_w, rollout.object_vertices_o,
                            rollout.body_object_ids, invalid_worlds=rollout.non_finite_worlds,
                            episode=args.episode)
    write_json(out / "phase_metrics.json", phases)
    scores = phases["phases"]["full_sequence"].get("scores", dict.fromkeys(("add_auc", "spider_sr", "maniptrans_sr", "mppe_cm")))
    if not rollout.non_finite_worlds:
        native_scores = leaderboard_scores(data)
        for name, value in scores.items():
            if value is None or native_scores[name] is None or not np.isclose(value, native_scores[name], rtol=1e-4, atol=1e-4):
                raise RuntimeError(f"Independent {name} disagrees: {value} vs {native_scores[name]}")
    saved_env = captured["saved_environment_config"]
    data.update({
        "schema": "v2d_overnight_evaluation_v1", "leaderboard": {"scores": scores},
        "metric_reference": f"tier{args.tier}_input_reference", "official_leaderboard_score": False,
        "elapsed_s": time.monotonic() - started,
        "evaluation_contract": {"frame_zero": True, "explicit_reset": True, "within_attempt_resets": 0,
                                "voc_scale": 0.0, "reset_voc_scale": 0.0, "settling_steps": 0,
                                "training_termination_disabled_during_rollout": True},
        "training_termination_config": saved_env["termination"],
        "reference_tracking_note": "Native report uses checkpoint numeric cutoffs only; custom training dwell semantics are not reconstructed. Use common_native_reference_tracking for cross-checkpoint diagnostic comparisons.",
        "common_native_reference_tracking": common_tracking_analysis(
            rollout.tracking_error, rollout.tracking_error_names,
            freeze_steps=ReconBodyTerminationConfig().reset_freeze_steps,
            invalid_worlds=rollout.non_finite_worlds),
        "phase_metrics": phases,
        "diagnostics": captured["trace"].write(out, captured["network"]),
    })
    write_json(out / "evaluation_environment_config.json", captured["evaluation_environment_config"])
    write_json(out / "evaluation.json", data)
    write_report(out, [leaderboard_row(data, tier=args.tier, episode=args.episode, variant=args.variant,
                                      training_s=args.training_seconds, transitions=outcome.environment_steps)])
    print(json.dumps({"output": str(out), "scores": scores, "elapsed_s": data["elapsed_s"]}), flush=True)
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint")
    parser.add_argument("--runtime", help="Pinned FlashCHORD repository; omit when already on PYTHONPATH")
    parser.add_argument("--source-root", help="Native evaluator source-root mapping for checkpoint asset paths")
    parser.add_argument("--output", required=True)
    parser.add_argument("--worlds", type=int, default=16)
    parser.add_argument("--tier", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--episode", type=int, default=5)
    parser.add_argument("--variant", default="overnight_refinement")
    parser.add_argument("--training-seconds", type=float)
    parser.add_argument("--diagnostic-stride", type=int, default=5)
    parser.add_argument("--max-diagnostic-samples", type=int, default=400)
    parser.add_argument("--analyze-rollout", help="CPU-only: recompute phase metrics from rollout_arrays.npz")
    args = parser.parse_args()
    if args.analyze_rollout:
        with np.load(args.analyze_rollout, allow_pickle=False) as arrays:
            report = phase_analysis(arrays["achieved_pose_w"], arrays["reference_pose_w"], arrays["object_vertices_o"],
                                    arrays["body_object_ids"], invalid_worlds=arrays["invalid_worlds"],
                                    episode=args.episode)
        destination = Path(args.output)
        destination.mkdir(parents=True, exist_ok=True)
        write_json(destination / "phase_metrics.json", report)
        print(json.dumps(report), flush=True)
        return
    if not args.checkpoint:
        parser.error("--checkpoint is required unless --analyze-rollout is used")
    if not 1 <= args.worlds <= 64 or args.diagnostic_stride < 1 or not 2 <= args.max_diagnostic_samples <= 2000:
        parser.error("worlds must be 1..64; positive stride; max diagnostic samples 2..2000")
    run(args)


if __name__ == "__main__":
    main()

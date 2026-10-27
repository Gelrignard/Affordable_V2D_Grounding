"""Native full-sequence evaluation plus independent metric and physics recording.

Positive --probe-assistance is exclusively a curriculum diagnostic. Only scale
zero writes a leaderboard table. No probe ever enters replay or policy updates.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def run(args):
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", "/root/autodl-tmp/v2d/jax_cache")
    import numpy as np
    from flash_chord.evaluation import harness

    from training.leaderboard import leaderboard_row, leaderboard_scores, write_report
    from training.leaderboard_metrics import pose_metrics
    from training.native_guidance import DynamicsObjective, GuidedRLEnv

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    captured = {}

    class MeasuredEnv(GuidedRLEnv):
        def step(self, *pos, **kw):
            result = super().step(*pos, **kw)
            if captured.get("recording") and isinstance(
                self.objective, DynamicsObjective
            ):
                captured.setdefault("physical", []).append(
                    self.objective.diagnostics.numpy()
                )
                scale = self.reset_policy.applied_voc_scale.numpy()
                if not np.allclose(scale, args.probe_assistance, atol=1e-6):
                    raise RuntimeError(
                        "Evaluation assistance differs from the declared scale"
                    )
            return result

    harness.RLEnv = MeasuredEnv
    native_composition = harness.apply_evaluation_composition
    native_stage = harness._evaluation_curriculum_stage
    native_record = harness.record_object_pose_rollout

    def composition(config, **kwargs):
        force, torque = (
            config["env"]["voc"]["max_force"],
            config["env"]["voc"]["max_torque"],
        )
        native_composition(config, **kwargs)
        if args.probe_assistance > 0:
            config["env"]["voc_scale"] = args.probe_assistance
            config["env"]["voc"].update(max_force=force, max_torque=torque)
        return config

    def stage(original, mode):
        return dataclasses.replace(
            native_stage(original, mode), voc_scale=args.probe_assistance
        )

    def record(*pos, **kwargs):
        captured["recording"] = True
        try:
            result = native_record(*pos, **kwargs)
            captured["rollout"] = result
            return result
        finally:
            captured["recording"] = False

    harness.apply_evaluation_composition = composition
    harness._evaluation_curriculum_stage = stage
    harness.record_object_pose_rollout = record
    outcome = harness.evaluate_checkpoint(
        args.checkpoint,
        world_count=args.worlds,
        start_frame=0,
        reset_mode="explicit",
        object_trajectories_output=out / "object_rollouts.parquet",
        eval_script=__file__,
    )
    harness.write_evaluation_report(out / "evaluation.json", outcome)
    data = json.loads((out / "evaluation.json").read_text())
    rollout = captured["rollout"]
    np.savez_compressed(out / "auc_vertices.npz", vertices=rollout.object_vertices_o)
    independent = pose_metrics(
        rollout.achieved_pose_w,
        rollout.reference_pose_w,
        rollout.object_vertices_o,
        rollout.body_object_ids,
    )
    if rollout.non_finite_worlds:
        independent["scores"] = dict.fromkeys(independent["scores"])
    else:
        native_scores = leaderboard_scores(data)
        for name, value in independent["scores"].items():
            if (
                value is None
                or native_scores[name] is None
                or not np.isclose(value, native_scores[name], rtol=1e-4, atol=1e-4)
            ):
                raise RuntimeError(
                    f"Independent {name} disagrees: {value} vs {native_scores[name]}"
                )
    data["metric_verification"] = independent
    data["probe_assistance"] = args.probe_assistance
    data["metric_reference"] = f"tier{args.tier}_input_reference"
    data["official_leaderboard_score"] = False
    data["elapsed_s"] = time.monotonic() - start
    physical = captured.get("physical")
    if physical:
        physical = np.asarray(physical)
        np.savez_compressed(
            out / "physical_diagnostics.npz",
            samples=physical,
            names=np.asarray(
                [
                    "wrench_deficit",
                    "hand_force_present",
                    "reference_contact_active",
                    "force_deficit",
                    "torque_deficit",
                    "reference_confidence",
                ]
            ),
        )
        active = physical[..., 2] > 0.5
        data["physical_diagnostics"] = {
            "active_phase_contact_fraction": float(physical[..., 1][active].mean())
            if active.any()
            else None,
            "wrench_deficit_mean": float(physical[..., 0].mean()),
            "force_deficit_mean": float(physical[..., 3].mean()),
            "torque_deficit_mean": float(physical[..., 4].mean()),
            "reference_confidence_mean": float(physical[..., 5].mean()),
        }
    if args.probe_assistance == 0:
        data["leaderboard"] = {"scores": independent["scores"]}
        write_report(
            out,
            [
                leaderboard_row(
                    data,
                    tier=args.tier,
                    episode=args.episode,
                    variant=args.variant,
                    training_s=args.training_seconds,
                    transitions=outcome.environment_steps,
                )
            ],
        )
    temporary = out / "evaluation.json.tmp"
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(out / "evaluation.json")
    print(
        json.dumps(
            {
                "output": str(out),
                "scores": independent["scores"],
                "elapsed_s": data["elapsed_s"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--worlds", type=int, default=16)
    p.add_argument("--probe-assistance", type=float, default=0.0)
    p.add_argument("--tier", type=int, default=1)
    p.add_argument("--episode", type=int, default=5)
    p.add_argument("--variant", default="corrected")
    p.add_argument("--training-seconds", type=float, default=None)
    args = p.parse_args()
    if not 0 <= args.probe_assistance <= 1 or args.worlds <= 0:
        p.error("Invalid probe assistance or world count")
    run(args)

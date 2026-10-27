"""Bounded probe/evaluation hooks around the unchanged native FlashSAC runner."""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from training.adaptive_chord import AdaptiveCurriculum
from training.leaderboard import leaderboard_row, write_report


class CandidateStalled(Exception):
    """Controlled stop after a saved checkpoint and a failed competence gate."""


def atomic_json(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False))
    temp.replace(path)


class NativeExperiment:
    def __init__(self, args, out):
        self.args, self.out = args, Path(out)
        self.started = time.monotonic()
        self.evaluation_seconds = 0.0
        self.rows = []
        self.curriculum = None

    def bind(self, env, config):
        self.env = env
        self.curriculum = AdaptiveCurriculum(
            config.curriculum, dynamics_weight=self.args.dynamics_weight
        )
        np.savez_compressed(
            self.out / "dynamics_targets.npz", **env.env.objective.targets
        )
        atomic_json(
            self.out / "extension.json",
            {
                "method": self.args.method,
                "world_model": False,
                "gate": dataclasses.asdict(self.curriculum.gate),
                "stages": [dataclasses.asdict(s) for s in self.curriculum.stages],
                "stall_transitions": self.args.stall_transitions,
                "maximum_transitions": self.args.transitions,
                "native_optimizer_horizon": 349999104,
                "object_mass_kg": env.env.objective.mass,
                "object_radius_m": env.env.objective.radius,
                "wrench_target": "smoothed required object-COM contact wrench; measured support subtracted; no VOC credit",
                "phase_resets": "native random recovery starts plus earlier explicit frame-zero mixture; learned phase-state bank not enabled",
            },
        )
        return dataclasses.replace(config, curriculum=self.curriculum)

    def log(self, metrics, step):
        physical = self.env.env.objective.diagnostics.numpy()
        metrics.update(
            {
                "dynamics/wrench_deficit_sample_mean": float(physical[:, 0].mean()),
                "dynamics/hand_force_present_sample_fraction": float(
                    physical[:, 1].mean()
                ),
            }
        )

    def evaluate(self, checkpoint, step, scale, label, training_s):
        dest = self.out / "evaluations" / f"{step}_{label}"
        command = [
            sys.executable,
            "-u",
            str(
                Path(__file__).resolve().parents[1]
                / "experiments/evaluate_chord_candidate.py"
            ),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(dest),
            "--worlds",
            str(self.args.evaluation_worlds),
            "--probe-assistance",
            str(scale),
            "--variant",
            self.args.method,
            "--training-seconds",
            str(training_s),
        ]
        started = time.monotonic()
        dest.parent.mkdir(exist_ok=True)
        with (dest.parent / f"{step}_{label}.log").open("w") as log:
            subprocess.run(
                command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=360
            )
        self.evaluation_seconds += time.monotonic() - started
        return json.loads((dest / "evaluation.json").read_text())

    def log_checkpoint(self, paths, *, environment_steps, metadata):
        del metadata
        step = environment_steps
        training_s = time.monotonic() - self.started - self.evaluation_seconds
        status = {
            "state": "evaluating",
            "transitions": step,
            "stage": self.curriculum.index,
            "assistance_scale": self.curriculum.stages[self.curriculum.index].voc_scale,
            "checkpoint": str(paths.policy),
            "training_elapsed_s": training_s,
            "evaluation_elapsed_s": self.evaluation_seconds,
        }
        atomic_json(self.out / "extension_status.json", status)
        unassisted = self.evaluate(paths.policy, step, 0, "unassisted", training_s)
        self.rows.append(
            leaderboard_row(
                unassisted,
                tier=1,
                episode=5,
                variant=self.args.method,
                training_s=training_s,
                transitions=step,
            )
        )
        write_report(self.out, self.rows)
        scale = self.curriculum.next_scale
        probe = (
            unassisted
            if scale == 0
            else self.evaluate(paths.policy, step, scale, "probe", training_s)
        )
        decision = self.curriculum.observe(probe, step)
        status.update(
            state="training",
            decision=decision,
            latest_unassisted=unassisted,
            latest_probe=probe,
            evaluation_elapsed_s=self.evaluation_seconds,
            total_elapsed_s=time.monotonic() - self.started,
        )
        atomic_json(self.out / "curriculum_decisions.json", self.curriculum.decisions)
        stopped = self.curriculum.stalled(step, self.args.stall_transitions)
        if stopped:
            status["state"] = "stopped_at_stall_gate"
        elif step >= self.args.transitions:
            status["state"] = "budget_complete"
        atomic_json(self.out / "extension_status.json", status)
        if stopped:
            raise CandidateStalled(
                f"No accepted curriculum advance for {self.args.stall_transitions} transitions; checkpoint saved at {step}"
            )

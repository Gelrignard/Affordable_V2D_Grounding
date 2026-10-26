"""DexMachina-inspired assistance curriculum for native CHORD PPO.

This adapts rolling reward, completion, plateau and recovery ideas; it is not a
reproduction of DexMachina's floating-hand environment or separate PD gains.
Only native CHORD stages/reset mixtures change. No learned dynamics are used.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import asdict, dataclass, replace
from statistics import mean


@dataclass(frozen=True)
class DexGate:
    window: int = 40
    warmup_iterations: int = 120
    minimum_stage_iterations: int = 80
    recovery_iterations: int = 120
    task_minimum: float = 0.5
    contact_minimum: float = 0.05
    imitation_minimum: float = 0.05
    completion_minimum: float = 0.8
    progress_minimum: float = 0.9
    plateau_slope_maximum: float = 0.001
    recovery_progress_maximum: float = 0.25
    immediate_first_frame_probability: float = 0.25

    def __post_init__(self):
        if (
            self.window < 4
            or min(
                self.warmup_iterations,
                self.minimum_stage_iterations,
                self.recovery_iterations,
            )
            < self.window
        ):
            raise ValueError("Curriculum windows must fit warmup and dwell times")
        for name, value in asdict(self).items():
            if isinstance(value, float) and (
                not math.isfinite(value) or not 0 <= value <= 1
            ):
                raise ValueError(f"Invalid curriculum value: {name}")


class DexCurriculum:
    def __init__(self, native, *, frame_dt, gate=None):
        self.gate = gate or DexGate()
        if not math.isfinite(frame_dt) or frame_dt <= 0:
            raise ValueError("Positive finite frame_dt required")
        self.frame_dt = frame_dt
        # Keep CHORD's native reward weights and discrete assistance levels.
        self.stages = tuple(
            replace(
                stage,
                immediate_first_frame_probability=(
                    0.5
                    if stage.voc_scale == 0
                    else self.gate.immediate_first_frame_probability
                ),
            )
            for stage in native.stages[:7]
        )
        if not self.stages or self.stages[-1].voc_scale != 0:
            raise ValueError("Expected native assistance stages ending at zero")
        self.index = 0
        self.last_change = 0
        self.history = deque(maxlen=self.gate.window)
        self.decisions = []

    def stage_index(self, step):
        if step < 0:
            raise ValueError("Negative iteration")
        return self.index

    def _term(self, metrics, name):
        weight = self.stages[self.index].objective_weights[name]
        if weight <= 0:
            raise ValueError(f"Positive normalization weight required for {name}")
        # Upstream logs dt * weight * mean(raw term). Undo both factors.
        value = metrics[f"reward_components/{name}_weighted_mean_per_env_step"]
        return float(value) / (self.frame_dt * weight)

    def observe(self, metrics, iteration):
        sample = {
            "task": self._term(metrics, "object_position"),
            "contact": self._term(metrics, "force_closure"),
            "imitation": mean(
                self._term(metrics, name)
                for name in ("body_joint_position", "palm_position", "palm_orientation")
            ),
            "completion": float(
                metrics["termination/reference_end_fraction_per_completed_episode"]
            ),
            "progress": float(
                metrics["episode/reference_progress_mean_per_completed_episode"]
            ),
            "episodes": float(metrics["episode/completed_count_per_iteration"]),
        }
        if not all(math.isfinite(v) for v in sample.values()):
            raise ValueError("Non-finite curriculum observations")
        # A rollout with no completed episodes carries no evidence of completion.
        self.history.append(sample)
        result = {"iteration": iteration, "stage": self.index, "action": "hold"}
        if len(self.history) < self.gate.window:
            return {**result, "reason": "collecting_window", "sample": sample}
        samples = list(self.history)
        episodes = sum(s["episodes"] for s in samples)
        averages = {
            k: mean(s[k] for s in samples) for k in ("task", "contact", "imitation")
        }
        averages.update(
            {
                k: sum(s[k] * s["episodes"] for s in samples) / max(episodes, 1)
                for k in ("completion", "progress")
            }
        )
        midpoint = (len(samples) - 1) / 2
        denominator = sum((i - midpoint) ** 2 for i in range(len(samples)))
        slopes = {
            key: sum((i - midpoint) * s[key] for i, s in enumerate(samples))
            / denominator
            for key in ("task", "contact", "imitation")
        }
        age = iteration - self.last_change
        result.update(averages=averages, slopes=slopes, completed_episodes=episodes)
        gates = {
            "warmup": iteration >= self.gate.warmup_iterations,
            "stage_dwell": age >= self.gate.minimum_stage_iterations,
            "completed_episodes": episodes >= 100,
            "task": averages["task"] >= self.gate.task_minimum,
            "contact": averages["contact"] >= self.gate.contact_minimum,
            "imitation": averages["imitation"] >= self.gate.imitation_minimum,
            "completion": averages["completion"] >= self.gate.completion_minimum,
            "progress": averages["progress"] >= self.gate.progress_minimum,
            "plateau": max(abs(v) for v in slopes.values())
            <= self.gate.plateau_slope_maximum,
        }
        result["gates"] = gates
        if (
            self.index > 0
            and age >= self.gate.recovery_iterations
            and episodes >= 100
            and averages["progress"] < self.gate.recovery_progress_maximum
        ):
            self.index -= 1
            result.update(action="recover", reason="sustained_short_episodes")
        elif all(gates.values()) and self.index + 1 < len(self.stages):
            self.index += 1
            result.update(action="advance", reason="reward_plateau_and_completion")
        if result["action"] != "hold":
            self.last_change = iteration
            self.history.clear()  # Never mix evidence from different assistance levels.
            result["next_stage"] = self.index
            result["next_assistance"] = self.stages[self.index].voc_scale
            self.decisions.append(result)
        return result

    def state_dict(self):
        return {
            "index": self.index,
            "last_change": self.last_change,
            "history": list(self.history),
            "decisions": self.decisions,
            "gate": asdict(self.gate),
            "frame_dt": self.frame_dt,
            "stages": [asdict(s) for s in self.stages],
        }

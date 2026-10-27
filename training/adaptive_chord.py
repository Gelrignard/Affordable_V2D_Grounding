"""Host-side, probe-gated native CHORD curriculum without a learned world model."""

from __future__ import annotations

import dataclasses
import math


@dataclasses.dataclass
class ProbeGate:
    """Predeclared exploratory thresholds; assisted reward never enters this gate."""

    max_position_m: float = 0.15
    max_orientation_rad: float = 0.8
    max_pelvis_failure: float = 0.25
    min_contact_fraction: float = 0.20
    max_wrench_deficit: float = 1.25

    def accepts(self, report, expected_scale):
        if (
            report.get("non_finite_worlds")
            or report.get("probe_assistance") != expected_scale
        ):
            return False
        metrics = report.get("metrics", {})
        physical = report.get("physical_diagnostics", {})
        values = (
            metrics.get("object_position_error_m"),
            metrics.get("object_orientation_error_rad"),
            report.get("reference_tracking", {})
            .get("failure_cause_fraction", {})
            .get("pelvis"),
            physical.get("active_phase_contact_fraction"),
            physical.get("wrench_deficit_mean"),
        )
        if any(v is None or not math.isfinite(v) for v in values):
            return False
        return (
            values[0] <= self.max_position_m
            and values[1] <= self.max_orientation_rad
            and values[2] <= self.max_pelvis_failure
            and values[3] >= self.min_contact_fraction
            and values[4] <= self.max_wrench_deficit
        )


class AdaptiveCurriculum:
    """Compatible with native runner.stage_index/stages; changes only at probes.

    No forced assistance reduction disguised as measured competence. If progress
    stalls, the bounded runner stops and saves instead of claiming completion.
    """

    def __init__(self, native, *, dynamics_weight=0.0):
        self.thresholds = native.thresholds
        self.stages = tuple(
            dataclasses.replace(
                stage,
                objective_weights=dict(
                    stage.objective_weights, dynamics_wrench=dynamics_weight
                ),
                # Train some prescribed-start episodes before the end of training.
                immediate_first_frame_probability=(
                    0.1 if i < 4 else 0.25 if i < 6 else 0.5
                ),
            )
            for i, stage in enumerate(native.stages[:7])
        )
        self.index = 0
        self.last_advance_step = 0
        self.gate = ProbeGate()
        self.decisions = []

    def stage_index(self, step):
        if step < 0:
            raise ValueError("Negative transition counter")
        return self.index

    @property
    def next_scale(self):
        return self.stages[min(self.index + 1, len(self.stages) - 1)].voc_scale

    def observe(self, report, step):
        if self.decisions and step <= self.decisions[-1]["transitions"]:
            raise ValueError("Probe decisions must have increasing transition counters")
        before = self.index
        passed = self.gate.accepts(report, self.next_scale)
        if passed and self.index < len(self.stages) - 1:
            self.index += 1
            self.last_advance_step = step
        decision = {
            "transitions": step,
            "previous_stage": before,
            "stage": self.index,
            "probe_scale": report.get("probe_assistance"),
            "passed": passed,
            "reason": "lower_assistance_physical_probe"
            if passed
            else "probe_gate_not_met",
        }
        self.decisions.append(decision)
        return decision

    def stalled(self, step, stall_budget):
        return (
            self.index < len(self.stages) - 1
            and step - self.last_advance_step >= stall_budget
        )

"""Explicit resource budgets; no paid runs or implicit data downloads."""

import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass
class TrainConfig:
    algorithm: str = "model_based"
    seed: int = 0
    worlds: int = 256
    device: str = "cuda:0"
    replay_capacity: int = 131072
    replay_device: str = "cpu"
    batch_size: int = 256
    total_transitions: int = 10000000
    max_hours: float = 2.0
    warmup_transitions: int = 16384
    collection_steps: int = 8
    updates_per_transition: float = 1.0
    hidden: int = 256
    model_hidden: int = 512
    model_window: int = 8
    imagination_horizon: int = 8
    imagination_batch: int = 128
    model_updates: int = 4
    model_min_updates: int = 100
    model_validation_interval: int = 25
    model_validation_threshold: float = 1.0
    imagination_weight: float = 0.1
    model_trust_weight: float = 0.1
    gamma: float = 0.99
    learning_rate: float = 0.0003
    tau: float = 0.005
    initial_alpha: float = 0.05
    exploration_std: float = 0.15
    action_bound: float = 3.0
    lower_body_residual: float = 0.0
    reference_weight: float = 0.2
    wrench_weight: float = 1.0
    orientation_weight: float = 1.0
    wrench_basis: int = 64
    friction: float = 0.8
    checkpoint_interval: int = 250000
    log_interval: int = 10
    first_frame_probability: float = 0.25
    capture_sim: bool = True
    curriculum: str = "none"
    curriculum_transitions: int = 349999104
    evaluation_interval: int = 0
    initial_action_std: float = 0.1353352832366127
    sonic_reference_mode: str = "task"

    def validate(self):
        for key, value in self.to_dict().items():
            if isinstance(value, (float, int)) and not math.isfinite(value):
                raise ValueError(f"{key} must be finite")
        if self.algorithm not in ("sac", "model_based"):
            raise ValueError("algorithm must be sac or model_based")
        if self.sonic_reference_mode not in ("task", "original_body"):
            raise ValueError("sonic_reference_mode must be task or original_body")
        if self.curriculum not in ("none", "chord"):
            raise ValueError("curriculum must be none or chord")
        if self.curriculum_transitions <= 0 or self.evaluation_interval < 0:
            raise ValueError("invalid curriculum/evaluation budget")
        if not 0 < self.initial_action_std <= 1:
            raise ValueError("initial_action_std must be in (0,1]")
        positive = (
            "worlds",
            "replay_capacity",
            "batch_size",
            "total_transitions",
            "max_hours",
            "collection_steps",
            "hidden",
            "model_hidden",
            "model_window",
            "imagination_horizon",
            "imagination_batch",
            "model_updates",
            "model_validation_interval",
            "action_bound",
            "learning_rate",
            "checkpoint_interval",
            "log_interval",
            "wrench_basis",
        )
        for key in positive:
            if not float(getattr(self, key)) > 0:
                raise ValueError(f"{key} must be positive")
        if (
            self.replay_capacity // self.worlds
            < 2 * max(self.model_window, self.imagination_horizon) + 2
        ):
            raise ValueError("replay must hold at least two model windows per world")
        if not 0 < self.gamma < 1 or not 0 < self.tau <= 1:
            raise ValueError("invalid discount or target-network update rate")
        if not 0 <= self.first_frame_probability <= 1:
            raise ValueError("invalid reset probability")
        for key in (
            "reference_weight",
            "wrench_weight",
            "orientation_weight",
            "lower_body_residual",
            "imagination_weight",
            "model_trust_weight",
            "exploration_std",
        ):
            if getattr(self, key) < 0:
                raise ValueError(f"{key} must be nonnegative")
        if self.updates_per_transition <= 0 or self.initial_alpha <= 0:
            raise ValueError("update ratio and entropy coefficient must be positive")
        if not 0 <= self.lower_body_residual <= 1:
            raise ValueError("lower_body_residual must be in [0,1]")
        if (
            self.model_min_updates < 1
            or self.warmup_transitions < 0
            or self.model_validation_threshold <= 0
            or self.friction < 0
        ):
            raise ValueError("invalid model, warmup or friction setting")
        if (
            self.warmup_transitions > self.replay_capacity
            or self.batch_size > self.replay_capacity
        ):
            raise ValueError("warmup and minibatch must fit replay capacity")
        return self

    def to_dict(self):
        return asdict(self)

    @classmethod
    def load(cls, path=None, **overrides):
        values = json.loads(Path(path).read_text()) if path else {}
        values.update({k: v for k, v in overrides.items() if v is not None})
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown training options: {sorted(unknown)}")
        return cls(**values).validate()

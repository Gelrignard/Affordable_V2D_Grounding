from dataclasses import dataclass, field, replace
from pathlib import Path
import numpy as np
import warnings
from .completion import fill_missing


@dataclass
class HumanMotion:
    """Metres, seconds, Z-up world; joint rotations are WORLD wxyz quaternions.

    The current contact solver supports one rigid object. Missing observations
    are completed on their own timeline and recorded; finite samples are retained.
    """

    joint_names: tuple[str, ...]
    positions: np.ndarray
    rotations_wxyz: np.ndarray
    timestamps: np.ndarray
    object_positions: np.ndarray
    object_wxyz: np.ndarray
    object_mesh: Path
    sequence_id: str
    metadata: dict = field(default_factory=dict)
    upstream_motion: object | None = field(default=None, repr=False)

    def __post_init__(self):
        n, j = len(self.timestamps), len(self.joint_names)
        shapes = {
            "positions": (n, j, 3),
            "rotations_wxyz": (n, j, 4),
            "timestamps": (n,),
            "object_positions": (n, 3),
            "object_wxyz": (n, 4),
        }
        if n < 1 or len(set(self.joint_names)) != j:
            raise ValueError("Need frames and unique joint names")
        for name, shape in shapes.items():
            a = np.asarray(getattr(self, name), dtype=np.float64)
            if a.shape != shape:
                raise ValueError(
                    f"Invalid {name}; expected finite {shape}, got {a.shape}"
                )
            setattr(self, name, a)
        if not np.isfinite(self.timestamps).all() or (n > 1 and np.any(np.diff(self.timestamps) <= 0)):
            raise ValueError("Timestamps must increase strictly")
        repairs = {}
        for name in ("positions", "rotations_wxyz", "object_positions", "object_wxyz"):
            filled, observed = fill_missing(self.timestamps, getattr(self, name),
                                            quaternion=name.endswith("wxyz"))
            missing = np.flatnonzero(~observed.reshape(n, -1).all(axis=1))
            if len(missing):
                setattr(self, name, filled)
                repairs[name] = missing.tolist()
        if repairs:
            self.metadata = {**self.metadata, "missing_source_samples": repairs}
            object_missing = sorted(set(repairs.get("object_positions", [])) | set(repairs.get("object_wxyz", [])))
            if object_missing:
                observed = np.asarray(self.metadata.get("object_observed_mask", np.ones(n, bool)), dtype=bool).copy()
                observed[object_missing] = False
                self.metadata["object_observed_mask"] = observed.tolist()
            warnings.warn(f"Filled missing source samples in {self.sequence_id}: {repairs}", RuntimeWarning)
        for name in ("rotations_wxyz", "object_wxyz"):
            a = getattr(self, name)
            norms = np.linalg.norm(a, axis=-1, keepdims=True)
            if np.max(np.abs(norms - 1)) > 1e-3:
                raise ValueError(f"{name} must contain unit wxyz quaternions")
            setattr(self, name, a / norms)
        self.object_mesh = Path(self.object_mesh).resolve()
        if not self.object_mesh.is_file():
            raise FileNotFoundError(self.object_mesh)

    def index(self, name):
        return self.joint_names.index(name)

    def subset(self, start=0, stop=None, stride=1):
        if start < 0 or stride < 1 or (stop is not None and stop <= start):
            raise ValueError("Invalid frame selection")
        selection = slice(start, stop, stride)
        fields = {
            k: getattr(self, k)[selection].copy()
            for k in (
                "positions",
                "rotations_wxyz",
                "timestamps",
                "object_positions",
                "object_wxyz",
            )
        }
        indices = np.asarray(
            self.metadata.get(
                "selected_source_indices", np.arange(len(self.timestamps))
            )
        )
        metadata = dict(
            self.metadata, selected_source_indices=indices[selection].tolist()
        )
        return replace(self, **fields, metadata=metadata)

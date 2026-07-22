"""The canonical motion representation every input adapter produces.

:class:`MotionTrack` is the contract between data ingestion and the
radiometric engine (paper Sec. 3.1): a sequence of timestamped 3-D cell
positions ``p_c(t)`` and outward normals ``n_c(t)``.  The engine consumes
nothing else, so supporting a new motion-capture system means writing an
adapter that emits a ``MotionTrack`` -- no change to the physics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from .kernel import EPS_NORM, normalize_rows

#: Multipliers converting a declared length unit into millimetres.
LENGTH_UNITS_TO_MM = {"mm": 1.0, "cm": 10.0, "m": 1000.0, "in": 25.4}


def to_mm(value: np.ndarray, units: str) -> np.ndarray:
    """Scale a length from ``units`` into millimetres."""
    key = str(units).lower()
    if key not in LENGTH_UNITS_TO_MM:
        raise ValueError(
            f"Unknown length unit {units!r}; expected one of "
            f"{sorted(LENGTH_UNITS_TO_MM)}"
        )
    return np.asarray(value, dtype=np.float64) * LENGTH_UNITS_TO_MM[key]


@dataclass
class MotionTrack:
    """A timestamped cell trajectory in the scene's coordinate frame.

    Attributes:
        time: ``(T,)`` timestamps in seconds.
        position: ``(T, 3)`` cell positions in millimetres.
        normal: ``(T, 3)`` outward cell normals, unit length.
        valid: ``(T,)`` boolean mask; ``False`` marks tracker dropout.
        name: identifier for the recording, used in reports and outputs.
        sample_rate_hz: nominal rate, when the source declares one.
        meta: free-form provenance carried through the pipeline.
    """

    time: np.ndarray
    position: np.ndarray
    normal: np.ndarray
    valid: np.ndarray
    name: str = ""
    sample_rate_hz: Optional[float] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.time = np.asarray(self.time, dtype=np.float64).reshape(-1)
        self.position = np.asarray(self.position, dtype=np.float64)
        self.normal = np.asarray(self.normal, dtype=np.float64)
        self.valid = np.asarray(self.valid, dtype=bool).reshape(-1)

        T = self.time.shape[0]
        if self.position.shape != (T, 3):
            raise ValueError(
                f"position must have shape ({T}, 3), got {self.position.shape}"
            )
        if self.normal.shape != (T, 3):
            raise ValueError(
                f"normal must have shape ({T}, 3), got {self.normal.shape}"
            )
        if self.valid.shape != (T,):
            raise ValueError(f"valid must have shape ({T},), got {self.valid.shape}")

        # Degenerate normals cannot define a cell orientation, so they are
        # dropout by definition rather than a separate failure mode.
        self.valid &= np.linalg.norm(self.normal, axis=1) >= EPS_NORM
        self.valid &= np.isfinite(self.position).all(axis=1)
        self.valid &= np.isfinite(self.normal).all(axis=1)
        self.normal = normalize_rows(self.normal)

    def __len__(self) -> int:
        return int(self.time.shape[0])

    @property
    def n_valid(self) -> int:
        """Number of frames usable by the simulator."""
        return int(self.valid.sum())

    @property
    def duration_s(self) -> float:
        """Wall-clock span of the recording in seconds."""
        if len(self) < 2:
            return 0.0
        return float(self.time[-1] - self.time[0])

    def compress(self) -> "MotionTrack":
        """Drop invalid frames, returning a track where every frame is usable."""
        keep = self.valid
        return MotionTrack(
            time=self.time[keep],
            position=self.position[keep],
            normal=self.normal[keep],
            valid=np.ones(int(keep.sum()), dtype=bool),
            name=self.name,
            sample_rate_hz=self.sample_rate_hz,
            meta=dict(self.meta),
        )

    def head(self, seconds: float) -> "MotionTrack":
        """First ``seconds`` of the recording.

        Calibration uses a short paired excerpt (paper Sec. 3.3: ~60 s of free
        hand motion), so this trims a reference window without loading a
        separate file.
        """
        if len(self) == 0 or seconds is None:
            return self
        keep = self.time <= self.time[0] + float(seconds)
        return MotionTrack(
            time=self.time[keep],
            position=self.position[keep],
            normal=self.normal[keep],
            valid=self.valid[keep],
            name=self.name,
            sample_rate_hz=self.sample_rate_hz,
            meta=dict(self.meta),
        )

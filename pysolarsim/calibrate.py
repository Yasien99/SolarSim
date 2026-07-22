"""Fitting the per-emitter calibration scalars kappa (paper Sec. 3.3).

Each kappa absorbs everything that is not geometry -- cell efficiency and
area, source intensity, spectral response -- into a single scalar per
emitter.  That factorisation is what lets a scene transfer: changing the room
means editing geometry, and changing the harvester means refitting kappa,
never retraining anything.

Given ``N`` synchronised frames, a measured power vector ``P`` and a design
matrix ``F`` whose column ``i`` holds emitter ``i``'s view factors, we solve

.. math::  \\hat{\\kappa} = \\arg\\min_{\\kappa \\succeq 0} \\lVert F\\kappa - P \\rVert_2^2

Non-negativity keeps each source's contribution physical and stops noise
being absorbed into offsetting positive and negative weights.  With a single
emitter this reduces to scalar least squares.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from scipy.optimize import nnls


@dataclass
class CalibrationResult:
    """Fitted scalars and the diagnostics needed to judge them."""

    kappas: np.ndarray
    method: str
    emitter_ids: List[str]
    n_frames: int
    residual: float

    def as_dict(self) -> dict:
        """JSON-serialisable summary for saving beside simulated data."""
        return {
            "method": self.method,
            "n_frames": int(self.n_frames),
            "residual_uW": float(self.residual),
            "kappas": {
                eid: float(k) for eid, k in zip(self.emitter_ids, self.kappas)
            },
        }


def fit_kappa(
    view_factors: np.ndarray,
    measured_power_uW: np.ndarray,
    method: str = "auto",
    emitter_ids: Optional[List[str]] = None,
) -> CalibrationResult:
    """Fit one kappa per emitter against a paired motion-PV recording.

    Args:
        view_factors: ``(N, M)`` design matrix, one column per active emitter.
        measured_power_uW: ``(N,)`` measured power in microwatts.
        method: ``auto`` (scalar least squares when ``M == 1``, else NNLS),
            ``scalar_ls``, or ``nnls``.
        emitter_ids: names for reporting; defaults to positional labels.

    Returns:
        The fitted :class:`CalibrationResult`.

    Raises:
        ValueError: if the inputs disagree in length, no frame is usable, or
            the requested method is unknown.
    """
    F = np.asarray(view_factors, dtype=np.float64)
    if F.ndim == 1:
        F = F.reshape(-1, 1)
    P = np.asarray(measured_power_uW, dtype=np.float64).reshape(-1)

    if F.shape[0] != P.shape[0]:
        raise ValueError(
            f"view factors have {F.shape[0]} frames but power has {P.shape[0]}"
        )

    keep = np.isfinite(P) & np.isfinite(F).all(axis=1)
    F, P = F[keep], P[keep]
    if F.shape[0] == 0:
        raise ValueError("no finite paired frames available for calibration")

    M = F.shape[1]
    if emitter_ids is None:
        emitter_ids = [f"emitter_{i}" for i in range(M)]

    resolved = method.lower()
    if resolved == "auto":
        resolved = "scalar_ls" if M == 1 else "nnls"

    if resolved == "scalar_ls":
        if M != 1:
            raise ValueError(
                f"scalar_ls needs exactly one active emitter, scene has {M}; "
                "use method 'nnls'"
            )
        denom = float(np.sum(F[:, 0] ** 2))
        kappas = np.array([0.0 if denom == 0.0 else float(F[:, 0] @ P) / denom])
    elif resolved == "nnls":
        kappas, _ = nnls(F, P)
    else:
        raise ValueError(
            f"unknown calibration method {method!r}; expected 'auto', "
            "'scalar_ls', or 'nnls'"
        )

    residual = float(np.sqrt(np.mean((F @ kappas - P) ** 2)))
    return CalibrationResult(
        kappas=np.asarray(kappas, dtype=np.float64),
        method=resolved,
        emitter_ids=list(emitter_ids),
        n_frames=int(F.shape[0]),
        residual=residual,
    )

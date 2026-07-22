"""Fidelity metrics for comparing simulated against measured PV traces.

These are the three metrics reported in the paper (Sec. 4.2), plus SSIM for
the XSolar comparison (Sec. 5.3):

- Pearson correlation captures temporal shape fidelity and is invariant to
  absolute level, which is what downstream models actually consume.
- RMSE reports absolute error in physical units (microwatts).
- nRMSE divides RMSE by the per-recording signal range, so configurations
  whose peak power differs by two orders of magnitude stay comparable.
"""

from __future__ import annotations

from typing import Dict

import numpy as np


def _aligned(actual: np.ndarray, predicted: np.ndarray) -> tuple:
    """Flatten both traces to finite, equal-length arrays."""
    a = np.asarray(actual, dtype=np.float64).reshape(-1)
    p = np.asarray(predicted, dtype=np.float64).reshape(-1)
    if a.shape != p.shape:
        raise ValueError(f"length mismatch: {a.shape[0]} vs {p.shape[0]}")
    keep = np.isfinite(a) & np.isfinite(p)
    return a[keep], p[keep]


def pearson(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Pearson correlation; ``nan`` when either trace is constant."""
    a, p = _aligned(actual, predicted)
    if a.size < 2 or np.std(a) == 0.0 or np.std(p) == 0.0:
        return float("nan")
    return float(np.corrcoef(a, p)[0, 1])


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Root-mean-square error in the units of the inputs."""
    a, p = _aligned(actual, predicted)
    if a.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((a - p) ** 2)))


def nrmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """RMSE normalised by the range of the measured trace."""
    a, p = _aligned(actual, predicted)
    if a.size == 0:
        return float("nan")
    span = float(np.max(a) - np.min(a))
    if span <= 0.0:
        return float("nan")
    return rmse(a, p) / span


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean absolute error in the units of the inputs."""
    a, p = _aligned(actual, predicted)
    if a.size == 0:
        return float("nan")
    return float(np.mean(np.abs(a - p)))


def r2(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Coefficient of determination against the mean of the measured trace."""
    a, p = _aligned(actual, predicted)
    if a.size < 2:
        return float("nan")
    ss_res = float(np.sum((a - p) ** 2))
    ss_tot = float(np.sum((a - np.mean(a)) ** 2))
    if ss_tot == 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def ssim(actual: np.ndarray, predicted: np.ndarray, data_range: float = None) -> float:
    """Global structural similarity between two 1-D traces.

    Matches the XSolar comparison protocol (paper Sec. 5.3).  ``data_range``
    defaults to the range of the measured trace.
    """
    a, p = _aligned(actual, predicted)
    if a.size < 2:
        return float("nan")
    if data_range is None:
        data_range = float(np.max(a) - np.min(a))
    if data_range <= 0.0:
        return float("nan")

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    mu_a, mu_p = float(np.mean(a)), float(np.mean(p))
    va, vp = float(np.var(a)), float(np.var(p))
    cov = float(np.mean((a - mu_a) * (p - mu_p)))

    num = (2 * mu_a * mu_p + c1) * (2 * cov + c2)
    den = (mu_a ** 2 + mu_p ** 2 + c1) * (va + vp + c2)
    if den == 0.0:
        return float("nan")
    return float(num / den)


def all_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    """Every fidelity metric in one dict, keyed as reported in the paper."""
    return {
        "pearson": pearson(actual, predicted),
        "rmse": rmse(actual, predicted),
        "nrmse": nrmse(actual, predicted),
        "mae": mae(actual, predicted),
        "r2": r2(actual, predicted),
        "ssim": ssim(actual, predicted),
    }

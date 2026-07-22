"""Composing per-emitter view factors into a predicted power trace.

This implements paper Eq. 1,

.. math::  P(t) = \\sum_{i=1}^{M} \\kappa_i \\, F_i(p_c(t), n_c(t); p_{s_i}, n_{s_i}, \\Gamma_i)

as two separable steps: build the purely geometric design matrix ``F``, then
scale and sum it with the calibration scalars.  Keeping them separate is what
lets the same view factors be reused to fit kappa and to synthesise power,
without recomputing the expensive quadrature.
"""

from __future__ import annotations

import numpy as np

from .kernel import DEFAULT_CHUNK
from .motion import MotionTrack
from .scene import Scene, default_integration


def view_factor_matrix(
    scene: Scene,
    track: MotionTrack,
    chunk: int = DEFAULT_CHUNK,
) -> np.ndarray:
    """Geometric design matrix for every active emitter.

    Args:
        scene: the illumination environment.
        track: the motion trajectory; only valid frames are evaluated, and
            invalid frames are returned as zero.
        chunk: frames evaluated per block inside the quadrature.

    Returns:
        ``(T, M)`` array where column ``i`` is emitter ``i``'s view factor and
        ``T == len(track)``.

    Raises:
        ValueError: if the scene has no enabled emitter.
    """
    active = scene.active
    if not active:
        raise ValueError(f"scene {scene.name!r} has no enabled emitters")

    integration = {**default_integration(), **scene.integration}

    T = len(track)
    F = np.zeros((T, len(active)), dtype=np.float64)
    idx = np.flatnonzero(track.valid)
    if idx.size == 0:
        return F

    positions = track.position[idx]
    normals = track.normal[idx]
    for i, emitter in enumerate(active):
        F[idx, i] = emitter.view_factor(positions, normals, integration, chunk)
    return F


def power_from_view_factors(
    view_factors: np.ndarray,
    kappas: np.ndarray,
) -> np.ndarray:
    """Superpose per-emitter contributions into total power.

    Args:
        view_factors: ``(T, M)`` design matrix.
        kappas: ``(M,)`` calibration scalars.

    Returns:
        ``(T,)`` power trace in the units of ``kappas`` (microwatts, when
        kappa was fitted against microwatt measurements).
    """
    F = np.asarray(view_factors, dtype=np.float64)
    k = np.asarray(kappas, dtype=np.float64).reshape(-1)
    if F.shape[1] != k.shape[0]:
        raise ValueError(
            f"design matrix has {F.shape[1]} emitters but {k.shape[0]} kappas given"
        )
    return F @ k

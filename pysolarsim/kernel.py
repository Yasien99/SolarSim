"""Radiometric view-factor kernels.

Every emitter geometry shares the same kernel (paper Eq. 2)

.. math::

    K(\\mathbf{R}, \\mathbf{n}_s, \\mathbf{n}_c) =
        \\frac{(\\mathbf{R}\\cdot\\mathbf{n}_s)(-\\mathbf{R}\\cdot\\mathbf{n}_c)}
             {\\lVert\\mathbf{R}\\rVert^4},
    \\qquad \\mathbf{R} = \\mathbf{p}_c - \\mathbf{x}'

and differs only in how the emitter surface :math:`A_i` is parametrised.
Changing geometry therefore changes the quadrature points, never the kernel.

Both integrals are evaluated with the composite 2-D Simpson rule (paper
Sec. 3.2): radial x angular for the disc (Eq. 3), u x v for the rectangle
(Eq. 4).  Spatial units are millimetres throughout; the coordinate frame is
Y-up, and the default emitter normal is ``[0, -1, 0]`` (pointing down).

All functions are vectorised over the whole trajectory and evaluate in
chunks, so memory stays bounded regardless of recording length.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

#: Emitter normal used when a scene does not specify one: straight down.
DEFAULT_EMITTER_NORMAL = np.array([0.0, -1.0, 0.0])

#: Squared distances below this are treated as coincident (kernel -> 0).
EPS_R2 = 1e-12

#: Normals shorter than this are treated as undefined (view factor -> 0).
EPS_NORM = 1e-6

#: Trajectory frames evaluated per chunk; caps peak memory at ~chunk x P x 3.
DEFAULT_CHUNK = 4096


def normalize_rows(v: np.ndarray) -> np.ndarray:
    """Return ``v`` with each row scaled to unit length.

    Rows shorter than :data:`EPS_NORM` are left unscaled; callers gate them
    out separately rather than relying on a divide-by-zero guard.
    """
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n < EPS_NORM, 1.0, n)


def plane_tangents(n: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Two orthonormal vectors spanning the plane whose normal is ``n``.

    The reference axis is switched when ``n`` is nearly parallel to X so the
    cross product never degenerates.
    """
    n = normalize_rows(np.asarray(n, dtype=np.float64))
    ref = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    t1 = np.cross(n, ref)
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(n, t1)
    return t1, t2


def simpson_weights(n: int) -> Tuple[np.ndarray, int]:
    """1-D composite Simpson weights over ``n`` intervals.

    ``n`` is rounded up to the next even number when odd, since Simpson's rule
    requires an even interval count.  Returns the weights and the ``n``
    actually used.
    """
    if n < 2:
        n = 2
    if n % 2 != 0:
        n += 1
    w = np.ones(n + 1, dtype=np.float64)
    w[1:-1:2] = 4.0
    w[2:-1:2] = 2.0
    return w, n


def resolve_emitter_normal(normal: Optional[np.ndarray]) -> np.ndarray:
    """Unit emitter normal, falling back to straight-down when unspecified."""
    if normal is None:
        return DEFAULT_EMITTER_NORMAL.copy()
    n = np.asarray(normal, dtype=np.float64).reshape(3)
    if np.linalg.norm(n) < EPS_NORM:
        return DEFAULT_EMITTER_NORMAL.copy()
    return n / np.linalg.norm(n)


def integrate_surface(
    positions: np.ndarray,
    normals: np.ndarray,
    center: np.ndarray,
    patches: np.ndarray,
    weights: np.ndarray,
    emitter_normal: np.ndarray,
    chunk: int = DEFAULT_CHUNK,
) -> np.ndarray:
    """Integrate the radiometric kernel over a discretised emitter surface.

    This is the single shared engine behind every geometry: a shape supplies
    its quadrature points and weights, and the kernel is reused unchanged.

    Args:
        positions: ``(T, 3)`` cell positions in mm.
        normals: ``(T, 3)`` outward cell normals; need not be unit length.
        center: ``(3,)`` emitter centre in mm, used for the front-side test.
        patches: ``(P, 3)`` world-space quadrature points on the emitter.
        weights: ``(P,)`` quadrature weights, including any Jacobian.
        emitter_normal: ``(3,)`` unit outward emitter normal.
        chunk: frames evaluated per block.

    Returns:
        ``(T,)`` dimensionless view factors.
    """
    positions = np.asarray(positions, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64).reshape(3)

    T = positions.shape[0]
    out = np.zeros(T, dtype=np.float64)

    n_c = normalize_rows(normals)
    has_normal = np.linalg.norm(normals, axis=1) >= EPS_NORM

    # A planar emitter's tangents are orthogonal to its normal, so every patch
    # shares the sign of (p_c - center) . n_s.  Cells behind the emitter can be
    # skipped wholesale instead of per patch.
    in_front = (positions - center) @ emitter_normal > 0.0
    active = np.flatnonzero(has_normal & in_front)
    if active.size == 0:
        return out

    for start in range(0, active.size, chunk):
        idx = active[start:start + chunk]
        R = positions[idx, None, :] - patches[None, :, :]      # (n, P, 3)
        R2 = np.einsum("npk,npk->np", R, R)
        Rs = R @ emitter_normal                                 # (n, P)
        Rc = np.einsum("npk,nk->np", R, n_c[idx])

        visible = (Rs > 0.0) & (Rc < 0.0) & (R2 >= EPS_R2)
        integrand = np.where(visible, Rs * (-Rc) / np.square(R2), 0.0)
        out[idx] = integrand @ weights

    return out / np.pi


def disc_patches(
    center: np.ndarray,
    diameter: float,
    emitter_normal: np.ndarray,
    n_r: int = 20,
    n_phi: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """Polar quadrature points and weights for a circular emitter (Eq. 3).

    The radial Jacobian ``r'`` is folded into the weights, and the degenerate
    ``r' = 0`` ring is zero-weighted.
    """
    center = np.asarray(center, dtype=np.float64).reshape(3)
    radius = float(diameter) / 2.0
    t1, t2 = plane_tangents(emitter_normal)

    w_r, n_r = simpson_weights(n_r)
    w_phi, n_phi = simpson_weights(n_phi)
    r_vals = np.linspace(0.0, radius, n_r + 1)
    phi_vals = np.linspace(0.0, 2.0 * np.pi, n_phi + 1)

    dr = radius / n_r
    d_phi = (2.0 * np.pi) / n_phi

    r_grid, phi_grid = np.meshgrid(r_vals, phi_vals, indexing="ij")
    patches = (
        center
        + r_grid[..., None] * np.cos(phi_grid)[..., None] * t1
        + r_grid[..., None] * np.sin(phi_grid)[..., None] * t2
    )

    weights = np.outer(w_r, w_phi) * (dr / 3.0) * (d_phi / 3.0) * r_grid
    weights = np.where(r_grid >= EPS_NORM, weights, 0.0)

    return patches.reshape(-1, 3), weights.reshape(-1)


def rect_patches(
    center: np.ndarray,
    width: float,
    height: float,
    emitter_normal: np.ndarray,
    n_u: int = 20,
    n_v: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """Cartesian quadrature points and weights for a rectangular emitter (Eq. 4).

    ``width`` runs along the first tangent and ``height`` along the second.
    """
    center = np.asarray(center, dtype=np.float64).reshape(3)
    t1, t2 = plane_tangents(emitter_normal)

    w_u, n_u = simpson_weights(n_u)
    w_v, n_v = simpson_weights(n_v)
    half_w = float(width) / 2.0
    half_h = float(height) / 2.0
    u_vals = np.linspace(-half_w, half_w, n_u + 1)
    v_vals = np.linspace(-half_h, half_h, n_v + 1)

    h_u = float(width) / n_u
    h_v = float(height) / n_v

    u_grid, v_grid = np.meshgrid(u_vals, v_vals, indexing="ij")
    patches = center + u_grid[..., None] * t1 + v_grid[..., None] * t2
    weights = np.outer(w_u, w_v) * (h_u / 3.0) * (h_v / 3.0)

    return patches.reshape(-1, 3), weights.reshape(-1)


def disc_view_factor(
    positions: np.ndarray,
    normals: np.ndarray,
    center: np.ndarray,
    diameter: float,
    emitter_normal: Optional[np.ndarray] = None,
    n_r: int = 20,
    n_phi: int = 20,
    chunk: int = DEFAULT_CHUNK,
) -> np.ndarray:
    """View factor for a circular emitter, evaluated over a trajectory."""
    n_e = resolve_emitter_normal(emitter_normal)
    patches, weights = disc_patches(center, diameter, n_e, n_r, n_phi)
    return integrate_surface(positions, normals, center, patches, weights, n_e, chunk)


def rect_view_factor(
    positions: np.ndarray,
    normals: np.ndarray,
    center: np.ndarray,
    width: float,
    height: float,
    emitter_normal: Optional[np.ndarray] = None,
    n_u: int = 20,
    n_v: int = 20,
    chunk: int = DEFAULT_CHUNK,
) -> np.ndarray:
    """View factor for a rectangular emitter, evaluated over a trajectory."""
    n_e = resolve_emitter_normal(emitter_normal)
    patches, weights = rect_patches(center, width, height, n_e, n_u, n_v)
    return integrate_surface(positions, normals, center, patches, weights, n_e, chunk)


def sphere_view_factor(
    positions: np.ndarray,
    normals: np.ndarray,
    center: np.ndarray,
    radius: float,
    emitter_normal: Optional[np.ndarray] = None,
) -> np.ndarray:
    """View factor for a small spherical emitter.

    No surface integration is needed: the sphere is treated as a point source
    whose projected area is ``pi r^2``, so a single kernel evaluation at the
    centre suffices.
    """
    n_e = resolve_emitter_normal(emitter_normal)
    center = np.asarray(center, dtype=np.float64).reshape(3)
    area = np.pi * float(radius) ** 2
    patches = center.reshape(1, 3)
    weights = np.array([area], dtype=np.float64)
    return integrate_surface(positions, normals, center, patches, weights, n_e)

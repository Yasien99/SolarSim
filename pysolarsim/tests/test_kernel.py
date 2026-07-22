"""Physical-invariant tests for the radiometric kernels.

These check properties the formulation must satisfy regardless of
implementation: visibility gating, the inverse-square law, convergence of the
quadrature, and geometry-agnosticism of the shared kernel.
"""

import numpy as np
import pytest

from pysolarsim.kernel import (
    disc_view_factor,
    plane_tangents,
    rect_view_factor,
    resolve_emitter_normal,
    simpson_weights,
    sphere_view_factor,
)

CENTER = np.array([0.0, 500.0, 0.0])
UP = np.array([[0.0, 1.0, 0.0]])
ORIGIN = np.array([[0.0, 0.0, 0.0]])


def test_cell_facing_away_sees_nothing():
    """A cell whose normal points away from the emitter harvests nothing."""
    down = np.array([[0.0, -1.0, 0.0]])
    assert disc_view_factor(ORIGIN, down, CENTER, 200.0)[0] == 0.0
    assert rect_view_factor(ORIGIN, down, CENTER, 100.0, 100.0)[0] == 0.0
    assert sphere_view_factor(ORIGIN, down, CENTER, 50.0)[0] == 0.0


def test_cell_behind_emitter_sees_nothing():
    """A cell above a downward-facing emitter is outside its emitting hemisphere."""
    above = np.array([[0.0, 600.0, 0.0]])
    assert disc_view_factor(above, UP, CENTER, 200.0)[0] == 0.0
    assert rect_view_factor(above, UP, CENTER, 100.0, 100.0)[0] == 0.0


def test_zero_normal_is_treated_as_dropout():
    """An undefined cell normal yields no power rather than a divide-by-zero."""
    assert disc_view_factor(ORIGIN, np.zeros((1, 3)), CENTER, 200.0)[0] == 0.0


def test_view_factor_falls_with_distance():
    """Moving the cell away monotonically reduces the view factor."""
    positions = np.array([[0.0, y, 0.0] for y in (400.0, 300.0, 200.0, 100.0, 0.0)])
    normals = np.repeat(UP, len(positions), axis=0)
    F = disc_view_factor(positions, normals, CENTER, 200.0)
    assert np.all(np.diff(F) < 0)


def test_small_source_approaches_inverse_square():
    """For a source small against its distance, the kernel is inverse-square."""
    near = np.array([[0.0, 0.0, 0.0]])
    far = np.array([[0.0, -500.0, 0.0]])
    normals = UP
    f_near = disc_view_factor(near, normals, CENTER, 2.0)[0]
    f_far = disc_view_factor(far, normals, CENTER, 2.0)[0]
    # Distance doubles, so irradiance should fall by ~4x.
    assert f_far == pytest.approx(f_near / 4.0, rel=1e-3)


def test_quadrature_converges():
    """Refining the Simpson grid changes the answer only marginally."""
    coarse = disc_view_factor(ORIGIN, UP, CENTER, 200.0, n_r=20, n_phi=20)[0]
    fine = disc_view_factor(ORIGIN, UP, CENTER, 200.0, n_r=80, n_phi=80)[0]
    assert coarse == pytest.approx(fine, rel=1e-4)


def test_square_and_equal_area_disc_agree_closely():
    """The kernel is geometry-agnostic: equal-area shapes give similar answers.

    Exercises the claim in paper Sec. 3.2 that only the integration domain
    changes between Eq. 3 and Eq. 4.
    """
    side = 100.0
    equivalent_diameter = 2.0 * np.sqrt(side * side / np.pi)
    f_rect = rect_view_factor(ORIGIN, UP, CENTER, side, side, n_u=60, n_v=60)[0]
    f_disc = disc_view_factor(ORIGIN, UP, CENTER, equivalent_diameter, n_r=60, n_phi=60)[0]
    assert f_rect == pytest.approx(f_disc, rel=0.02)


def test_disc_is_rotationally_symmetric():
    """A disc directly overhead gives the same answer from any azimuth."""
    radius = 120.0
    angles = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    positions = np.stack(
        [radius * np.cos(angles), np.zeros_like(angles), radius * np.sin(angles)], axis=1
    )
    normals = np.repeat(UP, len(positions), axis=0)
    F = disc_view_factor(positions, normals, CENTER, 200.0)
    assert np.allclose(F, F[0], rtol=1e-9)


def test_rect_orientation_matters_but_area_dominates():
    """Rotating a rectangle in its own plane preserves area and view factor."""
    f_wide = rect_view_factor(ORIGIN, UP, CENTER, 200.0, 50.0, n_u=60, n_v=60)[0]
    f_tall = rect_view_factor(ORIGIN, UP, CENTER, 50.0, 200.0, n_u=60, n_v=60)[0]
    assert f_wide == pytest.approx(f_tall, rel=1e-6)


def test_chunking_does_not_change_results():
    """Chunked evaluation agrees with a single pass to floating-point precision.

    Not bit-exact: the matrix product dispatches to different BLAS kernels for
    different block shapes, so agreement is to rounding, not to the last bit.
    """
    rng = np.random.default_rng(0)
    positions = rng.uniform(-300.0, 300.0, size=(50, 3))
    normals = rng.normal(size=(50, 3))
    a = disc_view_factor(positions, normals, CENTER, 200.0, chunk=7)
    b = disc_view_factor(positions, normals, CENTER, 200.0, chunk=10_000)
    assert np.allclose(a, b, rtol=1e-12, atol=1e-18)


def test_simpson_weights_round_up_to_even():
    """Simpson's rule needs an even interval count."""
    w, n = simpson_weights(5)
    assert n == 6 and w.shape == (7,)
    assert w[0] == 1 and w[-1] == 1 and w[1] == 4 and w[2] == 2


def test_plane_tangents_are_orthonormal():
    """Tangents span the emitter plane for any normal, including near-X."""
    for n in ([0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.3, -0.9, 0.2]):
        n = resolve_emitter_normal(n)
        t1, t2 = plane_tangents(n)
        assert np.dot(t1, n) == pytest.approx(0.0, abs=1e-12)
        assert np.dot(t2, n) == pytest.approx(0.0, abs=1e-12)
        assert np.dot(t1, t2) == pytest.approx(0.0, abs=1e-12)
        assert np.linalg.norm(t1) == pytest.approx(1.0)
        assert np.linalg.norm(t2) == pytest.approx(1.0)


def test_default_emitter_normal_points_down():
    """An unspecified emitter is a ceiling fixture pointing at the floor."""
    assert np.allclose(resolve_emitter_normal(None), [0.0, -1.0, 0.0])
    assert np.allclose(resolve_emitter_normal([0.0, -2.0, 0.0]), [0.0, -1.0, 0.0])

"""Tests for the 1-D BB-PD surface-correction weights.

The reference values are the published results of Example 1 (Section 5.1 of
the paper): a unit bar under uniaxial tension with L = E = 1, u_ref(x) = x, the
left end clamped and a concentrated traction P = 1 applied at the right end,
both imposed at a single node.
"""
import numpy as np
import pytest
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import spsolve

from perifit import (
    build_families_1d,
    build_targets_bb_1d,
    compute_weights_1d,
    full_segment_weighted_volume,
    segment_moment,
    segment_moment_over_r,
    segment_moment_over_r3,
)


def bar(dx, m, L=1.0):
    n = int(round(L / dx))
    x = (np.arange(n) + 0.5) * dx
    return x, np.full(n, dx), m * dx


def solve_bar(x, vols, delta, w=None):
    """Static 1-D BB-PD solve with c = 2E/delta^2 and no volume correction."""
    n = len(x)
    w = np.ones(n) if w is None else w
    fam = build_families_1d(x, delta)
    c = 2.0 / delta ** 2
    A = lil_matrix((n, n))
    for i in range(n):
        j = np.asarray(fam[i])
        k = c * 0.5 * (w[i] + w[j]) * vols[j] / np.abs(x[j] - x[i])
        A[i, j] = k
        A[i, i] = -k.sum()
    rhs = np.zeros(n)
    rhs[-1] = -1.0 / vols[-1]
    A[0, :] = 0.0
    A[0, 0] = 1.0
    rhs[0] = 0.0
    return spsolve(csr_matrix(A), rhs)


def l2_error(u, u_ref):
    return float(np.sqrt(np.sum((u - u_ref) ** 2) / np.sum(u_ref ** 2)))


class TestMoments1D:
    def test_odd_exponents_vanish(self):
        for p in (1, 3, 5):
            assert segment_moment(0.3, p) == 0.0
            assert segment_moment_over_r(0.3, p) == 0.0
            assert segment_moment_over_r3(0.3, p) == 0.0

    def test_closed_form_against_quadrature(self):
        delta = 0.17
        xi = np.linspace(-delta, delta, 2_000_001)
        xi = xi[np.abs(xi) > 1e-13]
        for p in (2, 4):
            assert segment_moment(delta, p) == pytest.approx(
                np.trapezoid(xi ** p, xi), rel=1e-6)
            assert segment_moment_over_r(delta, p) == pytest.approx(
                np.trapezoid(xi ** p / np.abs(xi), xi), rel=1e-6)
        for p in (4, 6):
            assert segment_moment_over_r3(delta, p) == pytest.approx(
                np.trapezoid(xi ** p / np.abs(xi) ** 3, xi), rel=1e-6)

    def test_weighted_volume(self):
        delta = 0.25
        assert full_segment_weighted_volume(delta) == pytest.approx(
            segment_moment(delta, 2))

    def test_singular_moments_raise(self):
        with pytest.raises(ValueError):
            segment_moment_over_r(0.3, 0)
        with pytest.raises(ValueError):
            segment_moment_over_r3(0.3, 2)


class TestTargets1D:
    def test_values(self):
        delta = 0.13
        d_star, e_star = build_targets_bb_1d(delta)
        # int (xi/|xi|)(xi/delta) dxi = delta ; the quadratic test is odd -> 0
        assert d_star == pytest.approx([delta, 0.0], abs=1e-15)
        # 0.25 * int phi_k^2 / |xi| dxi  for phi = xi/delta, xi^2/delta^2
        assert e_star == pytest.approx([0.25, 0.125], rel=1e-14)

    def test_scaling_with_horizon(self):
        d1, e1 = build_targets_bb_1d(0.1)
        d2, e2 = build_targets_bb_1d(0.4)
        assert d2[0] / d1[0] == pytest.approx(4.0)
        assert e2 == pytest.approx(e1)  # energy targets are dimensionless


class TestWeights1D:
    @pytest.mark.parametrize("m", [4, 8, 16])
    def test_interior_weight_matches_quadrature_defect(self, m):
        """Midpoint quadrature over-stiffens the bulk by (m+1)/m, so the
        interior weights must sit near m/(m+1) -- not near 1."""
        x, vols, delta = bar(0.005, m)
        w = compute_weights_1d(x, vols, delta)
        interior = w[(x > 0.4) & (x < 0.6)].mean()
        assert interior == pytest.approx(m / (m + 1), rel=0.06)

    def test_boundary_layer(self):
        x, vols, delta = bar(0.005, 4)
        w = compute_weights_1d(x, vols, delta)
        interior = w[(x > 0.4) & (x < 0.6)].mean()
        assert w[0] > 1.5 * interior            # amplified at a truncated horizon
        assert w.min() < interior               # reduced just inside
        assert np.allclose(w, w[::-1], atol=1e-8)  # symmetric bar

    def test_accepts_column_vector(self):
        x, vols, delta = bar(0.01, 4)
        assert np.allclose(compute_weights_1d(x, vols, delta),
                           compute_weights_1d(x.reshape(-1, 1), vols, delta))

    def test_per_row_norm_keeps_interior_but_smooths_boundary(self):
        x, vols, delta = bar(0.005, 4)
        w_paper = compute_weights_1d(x, vols, delta, row_norm="none")
        w_norm = compute_weights_1d(x, vols, delta, row_norm="per-row")
        inner = (x > 0.4) & (x < 0.6)
        assert w_norm[inner].mean() == pytest.approx(w_paper[inner].mean(), rel=0.02)
        assert w_norm[0] < w_paper[0]

    def test_rejects_bad_arguments(self):
        x, vols, delta = bar(0.05, 4)
        with pytest.raises(ValueError):
            compute_weights_1d(x, vols, delta, row_norm="bogus")
        with pytest.raises(ValueError):
            compute_weights_1d(x, vols, -1.0)
        with pytest.raises(ValueError):
            compute_weights_1d(x, vols[:-1], delta)


class TestExample1:
    """End-to-end reproduction of Section 5.1 of the paper."""

    def test_uncorrected_pd_is_too_stiff(self):
        x, vols, delta = bar(0.005, 4)
        u = solve_bar(x, vols, delta)
        # standard PD recovers a slope of m/(m+1) = 0.8 instead of 1
        assert u[-1] == pytest.approx(0.82, abs=0.02)
        assert l2_error(u, x) == pytest.approx(0.183, abs=0.02)

    def test_modified_pd_recovers_the_exact_solution(self):
        x, vols, delta = bar(0.005, 4)
        w = compute_weights_1d(x, vols, delta)
        u = solve_bar(x, vols, delta, w)
        assert u[-1] == pytest.approx(1.04, abs=0.03)      # paper, Fig. 7 (top)
        assert l2_error(u, x) == pytest.approx(0.047, abs=0.01)  # paper, Fig. 6

    @pytest.mark.parametrize("m", [4, 8, 16])
    def test_modified_pd_beats_standard_pd(self, m):
        x, vols, delta = bar(0.005, m)
        w = compute_weights_1d(x, vols, delta)
        assert l2_error(solve_bar(x, vols, delta, w), x) < \
            l2_error(solve_bar(x, vols, delta), x)

    def test_delta_convergence_is_monotonic(self):
        errs = []
        for dx in (0.01, 0.005, 0.0025):
            x, vols, delta = bar(dx, 8)
            w = compute_weights_1d(x, vols, delta)
            errs.append(l2_error(solve_bar(x, vols, delta, w), x))
        assert errs[0] > errs[1] > errs[2]

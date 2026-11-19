"""Nodal weights in 1-D and 2-D (bond-based PD).

Checks the calibration itself (targets, bulk weight, unit invariance) and its
effect on the two benchmark problems of the IJNME paper: the 1-D bar under
tension (Example 1) and the 2-D cantilever with parabolic end shear
(Example 2).  The convergence checks use a minimal static PD solver
(tests/_pdsolver.py) shared with the bond-weight tests.
"""
import numpy as np
import pytest

from perifit.core.families import families_from_bonds
from perifit.core.moments_1d import (
    full_segment_weighted_volume,
    segment_moment,
    segment_moment_over_r,
    segment_moment_over_r3,
)
from perifit.core.targets_1d import build_targets_bb_1d
from perifit.core.targets_2d import build_targets_bb_2d
from perifit.core.weights_1d import compute_weights_1d
from perifit.core.weights_2d import compute_weights_2d

from ._pdsolver import bar_problem, cantilever_problem, nodal_factor


# ------------------------------------------------------------------ moments
class TestMoments1D:
    def test_odd_exponents_vanish(self):
        for p in (1, 3, 5):
            assert segment_moment(0.3, p) == 0.0
            assert segment_moment_over_r(0.3, p) == 0.0
            assert segment_moment_over_r3(0.3, p) == 0.0

    def test_closed_form_against_quadrature(self):
        delta = 0.17
        xi = np.linspace(-delta, delta, 400_001)
        xi = xi[np.abs(xi) > 1e-13]
        trap = getattr(np, "trapezoid", None) or np.trapz
        for p in (2, 4):
            assert segment_moment(delta, p) == pytest.approx(trap(xi ** p, xi), rel=1e-6)
            assert segment_moment_over_r(delta, p) == pytest.approx(
                trap(xi ** p / np.abs(xi), xi), rel=1e-6)
        assert segment_moment_over_r3(delta, 4) == pytest.approx(
            trap(xi ** 4 / np.abs(xi) ** 3, xi), rel=1e-6)

    def test_weighted_volume(self):
        assert full_segment_weighted_volume(0.25) == pytest.approx(segment_moment(0.25, 2))

    def test_singular_moments_raise(self):
        with pytest.raises(ValueError):
            segment_moment_over_r(0.3, 0)
        with pytest.raises(ValueError):
            segment_moment_over_r3(0.3, 2)


# ------------------------------------------------------------------ targets
class TestTargets:
    def test_1d_values(self):
        d_star, e_star = build_targets_bb_1d(0.13)
        assert d_star == pytest.approx([0.13, 0.0], abs=1e-15)   # quadratic test is odd
        assert e_star == pytest.approx([0.25])                    # dimensionless

    def test_2d_shapes_and_values(self):
        d_star, e_star = build_targets_bb_2d(0.2)
        assert d_star.shape == (6,) and e_star.shape == (4,)
        assert np.all(d_star == 0.0)        # third moments vanish on a full disc
        # 1/4 int xi_x^4 / r^3 dA / delta^2 = 1/4 * (3 pi / 4) * delta^3 / 3 / delta^2
        assert e_star[0] == pytest.approx(0.25 * 0.75 * np.pi * 0.2 / 3)
        assert e_star[0] == pytest.approx(3 * e_star[1])          # x^4 vs x^2 y^2


# ------------------------------------------------------------- calibration
class TestCalibration1D:
    @pytest.mark.parametrize("m", [3, 4, 5, 8])
    def test_bulk_weight_is_exact(self, m):
        """Interior weight m/(m+1) exactly removes the midpoint-quadrature error."""
        x, V, delta, _, _ = bar_problem(0.0025, m)      # centre >= 25 horizons from the ends
        w = compute_weights_1d(x, V, delta)
        assert w[len(w) // 2] == pytest.approx(m / (m + 1), abs=1e-6)

    def test_boundary_layer_is_amplified(self):
        x, V, delta, _, _ = bar_problem(0.01)
        w = compute_weights_1d(x, V, delta)
        assert w[0] > 1.2 and w[-1] > 1.2
        assert np.allclose(w, w[::-1], atol=1e-8)                 # symmetric bar

    def test_unit_invariance(self):
        x, V, delta, _, _ = bar_problem(0.01)
        w = compute_weights_1d(x, V, delta)
        for s in (1e-3, 1e3):
            assert np.abs(compute_weights_1d(x * s, V * s, delta * s) - w).max() < 1e-10

    def test_removed_option_raises(self):
        x, V, delta, _, _ = bar_problem(0.05)
        with pytest.raises(TypeError, match="removed"):
            compute_weights_1d(x, V, delta, row_norm="none")


class TestCalibration2D:
    def test_symmetric_square(self):
        nodes, V, delta, bonds, _ = cantilever_problem(0.05)
        w = compute_weights_2d(nodes, V, delta, families=families_from_bonds(len(nodes), bonds))
        W = w.reshape(21, 21)
        assert np.allclose(W, W.T, atol=1e-8) and np.allclose(W, W[::-1], atol=1e-8)
        assert W[0, 0] == pytest.approx(W.max())                  # corners need most correction
        assert 1.0 < W[10, 10] < 1.1                              # bulk close to 1

    def test_unit_invariance(self):
        nodes, V, delta, _, _ = cantilever_problem(0.05)
        w = compute_weights_2d(nodes, V, delta)
        w2 = compute_weights_2d(nodes * 1e3, V * 1e6, delta * 1e3)
        assert np.abs(w - w2).max() < 1e-10


# ------------------------------------------------------------- convergence
class TestBar:
    """Example 1: standard PD does not converge; nodal weights converge at first order."""

    @pytest.fixture(scope="class")
    @classmethod
    def errors(cls):
        out = {"pd": [], "nodal": []}
        for dx in (0.02, 0.01, 0.005):
            x, V, delta, bonds, error = bar_problem(dx)
            out["pd"].append(error(np.ones(len(bonds))))
            out["nodal"].append(error(nodal_factor(compute_weights_1d(x, V, delta), bonds)))
        return {k: np.array(v) for k, v in out.items()}

    def test_standard_pd_does_not_converge(self, errors):
        assert errors["pd"][-1] > errors["pd"][0]

    def test_nodal_converges_first_order(self, errors):
        slopes = np.log2(errors["nodal"][:-1] / errors["nodal"][1:])
        assert np.all(slopes > 0.95) and np.all(slopes < 1.3)
        assert errors["nodal"][-1] < 0.004                        # 0.35 % at dx = 0.005

    def test_nodal_beats_standard_pd(self, errors):
        assert np.all(errors["nodal"] < 0.2 * errors["pd"])


class TestCantilever:
    """Example 2 (paper setup, m = 4)."""

    @pytest.fixture(scope="class")
    @classmethod
    def errors(cls):
        out = {"pd": [], "nodal": []}
        for dx in (0.04, 0.02):
            nodes, V, delta, bonds, error = cantilever_problem(dx)
            out["pd"].append(error(np.ones(len(bonds))))
            w = compute_weights_2d(nodes, V, delta, families=families_from_bonds(len(nodes), bonds))
            out["nodal"].append(error(nodal_factor(w, bonds)))
        return {k: np.array(v) for k, v in out.items()}

    def test_regression_values(self, errors):
        assert np.log10(errors["nodal"]) == pytest.approx([-0.945, -1.209], abs=0.01)
        assert np.log10(errors["pd"]) == pytest.approx([-0.631, -0.846], abs=0.01)

    def test_nodal_converges_and_beats_pd(self, errors):
        assert errors["nodal"][1] < errors["nodal"][0]
        assert np.all(errors["nodal"] < 0.5 * errors["pd"])

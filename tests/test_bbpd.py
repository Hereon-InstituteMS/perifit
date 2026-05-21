"""Tests for bond-based PD surface correction (moments, targets, local system, weights)."""

import math
import numpy as np
import pytest
from perifit.core.moments import ball_moment_over_r3, ball_moment_over_r2
from perifit.core.targets import build_targets_bb
from perifit.core.local_system import build_local_system_bb
from perifit.core.weights import compute_weights, build_families

DELTA = 0.15


# ---------------------------------------------------------------------------
# ball_moment_over_r3 tests
# ---------------------------------------------------------------------------

class TestBallMomentOverR3:
    def test_known_value_200(self):
        # M_3(2,0,0) = delta^2 / 2 * I(2,0,0)
        # I(2,0,0) = 4*pi * 1!! * (-1)!! * (-1)!! / 3!! = 4*pi / 3
        # => M_3(2,0,0) = delta^2 / 2 * 4*pi/3 = 2*pi*delta^2/3
        expected = 2.0 * math.pi * DELTA ** 2 / 3.0
        assert abs(ball_moment_over_r3(DELTA, 2, 0, 0) - expected) < 1e-12

    def test_symmetry(self):
        assert abs(ball_moment_over_r3(DELTA, 2, 0, 0) - ball_moment_over_r3(DELTA, 0, 2, 0)) < 1e-15
        assert abs(ball_moment_over_r3(DELTA, 2, 0, 0) - ball_moment_over_r3(DELTA, 0, 0, 2)) < 1e-15

    def test_known_value_400(self):
        # M_3(4,0,0) = delta^4 / 4 * I(4,0,0)
        # Using Gamma-based angular:
        # angular = 2*Gamma(5/2)*Gamma(1/2)*Gamma(1/2)/Gamma(7/2)
        # = 2*(3*sqrt(pi)/4)*sqrt(pi)*sqrt(pi)/(15*sqrt(pi)/8) = 4*pi/5
        # result = (4*pi/5) * delta^4 / 4
        expected = DELTA ** 4 / 4.0 * 4.0 * math.pi / 5.0
        assert abs(ball_moment_over_r3(DELTA, 4, 0, 0) - expected) < 1e-12

    def test_odd_exponents_zero(self):
        for delta in (0.1, 0.3, 1.0):
            for (p, q, r) in [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 2, 0), (3, 1, 2)]:
                assert ball_moment_over_r3(delta, p, q, r) == 0.0

    def test_zero_total_exponent(self):
        # P=0 -> singular; the reference raises ValueError
        with pytest.raises(ValueError, match="singular"):
            ball_moment_over_r3(DELTA, 0, 0, 0)

    def test_moment_hierarchy(self):
        # M_2(p,q,r) / M_3(p,q,r) = delta * P / (P+1)  for even exponents P>0
        for (p, q, r) in [(2, 0, 0), (0, 2, 0), (0, 0, 2), (4, 0, 0), (2, 2, 0)]:
            P = p + q + r
            m2 = ball_moment_over_r2(DELTA, p, q, r)
            m3 = ball_moment_over_r3(DELTA, p, q, r)
            if m3 > 0:
                ratio = m2 / m3
                expected_ratio = DELTA * P / (P + 1)
                assert abs(ratio - expected_ratio) < 1e-12, \
                    f"Ratio test failed for ({p},{q},{r}): {ratio} != {expected_ratio}"


# ---------------------------------------------------------------------------
# build_targets_bb tests
# ---------------------------------------------------------------------------

class TestTargetsBBPD:
    def test_output_shapes(self):
        d, e = build_targets_bb(DELTA)
        assert d.shape == (27,)
        assert e.shape == (27,)

    def test_d_star_known_values(self):
        """Derivative targets: d_star[alpha*9 + (k-1)] = ball_moment_over_r3 of appropriate exponents."""
        d, _ = build_targets_bb(DELTA)
        # alpha=0, k=1 (dphi=xi_x): px=1+1=2, py=0, pz=0
        expected = ball_moment_over_r3(DELTA, 2, 0, 0)
        assert abs(d[0 * 9 + 0] - expected) < 1e-12

    def test_d_star_symmetry(self):
        """Linear test entries should be symmetric across directions."""
        d, _ = build_targets_bb(DELTA)
        # alpha=0,k=1 vs alpha=1,k=2 vs alpha=2,k=3
        assert abs(d[0 * 9 + 0] - d[1 * 9 + 1]) < 1e-15
        assert abs(d[0 * 9 + 0] - d[2 * 9 + 2]) < 1e-15

    def test_e_star_symmetry(self):
        """Energy targets should be symmetric under ell permutation
        for test functions with matching symmetry."""
        _, e = build_targets_bb(DELTA)
        # ell=0,k=1 vs ell=1,k=2 vs ell=2,k=3
        # These are 0.25*M3(4,0,0), 0.25*M3(0,4,0), 0.25*M3(0,0,4)
        assert abs(e[0 * 9 + 0] - e[1 * 9 + 1]) < 1e-15
        assert abs(e[0 * 9 + 0] - e[2 * 9 + 2]) < 1e-15


# ---------------------------------------------------------------------------
# build_local_system_bb tests
# ---------------------------------------------------------------------------

class TestLocalSystemBBPD:
    @pytest.fixture
    def interior_system(self):
        """A well-resolved interior node with ~100 neighbours."""
        dx = 0.1
        delta = 0.3
        xs = np.arange(dx / 2, 1.0, dx)
        gx, gy, gz = np.meshgrid(xs, xs, xs, indexing='ij')
        coords = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
        # Centre node
        centre = np.argmin(np.sum((coords - 0.5) ** 2, axis=1))
        from scipy.spatial import KDTree
        tree = KDTree(coords)
        nbrs = tree.query_ball_point(coords[centre], delta + 1e-10)
        nbrs = [j for j in nbrs if j != centre]
        xi = coords[nbrs] - coords[centre]
        vol = np.full(len(nbrs), dx ** 3)
        d_star, e_star = build_targets_bb(delta)
        return xi, vol, delta, d_star, e_star

    def test_too_few_neighbors(self):
        d, e = build_targets_bb(0.15)
        xi = np.random.randn(5, 3) * 0.1
        vol = np.ones(5) * 0.001
        assert build_local_system_bb(xi, vol, 0.15, d, e) is None

    def test_output_shapes(self, interior_system):
        xi, vol, delta, d_star, e_star = interior_system
        result = build_local_system_bb(xi, vol, delta, d_star, e_star)
        assert result is not None
        beta, coupling = result
        assert isinstance(beta, float)
        assert coupling.shape == (len(xi),)

    def test_beta_finite(self, interior_system):
        xi, vol, delta, d_star, e_star = interior_system
        beta, coupling = build_local_system_bb(xi, vol, delta, d_star, e_star)
        assert np.isfinite(beta)
        assert np.all(np.isfinite(coupling))


# ---------------------------------------------------------------------------
# End-to-end BB-PD weight tests on unit cube
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def cube_10():
    """10x10x10 cube, m=3, cached for the module."""
    dx = 0.1
    xs = np.arange(dx / 2, 1.0, dx)
    gx, gy, gz = np.meshgrid(xs, xs, xs, indexing='ij')
    coords = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
    volumes = np.full(len(coords), dx ** 3)
    delta = 3 * dx
    families = build_families(coords, delta)
    w = compute_weights(coords, volumes, delta, model='bb', families=families)
    return coords, volumes, delta, w


class TestComputeWeightsBBPD:
    def test_positive(self, cube_10):
        _, _, _, w = cube_10
        assert (w > 0).all()

    def test_all_finite(self, cube_10):
        _, _, _, w = cube_10
        assert np.isfinite(w).all()

    def test_interior_near_one(self, cube_10):
        coords, _, _, w = cube_10
        centre = np.argmin(np.sum((coords - 0.5) ** 2, axis=1))
        assert abs(w[centre] - 1.0) < 0.30  # BB-PD converges slower than SB-PD

    def test_length(self, cube_10):
        coords, _, _, w = cube_10
        assert len(w) == len(coords)

    def test_symmetric_on_symmetric_mesh(self, cube_10):
        """Weights should respect the mesh symmetry under axis reflections."""
        coords, _, _, w = cube_10
        # Reflect x -> 1-x
        reflected_x = coords.copy()
        reflected_x[:, 0] = 1.0 - reflected_x[:, 0]
        # Find matching nodes
        from scipy.spatial import KDTree
        tree = KDTree(coords)
        _, idx = tree.query(reflected_x)
        # BB-PD has a 1/r^3 singularity, so iterative solver residuals
        # break exact symmetry more than SB-PD.  Tolerance is generous but
        # still catches gross asymmetry bugs.
        np.testing.assert_allclose(w, w[idx], atol=0.08,
                                   err_msg="Weights not symmetric under x-reflection")

    def test_different_from_osb(self, cube_10):
        """BB-PD and OSB-PD should give different weights on the same mesh."""
        coords, volumes, delta, w_bb = cube_10
        families = build_families(coords, delta)
        w_osb = compute_weights(coords, volumes, delta,
                                model='osb', families=families)
        assert not np.allclose(w_bb, w_osb, atol=0.01)

    def test_invalid_model_raises(self):
        coords = np.random.randn(20, 3)
        volumes = np.full(20, 0.001)
        with pytest.raises(ValueError, match="model must be"):
            compute_weights(coords, volumes, 0.3, model='invalid_model')

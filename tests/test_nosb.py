"""Tests for NOSB-PD (correspondence model) surface correction."""

import math
import numpy as np
import pytest
from perifit.core.moments import ball_moment
from perifit.core.targets import build_targets_nosb
from perifit.core.local_system import build_local_system_nosb
from perifit.core.weights import compute_weights, build_families

DELTA = 0.15


# ---------------------------------------------------------------------------
# build_targets_nosb tests
# ---------------------------------------------------------------------------

class TestTargetsNOSB:
    def test_output_shape(self):
        n = build_targets_nosb(DELTA)
        assert n.shape == (27,)

    def test_n_star_symmetry(self):
        """Gradient targets for k=1 (dphi=xi_x) should be symmetric
        across the three b-directions for the matching indices."""
        n = build_targets_nosb(DELTA)
        # b=0, k=1: N_star[0*9+0] = ball_moment(delta, 2, 0, 0)  (ax=1, +1 for b==0)
        # b=1, k=2: N_star[1*9+1] = ball_moment(delta, 0, 2, 0)
        # b=2, k=3: N_star[2*9+2] = ball_moment(delta, 0, 0, 2)
        assert abs(n[0] - n[1 * 9 + 1]) < 1e-15
        assert abs(n[0] - n[2 * 9 + 2]) < 1e-15

    def test_n_star_odd_vanish(self):
        """Gradient targets with odd total exponent should be zero."""
        n = build_targets_nosb(DELTA)
        # For b=0, k=2 (dphi=xi_y): exponents = (0+1, 1+0, 0+0) = (1,1,0) — odd x
        assert n[0 * 9 + 1] == 0.0  # b=0, k=2: exps (1,1,0) -> odd x

    def test_n_star_hand_computed(self):
        """Spot-check N_star[b=0, k=4] where dphi_4 = xi_x^2.

        Exponents: ax=2, ay=0, az=0 + (b==0 -> +1) = (3, 0, 0).
        Total exponent sum P=3, which is odd -> ball_moment = 0.
        """
        n = build_targets_nosb(DELTA)
        assert n[0 * 9 + 3] == 0.0  # b=0, k=4: exps (3,0,0) odd

    def test_n_star_nonzero_entry(self):
        """Spot-check a nonzero N_star entry.

        b=0, k=1 (dphi_1 = xi_x): ax=1, ay=0, az=0.
        Add b==0 -> (2, 0, 0).  All even -> nonzero.
        """
        n = build_targets_nosb(DELTA)
        # b=0, k=1: exponents (2,0,0) -> even -> nonzero
        expected = ball_moment(DELTA, 2, 0, 0)
        assert abs(n[0 * 9 + 0] - expected) < 1e-14
        assert expected > 0  # ensure truly nonzero

    def test_scale_with_delta(self):
        """Targets should scale with delta to appropriate powers."""
        n_a = build_targets_nosb(0.1)
        n_b = build_targets_nosb(0.2)
        # N_star[0] = ball_moment(delta, 2, 0, 0) = delta^5/5 * angular
        # Ratio should be (0.2/0.1)^5 = 32
        ratio = n_b[0] / n_a[0]
        assert abs(ratio - 32.0) < 1e-10


# ---------------------------------------------------------------------------
# build_local_system_nosb tests
# ---------------------------------------------------------------------------

class TestLocalSystemNOSB:
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
        n_star = build_targets_nosb(delta)
        return xi, vol, delta, n_star

    def test_too_few_neighbors(self):
        n = build_targets_nosb(0.15)
        xi = np.random.randn(5, 3) * 0.1
        vol = np.ones(5) * 0.001
        assert build_local_system_nosb(xi, vol, 0.15, n) is None

    def test_output_shapes(self, interior_system):
        xi, vol, delta, n_star = interior_system
        result = build_local_system_nosb(xi, vol, delta, n_star)
        assert result is not None
        beta, coupling = result
        assert isinstance(beta, float)
        assert coupling.shape == (len(xi),)

    def test_beta_finite(self, interior_system):
        xi, vol, delta, n_star = interior_system
        beta, coupling = build_local_system_nosb(xi, vol, delta, n_star)
        assert np.isfinite(beta)
        assert np.all(np.isfinite(coupling))

    def test_exactly_10_neighbors(self):
        """The minimum case (n_F = NP = 10) should still work."""
        rng = np.random.default_rng(42)
        delta = 0.3
        n_F = 10
        # Generate 10 random bonds inside the ball
        phi = rng.uniform(0, 2 * math.pi, n_F)
        cos_t = rng.uniform(-0.99, 0.99, n_F)
        r = delta * rng.uniform(0.2, 0.98, n_F) ** (1 / 3)
        sin_t = np.sqrt(1 - cos_t ** 2)
        xi = np.column_stack([r * sin_t * np.cos(phi),
                               r * sin_t * np.sin(phi),
                               r * cos_t])
        vol = np.full(n_F, 0.001)
        n = build_targets_nosb(delta)
        result = build_local_system_nosb(xi, vol, delta, n)
        assert result is not None
        beta, coupling = result
        assert np.isfinite(beta)
        assert np.all(np.isfinite(coupling))

    def test_tikhonov_effect(self, interior_system):
        """Higher Tikhonov regularisation should change the result but keep it finite."""
        xi, vol, delta, n_star = interior_system
        beta1, c1 = build_local_system_nosb(xi, vol, delta, n_star,
                                             tikhonov_rel=1e-8)
        beta2, c2 = build_local_system_nosb(xi, vol, delta, n_star,
                                             tikhonov_rel=1e-2)
        assert np.isfinite(beta2)
        assert np.all(np.isfinite(c2))
        # Results should differ
        assert not np.allclose(c1, c2)

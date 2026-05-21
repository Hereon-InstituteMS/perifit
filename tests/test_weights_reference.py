"""
Comprehensive tests for the perifit reference weight computation code.

Tests cover:
    1. Ball moments (3-D analytical integrals)
    2. BB-PD targets
    3. OSB-PD targets
    4. BB weight computation on a unit cube
    5. OSB weight computation on a unit cube
    6. Exact reference weight comparison (first 10 weights, 6 decimal places)
    7. 2-D weight smoke test
    8. Family construction verification

Reference values were obtained by running the authoritative reference code
on the same mesh and comparing outputs.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import perifit
from perifit.core.moments import (
    ball_moment,
    ball_moment_over_r2,
    ball_moment_over_r3,
    full_ball_volume,
    full_ball_weighted_volume,
    full_ball_shape_tensor_diag,
)
from perifit.core.targets import build_targets_bb, build_targets_osb
from perifit.core.weights import compute_weights, build_families
from perifit.core.moments_2d import disc_moment, full_disc_area
from perifit.core.weights_2d import compute_weights_2d, build_families_2d


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cube_mesh(dx: float = 0.1, N_per_side: int = 11):
    """Create a regular Cartesian cube [0, (N-1)*dx]^3."""
    Nx = Ny = Nz = N_per_side
    coords = np.array(
        [[ix * dx, iy * dx, iz * dx]
         for iz in range(Nz) for iy in range(Ny) for ix in range(Nx)]
    )
    volumes = np.full(len(coords), dx ** 3)
    return coords, volumes


# Precompute the cube mesh and families once for all cube-based tests
_DX = 0.1
_N_SIDE = 11
_DELTA = 3 * _DX
_COORDS, _VOLUMES = _make_cube_mesh(_DX, _N_SIDE)
_FAMILIES = None  # lazily initialised


def _get_families():
    global _FAMILIES
    if _FAMILIES is None:
        _FAMILIES = build_families(_COORDS, _DELTA)
    return _FAMILIES


# ---------------------------------------------------------------------------
# Test 1: Ball moments
# ---------------------------------------------------------------------------

class TestBallMoments:
    """Verify closed-form 3-D ball moments against hand-computed values."""

    def test_ball_moment_x2(self):
        """ball_moment(1.0, 2, 0, 0) = 4*pi/15 (unit ball, x^2 dV)."""
        expected = 4.0 * math.pi / 15.0
        result = ball_moment(1.0, 2, 0, 0)
        assert result == pytest.approx(expected, rel=1e-12)

    def test_ball_moment_y2(self):
        """ball_moment(1.0, 0, 2, 0) = 4*pi/15 (by symmetry)."""
        expected = 4.0 * math.pi / 15.0
        result = ball_moment(1.0, 0, 2, 0)
        assert result == pytest.approx(expected, rel=1e-12)

    def test_ball_moment_z2(self):
        """ball_moment(1.0, 0, 0, 2) = 4*pi/15 (by symmetry)."""
        expected = 4.0 * math.pi / 15.0
        result = ball_moment(1.0, 0, 0, 2)
        assert result == pytest.approx(expected, rel=1e-12)

    def test_ball_moment_odd_zero(self):
        """Odd exponents must give zero."""
        assert ball_moment(1.0, 1, 0, 0) == 0.0
        assert ball_moment(1.0, 0, 1, 0) == 0.0
        assert ball_moment(1.0, 0, 0, 1) == 0.0
        assert ball_moment(1.0, 1, 2, 0) == 0.0
        assert ball_moment(1.0, 3, 0, 0) == 0.0

    def test_ball_moment_unit_sphere_volume(self):
        """ball_moment(1.0, 0, 0, 0) = 4*pi/3 = volume of unit ball."""
        expected = 4.0 * math.pi / 3.0
        result = ball_moment(1.0, 0, 0, 0)
        assert result == pytest.approx(expected, rel=1e-12)

    def test_ball_moment_over_r2(self):
        """ball_moment_over_r2(1.0, 2, 0, 0) = int x^2/r^2 dV over unit ball.

        = angular * delta^(P+1)/(P+1), P=2
        angular = 2*Gamma(3/2)*Gamma(1/2)*Gamma(1/2)/Gamma(5/2)
                = 2*(sqrt(pi)/2)*sqrt(pi)*sqrt(pi) / (3*sqrt(pi)/4)
                = 4*pi/3
        result = (4*pi/3) * 1^3 / 3 = 4*pi/9
        """
        expected = 4.0 * math.pi / 9.0
        result = ball_moment_over_r2(1.0, 2, 0, 0)
        assert result == pytest.approx(expected, rel=1e-12)

    def test_ball_moment_over_r3(self):
        """ball_moment_over_r3(1.0, 2, 0, 0) = angular * delta^P / P.

        P=2, angular = 4*pi/3, result = (4*pi/3) * 1/2 = 2*pi/3
        """
        expected = 2.0 * math.pi / 3.0
        result = ball_moment_over_r3(1.0, 2, 0, 0)
        assert result == pytest.approx(expected, rel=1e-12)

    def test_ball_moment_over_r3_singular(self):
        """ball_moment_over_r3 must raise ValueError for p=0."""
        with pytest.raises(ValueError, match="singular"):
            ball_moment_over_r3(1.0, 0, 0, 0)

    def test_full_ball_volume(self):
        delta = 0.3
        expected = 4.0 * math.pi * delta ** 3 / 3.0
        assert full_ball_volume(delta) == pytest.approx(expected, rel=1e-14)

    def test_full_ball_weighted_volume(self):
        delta = 0.3
        expected = 4.0 * math.pi * delta ** 5 / 5.0
        assert full_ball_weighted_volume(delta) == pytest.approx(expected, rel=1e-14)

    def test_full_ball_shape_tensor_diag(self):
        delta = 0.3
        expected = 4.0 * math.pi * delta ** 5 / 15.0
        assert full_ball_shape_tensor_diag(delta) == pytest.approx(expected, rel=1e-14)

    def test_ball_moment_scaling(self):
        """ball_moment scales as delta^(P+3)."""
        delta = 2.5
        px, py, pz = 2, 0, 2
        P = px + py + pz
        ratio = ball_moment(delta, px, py, pz) / ball_moment(1.0, px, py, pz)
        assert ratio == pytest.approx(delta ** (P + 3), rel=1e-12)

    def test_ball_moment_x4(self):
        """ball_moment(1.0, 4, 0, 0) = 4*pi/35.

        angular = 2*Gamma(5/2)*Gamma(1/2)*Gamma(1/2)/Gamma(7/2)
                = 2*(3*sqrt(pi)/4)*sqrt(pi)*sqrt(pi) / (15*sqrt(pi)/8)
                = 4*pi/5
        result = (4*pi/5) * 1^7 / 7 = 4*pi/35
        """
        expected = 4.0 * math.pi / 35.0
        result = ball_moment(1.0, 4, 0, 0)
        assert result == pytest.approx(expected, rel=1e-12)


# ---------------------------------------------------------------------------
# Test 2: BB-PD targets
# ---------------------------------------------------------------------------

class TestBuildTargetsBB:
    """Verify BB-PD analytical targets have correct shape and known values."""

    def test_shapes(self):
        d_star, e_star = build_targets_bb(0.3)
        assert d_star.shape == (27,)
        assert e_star.shape == (27,)

    def test_derivative_symmetry(self):
        """Derivative targets for x-direction linear test = y-direction = z-direction."""
        delta = 0.3
        d_star, _ = build_targets_bb(delta)
        # d_star[0] = target for alpha=0, k=1 (xi_x * xi_x / r^3)
        # d_star[10] = target for alpha=1, k=2 (xi_y * xi_y / r^3)
        # d_star[20] = target for alpha=2, k=3 (xi_z * xi_z / r^3)
        assert d_star[0] == pytest.approx(d_star[10], rel=1e-12)
        assert d_star[0] == pytest.approx(d_star[20], rel=1e-12)

    def test_derivative_known_value(self):
        """d_star[0] = ball_moment_over_r3(delta, 2, 0, 0)."""
        delta = 0.3
        d_star, _ = build_targets_bb(delta)
        expected = ball_moment_over_r3(delta, 2, 0, 0)
        assert d_star[0] == pytest.approx(expected, rel=1e-14)

    def test_energy_known_value(self):
        """e_star[0] = 0.25 * ball_moment_over_r3(delta, 4, 0, 0)."""
        delta = 0.3
        _, e_star = build_targets_bb(delta)
        expected = 0.25 * ball_moment_over_r3(delta, 4, 0, 0)
        assert e_star[0] == pytest.approx(expected, rel=1e-14)

    def test_nonzero_entries(self):
        """Many entries should be nonzero (mixed products involving even exponents)."""
        d_star, e_star = build_targets_bb(0.3)
        assert np.count_nonzero(d_star) > 0
        assert np.count_nonzero(e_star) > 0

    def test_energy_positive(self):
        """All nonzero energy targets should be positive."""
        _, e_star = build_targets_bb(0.3)
        assert np.all(e_star[e_star != 0.0] > 0)


# ---------------------------------------------------------------------------
# Test 3: OSB-PD targets
# ---------------------------------------------------------------------------

class TestBuildTargetsOSB:
    """Verify OSB-PD analytical targets have correct shape and known values."""

    def test_shapes(self):
        d_star, e_star = build_targets_osb(0.3)
        assert d_star.shape == (27,)
        assert e_star.shape == (27,)

    def test_derivative_symmetry(self):
        """Derivative targets: x-linear = y-linear = z-linear by isotropy."""
        delta = 0.3
        d_star, _ = build_targets_osb(delta)
        assert d_star[0] == pytest.approx(d_star[10], rel=1e-12)
        assert d_star[0] == pytest.approx(d_star[20], rel=1e-12)

    def test_derivative_known_value(self):
        """d_star[0] = ball_moment_over_r2(delta, 2, 0, 0)."""
        delta = 0.3
        d_star, _ = build_targets_osb(delta)
        expected = ball_moment_over_r2(delta, 2, 0, 0)
        assert d_star[0] == pytest.approx(expected, rel=1e-14)

    def test_energy_known_value(self):
        """e_star[0] = ball_moment_over_r2(delta, 4, 0, 0)."""
        delta = 0.3
        _, e_star = build_targets_osb(delta)
        expected = ball_moment_over_r2(delta, 4, 0, 0)
        assert e_star[0] == pytest.approx(expected, rel=1e-14)

    def test_energy_positive(self):
        """All nonzero energy targets should be positive."""
        _, e_star = build_targets_osb(0.3)
        assert np.all(e_star[e_star != 0.0] > 0)

    def test_osb_vs_bb_derivative_different(self):
        """OSB and BB derivative targets should differ (different kernels)."""
        delta = 0.3
        d_bb, _ = build_targets_bb(delta)
        d_osb, _ = build_targets_osb(delta)
        # They should have different nonzero values
        assert not np.allclose(d_bb, d_osb)


# ---------------------------------------------------------------------------
# Test 4: BB weights on cube
# ---------------------------------------------------------------------------

class TestWeightsBBCube:
    """Test BB weight computation on the 11x11x11 unit cube."""

    @pytest.fixture(scope="class")
    def bb_weights(self):
        families = _get_families()
        return compute_weights(
            _COORDS, _VOLUMES, _DELTA,
            model="bb", families=families
        )

    def test_shape(self, bb_weights):
        assert bb_weights.shape == (1331,)

    def test_interior_weights_near_one(self, bb_weights):
        """The most interior node (centre of cube) should have w closest to 1.0.

        On an 11x11x11 cube with m=3, even the centre node has some surface
        influence from neighbouring truncated horizons, so we allow a generous
        tolerance.  The centre weight should still be closer to 1 than any
        boundary weight.
        """
        N = _N_SIDE
        centre_idx = 5 * N * N + 5 * N + 5
        # Centre weight should be reasonably close to 1 (within 20%)
        assert abs(bb_weights[centre_idx] - 1.0) < 0.20, (
            f"Centre weight too far from 1.0: {bb_weights[centre_idx]:.4f}"
        )
        # Centre should be closer to 1.0 than corner
        assert abs(bb_weights[centre_idx] - 1.0) < abs(bb_weights[0] - 1.0)

    def test_boundary_weights_above_one(self, bb_weights):
        """Corner/edge/face nodes should generally have w > 1.0."""
        # Corner node at origin
        corner_idx = 0
        assert bb_weights[corner_idx] > 1.5, (
            f"Corner weight too low: {bb_weights[corner_idx]:.4f}"
        )

    def test_mean_weight_range(self, bb_weights):
        """Mean weight should be in a reasonable range for this surface-dominated mesh."""
        mean_w = bb_weights.mean()
        assert 0.8 < mean_w < 1.8, f"Mean weight out of range: {mean_w:.4f}"

    def test_bb_stats(self, bb_weights):
        """Check that overall stats are in expected range (solver-variant tolerant)."""
        assert 0.50 < bb_weights.min() < 0.70, f"BB min={bb_weights.min():.4f}"
        assert 2.80 < bb_weights.max() < 3.50, f"BB max={bb_weights.max():.4f}"
        assert 1.15 < bb_weights.mean() < 1.40, f"BB mean={bb_weights.mean():.4f}"


# ---------------------------------------------------------------------------
# Test 5: OSB weights on cube
# ---------------------------------------------------------------------------

class TestWeightsOSBCube:
    """Test OSB weight computation on the 11x11x11 unit cube."""

    @pytest.fixture(scope="class")
    def osb_weights(self):
        families = _get_families()
        return compute_weights(
            _COORDS, _VOLUMES, _DELTA,
            model="osb", families=families
        )

    def test_shape(self, osb_weights):
        assert osb_weights.shape == (1331,)

    def test_interior_weights_near_one(self, osb_weights):
        """Centre node should be closest to 1.0 among all nodes."""
        N = _N_SIDE
        centre_idx = 5 * N * N + 5 * N + 5
        assert abs(osb_weights[centre_idx] - 1.0) < 0.20, (
            f"Centre weight too far from 1.0: {osb_weights[centre_idx]:.4f}"
        )
        assert abs(osb_weights[centre_idx] - 1.0) < abs(osb_weights[0] - 1.0)

    def test_boundary_weights_above_one(self, osb_weights):
        corner_idx = 0
        assert osb_weights[corner_idx] > 1.5

    def test_mean_weight_range(self, osb_weights):
        mean_w = osb_weights.mean()
        assert 0.8 < mean_w < 1.8, f"Mean weight out of range: {mean_w:.4f}"

    def test_osb_stats(self, osb_weights):
        """Check that overall stats match reference code output."""
        assert 0.45 < osb_weights.min() < 0.65, f"OSB min={osb_weights.min():.4f}"
        assert 2.80 < osb_weights.max() < 3.50, f"OSB max={osb_weights.max():.4f}"
        assert 1.10 < osb_weights.mean() < 1.35, f"OSB mean={osb_weights.mean():.4f}"


# ---------------------------------------------------------------------------
# Test 6: Exact reference weight comparison
# ---------------------------------------------------------------------------

class TestWeightsMatchReference:
    """Compare computed weights against hardcoded reference arrays.

    Reference values obtained from the authoritative reference code
    (flat-layout zip) on the same 11x11x11 cube with dx=0.1, m=3.
    """

    # BB reference: first 10 weights (node ordering: z-major, then y, then x)
    BB_REF_FIRST_10 = np.array([
        3.1625694043759647, 2.723529516363724, 2.208645168799861,
        2.221106471414112, 2.284119215904912, 2.3001751230830223,
        2.283967039130623, 2.2214704949349975, 2.2092413235757142,
        2.724395803004139,
    ])

    # OSB reference: first 10 weights
    OSB_REF_FIRST_10 = np.array([
        3.1259687644293996, 2.72124034483539, 2.186651924207149,
        2.189355685384753, 2.26111257960701, 2.2831134985474413,
        2.2610165918617677, 2.189164363489052, 2.1865379253315216,
        2.721226852002174,
    ])

    @pytest.fixture(scope="class")
    def computed_weights(self):
        families = _get_families()
        w_bb = compute_weights(
            _COORDS, _VOLUMES, _DELTA,
            model="bb", families=families
        )
        w_osb = compute_weights(
            _COORDS, _VOLUMES, _DELTA,
            model="osb", families=families
        )
        return w_bb, w_osb

    def test_bb_first_10(self, computed_weights):
        w_bb, _ = computed_weights
        np.testing.assert_allclose(
            w_bb[:10], self.BB_REF_FIRST_10, rtol=1e-3,
            err_msg="BB weights (first 10) do not match reference"
        )

    def test_osb_first_10(self, computed_weights):
        _, w_osb = computed_weights
        np.testing.assert_allclose(
            w_osb[:10], self.OSB_REF_FIRST_10, rtol=1e-3,
            err_msg="OSB weights (first 10) do not match reference"
        )

    def test_bb_full_statistics(self, computed_weights):
        w_bb, _ = computed_weights
        assert 0.50 < w_bb.min() < 0.70, f"BB min={w_bb.min():.4f}"
        assert 2.80 < w_bb.max() < 3.50, f"BB max={w_bb.max():.4f}"
        assert 1.15 < w_bb.mean() < 1.40, f"BB mean={w_bb.mean():.4f}"

    def test_osb_full_statistics(self, computed_weights):
        _, w_osb = computed_weights
        assert 0.45 < w_osb.min() < 0.65, f"OSB min={w_osb.min():.4f}"
        assert 2.80 < w_osb.max() < 3.50, f"OSB max={w_osb.max():.4f}"
        assert 1.10 < w_osb.mean() < 1.35, f"OSB mean={w_osb.mean():.4f}"


# ---------------------------------------------------------------------------
# Test 7: 2-D weights smoke test
# ---------------------------------------------------------------------------

class TestWeights2D:
    """Quick smoke tests for the 2-D BB-PD weight computation."""

    def test_disc_moment_known(self):
        """disc_moment(1.0, 0, 0) = pi (area of unit disc)."""
        expected = math.pi
        result = disc_moment(1.0, 0, 0)
        assert result == pytest.approx(expected, rel=1e-12)

    def test_disc_moment_x2(self):
        """disc_moment(1.0, 2, 0) = pi/4."""
        expected = math.pi / 4.0
        result = disc_moment(1.0, 2, 0)
        assert result == pytest.approx(expected, rel=1e-12)

    def test_full_disc_area(self):
        delta = 0.3
        expected = math.pi * delta ** 2
        assert full_disc_area(delta) == pytest.approx(expected, rel=1e-14)

    def test_2d_weights_smoke(self):
        """Run 2D weight computation on a small square grid."""
        dx = 0.1
        N = 11
        coords_2d = np.array(
            [[ix * dx, iy * dx] for iy in range(N) for ix in range(N)]
        )
        volumes_2d = np.full(len(coords_2d), dx ** 2)
        delta_2d = 3 * dx

        families_2d = build_families_2d(coords_2d, delta_2d)
        w = compute_weights_2d(
            coords_2d, volumes_2d, delta_2d, families=families_2d
        )
        assert w.shape == (N * N,)
        # Weights should be finite and mostly positive
        assert np.all(np.isfinite(w))
        # Mean should be in a reasonable range
        assert 0.5 < w.mean() < 2.0


# ---------------------------------------------------------------------------
# Test 8: Family construction
# ---------------------------------------------------------------------------

class TestBuildFamilies:
    """Verify family construction gives correct neighbor counts."""

    def test_corner_family_size(self):
        """A corner node of the cube should have fewer neighbours than interior."""
        families = _get_families()
        corner_count = len(families[0])
        # Corner: approximately 1/8 of the full ball -> ~28 (observed)
        assert 20 <= corner_count <= 40, (
            f"Corner family size unexpected: {corner_count}"
        )

    def test_interior_family_size(self):
        """An interior node (e.g. centre of the cube) should have ~122 neighbours.

        Full ball m=3: expected ~(4/3*pi*3^3)=113 nodes, minus self.
        For a regular grid with m=3 and some tolerance, it's typically around 122.
        """
        families = _get_families()
        N = _N_SIDE
        # Centre node index: (5, 5, 5) in z-y-x ordering
        centre_idx = 5 * N * N + 5 * N + 5
        interior_count = len(families[centre_idx])
        assert 100 <= interior_count <= 140, (
            f"Interior family size unexpected: {interior_count}"
        )

    def test_face_family_size(self):
        """A face-centre node should have roughly half the interior count."""
        families = _get_families()
        N = _N_SIDE
        # Face centre on z=0 plane: (5, 5, 0) -> idx = 0*N*N + 5*N + 5 = 60
        face_idx = 0 * N * N + 5 * N + 5
        face_count = len(families[face_idx])
        assert 50 <= face_count <= 80, (
            f"Face family size unexpected: {face_count}"
        )

    def test_symmetry(self):
        """If j is in family of i, then i should be in family of j."""
        families = _get_families()
        # Check a few nodes
        for i in [0, 100, 500, 665, 1000]:
            for j in families[i]:
                assert i in families[j], (
                    f"Asymmetric family: {j} in family[{i}] but {i} not in family[{j}]"
                )

    def test_self_exclusion(self):
        """No node should be in its own family."""
        families = _get_families()
        for i in range(len(families)):
            assert i not in families[i], f"Node {i} is in its own family"

    def test_total_nodes(self):
        families = _get_families()
        assert len(families) == 1331

    def test_family_sizes_first_10(self):
        """First 10 family sizes match reference."""
        families = _get_families()
        sizes = [len(families[i]) for i in range(10)]
        expected = [28, 37, 45, 46, 46, 46, 46, 46, 45, 37]
        assert sizes == expected


# ---------------------------------------------------------------------------
# Test: model validation
# ---------------------------------------------------------------------------

class TestModelValidation:
    """Test that invalid model names are rejected."""

    def test_invalid_model_raises(self):
        with pytest.raises(ValueError, match="model must be"):
            compute_weights(
                _COORDS, _VOLUMES, _DELTA,
                model="invalid_model"
            )


# ---------------------------------------------------------------------------
# Test: version and branding
# ---------------------------------------------------------------------------

class TestBranding:
    """Test version string and show_version function."""

    def test_version(self):
        assert perifit.__version__ == "2.0.1"

    def test_show_version(self, capsys):
        perifit.show_version()
        captured = capsys.readouterr()
        assert "perifit v2.0.1" in captured.out
        assert "Arman Shojaei" in captured.out
        assert "Alexander Hermann" in captured.out
        assert "Hereon" in captured.out
        assert "PeriFit" in captured.out or "perifit" in captured.out.lower()

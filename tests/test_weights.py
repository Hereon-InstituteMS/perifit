"""End-to-end tests for the weight computation and output writers."""
import math
import tempfile
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial import KDTree

from perifit import compute_weights, write_weights
from perifit.core.targets import build_targets_osb
from perifit.core.moments import full_ball_weighted_volume
from perifit.core.local_system import build_local_system_osb
from perifit.core.weights import build_families


def _cube_mesh(dx: float):
    """Structured Cartesian [0,1]^3 mesh."""
    xs = np.arange(dx / 2, 1.0, dx)
    gx, gy, gz = np.meshgrid(xs, xs, xs, indexing='ij')
    coords  = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
    volumes = np.full(len(coords), dx ** 3)
    return coords, volumes


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------

class TestTargets:
    def test_output_shapes(self):
        delta = 0.15
        d, e = build_targets_osb(delta)
        assert d.shape == (27,)
        assert e.shape == (27,)

    def test_targets_symmetry(self):
        """OSB derivative targets for linear test functions are symmetric."""
        delta = 0.15
        d, _ = build_targets_osb(delta)
        # alpha=0,k=1 vs alpha=1,k=2 vs alpha=2,k=3
        assert abs(d[0] - d[10]) < 1e-15
        assert abs(d[0] - d[20]) < 1e-15

    def test_targets_scale_with_delta(self):
        """Verify targets scale correctly with delta."""
        from perifit.core.moments import ball_moment_over_r2
        for delta in (0.1, 0.2, 0.5):
            d, _ = build_targets_osb(delta)
            # d[0] = ball_moment_over_r2(delta, 2, 0, 0)
            expected = ball_moment_over_r2(delta, 2, 0, 0)
            assert abs(d[0] - expected) < 1e-14


# ---------------------------------------------------------------------------
# Local system
# ---------------------------------------------------------------------------

class TestLocalSystem:
    def _random_ball(self, n_F: int, delta: float, seed: int = 42):
        rng = np.random.default_rng(seed)
        phi = rng.uniform(0, 2 * math.pi, n_F)
        cos_t = rng.uniform(-0.99, 0.99, n_F)
        r = delta * rng.uniform(0.2, 0.98, n_F) ** (1 / 3)
        sin_t = np.sqrt(1 - cos_t ** 2)
        xi = np.column_stack([r * sin_t * np.cos(phi),
                               r * sin_t * np.sin(phi),
                               r * cos_t])
        return xi

    def test_too_few_neighbors_returns_none(self):
        delta = 0.15
        d, e = build_targets_osb(delta)
        xi = self._random_ball(9, delta)   # 9 < NP=10
        vols = np.full(9, 0.001)
        result = build_local_system_osb(xi, vols, delta, d, e)
        assert result is None

    def test_output_shapes(self):
        delta = 0.15
        d, e = build_targets_osb(delta)
        n_F = 30
        xi = self._random_ball(n_F, delta)
        vols = np.full(n_F, 0.001)
        result = build_local_system_osb(xi, vols, delta, d, e)
        assert result is not None
        beta_i, coupling = result
        assert isinstance(beta_i, float)
        assert coupling.shape == (n_F,)

    def test_beta_is_finite(self):
        delta = 0.15
        d, e = build_targets_osb(delta)
        n_F = 50
        xi = self._random_ball(n_F, delta)
        vols = np.full(n_F, 0.001)
        beta_i, coupling = build_local_system_osb(xi, vols, delta, d, e)
        assert math.isfinite(beta_i)
        assert np.all(np.isfinite(coupling))

    def test_interior_local_system_finite(self):
        """For a complete (interior) neighbourhood, beta_i and coupling are finite."""
        dx = 0.1
        delta = 3.0 * dx
        coords, volumes = _cube_mesh(dx)
        # Find a node near the center
        center_idx = np.argmin(np.sum((coords - 0.5) ** 2, axis=1))
        tree = KDTree(coords)
        nbrs = [j for j in tree.query_ball_point(coords[center_idx], delta)
                if j != center_idx]
        xi = coords[nbrs] - coords[center_idx]
        vols = volumes[nbrs]
        d, e = build_targets_osb(delta)
        beta_i, coupling = build_local_system_osb(xi, vols, delta, d, e)
        assert math.isfinite(beta_i)
        assert np.all(np.isfinite(coupling))
        assert len(nbrs) == len(coupling)


# ---------------------------------------------------------------------------
# Full weight computation
# ---------------------------------------------------------------------------

class TestComputeWeights:
    """Tests on a small 5x5x5 = 125 node mesh (fast)."""

    @pytest.fixture(scope='class')
    def small_mesh(self):
        return _cube_mesh(dx=0.2)

    @pytest.fixture(scope='class')
    def weights(self, small_mesh):
        coords, volumes = small_mesh
        delta = 2 * 0.2  # m=2
        families = build_families(coords, delta)
        return compute_weights(coords, volumes, delta, model='osb', families=families)

    def test_length(self, small_mesh, weights):
        coords, _ = small_mesh
        assert len(weights) == len(coords)

    def test_positive(self, weights):
        assert weights.min() > 0

    def test_all_finite(self, weights):
        assert np.all(np.isfinite(weights))

    def test_interior_weights_near_one(self, small_mesh, weights):
        """Interior nodes should have weights close to 1."""
        coords, _ = small_mesh
        delta = 0.4
        tree = KDTree(coords)
        pairs = tree.query_ball_tree(tree, r=delta)
        sizes = np.array([len([j for j in pairs[i] if j != i])
                          for i in range(len(coords))])
        max_s = sizes.max()
        interior = sizes >= 0.95 * max_s
        if interior.sum() > 0:
            assert abs(weights[interior].mean() - 1.0) < 0.25

    def test_surface_weights_greater_than_interior(self, small_mesh, weights):
        """Surface nodes must have larger weights than interior nodes."""
        coords, _ = small_mesh
        delta = 0.4
        tree = KDTree(coords)
        pairs = tree.query_ball_tree(tree, r=delta)
        sizes = np.array([len([j for j in pairs[i] if j != i])
                          for i in range(len(coords))])
        max_s = sizes.max()
        interior = sizes >= 0.95 * max_s
        surface  = sizes <  0.60 * max_s
        if interior.sum() > 0 and surface.sum() > 0:
            assert weights[surface].mean() > weights[interior].mean()

    def test_weights_geometry_only(self, small_mesh):
        """Weights must be identical when called twice (deterministic)."""
        coords, volumes = small_mesh
        delta = 0.4
        families = build_families(coords, delta)
        w1 = compute_weights(coords, volumes, delta, model='osb', families=families)
        w2 = compute_weights(coords, volumes, delta, model='osb', families=families)
        assert np.allclose(w1, w2)

    def test_degenerate_isolated_node(self):
        """A single isolated node gets weight = 1."""
        coords = np.array([[0.0, 0.0, 0.0]])
        volumes = np.array([1.0])
        w = compute_weights(coords, volumes, 0.1, model='osb')
        assert len(w) == 1
        assert w[0] == pytest.approx(1.0)

    def test_weights_symmetric_on_symmetric_mesh(self):
        """
        On a node-centred cube [0,1]^3 the weights must be (nearly) symmetric
        under reflection and axis permutation.
        """
        dx = 0.2
        xs = np.arange(0, 1.0 + 0.5*dx, dx)   # 6 nodes: 0, 0.2, ..., 1.0
        Ns = len(xs)
        gx, gy, gz = np.meshgrid(xs, xs, xs, indexing='ij')
        coords = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
        volumes = np.full(len(coords), dx**3)
        delta = 3 * dx

        families = build_families(coords, delta)
        w = compute_weights(coords, volumes, delta, model='osb', families=families)

        lut = {}
        for k in range(len(w)):
            ix = round(coords[k, 0] / dx)
            iy = round(coords[k, 1] / dx)
            iz = round(coords[k, 2] / dx)
            lut[(ix, iy, iz)] = w[k]

        max_refl_x = max(
            abs(lut[(ix, iy, iz)] - lut[(Ns - 1 - ix, iy, iz)])
            for ix in range(Ns) for iy in range(Ns) for iz in range(Ns)
        )
        max_perm_xy = max(
            abs(lut[(ix, iy, iz)] - lut[(iy, ix, iz)])
            for ix in range(Ns) for iy in range(Ns) for iz in range(Ns)
        )
        # Tolerance is generous to allow BiCGSTAB solver non-symmetry
        assert max_refl_x < 0.12, (
            f"Reflection-x symmetry violated: max|w(i)-w(mirror)|={max_refl_x:.3f}")
        assert max_perm_xy < 0.12, (
            f"xy-permutation symmetry violated: max|w(i)-w(perm)|={max_perm_xy:.3f}")


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

class TestWriters:
    @pytest.fixture(scope='class')
    def small_result(self):
        coords, volumes = _cube_mesh(dx=0.2)
        delta = 0.4
        families = build_families(coords, delta)
        weights = compute_weights(coords, volumes, delta, model='osb', families=families)
        return coords, volumes, weights

    def _write(self, small_result, fmt, **kw):
        coords, volumes, weights = small_result
        with tempfile.TemporaryDirectory() as td:
            paths = write_weights(
                coords, volumes, weights, outdir=td, stem='t',
                formats=[fmt], **kw
            )
            assert fmt in paths
            p = paths[fmt]
            assert p.exists()
            assert p.stat().st_size > 100
            return p.read_text() if p.suffix in ('.csv', '.dat', '.vtk') else None

    def test_csv_roundtrip(self, small_result):
        coords, volumes, weights = small_result
        with tempfile.TemporaryDirectory() as td:
            paths = write_weights(
                coords, volumes, weights, outdir=td, stem='t', formats=['csv']
            )
            text = paths['csv'].read_text()
            lines = [l for l in text.splitlines() if not l.startswith('#')]
            assert 'weight' in lines[0]       # column header
            import io
            data = np.loadtxt(io.StringIO('\n'.join(lines[1:])), delimiter=',')
            assert data.shape == (len(coords), 6)
            assert np.allclose(data[:, 5], weights, atol=1e-8)

    def test_dat(self, small_result):
        text = self._write(small_result, 'dat')
        data_lines = [l for l in text.splitlines() if not l.startswith('#')]
        coords, _, _ = small_result
        assert len(data_lines) == len(coords)

    def test_vtk(self, small_result):
        text = self._write(small_result, 'vtk')
        assert 'vtk DataFile' in text
        assert 'SCALARS weight' in text

    def test_peridigm(self, small_result):
        self._write(small_result, 'peridigm')

    def test_perilab(self, small_result):
        self._write(small_result, 'perilab')

    def test_multiple_formats(self, small_result):
        coords, volumes, weights = small_result
        with tempfile.TemporaryDirectory() as td:
            paths = write_weights(
                coords, volumes, weights, outdir=td, stem='t',
                formats=['csv', 'dat', 'vtk']
            )
            assert set(paths.keys()) >= {'csv', 'dat', 'vtk'}
            for p in paths.values():
                assert p.exists()


# ---------------------------------------------------------------------------
# Mesh reader
# ---------------------------------------------------------------------------

class TestCSVReader:
    def test_csv_3col(self, tmp_path):
        from perifit import load_mesh
        csv_file = tmp_path / 'test.csv'
        coords_in = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        np.savetxt(csv_file, coords_in, delimiter=',', header='x,y,z')
        coords, volumes = load_mesh(csv_file)
        assert coords.shape == (2, 3)
        assert np.allclose(coords, coords_in)
        assert volumes.shape == (2,)
        assert volumes[0] > 0

    def test_csv_4col(self, tmp_path):
        from perifit import load_mesh
        csv_file = tmp_path / 'test4.csv'
        data = np.array([[0.1, 0.2, 0.3, 0.001], [0.4, 0.5, 0.6, 0.002]])
        np.savetxt(csv_file, data, delimiter=',', header='x,y,z,vol')
        coords, volumes = load_mesh(csv_file, vol_col=3)
        assert np.allclose(volumes, [0.001, 0.002])

    def test_csv_custom_columns(self, tmp_path):
        from perifit import load_mesh
        csv_file = tmp_path / 'custom.csv'
        # id, x, y, z order
        data = np.array([[1, 0.1, 0.2, 0.3], [2, 0.4, 0.5, 0.6]])
        np.savetxt(csv_file, data, delimiter=',', header='id,x,y,z')
        coords, _ = load_mesh(csv_file, x_col=1, y_col=2, z_col=3)
        assert coords.shape == (2, 3)
        assert np.allclose(coords[:, 0], [0.1, 0.4])

    def test_example_mesh_loads(self):
        from perifit import get_example_mesh, load_mesh
        coords, volumes = load_mesh(get_example_mesh())
        assert coords.shape == (1000, 3)
        assert volumes.shape == (1000,)
        assert np.all(volumes > 0)


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

class TestConfig:
    def test_defaults(self):
        from perifit import CorrectionConfig
        cfg = CorrectionConfig()
        assert cfg.m_ratio == 3.0
        assert cfg.horizon is None
        assert cfg.tol == 1e-10
        assert cfg.formats == ['csv']

    def test_invalid_horizon(self):
        from perifit import CorrectionConfig
        with pytest.raises(ValueError):
            CorrectionConfig(horizon=-0.1)

    def test_invalid_m_ratio(self):
        from perifit import CorrectionConfig
        with pytest.raises(ValueError):
            CorrectionConfig(m_ratio=0.0)

    def test_explicit_horizon(self):
        from perifit import CorrectionConfig
        cfg = CorrectionConfig(horizon=0.3)
        assert cfg.horizon == 0.3

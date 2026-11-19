"""Bond weights (one weight per bond) in 1-D, 2-D and 3-D."""
import numpy as np
import pytest

from perifit.core.bond_weights import MOMENTS, compute_bond_weights, sphere_moment
from perifit.core.families import build_bonds

from ._pdsolver import bar_problem, cantilever_problem


def _square(dx=0.05):
    s = np.linspace(0, 1, int(round(1 / dx)) + 1)
    X, Y = np.meshgrid(s, s, indexing="ij")
    return np.column_stack([X.ravel(), Y.ravel()]), 4 * dx * (1 + dx / 100)


class TestSphereMoments:
    def test_known_values(self):
        assert sphere_moment((0,)) == pytest.approx(2.0)                  # S^0 = {-1, +1}
        assert sphere_moment((0, 0)) == pytest.approx(2 * np.pi)          # circle
        assert sphere_moment((0, 0, 0)) == pytest.approx(4 * np.pi)       # sphere
        assert sphere_moment((4, 0)) == pytest.approx(3 * sphere_moment((2, 2)))
        assert sphere_moment((1, 1)) == 0.0

    def test_moments_are_even(self):
        for rows in MOMENTS.values():
            assert all(sum(a) % 2 == 0 for _, a in rows)


class TestSolution:
    @pytest.fixture(scope="class")
    def square(self):
        nodes, delta = _square()
        V = np.full(len(nodes), 0.05 ** 2)
        bonds, beta, info = compute_bond_weights(nodes, V, delta)
        return nodes, V, delta, bonds, beta, info

    def test_equations_satisfied_and_nonnegative(self, square):
        *_, beta, info = square
        assert info.residual < 1e-8
        assert beta.min() >= 0.0

    def test_bulk_close_to_one_boundary_amplified(self, square):
        nodes, _, _, bonds, beta, _ = square
        mid = np.all(np.abs(nodes[bonds].mean(axis=1) - 0.5) < 0.2, axis=1)
        edge = np.any(nodes[bonds].min(axis=1) < 1e-12, axis=1)
        # the bulk is also corrected: at m = 4 the grid sums of the energy moments
        # fall 3-13 % short of the exact values, hence a bulk median slightly above 1
        assert 1.0 < np.median(beta[mid]) < 1.15
        assert np.median(beta[edge]) > 1.2

    def test_bonds_sorted(self, square):
        bonds = square[3]
        assert np.all(bonds[:, 0] < bonds[:, 1])
        assert np.array_equal(bonds, build_bonds(square[0], square[2]))

    def test_unit_invariance(self, square):
        nodes, V, delta, _, beta, _ = square
        _, beta2, _ = compute_bond_weights(nodes * 1e3, V * 1e6, delta * 1e3)
        assert np.abs(beta - beta2).max() < 1e-8

    def test_nonuniform_volumes(self, square):
        nodes, V, delta, *_ = square
        Vr = V * np.random.default_rng(0).uniform(0.7, 1.3, len(V))
        _, beta, info = compute_bond_weights(nodes, Vr, delta)
        assert info.residual < 1e-8 and beta.min() >= 0.0

    def test_given_bonds_are_used(self, square):
        nodes, V, delta, bonds, beta, _ = square
        _, beta2, _ = compute_bond_weights(nodes, V, delta, bonds=bonds)
        assert np.allclose(beta, beta2, atol=1e-10)

    def test_3d_cube(self):
        s = np.arange(7) * 0.1
        X, Y, Z = np.meshgrid(s, s, s, indexing="ij")
        nodes = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        _, beta, info = compute_bond_weights(nodes, np.full(len(nodes), 1e-3), 0.3)
        assert info.residual < 1e-8 and beta.min() >= 0.0

    def test_invalid_input(self):
        with pytest.raises(ValueError, match="coincident"):
            compute_bond_weights(np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0]]),
                                 np.ones(3), 2.0)
        with pytest.raises(ValueError):
            compute_bond_weights(np.zeros((4, 4)), np.ones(4), 1.0)


class TestBar:
    def test_first_order_convergence(self):
        errs, pd = [], []
        for dx in (0.02, 0.01, 0.005):
            x, V, delta, bonds, error = bar_problem(dx)
            _, beta, _ = compute_bond_weights(x, V, delta, bonds=bonds)
            errs.append(error(beta))
            pd.append(error(np.ones(len(bonds))))
        slopes = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
        assert np.all(slopes > 0.95)
        assert errs[-1] < 0.003 and errs[-1] < 0.05 * pd[-1]


class TestCantilever:
    def test_regression_values(self):
        out = []
        for dx in (0.04, 0.02):
            nodes, V, delta, bonds, error = cantilever_problem(dx)
            _, beta, _ = compute_bond_weights(nodes, V, delta, bonds=bonds)
            out.append(error(beta))
        assert np.log10(out) == pytest.approx([-1.469, -1.795], abs=0.01)

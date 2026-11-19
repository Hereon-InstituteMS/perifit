"""Robustness of the nodal weights: units, neighbour order, regularisation.

The weights are purely geometric, so they must not depend on the length unit
or on the order in which neighbours are listed.  These tests guard the
round-off handling of the local solve and the relative horizon tolerance.
"""
import numpy as np
import pytest

from perifit.core.families import build_bonds, build_families, families_from_bonds
from perifit.core.weights import compute_weights
from perifit.core.weights_2d import compute_weights_2d


def _cube(n=9, dx=0.1):
    s = np.arange(n) * dx
    X, Y, Z = np.meshgrid(s, s, s, indexing="ij")
    return np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])


def _square(n=21, dx=0.05):
    s = np.arange(n) * dx
    X, Y = np.meshgrid(s, s, indexing="ij")
    return np.column_stack([X.ravel(), Y.ravel()])


class TestNeighbourSearch:
    def test_bonds_and_families_agree(self):
        c = _cube()
        assert families_from_bonds(len(c), build_bonds(c, 0.3)) == build_families(c, 0.3)

    def test_bonds_sorted_and_unique(self):
        b = build_bonds(_cube(), 0.3)
        assert np.all(b[:, 0] < b[:, 1])
        assert np.all(np.diff(b[:, 0] * len(b) + b[:, 1]) > 0)

    @pytest.mark.parametrize("scale", [1e-6, 1e4])
    def test_horizon_nodes_included_in_any_unit(self, scale):
        c = _cube()
        assert len(build_bonds(c * scale, 0.3 * scale)) == len(build_bonds(c, 0.3))


@pytest.mark.parametrize("model", ["bb", "osb"])
class TestNodalWeights3D:
    def test_unit_invariance(self, model):
        c = _cube()
        ref = compute_weights(c, np.full(len(c), 1e-3), 0.3, model=model)
        for scale in (1e-6, 1e4):
            w = compute_weights(c * scale, np.full(len(c), 1e-3 * scale ** 3), 0.3 * scale,
                                model=model)
            assert np.abs(w - ref).max() < 1e-10

    def test_neighbour_order_invariance(self, model):
        c = _cube()
        V = np.full(len(c), 1e-3)
        fam = build_families(c, 0.3)
        rng = np.random.default_rng(0)
        shuffled = [list(rng.permutation(f)) for f in fam]
        w1 = compute_weights(c, V, 0.3, model=model, families=fam)
        w2 = compute_weights(c, V, 0.3, model=model, families=shuffled)
        assert np.abs(w1 - w2).max() < 1e-10


class TestNodalWeights2D:
    def test_unit_invariance(self):
        c = _square()
        ref = compute_weights_2d(c, np.full(len(c), 0.05 ** 2), 0.2)
        w = compute_weights_2d(c * 1e3, np.full(len(c), 50.0 ** 2), 200.0)
        assert np.abs(w - ref).max() < 1e-10

    def test_regularisation_reaches_local_solve(self):
        """tikhonov_rel must act as regularisation (it was once passed as c_bond)."""
        c = _square()
        V = np.full(len(c), 0.05 ** 2)
        w_default = compute_weights_2d(c, V, 0.2)
        w_strong = compute_weights_2d(c, V, 0.2, tikhonov_rel=1e-1)
        assert np.abs(w_default - w_strong).max() > 1e-3

"""Public API: nodal_weights, bond_weights, surface_correction."""
import warnings

import numpy as np
import pytest

import perifit
from perifit.core.weights_2d import compute_weights_2d


def _plate(n=17, dx=1 / 16):
    s = np.arange(n) * dx
    X, Y = np.meshgrid(s, s, indexing="ij")
    return np.column_stack([X.ravel(), Y.ravel()]), np.full(n * n, dx * dx), 4 * dx * 1.0001


class TestSurfaceCorrection:
    def test_method_is_required(self):
        c, V, d = _plate()
        with pytest.raises(ValueError, match="choose the surface correction explicitly"):
            perifit.surface_correction(c, V, d)
        with pytest.raises(ValueError, match="unknown method"):
            perifit.surface_correction(c, V, d, method="magic")

    def test_nodal(self):
        c, V, d = _plate()
        corr = perifit.surface_correction(c, V, d, method="nodal")
        w = corr.node_weights
        assert corr.method == "nodal" and corr.info is None
        assert np.allclose(corr.factor, 0.5 * (w[corr.bonds[:, 0]] + w[corr.bonds[:, 1]]))
        assert np.allclose(w, perifit.nodal_weights(c, V, d), atol=1e-12)

    def test_bond(self):
        c, V, d = _plate()
        corr = perifit.surface_correction(c, V, d, method="bond")
        bonds, beta = perifit.bond_weights(c, V, d)
        assert corr.method == "bond" and corr.node_weights is None
        assert np.array_equal(corr.bonds, bonds) and np.allclose(corr.factor, beta)
        assert corr.info.residual < 1e-8 and len(corr) == len(bonds)

    def test_same_bonds_for_both_methods(self):
        c, V, d = _plate()
        a = perifit.surface_correction(c, V, d, method="nodal")
        b = perifit.surface_correction(c, V, d, method="bond")
        assert np.array_equal(a.bonds, b.bonds)

    def test_bond_weights_bond_based_only(self):
        s = np.arange(5) * 0.1
        X, Y, Z = np.meshgrid(s, s, s, indexing="ij")
        c = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        with pytest.raises(ValueError, match="bond-based"):
            perifit.surface_correction(c, np.full(len(c), 1e-3), 0.3, method="bond", model="osb")


class TestDimension:
    def test_1d_2d_3d_dispatch(self):
        x = (np.arange(40) + 0.5) * 0.025
        assert perifit.nodal_weights(x, np.full(40, 0.025), 0.1).shape == (40,)
        c, V, d = _plate()
        assert np.allclose(perifit.nodal_weights(c, V, d), compute_weights_2d(c, V, d))

    def test_planar_mesh_in_3_columns_raises(self):
        c, V, d = _plate()
        c3 = np.column_stack([c, np.zeros(len(c))])
        with pytest.raises(ValueError, match="dim=2"):
            perifit.nodal_weights(c3, V, d)
        assert np.allclose(perifit.nodal_weights(c3, V, d, dim=2), perifit.nodal_weights(c, V, d))

    def test_state_based_is_3d_only(self):
        c, V, d = _plate()
        with pytest.raises(ValueError, match="3-D only"):
            perifit.nodal_weights(c, V, d, model="osb")

    def test_bad_input(self):
        c, V, d = _plate()
        with pytest.raises(ValueError):
            perifit.nodal_weights(c, V[:-1], d)
        with pytest.raises(ValueError):
            perifit.bond_weights(c, V, -1.0)


class TestDeprecatedAliases:
    def test_aliases_warn_and_forward(self):
        c, V, d = _plate()
        with pytest.warns(DeprecationWarning, match="nodal_weights"):
            w = perifit.compute_weights_2d(c, V, d)
        assert np.allclose(w, perifit.nodal_weights(c, V, d))

    def test_new_api_does_not_warn(self):
        c, V, d = _plate()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            perifit.nodal_weights(c, V, d)
            perifit.bond_weights(c, V, d)

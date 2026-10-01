"""Regression test: 2-D BB-PD cantilever with end shear (IJNME 2026, Example 2).

Plane stress, E = 1, nu = 1/3, L = H = 1, parabolic end shear P = 1, m_ratio = 4
(delta = 4 dx (1 + dx/100)), exact displacement prescribed on the left node column, nodal
shear forces on the right node column. With the plain-pseudoinverse local solve
(row_normalize=False) the corrected model reproduces the published error levels
(about 10 % at dx = 0.04 and 3-4 % at dx = 0.02).
"""
import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.spatial import cKDTree

from perifit.core.weights_2d import compute_weights_2d

E, NU, P, L, H = 1.0, 1.0 / 3.0, 1.0, 1.0, 1.0
I_B = H**3 / 12


def _exact(x, y):
    ux = P / (6 * E * I_B) * (y - H / 2) * ((6 * L * x - 3 * x**2) + (2 + NU) * (y**2 - H * y))
    uy = -P / (6 * E * I_B) * ((3 * L * x**2 - x**3) + 3 * NU * (L - x) * (y - H / 2) ** 2
                               + (4 + 5 * NU) * H**2 / 4 * x)
    return np.c_[ux, uy]


def _cantilever_error(dx, weights):
    delta = 4.0 * dx * (1 + dx / 100)
    c = 6 * E / (np.pi * delta**3 * (1 - NU))
    xs = np.arange(0, L + dx / 2, dx)
    X = np.array([(x, y) for x in xs for y in xs])
    N = len(X)
    w = np.ones(N) if weights is None else weights(X, np.full(N, dx * dx), delta)
    pr = cKDTree(X).query_pairs(delta, output_type="ndarray")
    i, j = pr[:, 0], pr[:, 1]
    xi = X[j] - X[i]
    r = np.linalg.norm(xi, axis=1)
    n = xi / r[:, None]
    k = 0.5 * (w[i] + w[j]) * c / r * dx**4
    rows, cols, vals = [], [], []
    for a in range(2):
        for b in range(2):
            kk = k * n[:, a] * n[:, b]
            rows += [2 * i + a, 2 * j + a, 2 * i + a, 2 * j + a]
            cols += [2 * i + b, 2 * j + b, 2 * j + b, 2 * i + b]
            vals += [kk, kk, -kk, -kk]
    K = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(2 * N, 2 * N))
    f = np.zeros(2 * N)
    right = np.where(np.abs(X[:, 0] - L) < dx / 2)[0]
    f[2 * right + 1] = -6 * P / H**3 * X[right, 1] * (H - X[right, 1]) * dx
    bc = np.where(X[:, 0] < dx / 2)[0]
    fixed = np.zeros(2 * N, bool)
    fixed[2 * bc] = True
    fixed[2 * bc + 1] = True
    u = np.zeros((N, 2))
    u[bc] = _exact(*X[bc].T)
    u = u.ravel()
    fr = ~fixed
    u[fr] = spla.spsolve(K[fr][:, fr].tocsc(), f[fr] - K[fr][:, fixed] @ u[fixed])
    ue = _exact(*X.T)
    return np.linalg.norm(u.reshape(-1, 2) - ue) / np.linalg.norm(ue)


@pytest.mark.parametrize("dx, bound", [(0.04, 0.12), (0.02, 0.05)])
def test_cantilever_pinv_reproduces_paper(dx, bound):
    e_pd = _cantilever_error(dx, None)
    e_mod = _cantilever_error(
        dx, lambda X, V, d: compute_weights_2d(X, V, d, row_normalize=False))
    assert e_pd > 0.13                 # uncorrected PD: surface effect
    assert e_mod < bound               # corrected: published error level
    assert e_mod < 0.5 * e_pd

"""Minimal static linearised bond-based PD solver used by the tests.

A bond (i, j) contributes the stiffness block

    k_b = f_b * c / |xi|^3 * V_i * V_j * (xi (x) xi)

where ``f_b`` is the bond multiplier of the surface correction (1 for standard
PD, (w_i + w_j)/2 for nodal weights, beta_b for bond weights).
"""
import math

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

from perifit.core.families import build_bonds


def stiffness(nodes, volumes, bonds, factor, c):
    nodes = np.asarray(nodes, float).reshape(len(nodes), -1)
    n, d = nodes.shape
    bi, bj = bonds[:, 0], bonds[:, 1]
    xi = nodes[bj] - nodes[bi]
    k = factor * c / np.linalg.norm(xi, axis=1) ** 3 * volumes[bi] * volumes[bj]
    rows, cols, vals = [], [], []
    for p in range(d):
        for q in range(d):
            kk = k * xi[:, p] * xi[:, q]
            rows += [d * bi + p, d * bj + p, d * bi + p, d * bj + p]
            cols += [d * bi + q, d * bj + q, d * bj + q, d * bi + q]
            vals += [kk, kk, -kk, -kk]
    return csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(d * n, d * n))


def solve_dirichlet(K, F, fixed, values):
    n = K.shape[0]
    free = np.setdiff1d(np.arange(n), fixed)
    u = np.zeros(n)
    u[fixed] = values
    u[free] = spsolve(K[free][:, free].tocsc(), F[free] - K[free][:, fixed] @ values)
    return u


# ----------------------------------------------------------------- 1-D bar
def bar_problem(dx, m=4):
    """Example 1: unit bar, E = 1, clamp at node 0, unit force at the last node.

    Returns ``(x, volumes, horizon, bonds, error(factor))``; exact u = x.
    """
    x = (np.arange(int(round(1 / dx))) + 0.5) * dx
    V = np.full(len(x), dx)
    delta = m * dx
    bonds = build_bonds(x, delta)
    c = 2.0 / delta ** 2

    def error(factor):
        K = stiffness(x, V, bonds, factor, c)
        F = np.zeros(len(x)); F[-1] = 1.0
        u = solve_dirichlet(K, F, np.array([0]), np.array([0.0]))
        return float(np.sqrt(np.sum((u - x) ** 2) / np.sum(x ** 2)))

    return x, V, delta, bonds, error


# ---------------------------------------------------------- 2-D cantilever
def cantilever_exact(x, y, P=1.0, E=1.0, nu=1.0 / 3.0, L=1.0, H=1.0):
    I = H ** 3 / 12.0
    ux = P / (6 * E * I) * (y - H / 2) * ((6 * L * x - 3 * x ** 2) + (2 + nu) * (y ** 2 - H * y))
    uy = -P / (6 * E * I) * ((3 * L * x ** 2 - x ** 3) + 3 * nu * (L - x) * (y - H / 2) ** 2
                             + (4 + 5 * nu) * H ** 2 / 4 * x)
    return ux, uy


def cantilever_problem(dx, m=4):
    """Example 2 (IJNME paper setup): unit square, plane stress nu = 1/3,
    exact displacement on the x = 0 node column, parabolic end shear on x = 1.

    Returns ``(nodes, volumes, horizon, bonds, error(factor))``.
    """
    s = np.linspace(0.0, 1.0, int(round(1 / dx)) + 1)
    X, Y = np.meshgrid(s, s, indexing="ij")
    nodes = np.column_stack([X.ravel(), Y.ravel()])
    V = np.full(len(nodes), dx * dx)
    delta = m * (1 + dx / 100) * dx
    bonds = build_bonds(nodes, delta)
    c = 9.0 / (math.pi * delta ** 3)
    x, y = nodes.T
    ue, ve = cantilever_exact(x, y)

    def error(factor):
        K = stiffness(nodes, V, bonds, factor, c)
        F = np.zeros(2 * len(nodes))
        right = np.where(x >= 1 - dx / 100)[0]
        F[2 * right + 1] = -6.0 * y[right] * (1 - y[right]) * dx
        left = np.where(x <= dx / 100)[0]
        fixed = np.concatenate([2 * left, 2 * left + 1])
        u = solve_dirichlet(K, F, fixed, np.concatenate([ue[left], ve[left]]))
        return float(np.sqrt(np.sum((u[0::2] - ue) ** 2 + (u[1::2] - ve) ** 2)
                             / np.sum(ue ** 2 + ve ** 2)))

    return nodes, V, delta, bonds, error


def nodal_factor(w, bonds):
    return 0.5 * (w[bonds[:, 0]] + w[bonds[:, 1]])

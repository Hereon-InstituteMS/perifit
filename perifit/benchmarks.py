"""Benchmark problems with exact solutions, and a minimal static PD solver.

Used by the test suite, ``examples/benchmarks.py`` and the README figures, and
handy for checking a surface correction in your own setting.  The solver is
linearised bond-based peridynamics; a surface correction enters only through
one *multiplier per bond*:

    standard PD    : 1
    nodal weights  : (w_i + w_j) / 2
    bond weights   : beta_b

Problems (horizon delta = m * dx, no partial-volume factor):

``bar(dx)``         1-D bar, E = 1, clamped at x = 0, unit force at x = 1; u = x.
``cantilever(dx)``  2-D unit square, plane stress nu = 1/3, exact displacement on the
                    x = 0 node column, parabolic end shear on x = 1 (Timoshenko).
``cylinder(dx)``    3-D circular cantilever, L = 1, R = 0.5, nu = 1/4, exact displacement
                    on the x = 0 node plane, Saint-Venant end shear on x = 1.

Example
-------
    >>> import perifit
    >>> from perifit.benchmarks import cantilever
    >>> prob = cantilever(0.04)
    >>> corr = perifit.surface_correction(prob.nodes, prob.volumes, prob.horizon, method="bond")
    >>> prob.error(corr.bonds, corr.factor)        # relative L2 displacement error
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import cg, spsolve

from .core.families import build_bonds

__all__ = ["Problem", "bar", "cantilever", "cylinder", "cantilever_exact", "cylinder_exact",
           "stiffness", "solve"]


# ---------------------------------------------------------------------------
# Static linearised bond-based PD
# ---------------------------------------------------------------------------
def _stiffness_pattern(nodes: np.ndarray, volumes: np.ndarray, bonds: np.ndarray):
    """Sparsity pattern and unit-multiplier values of the bond stiffness blocks.

    Bond b contributes  c / |xi|^3 * V_i V_j * (xi (x) xi)  to the blocks
    (i, i), (j, j) and minus that to (i, j), (j, i).  Returns ``(rows, cols,
    vals, bond_of_entry)`` for c = 1 and multiplier 1.
    """
    n, d = nodes.shape
    bi, bj = bonds[:, 0], bonds[:, 1]
    xi = nodes[bj] - nodes[bi]
    k = volumes[bi] * volumes[bj] / np.linalg.norm(xi, axis=1) ** 3
    rows, cols, vals = [], [], []
    for p, q in itertools.product(range(d), repeat=2):
        kk = k * xi[:, p] * xi[:, q]
        rows += [d * bi + p, d * bj + p, d * bi + p, d * bj + p]
        cols += [d * bi + q, d * bj + q, d * bj + q, d * bi + q]
        vals += [kk, kk, -kk, -kk]
    bond_of_entry = np.tile(np.arange(len(bonds)), 4 * d * d)
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(vals), bond_of_entry


def stiffness(nodes: np.ndarray, volumes: np.ndarray, bonds: np.ndarray,
              factor: np.ndarray, c: float) -> csr_matrix:
    """Stiffness matrix with bond blocks  factor_b * c / |xi|^3 * V_i V_j * (xi (x) xi)."""
    nodes = np.asarray(nodes, float).reshape(len(nodes), -1)
    rows, cols, vals, b = _stiffness_pattern(nodes, np.asarray(volumes, float), bonds)
    size = nodes.size
    return csr_matrix((vals * (c * np.asarray(factor))[b], (rows, cols)), shape=(size, size))


def solve(K: csr_matrix, F: np.ndarray, fixed: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Solve ``K u = F`` with ``u[fixed] = values`` (direct solver, or Jacobi-CG for large systems)."""
    free = np.setdiff1d(np.arange(K.shape[0]), fixed)
    u = np.zeros(K.shape[0])
    u[fixed] = values
    K_free = K[free]
    Kff = K_free[:, free]
    rhs = F[free] - K_free[:, fixed] @ values
    if len(free) < 60_000:
        u[free] = spsolve(Kff.tocsc(), rhs)
    else:
        u[free], _ = cg(Kff, rhs, rtol=1e-11, maxiter=200_000, M=diags(1.0 / Kff.diagonal()))
    return u


@dataclass
class Problem:
    """A discretised benchmark: geometry, loading, boundary conditions and exact solution."""
    nodes: np.ndarray            #: (N, d) nodal positions
    volumes: np.ndarray          #: (N,) nodal volumes
    horizon: float               #: peridynamic horizon delta
    c: float                     #: micromodulus
    load: np.ndarray             #: (N * d,) nodal forces
    fixed: np.ndarray            #: prescribed degrees of freedom
    exact: np.ndarray            #: (N, d) exact displacement
    fixed_values: np.ndarray | None = None   #: prescribed values (default: exact displacement)
    _pattern_bonds: np.ndarray | None = field(default=None, repr=False)

    @cached_property
    def bonds(self) -> np.ndarray:
        """All bonds within the horizon (``i < j``, sorted)."""
        return build_bonds(self.nodes, self.horizon)

    def error(self, bonds: np.ndarray, factor: np.ndarray) -> float:
        """Relative L2 displacement error of the PD solution with these bond multipliers."""
        if self._pattern_bonds is None or not np.array_equal(self._pattern_bonds, bonds):
            self._pattern = _stiffness_pattern(self.nodes, self.volumes, bonds)
            self._pattern_bonds = bonds
        rows, cols, vals, b = self._pattern
        size = self.nodes.size
        K = csr_matrix((vals * (self.c * np.asarray(factor))[b], (rows, cols)), shape=(size, size))
        values = self.exact.ravel()[self.fixed] if self.fixed_values is None else self.fixed_values
        u = solve(K, self.load, self.fixed, values)
        return float(np.linalg.norm(u - self.exact.ravel()) / np.linalg.norm(self.exact))


# ---------------------------------------------------------------------------
# 1-D bar
# ---------------------------------------------------------------------------
def bar(dx: float, m: float = 4) -> Problem:
    """Unit bar under a unit end force; the clamped node is held at u = 0 (exact: u = x)."""
    x = (np.arange(int(round(1 / dx))) + 0.5) * dx
    load = np.zeros(len(x))
    load[-1] = 1.0
    delta = m * dx
    return Problem(x[:, None], np.full(len(x), dx), delta, 2.0 / delta ** 2,
                   load, np.array([0]), x[:, None], fixed_values=np.array([0.0]))


# ---------------------------------------------------------------------------
# 2-D cantilever with parabolic end shear (Timoshenko)
# ---------------------------------------------------------------------------
def cantilever_exact(x, y, P=1.0, E=1.0, nu=1 / 3, L=1.0, H=1.0) -> np.ndarray:
    """Exact plane-stress displacement (N, 2) of the cantilever under end shear P."""
    I = H ** 3 / 12
    ux = P / (6 * E * I) * (y - H / 2) * ((6 * L * x - 3 * x ** 2) + (2 + nu) * (y ** 2 - H * y))
    uy = -P / (6 * E * I) * ((3 * L * x ** 2 - x ** 3) + 3 * nu * (L - x) * (y - H / 2) ** 2
                             + (4 + 5 * nu) * H ** 2 / 4 * x)
    return np.column_stack([ux, uy])


def cantilever(dx: float, m: float = 4) -> Problem:
    """Unit-square cantilever (vertex grid), horizon m (1 + dx/100) dx as in the IJNME paper."""
    s = np.linspace(0, 1, int(round(1 / dx)) + 1)
    X, Y = np.meshgrid(s, s, indexing="ij")
    nodes = np.column_stack([X.ravel(), Y.ravel()])
    delta = m * (1 + dx / 100) * dx
    x, y = nodes.T
    load = np.zeros(2 * len(nodes))
    right = np.flatnonzero(x >= 1 - dx / 100)
    load[2 * right + 1] = -6 * y[right] * (1 - y[right]) * dx     # parabolic shear, resultant 1
    left = np.flatnonzero(x <= dx / 100)
    return Problem(nodes, np.full(len(nodes), dx * dx), delta, 9 / (math.pi * delta ** 3), load,
                   np.concatenate([2 * left, 2 * left + 1]), cantilever_exact(x, y))


# ---------------------------------------------------------------------------
# 3-D circular cantilever (Saint-Venant flexure)
# ---------------------------------------------------------------------------
E3, NU3, L3, R3 = 1.0, 0.25, 1.0, 0.5
I3 = math.pi * R3 ** 4 / 4
K1 = (3 + 2 * NU3) / (8 * (1 + NU3) * I3)
K2 = (1 + 2 * NU3) / (4 * (1 + NU3) * I3)
A3 = (1 - 2 * NU3) / (3 + 2 * NU3)


def _monomial(P: np.ndarray, e: tuple, d: int | None = None) -> np.ndarray:
    """prod_k P[:, k]**e[k], or its derivative with respect to coordinate ``d``."""
    out = np.ones(len(P))
    for k in range(3):
        if k == d:
            out = out * (e[k] * P[:, k] ** (e[k] - 1) if e[k] else 0.0 * P[:, k])
        else:
            out = out * P[:, k] ** e[k]
    return out


def _saint_venant_coefficients() -> tuple[list, np.ndarray]:
    """Cubic displacement of the circular cantilever under unit end shear.

    Stresses (Timoshenko & Goodier):  s_xx = (L - x) y / I,
    t_xy = -K1 (R^2 - y^2 - A z^2),   t_xz = K2 y z,   all others zero.
    The displacement is the cubic polynomial whose strains equal these
    stresses / Hooke's law, without rigid motion at the centre of the clamp;
    the least-squares fit is exact to round-off.
    """
    exps = [e for e in itertools.product(range(4), repeat=3) if sum(e) <= 3]
    X = np.random.default_rng(0).uniform(-1, 1, (300, 3))
    x, y, z = X.T
    sxx = (L3 - x) * y / I3
    G = E3 / (2 * (1 + NU3))
    strains = [sxx / E3, -NU3 * sxx / E3, -NU3 * sxx / E3,
               -K1 * (R3 ** 2 - y ** 2 - A3 * z ** 2) / G, K2 * y * z / G, 0 * x]
    D = lambda P, d: np.column_stack([_monomial(P, e, d) for e in exps])
    Z = np.zeros((len(X), len(exps)))
    blocks = [[D(X, 0), Z, Z], [Z, D(X, 1), Z], [Z, Z, D(X, 2)],
              [D(X, 1), D(X, 0), Z], [D(X, 2), Z, D(X, 0)], [Z, D(X, 2), D(X, 1)]]
    O = np.zeros((1, 3))
    v = lambda d=None: D(O, d)[0] if d is not None else np.array([_monomial(O, e)[0] for e in exps])
    z0 = np.zeros(len(exps))
    rigid = [np.r_[v(), z0, z0], np.r_[z0, v(), z0], np.r_[z0, z0, v()],
             np.r_[z0, v(0), z0], np.r_[z0, z0, v(0)], np.r_[z0, -v(2), v(1)]]
    A = np.vstack([np.hstack(b) for b in blocks] + [np.array(rigid)])
    b = np.concatenate(strains + [np.zeros(6)])
    return exps, np.linalg.lstsq(A, b, rcond=None)[0]


_SV_EXPS, _SV_COEF = _saint_venant_coefficients()


def cylinder_exact(nodes: np.ndarray) -> np.ndarray:
    """Exact displacement (N, 3) of the circular cantilever under unit end shear."""
    M = np.column_stack([np.prod(nodes ** np.array(e), axis=1) for e in _SV_EXPS])
    n = len(_SV_EXPS)
    return np.column_stack([M @ _SV_COEF[:n], M @ _SV_COEF[n:2 * n], M @ _SV_COEF[2 * n:]])


def cylinder(dx: float, m: float = 4) -> Problem:
    """Circular cantilever on a cubic lattice (use R/dx integer for a symmetric lattice)."""
    xs = np.linspace(0, L3, int(round(L3 / dx)) + 1)
    ys = np.linspace(-R3, R3, int(round(2 * R3 / dx)) + 1)
    X, Y, Z = np.meshgrid(xs, ys, ys, indexing="ij")
    inside = Y ** 2 + Z ** 2 <= R3 ** 2 * (1 + 1e-9)
    nodes = np.column_stack([X[inside], Y[inside], Z[inside]])
    delta = m * (1 + dx / 100) * dx
    x, y, z = nodes.T
    load = np.zeros((len(nodes), 3))
    end = np.flatnonzero(x >= L3 - dx / 100)
    load[end, 1] = -K1 * (R3 ** 2 - y[end] ** 2 - A3 * z[end] ** 2) * dx * dx
    load[end, 2] = K2 * y[end] * z[end] * dx * dx
    clamp = np.flatnonzero(x <= dx / 100)
    K_bulk = E3 / (3 * (1 - 2 * NU3))
    return Problem(nodes, np.full(len(nodes), dx ** 3), delta, 18 * K_bulk / (math.pi * delta ** 4),
                   load.ravel(), np.concatenate([3 * clamp + k for k in range(3)]),
                   cylinder_exact(nodes))

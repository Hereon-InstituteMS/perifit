"""Per-node local least-squares system of the 1-D nodal weights (bond-based PD).

For node i with bonds xi_j = x_j - x_i the stacked system has

  * n_F collocation rows   p(xi_j) = [1, xi/delta, (xi/delta)^2]
  * 2 derivative rows      sum_j (xi_j/|xi_j|) dphi_k(xi_j) V_j p_eff(xi_j),  k = 1, 2
  * 1 energy row           sum_j 1/4 dphi_1^2 xi_j^2/|xi_j|^3 V_j p_eff(xi_j)

with dphi_k = (xi/delta)^k and p_eff = (p(xi) + p(0)) / 2, the polynomial
image of the bond-averaged weight (w_i + w_j) / 2.  The system is solved by
the shared routine in :mod:`perifit.core._lsq` (per-row normalisation,
Tikhonov regularisation, round-off cutoff), so the weights do not depend on
the length unit.
"""
from __future__ import annotations

import numpy as np

from ._lsq import bond_averaged, solve_local
from .targets_1d import DERIV_DEGREES, ENERGY_DEGREES

NP_1D = 3                     # polynomial basis 1, xn, xn^2
#: each operator row has its own test degree -> its own round-off group
_GROUPS = [[0], [1], [2]]


def _poly_basis_1d(xi: np.ndarray, delta: float) -> np.ndarray:
    """Basis [1, xi/delta, (xi/delta)^2] evaluated at the bonds."""
    xn = np.asarray(xi, dtype=np.float64).reshape(-1) / delta
    return np.column_stack([np.ones(len(xn)), xn, xn * xn])


def build_local_system_bb_1d(
    xi_bonds: np.ndarray,
    volumes: np.ndarray,
    delta: float,
    d_star: np.ndarray,
    e_star: np.ndarray,
    tikhonov_rel: float = 1e-8,
    tikhonov_abs: float = 1e-12,
) -> tuple[float, np.ndarray] | None:
    """Local system of node i; returns ``(beta_i, coupling)`` or ``None`` (too few bonds)."""
    xi = np.asarray(xi_bonds, dtype=np.float64).reshape(-1)
    V = np.asarray(volumes, dtype=np.float64).reshape(-1)
    n_F = len(xi)
    if n_F < NP_1D:
        return None
    r = np.abs(xi)

    p = _poly_basis_1d(xi, delta)
    p_eff = bond_averaged(p)

    rows = [p]
    for k in DERIV_DEGREES:
        rows.append(((xi / r * p[:, k] * V) @ p_eff)[None, :])
    for k in ENERGY_DEGREES:
        rows.append(((0.25 * p[:, k] ** 2 * xi ** 2 / r ** 3 * V) @ p_eff)[None, :])
    ML = np.vstack(rows)

    targets = np.concatenate([d_star, e_star])
    return solve_local(ML, targets, n_F, tikhonov_rel, tikhonov_abs, _GROUPS)

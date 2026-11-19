"""Per-node local least-squares system of the 2-D nodal weights (bond-based PD).

For node i with bonds xi_j = x_j - x_i the stacked system has

  * n_F collocation rows  p(xi_j): complete quadratic basis in xi/delta (6 terms)
  * 6 derivative rows     sum_j (xi_a/r^3) dphi_k V_j p_eff,  a in {x,y}, dphi_k quadratic
  * 4 energy rows         sum_j 1/4 dphi_k^2 xi_l^2/r^3 V_j p_eff,  l in {x,y}, dphi_k linear

with p_eff = (p(xi) + p(0)) / 2 (the bond-averaged weight).  See
:mod:`perifit.core.targets_2d` for the choice of rows; the solve is the shared
routine of :mod:`perifit.core._lsq`.
"""
from __future__ import annotations

import numpy as np

from ._lsq import bond_averaged, solve_local
from .targets_2d import DERIV_TESTS, ENERGY_TESTS

NP_2D = 6                     # basis 1, xn, yn, xn^2, xn*yn, yn^2
_N_D = 2 * len(DERIV_TESTS)   # 6 derivative rows
_N_E = 2 * len(ENERGY_TESTS)  # 4 energy rows
#: one test degree per block -> one round-off group per block
_GROUPS = [list(range(_N_D)), list(range(_N_D, _N_D + _N_E))]


def _poly_basis_2d(xi: np.ndarray, delta: float) -> np.ndarray:
    """Complete quadratic basis in xi/delta evaluated at the bonds (N, 6)."""
    xn, yn = xi[:, 0] / delta, xi[:, 1] / delta
    return np.column_stack([np.ones(len(xi)), xn, yn, xn * xn, xn * yn, yn * yn])


def _monomial(xi: np.ndarray, delta: float, ex: tuple[int, int]) -> np.ndarray:
    return (xi[:, 0] / delta) ** ex[0] * (xi[:, 1] / delta) ** ex[1]


def build_local_system_bb_2d(
    xi_bonds: np.ndarray,
    volumes: np.ndarray,
    delta: float,
    d_star: np.ndarray,
    e_star: np.ndarray,
    tikhonov_rel: float = 1e-8,
    tikhonov_abs: float = 1e-12,
) -> tuple[float, np.ndarray] | None:
    """Local system of node i; returns ``(beta_i, coupling)`` or ``None`` (too few bonds)."""
    xi = np.asarray(xi_bonds, dtype=np.float64)
    V = np.asarray(volumes, dtype=np.float64).reshape(-1)
    n_F = len(xi)
    if n_F < NP_2D:
        return None
    r3 = np.einsum("ij,ij->i", xi, xi) ** 1.5

    p = _poly_basis_2d(xi, delta)
    p_eff = bond_averaged(p)

    rows = [p]
    for a in range(2):
        kernel = xi[:, a] / r3 * V
        for ex in DERIV_TESTS:
            rows.append(((kernel * _monomial(xi, delta, ex)) @ p_eff)[None, :])
    for l in range(2):
        kernel = 0.25 * xi[:, l] ** 2 / r3 * V
        for ex in ENERGY_TESTS:
            rows.append(((kernel * _monomial(xi, delta, ex) ** 2) @ p_eff)[None, :])
    ML = np.vstack(rows)

    targets = np.concatenate([d_star, e_star])
    return solve_local(ML, targets, n_F, tikhonov_rel, tikhonov_abs, _GROUPS)

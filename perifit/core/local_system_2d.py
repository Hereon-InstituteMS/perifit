"""Per-node local least-squares system for 2-D BB-PD.

For each node i with neighbour cloud {x_j} and bond vectors xi_j = x_j - x_i,
we solve a Tikhonov-regularised LS problem to determine the local weight
polynomial contribution.

Polynomial basis (NP_2D = 6):
    1, xn, yn, xn^2, xn*yn, yn^2     (xn = xi_x/delta, yn = xi_y/delta)

Test functions dphi (5 monomials, degree 1+2):
    x, y, x^2, xy, y^2

Operator rows:
  - 10 derivative: 2 (alpha) x 5 (dphi)
  - 10 energy:     2 (ell)   x 5 (dphi)
  Total: 20 operator rows
"""
from __future__ import annotations

import numpy as np

NP_2D = 6  # 2D polynomial basis dimension
N_DPHI_2D = 5  # number of test functions (degree 1+2 monomials)


def _poly_basis_2d(xi: np.ndarray, delta: float) -> np.ndarray:
    """Evaluate 2D polynomial basis at bond vectors (N, 2)."""
    xn = xi[:, 0] / delta
    yn = xi[:, 1] / delta
    return np.column_stack([
        np.ones(len(xi)),
        xn, yn,
        xn * xn, xn * yn, yn * yn,
    ])


def _dphi_all_2d(xi: np.ndarray) -> np.ndarray:
    """Returns the 5 monomials of degree 1+2 in xi, evaluated bond-wise.

    dphi = [x, y, x^2, xy, y^2] (unscaled).
    """
    xx, yy = xi[:, 0], xi[:, 1]
    return np.column_stack([
        xx, yy,
        xx * xx, xx * yy, yy * yy,
    ])


def _solve_local_2d(
    ML: np.ndarray,
    targets: np.ndarray,
    n_F: int,
    tikhonov_rel: float,
    tikhonov_abs: float,
) -> tuple[float, np.ndarray]:
    """Per-row normalised, Tikhonov-regularised pseudoinverse solve."""
    nr = np.linalg.norm(ML, axis=1)
    valid = nr > 1e-20
    ML_norm = ML.copy()
    ML_norm[valid] /= nr[valid, np.newaxis]

    MtM = ML_norm.T @ ML_norm
    max_diag = float(np.max(np.abs(np.diag(MtM))))
    lam = max(tikhonov_abs, tikhonov_rel * max_diag)
    MtM_reg = MtM + lam * np.eye(NP_2D)
    try:
        MtM_inv = np.linalg.solve(MtM_reg, np.eye(NP_2D))
    except np.linalg.LinAlgError:
        MtM_inv = np.linalg.pinv(MtM_reg)
    Mplus = MtM_inv @ ML_norm.T

    n_op = ML.shape[0] - n_F
    beta_i = 0.0
    for k_op in range(n_op):
        row = n_F + k_op
        if valid[row]:
            beta_i += Mplus[0, row] * targets[k_op] / nr[row]

    coupling = np.zeros(n_F)
    for jj in range(n_F):
        if valid[jj]:
            coupling[jj] = Mplus[0, jj] / nr[jj]

    return beta_i, coupling


def build_local_system_bb_2d(
    xi_bonds: np.ndarray,
    volumes: np.ndarray,
    delta: float,
    d_star: np.ndarray,
    e_star: np.ndarray,
    c_bond: float = 1.0,
    tikhonov_rel: float = 1e-8,
    tikhonov_abs: float = 1e-12,
) -> tuple[float, np.ndarray] | None:
    """Build 2D BB-PD local system (20 operator rows: 10 derivative + 10 energy).

    Parameters
    ----------
    xi_bonds : (n_F, 2) bond vectors.
    volumes : (n_F,) neighbour volumes.
    delta : horizon radius.
    d_star : (10,) derivative targets.
    e_star : (10,) energy targets.
    c_bond : micromodulus constant (energy rows are scaled by c_bond).
    """
    n_F = len(xi_bonds)
    if n_F < NP_2D:
        return None

    n_op = 2 * N_DPHI_2D + 2 * N_DPHI_2D  # 10 + 10 = 20
    n_rows = n_F + n_op
    ML = np.zeros((n_rows, NP_2D))

    r2 = np.einsum('ij,ij->i', xi_bonds, xi_bonds)
    r = np.sqrt(r2)
    r3 = r2 * r
    r3_safe = np.where(r3 > 1e-30, r3, 1.0)
    mask = r3 > 1e-30

    p_arr = _poly_basis_2d(xi_bonds, delta)
    # dphi = p(xi) - p(0), dropping the constant = p_arr[:, 1:] (delta-scaled)
    dph = p_arr[:, 1:]

    # p_eff = (p(xi) + p(0))/2; p(0) = [1, 0, 0, 0, 0, 0]
    p_eff = np.empty_like(p_arr)
    p_eff[:, 0] = 1.0
    p_eff[:, 1:] = 0.5 * p_arr[:, 1:]

    # Collocation rows
    ML[:n_F, :] = p_arr

    # Derivative rows: M_D[alpha*5 + k, :] = sum_j (xi_alpha/r^3) * dphi_k * V_j * p_eff
    for alpha in range(2):
        xi_alpha_over_r3 = np.where(mask, xi_bonds[:, alpha] / r3_safe, 0.0)
        for k_idx in range(N_DPHI_2D):
            row = n_F + alpha * N_DPHI_2D + k_idx
            ML[row, :] = (xi_alpha_over_r3 * dph[:, k_idx] * volumes) @ p_eff

    # Energy rows: M_E[ell*5 + k, :] = c_bond * 0.25 * dphi_k^2 * xi_ell^2 / r^3 * V_j * p_eff
    for ell in range(2):
        xi_ell_sq_over_r3 = np.where(mask, xi_bonds[:, ell] ** 2 / r3_safe, 0.0)
        for k_idx in range(N_DPHI_2D):
            row = n_F + 2 * N_DPHI_2D + ell * N_DPHI_2D + k_idx
            ML[row, :] = (c_bond * 0.25 * dph[:, k_idx] ** 2 * xi_ell_sq_over_r3 * volumes) @ p_eff

    targets = np.concatenate([d_star, c_bond * e_star])
    return _solve_local_2d(ML, targets, n_F, tikhonov_rel, tikhonov_abs)

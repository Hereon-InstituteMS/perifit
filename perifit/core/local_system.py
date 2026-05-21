"""Per-node local least-squares systems for 3-D BB and OSB PD.

For each node i with family {j} and bond vectors xi_j = x_j - x_i,
a Tikhonov-regularised LS problem is solved:

    minimise || M * a - b* ||^2  +  lambda * || a ||^2

The stacked matrix M has (n_F + 54) rows x 10 columns:
  - n_F collocation rows: enforce polynomial interpolation at neighbours
  - 27 derivative rows:   match full-ball derivative targets (3 dirs x 9 tests)
  - 27 energy rows:       match full-ball energy targets (3 lifts x 9 tests)

After per-row normalisation and Tikhonov-regularised solve, the function
returns:
  - beta_i:   right-hand side contribution for node i in the global system
  - coupling: off-diagonal coefficients relating w_i to neighbour weights

Polynomial basis (NP = 10, complete quadratic in xi/delta):
    1, xn, yn, zn, xn^2, xn*yn, xn*zn, yn^2, yn*zn, zn^2

Authors: Arman Shojaei & Alexander Hermann
         Helmholtz-Zentrum Hereon, Geesthacht, Germany.
"""
from __future__ import annotations

import numpy as np

NP = 10  # polynomial basis dimension


def _poly_basis(xi: np.ndarray, delta: float) -> np.ndarray:
    xn = xi[:, 0] / delta
    yn = xi[:, 1] / delta
    zn = xi[:, 2] / delta
    return np.column_stack([
        np.ones(len(xi)),
        xn, yn, zn,
        xn * xn, xn * yn, xn * zn,
        yn * yn, yn * zn, zn * zn,
    ])


def _dphi_all(xi: np.ndarray) -> np.ndarray:
    """Returns the 9 monomials of degree 1+2 in xi, evaluated bond-wise."""
    xx, yy, zz = xi[:, 0], xi[:, 1], xi[:, 2]
    return np.column_stack([
        xx, yy, zz,
        xx * xx, xx * yy, xx * zz,
        yy * yy, yy * zz, zz * zz,
    ])


def _solve_local(
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
    MtM_reg = MtM + lam * np.eye(NP)
    try:
        MtM_inv = np.linalg.solve(MtM_reg, np.eye(NP))
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


# ---------------------------------------------------------------------------
# BB-PD local system  (54 operator rows: 27 derivative + 27 energy)
# ---------------------------------------------------------------------------
def build_local_system_bb(
    xi_bonds: np.ndarray,
    volumes: np.ndarray,
    delta: float,
    d_star: np.ndarray,
    e_star: np.ndarray,
    tikhonov_rel: float = 1e-8,
    tikhonov_abs: float = 1e-12,
) -> tuple[float, np.ndarray] | None:
    n_F = len(xi_bonds)
    if n_F < NP:
        return None

    n_rows = n_F + 54
    ML = np.zeros((n_rows, NP))

    r2 = np.einsum('ij,ij->i', xi_bonds, xi_bonds)
    r = np.sqrt(r2)
    r3 = r2 * r
    r3_safe = np.where(r3 > 1e-30, r3, 1.0)
    mask = r3 > 1e-30

    p_arr = _poly_basis(xi_bonds, delta)
    dph = _dphi_all(xi_bonds)
    p_eff = np.empty_like(p_arr)
    p_eff[:, 0] = 1.0
    p_eff[:, 1:] = 0.5 * p_arr[:, 1:]

    ML[:n_F, :] = p_arr

    for alpha in range(3):
        xi_alpha_over_r3 = np.where(mask, xi_bonds[:, alpha] / r3_safe, 0.0)
        for k_idx in range(9):
            row = n_F + alpha * 9 + k_idx
            ML[row, :] = (xi_alpha_over_r3 * dph[:, k_idx] * volumes) @ p_eff

    for ell in range(3):
        xi_ell_sq_over_r3 = np.where(mask, xi_bonds[:, ell] ** 2 / r3_safe, 0.0)
        for k_idx in range(9):
            row = n_F + 27 + ell * 9 + k_idx
            ML[row, :] = (0.25 * dph[:, k_idx] ** 2 * xi_ell_sq_over_r3 * volumes) @ p_eff

    targets = np.concatenate([d_star, e_star])
    return _solve_local(ML, targets, n_F, tikhonov_rel, tikhonov_abs)


# ---------------------------------------------------------------------------
# OSB-PD local system  (54 operator rows: 27 derivative + 27 energy)
# ---------------------------------------------------------------------------
def build_local_system_osb(
    xi_bonds: np.ndarray,
    volumes: np.ndarray,
    delta: float,
    d_star: np.ndarray,
    e_star: np.ndarray,
    tikhonov_rel: float = 1e-8,
    tikhonov_abs: float = 1e-12,
) -> tuple[float, np.ndarray] | None:
    n_F = len(xi_bonds)
    if n_F < NP:
        return None

    n_rows = n_F + 54
    ML = np.zeros((n_rows, NP))

    r2 = np.einsum('ij,ij->i', xi_bonds, xi_bonds)
    r2_safe = np.where(r2 > 1e-30, r2, 1.0)
    mask = r2 > 1e-30

    p_arr = _poly_basis(xi_bonds, delta)
    dph = _dphi_all(xi_bonds)
    p_eff = np.empty_like(p_arr)
    p_eff[:, 0] = 1.0
    p_eff[:, 1:] = 0.5 * p_arr[:, 1:]

    ML[:n_F, :] = p_arr

    for alpha in range(3):
        xi_alpha_over_r2 = np.where(mask, xi_bonds[:, alpha] / r2_safe, 0.0)
        for k_idx in range(9):
            row = n_F + alpha * 9 + k_idx
            ML[row, :] = (xi_alpha_over_r2 * dph[:, k_idx] * volumes) @ p_eff

    for ell in range(3):
        xi_ell_sq_over_r2 = np.where(mask, xi_bonds[:, ell] ** 2 / r2_safe, 0.0)
        for k_idx in range(9):
            row = n_F + 27 + ell * 9 + k_idx
            ML[row, :] = (dph[:, k_idx] ** 2 * xi_ell_sq_over_r2 * volumes) @ p_eff

    targets = np.concatenate([d_star, e_star])
    return _solve_local(ML, targets, n_F, tikhonov_rel, tikhonov_abs)


# ---------------------------------------------------------------------------
# NOSB-PD local system  (27 operator rows: gradient numerator only)
# ---------------------------------------------------------------------------
def build_local_system_nosb(
    xi_bonds: np.ndarray,
    volumes: np.ndarray,
    delta: float,
    n_star: np.ndarray,
    tikhonov_rel: float = 1e-8,
    tikhonov_abs: float = 1e-12,
) -> tuple[float, np.ndarray] | None:
    n_F = len(xi_bonds)
    if n_F < NP:
        return None

    n_rows = n_F + 27
    ML = np.zeros((n_rows, NP))

    p_arr = _poly_basis(xi_bonds, delta)
    dph = _dphi_all(xi_bonds)
    p_eff = np.empty_like(p_arr)
    p_eff[:, 0] = 1.0
    p_eff[:, 1:] = 0.5 * p_arr[:, 1:]

    ML[:n_F, :] = p_arr

    for b in range(3):
        for k_idx in range(9):
            row = n_F + b * 9 + k_idx
            ML[row, :] = (xi_bonds[:, b] * dph[:, k_idx] * volumes) @ p_eff

    return _solve_local(ML, n_star, n_F, tikhonov_rel, tikhonov_abs)

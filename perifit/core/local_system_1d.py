"""Per-node local least-squares system for 1-D BB-PD.

For each node i with neighbour cloud {x_j} and bond vectors xi_j = x_j - x_i,
a least-squares problem is solved to determine the local weight polynomial
contribution.

Polynomial basis (NP_1D = 3):
    1, xn, xn^2                      (xn = xi/delta)

Test functions dphi (2 monomials, degree 1+2):
    x, x^2

Operator rows:
  - 2 derivative: 1 (alpha) x 2 (dphi)
  - 2 energy:     1 (ell)   x 2 (dphi)
  Total: 4 operator rows

Row weighting
-------------
``row_norm="none"`` (default) leaves the stacked matrix at its natural block
magnitudes and inverts it with a Moore-Penrose pseudo-inverse.  This is the
combination that reproduces Figures 6 and 7 of the paper.  Because the blocks
are not rescaled, their relative magnitudes are part of the calibration: the
kernels and delta-scaled tests of ``targets_1d`` must be used unchanged.

``row_norm="per-row"`` reproduces the convention of the 2-D and 3-D modules,
where every row is scaled to unit norm before a Tikhonov-regularised solve.  It
weights the collocation and operator constraints equally, which smooths the
boundary layer: the interior weights are unchanged but the amplification at a
truncated horizon is roughly 25-30% smaller.
"""
from __future__ import annotations

import numpy as np

NP_1D = 3  # 1D polynomial basis dimension
N_DPHI_1D = 2  # number of test functions (degree 1+2 monomials)


def _poly_basis_1d(xi: np.ndarray, delta: float) -> np.ndarray:
    """Evaluate the 1D polynomial basis at bond vectors (N,) or (N, 1)."""
    xn = np.asarray(xi, dtype=np.float64).reshape(-1) / delta
    return np.column_stack([np.ones(len(xn)), xn, xn * xn])


def _solve_local_1d(
    ML: np.ndarray,
    targets: np.ndarray,
    n_F: int,
    row_norm: str,
    tikhonov_rel: float,
    tikhonov_abs: float,
) -> tuple[float, np.ndarray]:
    """Pseudo-inverse solve of the stacked local system; returns the row of eq. (68)."""
    if row_norm == "per-row":
        nr = np.linalg.norm(ML, axis=1)
    else:
        nr = np.ones(ML.shape[0])
    valid = nr > 1e-20
    ML_norm = ML.copy()
    ML_norm[valid] /= nr[valid, np.newaxis]
    ML_norm[~valid] = 0.0

    if row_norm == "per-row":
        MtM = ML_norm.T @ ML_norm
        max_diag = float(np.max(np.abs(np.diag(MtM))))
        lam = max(tikhonov_abs, tikhonov_rel * max_diag)
        try:
            MtM_inv = np.linalg.solve(MtM + lam * np.eye(NP_1D), np.eye(NP_1D))
        except np.linalg.LinAlgError:
            MtM_inv = np.linalg.pinv(MtM + lam * np.eye(NP_1D))
        Mplus = MtM_inv @ ML_norm.T
    else:
        Mplus = np.linalg.pinv(ML_norm, rcond=100 * np.finfo(float).eps)

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


def build_local_system_bb_1d(
    xi_bonds: np.ndarray,
    volumes: np.ndarray,
    delta: float,
    d_star: np.ndarray,
    e_star: np.ndarray,
    row_norm: str = "none",
    tikhonov_rel: float = 1e-8,
    tikhonov_abs: float = 1e-12,
) -> tuple[float, np.ndarray] | None:
    """Build the 1D BB-PD local system (4 operator rows: 2 derivative + 2 energy).

    Parameters
    ----------
    xi_bonds : (n_F,) or (n_F, 1) bond vectors.
    volumes : (n_F,) neighbour volumes.
    delta : horizon radius.
    d_star : (2,) derivative targets.
    e_star : (2,) energy targets.
    row_norm : "none" (paper convention) or "per-row" (2-D/3-D convention).
    """
    xi = np.asarray(xi_bonds, dtype=np.float64).reshape(-1)
    volumes = np.asarray(volumes, dtype=np.float64).reshape(-1)
    n_F = len(xi)
    if n_F < NP_1D:
        return None

    n_op = 2 * N_DPHI_1D  # 2 + 2 = 4
    ML = np.zeros((n_F + n_op, NP_1D))

    r = np.abs(xi)
    mask = r > 1e-30
    r_safe = np.where(mask, r, 1.0)
    r3_safe = r_safe ** 3

    p_arr = _poly_basis_1d(xi, delta)
    # dphi = p(xi) - p(0), dropping the constant = p_arr[:, 1:] (delta-scaled)
    dph = p_arr[:, 1:]

    # p_eff = (p(xi) + p(0))/2; p(0) = [1, 0, 0]
    p_eff = np.empty_like(p_arr)
    p_eff[:, 0] = 1.0
    p_eff[:, 1:] = 0.5 * p_arr[:, 1:]

    # Collocation rows
    ML[:n_F, :] = p_arr

    # Derivative rows: M_D[k, :] = sum_j (xi/|xi|) * dphi_k * V_j * p_eff
    xi_over_r = np.where(mask, xi / r_safe, 0.0)
    for k_idx in range(N_DPHI_1D):
        ML[n_F + k_idx, :] = (xi_over_r * dph[:, k_idx] * volumes) @ p_eff

    # Energy rows: M_E[k, :] = 0.25 * dphi_k^2 * xi^2 / |xi|^3 * V_j * p_eff
    xi_sq_over_r3 = np.where(mask, xi ** 2 / r3_safe, 0.0)
    for k_idx in range(N_DPHI_1D):
        ML[n_F + N_DPHI_1D + k_idx, :] = (
            0.25 * dph[:, k_idx] ** 2 * xi_sq_over_r3 * volumes) @ p_eff

    targets = np.concatenate([d_star, e_star])
    return _solve_local_1d(ML, targets, n_F, row_norm, tikhonov_rel, tikhonov_abs)

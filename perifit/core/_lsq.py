"""Shared local least-squares solve of the nodal-weight method (eqs. 64-68).

Every node ``i`` stacks a matrix

    M = [ collocation rows  p(xi_j)            (n_F rows, one per neighbour)
          operator rows     derivative/energy  (n_op rows) ]

with right-hand side ``[w_neighbours ; targets]`` and solves it in the
least-squares sense with Tikhonov regularisation.  Only the first row of the
generalised inverse is needed, because the weight at the node itself is the
reconstruction evaluated at ``xi = 0`` and ``p(0) = (1, 0, ..., 0)``:

    w_i = sum_j coupling_j * w_j + beta_i

``solve_local`` returns ``(beta_i, coupling)``; the global assembly then
collects ``w_i - sum_j coupling_ij w_j = beta_i`` for all nodes.
"""
from __future__ import annotations

import numpy as np


def solve_local(
    ML: np.ndarray,
    targets: np.ndarray,
    n_F: int,
    tikhonov_rel: float = 1e-8,
    tikhonov_abs: float = 1e-12,
) -> tuple[float, np.ndarray]:
    """Per-row normalised, Tikhonov-regularised solve; returns ``(beta_i, coupling)``.

    Parameters
    ----------
    ML : (n_F + n_op, n_p) stacked collocation and operator rows.
    targets : (n_op,) analytic full-horizon values of the operator rows.
    n_F : number of collocation rows (= number of neighbours).
    tikhonov_rel, tikhonov_abs : regularisation ``lam = max(abs, rel * max diag(M^T M))``.
    """
    n_p = ML.shape[1]
    nr = np.linalg.norm(ML, axis=1)
    valid = nr > 1e-20
    ML_norm = np.zeros_like(ML)
    ML_norm[valid] = ML[valid] / nr[valid, np.newaxis]

    MtM = ML_norm.T @ ML_norm
    lam = max(tikhonov_abs, tikhonov_rel * float(np.max(np.abs(np.diag(MtM)))))
    try:
        MtM_inv = np.linalg.solve(MtM + lam * np.eye(n_p), np.eye(n_p))
    except np.linalg.LinAlgError:
        MtM_inv = np.linalg.pinv(MtM + lam * np.eye(n_p))
    row0 = MtM_inv[0] @ ML_norm.T          # first row of the generalised inverse

    scale = np.where(valid, 1.0 / np.where(valid, nr, 1.0), 0.0)
    beta_i = float(np.sum(row0[n_F:] * scale[n_F:] * targets))
    coupling = row0[:n_F] * scale[:n_F]
    return beta_i, coupling

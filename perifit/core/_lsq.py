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

Round-off rows
--------------
On a symmetric (e.g. full, regular) horizon several operator rows vanish
analytically but come out as ~1e-17 noise.  Per-row normalisation would blow
such noise up to unit rows and make the weights depend on the floating-point
summation order.  Rows are therefore compared only with rows of the *same
group* (same operator block and same test-function degree, so they scale
identically with the length unit) and dropped when smaller than
``ROUNDOFF_RTOL`` times the largest row of that group.  No absolute threshold
is used anywhere, which keeps the weights invariant under a change of units.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

#: Operator rows smaller than this fraction of the largest row of their group
#: are treated as exact zeros (pure round-off).
ROUNDOFF_RTOL = 1e-10


def solve_local(
    ML: np.ndarray,
    targets: np.ndarray,
    n_F: int,
    tikhonov_rel: float = 1e-8,
    tikhonov_abs: float = 1e-12,
    row_groups: Sequence[Sequence[int]] | None = None,
) -> tuple[float, np.ndarray]:
    """Per-row normalised, Tikhonov-regularised solve; returns ``(beta_i, coupling)``.

    Parameters
    ----------
    ML : (n_F + n_op, n_p) stacked collocation and operator rows.
    targets : (n_op,) analytic full-horizon values of the operator rows.
    n_F : number of collocation rows (= number of neighbours).
    tikhonov_rel, tikhonov_abs : regularisation ``lam = max(abs, rel * max diag(M^T M))``
        of the *normalised* system (dimensionless, hence unit independent).
    row_groups : index sets (into the operator rows) that scale alike; the
        round-off cutoff is applied within each group.  ``None`` = one group.
    """
    n_p = ML.shape[1]
    n_op = ML.shape[0] - n_F
    nr = np.linalg.norm(ML, axis=1)

    valid = np.ones(len(nr), dtype=bool)
    valid[:n_F] = nr[:n_F] > 0.0           # collocation rows contain the constant 1
    groups = row_groups if row_groups is not None else [range(n_op)]
    for g in groups:
        idx = n_F + np.asarray(list(g), dtype=np.int64)
        if idx.size:
            gmax = nr[idx].max()
            valid[idx] = nr[idx] > ROUNDOFF_RTOL * gmax if gmax > 0.0 else False

    scale = np.zeros(len(nr))
    scale[valid] = 1.0 / nr[valid]
    ML_norm = ML * scale[:, np.newaxis]

    MtM = ML_norm.T @ ML_norm
    lam = max(tikhonov_abs, tikhonov_rel * float(np.max(np.abs(np.diag(MtM)))))
    try:
        MtM_inv = np.linalg.solve(MtM + lam * np.eye(n_p), np.eye(n_p))
    except np.linalg.LinAlgError:
        MtM_inv = np.linalg.pinv(MtM + lam * np.eye(n_p))
    row0 = MtM_inv[0] @ ML_norm.T          # first row of the generalised inverse

    beta_i = float(np.sum(row0[n_F:] * scale[n_F:] * targets))
    coupling = row0[:n_F] * scale[:n_F]
    return beta_i, coupling

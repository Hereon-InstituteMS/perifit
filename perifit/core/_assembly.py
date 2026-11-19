"""Global assembly and solve of the nodal-weight system ``K_w w = F_w`` (eq. 69).

Each node contributes one row  ``w_i - sum_j coupling_ij w_j = beta_i``.  The
rows are collected as COO triplets (vectorised, no per-entry sparse updates)
and the sparse non-symmetric system is solved with BiCGSTAB from ``w = 1``.

:func:`nodal_weights_from_local` is the driver shared by the 1-D, 2-D and 3-D
engines: it only needs the full-horizon targets and the local-system builder.
"""
from __future__ import annotations

import warnings
from typing import Callable, Optional, Sequence

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import bicgstab

LocalSolver = Callable[[int, np.ndarray], Optional[tuple]]


def nodal_weights_from_local(
    points: np.ndarray,
    volumes: np.ndarray,
    horizon: float,
    build: Callable[..., Optional[tuple]],
    targets: tuple[np.ndarray, ...],
    families: Sequence[Sequence[int]] | None = None,
    tol: float = 1e-10,
    max_iter: int = 5000,
    tikhonov_rel: float = 1e-8,
    tikhonov_abs: float = 1e-12,
    verbose: bool = False,
    label: str = "",
) -> np.ndarray:
    """Nodal weights from a local-system builder.

    ``build(xi_bonds, neighbour_volumes, horizon, *targets, tikhonov_rel,
    tikhonov_abs)`` returns ``(beta_i, coupling)`` or ``None`` for one node.
    """
    from .families import build_families

    if families is None:
        families = build_families(points, horizon)
    if verbose:
        print(f"perifit nodal weights  {label}  N={len(points):,d}  delta={horizon:.4g}")

    def local(i: int, nbrs: np.ndarray):
        return build(points[nbrs] - points[i], volumes[nbrs], horizon, *targets,
                     tikhonov_rel, tikhonov_abs)

    return solve_weight_system(len(points), families, local, tol=tol, max_iter=max_iter,
                               verbose=verbose)


def solve_weight_system(
    n_nodes: int,
    families: Sequence[Sequence[int]],
    local_solver: LocalSolver,
    tol: float = 1e-10,
    max_iter: int = 5000,
    verbose: bool = False,
) -> np.ndarray:
    """Assemble ``K_w w = F_w`` from per-node local solves and return ``w``.

    ``local_solver(i, nbrs)`` returns ``(beta_i, coupling)`` or ``None`` for a
    degenerate node (too few neighbours), whose weight is then fixed to 1.
    """
    rows, cols, vals = [], [], []
    rhs = np.ones(n_nodes)
    n_degenerate = 0
    for i in range(n_nodes):
        nbrs = np.asarray(families[i], dtype=np.int64)
        rows.append(np.array([i])); cols.append(np.array([i])); vals.append(np.array([1.0]))
        result = local_solver(i, nbrs) if len(nbrs) else None
        if result is None:
            n_degenerate += 1
            continue
        beta_i, coupling = result
        rows.append(np.full(len(nbrs), i)); cols.append(nbrs); vals.append(-np.asarray(coupling))
        rhs[i] = beta_i

    K = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                   shape=(n_nodes, n_nodes)).tocsr()
    if verbose:
        print(f"  assembled {n_nodes:,d} rows ({n_degenerate} degenerate nodes set to w=1)")

    w, info = bicgstab(K, rhs, x0=np.ones(n_nodes), rtol=tol, maxiter=max_iter)
    if info < 0:
        raise RuntimeError(f"BiCGSTAB failed with info={info}.")
    if info > 0:
        warnings.warn(f"BiCGSTAB did not converge after {info} iterations.",
                      RuntimeWarning, stacklevel=3)
    if verbose:
        print(f"  weights: min={w.min():.4f}  max={w.max():.4f}  mean={w.mean():.4f}")
    return w

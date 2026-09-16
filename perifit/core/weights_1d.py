"""1-D global weight assembly: solve (I - A) w = b with BiCGSTAB."""
from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import bicgstab
from scipy.spatial import KDTree

from .local_system_1d import build_local_system_bb_1d, NP_1D
from .targets_1d import build_targets_bb_1d


def build_families_1d(
    coords: np.ndarray, horizon: float, tol: float = 1e-12
) -> list[list[int]]:
    """KDTree-based 1D neighbour search; cutoff = ``horizon + tol``."""
    pts = np.asarray(coords, dtype=np.float64).reshape(-1, 1)
    tree = KDTree(pts)
    pairs = tree.query_ball_tree(tree, r=horizon + tol)
    return [[j for j in pairs[i] if j != i] for i in range(len(pts))]


def compute_weights_1d(
    coords: np.ndarray,
    volumes: np.ndarray,
    horizon: float,
    families: Sequence[Sequence[int]] | None = None,
    row_norm: str = "none",
    tol: float = 1e-10,
    max_iter: int = 5000,
    tikhonov_rel: float = 1e-8,
    tikhonov_abs: float = 1e-12,
    verbose: bool = False,
) -> np.ndarray:
    """Compute surface-correction nodal weights for a 1-D BB-PD discretisation.

    Parameters
    ----------
    coords : (N,) or (N, 1) array of nodal positions.
    volumes : (N,) cell volumes (spacing times cross-section).
    horizon : peridynamic horizon delta.
    families : optional precomputed neighbour lists.
    row_norm : "none" reproduces Figures 6-7 of the paper; "per-row" applies the
        unit-row scaling used by the 2-D and 3-D modules (see local_system_1d).
    tol, max_iter : BiCGSTAB controls.
    tikhonov_rel, tikhonov_abs : local-LS regularisation (``row_norm="per-row"``).
    verbose : print progress.

    Returns
    -------
    w : (N,) nodal influence weights.
        Apply as bond weight (w[i] + w[j]) / 2 in the PD solver.

    Notes
    -----
    No partial-volume factor beta must be applied, neither here nor in the
    solver that consumes the weights: the calibration targets are full-horizon
    integrals, so the weights already absorb the midpoint-quadrature error.  In
    1-D on a uniform grid that error is exactly (m+1)/m with m = delta/dx, which
    is why the interior weights tend to m/(m+1) rather than to 1.
    """
    if row_norm not in ("none", "per-row"):
        raise ValueError(f"row_norm must be 'none' or 'per-row', got {row_norm!r}")

    coords = np.asarray(coords, dtype=np.float64).reshape(-1)
    volumes = np.asarray(volumes, dtype=np.float64)
    N = len(coords)
    if volumes.shape != (N,):
        raise ValueError(f"volumes must be shape ({N},), got {volumes.shape}")
    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}")

    if families is None:
        if verbose:
            print(f"perifit BB-1D  N={N:,d}  delta={horizon:.4g}")
            print("  Building neighbour families ...", end=" ", flush=True)
        families = build_families_1d(coords, horizon)
        if verbose:
            avg = float(np.mean([len(f) for f in families]))
            print(f"done.  avg n_F={avg:.1f}")

    d_star, e_star = build_targets_bb_1d(horizon)

    if verbose:
        print("  Assembling global weight system ...", end=" ", flush=True)

    A = lil_matrix((N, N), dtype=np.float64)
    b = np.zeros(N, dtype=np.float64)
    n_deg = 0

    for i in range(N):
        nbrs = list(families[i])
        if not nbrs:
            A[i, i] = 1.0
            b[i] = 1.0
            n_deg += 1
            continue
        xi_bonds = coords[nbrs] - coords[i]
        result = build_local_system_bb_1d(
            xi_bonds, volumes[nbrs], horizon, d_star, e_star,
            row_norm, tikhonov_rel, tikhonov_abs)
        if result is None:
            A[i, i] = 1.0
            b[i] = 1.0
            n_deg += 1
            continue
        beta_i, coupling = result
        A[i, i] = 1.0
        for jj, j in enumerate(nbrs):
            a_ij = coupling[jj]
            if abs(a_ij) > 1e-15:
                A[i, j] -= a_ij
        b[i] = beta_i

    if verbose:
        print(f"done.  ({n_deg} degenerate nodes set to w=1)")
        print("  Solving global system (BiCGSTAB) ...", end=" ", flush=True)

    w0 = np.ones(N, dtype=np.float64)
    w, info = bicgstab(A.tocsr(), b, x0=w0, rtol=tol, maxiter=max_iter)

    if info < 0:
        raise RuntimeError(f"BiCGSTAB failed with info={info}.")

    if verbose:
        if info == 0:
            print("converged.")
        elif info > 0:
            print(f"info={info} (no convergence)")
        else:
            print(f"info={info} (illegal input)")
        print(f"  Weights: min={w.min():.4f}  max={w.max():.4f}  mean={w.mean():.4f}")

    if info > 0:
        warnings.warn(
            f"BiCGSTAB did not converge after {info} iterations.",
            RuntimeWarning, stacklevel=2)

    return w

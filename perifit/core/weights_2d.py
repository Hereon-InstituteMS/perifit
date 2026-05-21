"""2-D global weight assembly: solve (I - A) w = b with BiCGSTAB."""
from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import bicgstab
from scipy.spatial import KDTree

from .local_system_2d import build_local_system_bb_2d, NP_2D
from .targets_2d import build_targets_bb_2d


def build_families_2d(
    coords: np.ndarray, horizon: float, tol: float = 1e-12
) -> list[list[int]]:
    """KDTree-based 2D neighbour search; cutoff = ``horizon + tol``."""
    tree = KDTree(coords)
    pairs = tree.query_ball_tree(tree, r=horizon + tol)
    return [[j for j in pairs[i] if j != i] for i in range(len(coords))]


def compute_weights_2d(
    coords: np.ndarray,
    volumes: np.ndarray,
    horizon: float,
    families: Sequence[Sequence[int]] | None = None,
    tol: float = 1e-10,
    max_iter: int = 5000,
    tikhonov_rel: float = 1e-8,
    tikhonov_abs: float = 1e-12,
    verbose: bool = False,
) -> np.ndarray:
    """Compute surface-correction nodal weights for a 2-D BB-PD discretisation.

    Parameters
    ----------
    coords : (N, 2) array of nodal positions.
    volumes : (N,) cell volumes (areas times thickness).
    horizon : peridynamic horizon delta.
    families : optional precomputed neighbour lists.
    tol, max_iter : BiCGSTAB controls.
    tikhonov_rel, tikhonov_abs : local-LS regularisation.
    verbose : print progress.

    Returns
    -------
    w : (N,) nodal influence weights.
    """
    coords = np.asarray(coords, dtype=np.float64)
    volumes = np.asarray(volumes, dtype=np.float64)
    N = len(coords)

    if families is None:
        if verbose:
            print(f"perifit BB-2D  N={N:,d}  delta={horizon:.4g}")
            print("  Building neighbour families ...", end=" ", flush=True)
        families = build_families_2d(coords, horizon)
        if verbose:
            avg = float(np.mean([len(f) for f in families]))
            print(f"done.  avg n_F={avg:.1f}")

    d_star, e_star = build_targets_bb_2d(horizon)

    if verbose:
        print("  Assembling global weight system ...", end=" ", flush=True)

    A = lil_matrix((N, N), dtype=np.float64)
    b = np.zeros(N, dtype=np.float64)
    n_deg = 0

    for i in range(N):
        nbrs = families[i]
        if not nbrs:
            A[i, i] = 1.0
            b[i] = 1.0
            n_deg += 1
            continue
        xi_bonds = coords[nbrs] - coords[i]
        vol_nbrs = volumes[nbrs]
        result = build_local_system_bb_2d(
            xi_bonds, vol_nbrs, horizon, d_star, e_star,
            tikhonov_rel, tikhonov_abs)
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

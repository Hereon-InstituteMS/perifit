"""Global weight assembly and public API.

Assembles the sparse global system K_w * w = F_w from per-node local
least-squares relations, then solves with BiCGSTAB.

Authors: Arman Shojaei & Alexander Hermann
         Helmholtz-Zentrum Hereon, Geesthacht, Germany.
"""
from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import bicgstab
from scipy.spatial import KDTree

from .local_system import (
    build_local_system_bb,
    build_local_system_osb,
    build_local_system_nosb,
)
from .targets import build_targets_bb, build_targets_osb, build_targets_nosb


_MODELS = ("bb", "osb")


def build_families(
    coords: np.ndarray, horizon: float, tol: float = 1e-12
) -> list[list[int]]:
    """KDTree-based neighbour search; cutoff = ``horizon + tol``."""
    tree = KDTree(coords)
    pairs = tree.query_ball_tree(tree, r=horizon + tol)
    return [[j for j in pairs[i] if j != i] for i in range(len(coords))]


def compute_weights(
    coords: np.ndarray,
    volumes: np.ndarray,
    horizon: float,
    model: str = "bb",
    families: Sequence[Sequence[int]] | None = None,
    tol: float = 1e-10,
    max_iter: int = 5000,
    tikhonov_rel: float = 1e-8,
    tikhonov_abs: float = 1e-12,
    verbose: bool = False,
) -> np.ndarray:
    """Compute surface-correction nodal weights for a 3-D PD discretisation.

    Parameters
    ----------
    coords : (N, 3) array of nodal positions.
    volumes : (N,) cell volumes.
    horizon : peridynamic horizon delta.
    model : "bb" (bond-based, kernel q=3) or "osb" (state-based LPS, kernel q=2).
    families : optional precomputed neighbour lists (list of lists of int).
    tol, max_iter : BiCGSTAB convergence controls.
    tikhonov_rel, tikhonov_abs : local-LS regularisation parameters.
    verbose : print progress information.

    Returns
    -------
    w : (N,) nodal influence weights.
        Apply as bond weight (w[i] + w[j]) / 2 in the PD solver.

    Notes
    -----
    Material constants (micro-modulus c for BB, or a and b for OSB) are NOT
    needed: they cancel after per-row normalisation in the local LS.  The
    weights are purely geometric quantities depending only on the node
    positions, volumes, and horizon.
    """
    model = model.lower()
    if model not in _MODELS:
        raise ValueError(f"model must be one of {_MODELS}, got {model!r}")

    coords = np.asarray(coords, dtype=np.float64)
    volumes = np.asarray(volumes, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"coords must be shape (N, 3), got {coords.shape}")
    N = len(coords)
    if volumes.shape != (N,):
        raise ValueError(f"volumes must be shape ({N},), got {volumes.shape}")
    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}")

    if families is None:
        if verbose:
            print(f"perifit {model.upper()}  N={N:,d}  delta={horizon:.4g}")
            print("  Building neighbour families ...", end=" ", flush=True)
        families = build_families(coords, horizon)
        if verbose:
            avg = float(np.mean([len(f) for f in families]))
            print(f"done.  avg n_F={avg:.1f}")

    if model == "bb":
        d_star, e_star = build_targets_bb(horizon)
        local_call = lambda xi, V: build_local_system_bb(  # noqa: E731
            xi, V, horizon, d_star, e_star, tikhonov_rel, tikhonov_abs)
    elif model == "osb":
        d_star, e_star = build_targets_osb(horizon)
        local_call = lambda xi, V: build_local_system_osb(  # noqa: E731
            xi, V, horizon, d_star, e_star, tikhonov_rel, tikhonov_abs)
    else:  # nosb
        n_star = build_targets_nosb(horizon)
        local_call = lambda xi, V: build_local_system_nosb(  # noqa: E731
            xi, V, horizon, n_star, tikhonov_rel, tikhonov_abs)

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
        result = local_call(xi_bonds, vol_nbrs)
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

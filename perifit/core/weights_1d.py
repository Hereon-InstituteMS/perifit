"""1-D global weight assembly: solve (I - A) w = b with BiCGSTAB."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from ._assembly import solve_weight_system
from .families import build_families
from .local_system_1d import build_local_system_bb_1d
from .targets_1d import build_targets_bb_1d


def build_families_1d(
    coords: np.ndarray, horizon: float, tol: float = 1e-12
) -> list[list[int]]:
    """Per-node neighbour lists within ``horizon + tol`` (alias of :func:`build_families`)."""
    return build_families(np.asarray(coords, dtype=np.float64).reshape(-1), horizon, tol)


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

    def local(i: int, nbrs: np.ndarray):
        return build_local_system_bb_1d(coords[nbrs] - coords[i], volumes[nbrs], horizon,
                                        d_star, e_star, row_norm, tikhonov_rel, tikhonov_abs)

    w = solve_weight_system(N, families, local, tol=tol, max_iter=max_iter, verbose=verbose)
    return w

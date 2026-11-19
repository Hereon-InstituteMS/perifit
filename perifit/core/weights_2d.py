"""2-D nodal weights (bond-based PD): per-node fits assembled into ``K_w w = F_w``."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from ._assembly import solve_weight_system
from .families import build_families
from .local_system_2d import build_local_system_bb_2d
from .targets_2d import build_targets_bb_2d

__all__ = ["compute_weights_2d", "build_families_2d"]


def build_families_2d(coords: np.ndarray, horizon: float, tol: float = 1e-12) -> list[list[int]]:
    """Per-node neighbour lists within ``horizon + tol`` (alias of :func:`build_families`)."""
    return build_families(coords, horizon, tol)


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
    """Nodal surface-correction weights for a 2-D bond-based PD discretisation.

    Returns ``w`` of shape (N,); a bond (i, j) is scaled by ``(w[i] + w[j]) / 2``.
    """
    coords = np.asarray(coords, dtype=np.float64)
    volumes = np.asarray(volumes, dtype=np.float64)
    N = len(coords)
    if families is None:
        families = build_families(coords, horizon)
    if verbose:
        print(f"perifit nodal weights  2-D BB  N={N:,d}  delta={horizon:.4g}")
    d_star, e_star = build_targets_bb_2d(horizon)

    def local(i: int, nbrs: np.ndarray):
        return build_local_system_bb_2d(coords[nbrs] - coords[i], volumes[nbrs], horizon,
                                        d_star, e_star, tikhonov_rel, tikhonov_abs)

    return solve_weight_system(N, families, local, tol=tol, max_iter=max_iter, verbose=verbose)

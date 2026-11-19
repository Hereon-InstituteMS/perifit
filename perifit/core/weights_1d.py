"""1-D nodal weights (bond-based PD).

On a uniform grid the interior weights equal ``m / (m + 1)`` (m = delta/dx),
which exactly removes the midpoint-quadrature error of the bulk stiffness.
No partial-volume factor (beta) must be used together with these weights.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from ._assembly import nodal_weights_from_local
from .families import check_inputs
from .local_system_1d import build_local_system_bb_1d
from .targets_1d import build_targets_bb_1d

__all__ = ["compute_weights_1d"]


def compute_weights_1d(
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
    """Nodal weights ``w`` (N,) for a 1-D grid; a bond (i, j) is scaled by ``(w[i] + w[j]) / 2``."""
    pts, vol = check_inputs(coords, volumes, horizon, dims=(1,))
    return nodal_weights_from_local(pts, vol, horizon, build_local_system_bb_1d,
                                    build_targets_bb_1d(horizon), families, tol, max_iter,
                                    tikhonov_rel, tikhonov_abs, verbose, "1-D BB")

"""2-D nodal weights (bond-based PD); see :mod:`perifit.core.targets_2d` for the fitted rows."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from ._assembly import nodal_weights_from_local
from .families import check_inputs
from .local_system_2d import build_local_system_bb_2d
from .targets_2d import build_targets_bb_2d

__all__ = ["compute_weights_2d"]


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
    """Nodal weights ``w`` (N,) for a 2-D point cloud; a bond (i, j) is scaled by ``(w[i] + w[j]) / 2``.

    Validated for horizon ratios m = delta/dx of 3-4 (see the README).
    """
    pts, vol = check_inputs(coords, volumes, horizon, dims=(2,))
    return nodal_weights_from_local(pts, vol, horizon, build_local_system_bb_2d,
                                    build_targets_bb_2d(horizon), families, tol, max_iter,
                                    tikhonov_rel, tikhonov_abs, verbose, "2-D BB")

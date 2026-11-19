"""3-D nodal weights: per-node local fits assembled into ``K_w w = F_w``.

This is the 3-D engine behind :func:`perifit.nodal_weights`.  Each node solves
a small least-squares problem (:mod:`perifit.core.local_system`); the coupled
rows are assembled and solved in :mod:`perifit.core._assembly`.

Authors: Arman Shojaei & Alexander Hermann
         Helmholtz-Zentrum Hereon, Geesthacht, Germany.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from ._assembly import nodal_weights_from_local
from .families import build_families, check_inputs
from .local_system import build_local_system_bb, build_local_system_osb
from .targets import build_targets_bb, build_targets_osb

_MODELS = ("bb", "osb")

__all__ = ["compute_weights", "build_families"]


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
    """Nodal surface-correction weights for a 3-D PD discretisation.

    Parameters
    ----------
    coords : (N, 3) nodal positions.
    volumes : (N,) nodal volumes.
    horizon : peridynamic horizon delta.
    model : ``"bb"`` (bond-based) or ``"osb"`` (ordinary state-based / LPS).
    families : optional neighbour lists; built with :func:`build_families` if omitted.
    tol, max_iter : BiCGSTAB controls for the global system.
    tikhonov_rel, tikhonov_abs : local least-squares regularisation.
    verbose : print progress.

    Returns
    -------
    w : (N,) nodal weights; a bond (i, j) is scaled by ``(w[i] + w[j]) / 2``.

    The weights are purely geometric: material constants cancel after the
    per-row normalisation of the local systems.
    """
    model = model.lower()
    if model not in _MODELS:
        raise ValueError(f"model must be one of {_MODELS}, got {model!r}")
    pts, vol = check_inputs(coords, volumes, horizon, dims=(3,))
    if model == "bb":
        build, targets = build_local_system_bb, build_targets_bb(horizon)
    else:
        build, targets = build_local_system_osb, build_targets_osb(horizon)
    return nodal_weights_from_local(pts, vol, horizon, build, targets, families, tol, max_iter,
                                    tikhonov_rel, tikhonov_abs, verbose, f"3-D {model.upper()}")

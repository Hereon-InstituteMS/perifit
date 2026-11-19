"""Public API: nodal weights, bond weights and the surface-correction front door.

perifit offers two ways to correct the surface effect of peridynamics:

* **nodal weights** -- one weight ``w_i`` per node; a bond (i, j) is scaled by
  ``(w_i + w_j) / 2``.  Compact (N numbers), and accepted by solvers that
  only take per-node fields (Peridigm, PeriLab).  1-D/2-D/3-D bond-based PD
  and 3-D ordinary state-based (LPS) PD.
* **bond weights** -- one weight ``beta_b`` per bond.  More accurate and
  robust (it also corrects the anisotropy of regular grids), one number per
  bond; breaking a bond is just ``beta_b = 0``.  1-D/2-D/3-D bond-based PD.

Both are purely geometric (no material constants) and are computed once, as a
pre-processing step.  :func:`surface_correction` returns either of them in the
same form -- one *bond multiplier* per bond -- so a solver can use both
without changes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .core.bond_weights import BondWeightInfo, compute_bond_weights
from .core.families import build_bonds, families_from_bonds
from .core.weights import compute_weights as _weights_3d
from .core.weights_1d import compute_weights_1d as _weights_1d
from .core.weights_2d import compute_weights_2d as _weights_2d

METHODS = ("nodal", "bond")

_METHOD_HELP = (
    "choose the surface correction explicitly with method='nodal' or method='bond':\n"
    "  'nodal' : one weight per node, a bond is scaled by (w_i + w_j)/2 -- compact, works\n"
    "            with per-node solvers (Peridigm, PeriLab); also 3-D state-based PD\n"
    "  'bond'  : one weight per bond -- more accurate and robust, one number per bond"
)


@dataclass(frozen=True)
class SurfaceCorrection:
    """Result of :func:`surface_correction`.

    Attributes
    ----------
    method : ``"nodal"`` or ``"bond"``.
    bonds : (Nb, 2) int bond list, ``i < j``, lexicographically sorted.
    factor : (Nb,) bond multipliers -- multiply the stiffness / force of bond
        ``bonds[b]`` by ``factor[b]``.
    node_weights : (N,) nodal weights for ``method="nodal"``, else ``None``.
    info : solver diagnostics for ``method="bond"``, else ``None``.
    """
    method: str
    bonds: np.ndarray
    factor: np.ndarray
    node_weights: Optional[np.ndarray] = None
    info: Optional[BondWeightInfo] = None

    def __len__(self) -> int:
        return len(self.bonds)


# --------------------------------------------------------------------- input
def _prepare(coords, volumes, horizon, dim):
    """Validate input and return ``(points (N, d), volumes (N,), d)``."""
    pts = np.asarray(coords, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(-1, 1)
    if pts.ndim != 2 or not 1 <= pts.shape[1] <= 3:
        raise ValueError(f"coords must be shape (N,), (N, 1), (N, 2) or (N, 3), got {pts.shape}")
    vol = np.asarray(volumes, dtype=np.float64)
    if vol.shape != (len(pts),):
        raise ValueError(f"volumes must be shape ({len(pts)},), got {vol.shape}")
    if not np.isfinite(horizon) or horizon <= 0:
        raise ValueError(f"horizon must be a positive number, got {horizon}")

    extent = pts.max(axis=0) - pts.min(axis=0)
    flat = extent <= 1e-12 * max(float(extent.max()), horizon)
    if dim is None:
        if flat.any() and len(pts) > 1:
            raise ValueError(
                f"coordinate column(s) {np.flatnonzero(flat).tolist()} are constant: the mesh is "
                f"lower-dimensional than its {pts.shape[1]} columns. Pass dim={int((~flat).sum())} "
                "to drop the constant columns.")
    else:
        if dim not in (1, 2, 3):
            raise ValueError(f"dim must be 1, 2 or 3, got {dim}")
        if dim < pts.shape[1]:
            if int((~flat).sum()) > dim:
                raise ValueError(f"dim={dim} but {int((~flat).sum())} coordinate columns vary")
            keep = np.flatnonzero(~flat)
            keep = np.concatenate([keep, np.flatnonzero(flat)])[:dim]
            pts = pts[:, np.sort(keep)]
        elif dim > pts.shape[1]:
            raise ValueError(f"dim={dim} but coords have only {pts.shape[1]} column(s)")
    return pts, vol, pts.shape[1]


# ------------------------------------------------------------------ methods
def nodal_weights(
    coords: np.ndarray,
    volumes: np.ndarray,
    horizon: float,
    *,
    model: str = "bb",
    dim: int | None = None,
    families: Sequence[Sequence[int]] | None = None,
    tol: float = 1e-10,
    max_iter: int = 5000,
    verbose: bool = False,
) -> np.ndarray:
    """One weight per node; a bond (i, j) is scaled by ``(w[i] + w[j]) / 2``.

    Parameters
    ----------
    coords : (N,), (N, 1), (N, 2) or (N, 3) nodal positions.
    volumes : (N,) nodal volumes (lengths in 1-D, areas in 2-D).
    horizon : peridynamic horizon delta.
    model : ``"bb"`` (bond-based, 1-D/2-D/3-D) or ``"osb"`` (state-based LPS, 3-D).
    dim : spatial dimension; only needed for a planar/linear mesh given with
        constant extra columns (e.g. 2-D nodes stored as (x, y, 0)).
    families : optional neighbour lists (see :func:`perifit.build_families`).
    tol, max_iter : controls of the global BiCGSTAB solve.
    verbose : print progress.

    Returns
    -------
    w : (N,) nodal weights.

    Notes
    -----
    2-D nodal weights were validated for horizon ratios m = delta/dx of 3-4;
    for other ratios, or when accuracy matters most, use :func:`bond_weights`.
    """
    pts, vol, d = _prepare(coords, volumes, horizon, dim)
    model = model.lower()
    if model not in ("bb", "osb"):
        raise ValueError(f"model must be 'bb' or 'osb', got {model!r}")
    if model == "osb" and d != 3:
        raise ValueError("the state-based model ('osb') is available in 3-D only")
    kw = dict(families=families, tol=tol, max_iter=max_iter, verbose=verbose)
    if d == 1:
        return _weights_1d(pts[:, 0], vol, horizon, **kw)
    if d == 2:
        return _weights_2d(pts, vol, horizon, **kw)
    return _weights_3d(pts, vol, horizon, model=model, **kw)


def bond_weights(
    coords: np.ndarray,
    volumes: np.ndarray,
    horizon: float,
    *,
    dim: int | None = None,
    bonds: np.ndarray | None = None,
    nonneg: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """One weight per bond (bond-based PD, 1-D/2-D/3-D).

    Returns
    -------
    bonds : (Nb, 2) int bond list (``i < j``, sorted, unless ``bonds`` was given).
    beta : (Nb,) bond weights; multiply the stiffness / force of ``bonds[b]`` by ``beta[b]``.

    Use :func:`surface_correction` with ``method="bond"`` to also get the
    solver diagnostics.
    """
    pts, vol, _ = _prepare(coords, volumes, horizon, dim)
    bonds, beta, _ = compute_bond_weights(pts, vol, horizon, bonds=bonds, nonneg=nonneg)
    return bonds, beta


def surface_correction(
    coords: np.ndarray,
    volumes: np.ndarray,
    horizon: float,
    *,
    method: str | None = None,
    model: str = "bb",
    dim: int | None = None,
    bonds: np.ndarray | None = None,
    verbose: bool = False,
) -> SurfaceCorrection:
    """Surface correction as one multiplier per bond, with an explicit method.

    ``method`` is required: ``"nodal"`` (one weight per node) or ``"bond"``
    (one weight per bond).  The result holds the bond list and a multiplier
    for every bond, whichever method is chosen.
    """
    if method is None:
        raise ValueError("surface_correction(): " + _METHOD_HELP)
    method = method.lower()
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; " + _METHOD_HELP)
    pts, vol, _ = _prepare(coords, volumes, horizon, dim)
    if bonds is None:
        bonds = build_bonds(pts, horizon)

    if method == "nodal":
        w = nodal_weights(pts, vol, horizon, model=model,
                          families=families_from_bonds(len(pts), bonds), verbose=verbose)
        factor = 0.5 * (w[bonds[:, 0]] + w[bonds[:, 1]])
        return SurfaceCorrection("nodal", bonds, factor, node_weights=w)

    if model.lower() != "bb":
        raise ValueError("bond weights are defined for the bond-based model ('bb') only")
    bonds, beta, info = compute_bond_weights(pts, vol, horizon, bonds=bonds)
    return SurfaceCorrection("bond", bonds, beta, info=info)

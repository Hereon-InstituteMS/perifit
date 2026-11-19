"""Neighbour search shared by every perifit method and dimension.

Two equivalent descriptions of the same peridynamic bond set are used:

* **families** -- for each node ``i`` the list of neighbours ``j`` with
  ``0 < |x_j - x_i| <= horizon``; used by the per-node (nodal-weight) fits.
* **bonds** -- one row ``(i, j)`` with ``i < j`` per bond, sorted
  lexicographically; used by per-bond quantities (bond weights, solvers).

Both use the same (relative) cutoff, so ``families_from_bonds(build_bonds(...))`` and
``build_families(...)`` describe identical neighbourhoods.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

#: Relative tolerance on the horizon: nodes at distance <= horizon * (1 + HORIZON_RTOL)
#: are neighbours, so nodes lying exactly on the horizon (common on regular grids)
#: are included despite round-off -- identically in every unit system.
HORIZON_RTOL = 1e-10


def _cutoff(horizon: float, tol: float | None) -> float:
    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}")
    return horizon * (1.0 + HORIZON_RTOL) if tol is None else horizon + tol


def _as_points(coords: np.ndarray) -> np.ndarray:
    pts = np.asarray(coords, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(-1, 1)
    if pts.ndim != 2:
        raise ValueError(f"coords must be shape (N,) or (N, d), got {pts.shape}")
    return pts


def check_inputs(coords: np.ndarray, volumes: np.ndarray, horizon: float,
                 dims: tuple[int, ...] = (1, 2, 3)) -> tuple[np.ndarray, np.ndarray]:
    """Validate a point cloud; returns ``(points (N, d), volumes (N,))`` as float arrays."""
    pts = _as_points(coords)
    if pts.shape[1] not in dims:
        raise ValueError(f"coords must have {' or '.join(map(str, dims))} column(s), "
                         f"got shape {pts.shape}")
    vol = np.asarray(volumes, dtype=np.float64)
    if vol.shape != (len(pts),):
        raise ValueError(f"volumes must be shape ({len(pts)},), got {vol.shape}")
    if not np.isfinite(horizon) or horizon <= 0:
        raise ValueError(f"horizon must be a positive number, got {horizon}")
    return pts, vol


def build_bonds(coords: np.ndarray, horizon: float, tol: float | None = None) -> np.ndarray:
    """All bonds within the horizon as an ``(Nb, 2)`` int array, ``i < j``, sorted.

    ``tol=None`` (default) uses the relative cutoff ``horizon * (1 + HORIZON_RTOL)``;
    an explicit ``tol`` gives the absolute cutoff ``horizon + tol``.
    """
    pts = _as_points(coords)
    pairs = cKDTree(pts).query_pairs(_cutoff(horizon, tol), output_type="ndarray")
    if len(pairs) == 0:
        return np.zeros((0, 2), dtype=np.int64)
    pairs = np.sort(pairs.astype(np.int64), axis=1)
    order = np.lexsort((pairs[:, 1], pairs[:, 0]))
    return pairs[order]


def families_from_bonds(n_nodes: int, bonds: np.ndarray) -> list[list[int]]:
    """Convert a bond list into per-node neighbour lists (sorted ascending)."""
    bonds = np.asarray(bonds, dtype=np.int64).reshape(-1, 2)
    i = np.concatenate([bonds[:, 0], bonds[:, 1]])
    j = np.concatenate([bonds[:, 1], bonds[:, 0]])
    order = np.lexsort((j, i))
    i, j = i[order], j[order]
    starts = np.searchsorted(i, np.arange(n_nodes + 1))
    return [j[starts[k]:starts[k + 1]].tolist() for k in range(n_nodes)]


def build_families(coords: np.ndarray, horizon: float, tol: float | None = None) -> list[list[int]]:
    """Per-node neighbour lists within the horizon (self excluded); see :func:`build_bonds`."""
    pts = _as_points(coords)
    return families_from_bonds(len(pts), build_bonds(pts, horizon, tol))

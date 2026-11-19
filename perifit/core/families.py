"""Neighbour search shared by every perifit method and dimension.

Two equivalent descriptions of the same peridynamic bond set are used:

* **families** -- for each node ``i`` the list of neighbours ``j`` with
  ``0 < |x_j - x_i| <= horizon``; used by the per-node (nodal-weight) fits.
* **bonds** -- one row ``(i, j)`` with ``i < j`` per bond, sorted
  lexicographically; used by per-bond quantities (bond weights, solvers).

Both use the same cutoff, so ``families_from_bonds(build_bonds(...))`` and
``build_families(...)`` describe identical neighbourhoods.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

#: Absolute tolerance added to the horizon so that nodes lying exactly on the
#: horizon (common on regular grids) are included despite round-off.
HORIZON_TOL = 1e-12


def _as_points(coords: np.ndarray) -> np.ndarray:
    pts = np.asarray(coords, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(-1, 1)
    if pts.ndim != 2:
        raise ValueError(f"coords must be shape (N,) or (N, d), got {pts.shape}")
    return pts


def build_bonds(coords: np.ndarray, horizon: float, tol: float = HORIZON_TOL) -> np.ndarray:
    """All bonds within the horizon as an ``(Nb, 2)`` int array, ``i < j``, sorted."""
    pts = _as_points(coords)
    pairs = cKDTree(pts).query_pairs(horizon + tol, output_type="ndarray")
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


def build_families(coords: np.ndarray, horizon: float, tol: float = HORIZON_TOL) -> list[list[int]]:
    """Per-node neighbour lists within ``horizon + tol`` (self excluded)."""
    pts = _as_points(coords)
    return families_from_bonds(len(pts), build_bonds(pts, horizon, tol))

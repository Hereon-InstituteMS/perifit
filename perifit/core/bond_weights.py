"""Bond weights: one correction factor per bond (bond-based PD, 1-D/2-D/3-D).

Every bond b = (i, j) gets its own weight beta_b.  The weights are the
smallest change from 1 (no correction) for which the bond sums of every node
reproduce the full-horizon moments of the bond-based strain energy:

    minimise  || beta - 1 ||^2   subject to   A beta = t ,   beta >= 0

with, for every node i and every moment (p, a),

    sum_{j in H_i}  beta_ij  |xi_ij|^p  e_ij^a  V_j  =  int_{|xi| < delta} |xi|^p e^a dV
                                                   =  S_d(a) delta^(p+d) / (p+d)

where e = xi/|xi| is the bond direction and S_d(a) the integral of e^a over
the unit sphere.  The moments are those met by an affine displacement in the
bond energy (|xi| (e . F e)^2):

    1-D : p = 1 and p = 3                     (radial; e^a = 1)
    2-D : p = 1, e^a in {ex^2, ey^2, ex^4, ex^2 ey^2, ey^4}
    3-D : p = 1, e^a in {ex^2, ey^2, ez^2, ex^4, ey^4, ez^4,
                         ex^2 ey^2, ex^2 ez^2, ey^2 ez^2}

All moments are even in e, so a bond enters the rows of both end nodes with
the same geometric factor and one shared beta_b keeps the stiffness matrix
symmetric.  Rows are scaled by delta^-(p+d), which makes the problem
dimensionless and the weights independent of the length unit.

Because each bond has its own weight, bond weights also correct the
anisotropy of a regular grid (every node gets the exact fourth moments),
which per-node weights cannot do.  The price is one number per bond instead
of one per node.

Solution
--------
With delta = beta - 1 the problem is the minimum-norm solution of
``A delta = t - A 1``, computed with LSMR on the row-normalised sparse
system (well defined even where the equations are rank deficient, e.g. at
corners).  Weights that come out negative are fixed to zero and the others
re-solved (active set) until all are non-negative.  A bond with beta = 0
carries no force; for fracture, breaking a bond means setting its weight to
zero -- the other weights stay valid.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import LinearOperator, lsmr

from .families import build_bonds

#: (p, exponents of e) per dimension
MOMENTS = {
    1: [(1, (0,)), (3, (0,))],
    2: [(1, a) for a in [(2, 0), (0, 2), (4, 0), (2, 2), (0, 4)]],
    3: [(1, a) for a in [(2, 0, 0), (0, 2, 0), (0, 0, 2), (4, 0, 0), (0, 4, 0), (0, 0, 4),
                         (2, 2, 0), (2, 0, 2), (0, 2, 2)]],
}


@dataclass(frozen=True)
class BondWeightInfo:
    """Diagnostics of a bond-weight computation."""
    residual: float              #: relative residual of the (row-normalised) moment equations
    n_zero_bonds: int            #: bonds whose factor was fixed to 0 by beta >= 0
    active_set_iterations: int   #: number of solves of the active-set loop
    lsmr_iterations: int         #: LSMR iterations of the last solve
    n_dropped_rows: int          #: moment equations without any bond (ignored)


def sphere_moment(a: tuple[int, ...]) -> float:
    """Integral of prod e_i^a_i over the unit sphere S^(d-1) (d = len(a)); 0 if any a_i is odd."""
    if any(k % 2 for k in a):
        return 0.0
    b = [(k + 1) / 2 for k in a]
    return 2.0 * math.prod(math.gamma(x) for x in b) / math.gamma(sum(b))


def compute_bond_weights(
    coords: np.ndarray,
    volumes: np.ndarray,
    horizon: float,
    bonds: np.ndarray | None = None,
    nonneg: bool = True,
    rtol: float = 1e-10,
    max_iter: int | None = None,
) -> tuple[np.ndarray, np.ndarray, BondWeightInfo]:
    """Bond weights for a 1-, 2- or 3-D bond-based PD discretisation.

    Parameters
    ----------
    coords : (N,), (N, 1), (N, 2) or (N, 3) nodal positions; d = number of columns.
    volumes : (N,) nodal volumes (lengths in 1-D, areas in 2-D).
    horizon : peridynamic horizon delta.
    bonds : optional (Nb, 2) bond list; built with :func:`build_bonds` if omitted.
    nonneg : enforce beta >= 0 (recommended; a negative factor would make the bond push).
    rtol : LSMR tolerance (``atol = btol = rtol``).
    max_iter : LSMR iteration limit (default ``10 * N * n_moments``).

    Returns
    -------
    bonds : (Nb, 2) int bond list (``i < j``, sorted, when built here).
    beta : (Nb,) bond weights; the bond's stiffness is multiplied by ``beta[b]``.
    info : :class:`BondWeightInfo`.
    """
    pts = np.asarray(coords, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(-1, 1)
    n, d = pts.shape
    if d not in MOMENTS:
        raise ValueError(f"coords must have 1, 2 or 3 columns, got {d}")
    vol = np.asarray(volumes, dtype=np.float64)
    if vol.shape != (n,):
        raise ValueError(f"volumes must be shape ({n},), got {vol.shape}")
    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}")
    bonds = build_bonds(pts, horizon) if bonds is None else np.asarray(bonds, dtype=np.int64)
    if bonds.ndim != 2 or bonds.shape[1] != 2:
        raise ValueError(f"bonds must be shape (Nb, 2), got {bonds.shape}")
    nb = len(bonds)
    if nb == 0:
        return bonds, np.zeros(0), BondWeightInfo(0.0, 0, 0, 0, 0)

    bi, bj = bonds[:, 0], bonds[:, 1]
    xi = pts[bj] - pts[bi]
    r = np.linalg.norm(xi, axis=1)
    if np.any(r == 0.0):
        raise ValueError("coincident nodes: some bonds have zero length")
    e = xi / r[:, None]

    # geometric coefficient of every bond in every (dimensionless) moment row
    moments = MOMENTS[d]
    nc = len(moments)
    G = np.empty((nb, nc))
    t = np.empty(nc)
    for k, (p, a) in enumerate(moments):
        G[:, k] = (r / horizon) ** p * np.prod(e ** np.asarray(a), axis=1) / horizon ** d
        t[k] = sphere_moment(a) / (p + d)
    # sparse constraint matrix: row (node, moment); the row of node i uses V_j and
    # the row of node j uses V_i.  Rows are normalised (Jacobi scaling) and rows
    # without any bond are dropped.
    idx = np.arange(nb)
    rows = np.concatenate([(bi[:, None] * nc + np.arange(nc)).ravel(),
                           (bj[:, None] * nc + np.arange(nc)).ravel()])
    cols = np.concatenate([np.repeat(idx, nc), np.repeat(idx, nc)])
    vals = np.concatenate([(G * vol[bj][:, None]).ravel(), (G * vol[bi][:, None]).ravel()])
    A = csr_matrix((vals, (rows, cols)), shape=(n * nc, nb))
    del rows, cols, vals
    row_norm = np.sqrt(np.asarray(A.multiply(A).sum(axis=1)).ravel())
    live = row_norm > 0.0
    D = np.zeros(n * nc)
    D[live] = 1.0 / row_norm[live]
    A = diags(D) @ A                      # row-normalised, CSR
    At = A.T                              # CSC view, no copy
    target = D * np.tile(t, n)

    def solve(active: np.ndarray) -> tuple[np.ndarray, int]:
        op = LinearOperator((n * nc, nb), dtype=np.float64,
                            matvec=lambda x: A @ (active * x),
                            rmatvec=lambda y: active * (At @ y))
        sol = lsmr(op, target - A @ active, atol=rtol, btol=rtol,
                   maxiter=max_iter if max_iter is not None else 10 * n * nc)
        return active * (1.0 + sol[0]), int(sol[2])

    active = np.ones(nb)
    beta, its = solve(active)
    passes = 1
    if nonneg:
        while passes < 50:
            negative = (beta < 0.0) & (active > 0.0)
            if not negative.any():
                break
            active[negative] = 0.0
            beta, its = solve(active)
            passes += 1
        beta = np.maximum(beta, 0.0)

    residual = float(np.linalg.norm(A @ beta - target) / np.linalg.norm(target))
    info = BondWeightInfo(residual, int(np.sum(beta == 0.0)), passes, its, int(np.sum(~live)))
    if residual > 1e-6:
        warnings.warn(f"bond weights: moment equations satisfied only to a relative residual "
                      f"of {residual:.1e} (rank-deficient or very irregular neighbourhoods)",
                      RuntimeWarning, stacklevel=2)
    return bonds, beta, info

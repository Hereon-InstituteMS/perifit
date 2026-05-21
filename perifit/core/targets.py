"""Analytical operator targets (full-ball reference values).

For each PD model (BB or OSB), the surface-correction weights are obtained by
enforcing that the discrete derivative and energy operators, evaluated on the
truncated horizon, reproduce their full-ball analytical values for every
polynomial test function up to degree 2.

Each model produces 54 target equations per node:
  - 27 derivative rows: 3 directions (x, y, z) x 9 test monomials
  - 27 energy rows:     3 lift directions (x, y, z) x 9 test monomials

The test monomials (degree 1 + 2) in local coordinates xi/delta are:
    xi_x, xi_y, xi_z, xi_x^2, xi_x*xi_y, xi_x*xi_z, xi_y^2, xi_y*xi_z, xi_z^2

All target arrays are returned with material constants set to 1: those
constants cancel after per-row normalisation in the local LS, so the weights
are purely geometric.

Authors: Arman Shojaei & Alexander Hermann
         Helmholtz-Zentrum Hereon, Geesthacht, Germany.
"""
from __future__ import annotations

import numpy as np

from .moments import ball_moment, ball_moment_over_r2, ball_moment_over_r3


# Test polynomials phi_k for k = 1..9 are degree 1+2 monomials in (xi_x, xi_y, xi_z).
# _exponents_dphi(k) returns the exponents of the monomial.
_DPHI_EXPONENTS = {
    1: (1, 0, 0), 2: (0, 1, 0), 3: (0, 0, 1),
    4: (2, 0, 0), 5: (1, 1, 0), 6: (1, 0, 1),
    7: (0, 2, 0), 8: (0, 1, 1), 9: (0, 0, 2),
}


def _dphi_exponents(k: int) -> tuple[int, int, int]:
    return _DPHI_EXPONENTS[k]


# ---------------------------------------------------------------------------
# Bond-based PD targets
# ---------------------------------------------------------------------------
def build_targets_bb(delta: float) -> tuple[np.ndarray, np.ndarray]:
    """BB-PD analytical targets (54 = 27 derivative + 27 energy).

    Derivative kernel : (xi_alpha / r^3) * dphi_k
    Energy kernel     : (1/4) * dphi_k^2 * xi_ell^2 / r^3
    (The micro-modulus c cancels after per-row normalisation.)
    """
    d_star = np.zeros(27)
    for alpha in range(3):
        for k in range(1, 10):
            ax, ay, az = _dphi_exponents(k)
            px = ax + (1 if alpha == 0 else 0)
            py = ay + (1 if alpha == 1 else 0)
            pz = az + (1 if alpha == 2 else 0)
            d_star[alpha * 9 + (k - 1)] = ball_moment_over_r3(delta, px, py, pz)

    e_star = np.zeros(27)
    for ell in range(3):
        for k in range(1, 10):
            ax, ay, az = _dphi_exponents(k)
            px = 2 * ax + (2 if ell == 0 else 0)
            py = 2 * ay + (2 if ell == 1 else 0)
            pz = 2 * az + (2 if ell == 2 else 0)
            e_star[ell * 9 + (k - 1)] = 0.25 * ball_moment_over_r3(delta, px, py, pz)

    return d_star, e_star


# ---------------------------------------------------------------------------
# Ordinary state-based (LPS) PD targets
# ---------------------------------------------------------------------------
def build_targets_osb(delta: float) -> tuple[np.ndarray, np.ndarray]:
    """OSB-PD analytical targets (54 = 27 derivative + 27 energy).

    Derivative kernel : (xi_alpha / r^2) * dphi_k       (LPS gradient)
    Energy kernel     : (15 / m_a) * dphi_k^2 * xi_ell^2 / r^2
                        (G and m_a cancel after per-row normalisation)
    """
    d_star = np.zeros(27)
    for alpha in range(3):
        for k in range(1, 10):
            ax, ay, az = _dphi_exponents(k)
            px = ax + (1 if alpha == 0 else 0)
            py = ay + (1 if alpha == 1 else 0)
            pz = az + (1 if alpha == 2 else 0)
            d_star[alpha * 9 + (k - 1)] = ball_moment_over_r2(delta, px, py, pz)

    e_star = np.zeros(27)
    for ell in range(3):
        for k in range(1, 10):
            ax, ay, az = _dphi_exponents(k)
            px = 2 * ax + (2 if ell == 0 else 0)
            py = 2 * ay + (2 if ell == 1 else 0)
            pz = 2 * az + (2 if ell == 2 else 0)
            e_star[ell * 9 + (k - 1)] = ball_moment_over_r2(delta, px, py, pz)

    return d_star, e_star


# ---------------------------------------------------------------------------
# Correspondence (NOSB) PD targets
# ---------------------------------------------------------------------------
def build_targets_nosb(delta: float) -> np.ndarray:
    """NOSB-PD analytical derivative targets (27 rows).

    Kernel : xi_alpha * dphi_k    (gradient numerator; shape tensor K is
    inverted analytically in the solver, so no separate energy block is
    needed -- energy consistency follows from gradient consistency.)
    """
    n_star = np.zeros(27)
    for b in range(3):
        for k in range(1, 10):
            ax, ay, az = _dphi_exponents(k)
            px = ax + (1 if b == 0 else 0)
            py = ay + (1 if b == 1 else 0)
            pz = az + (1 if b == 2 else 0)
            n_star[b * 9 + (k - 1)] = ball_moment(delta, px, py, pz)
    return n_star

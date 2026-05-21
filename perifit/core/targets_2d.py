"""2-D analytical operator targets (full-disc reference values).

For 2-D BB-PD, the surface-correction weights are obtained by enforcing
that two non-local operators evaluated on the truncated horizon equal their
full-disc analytical values for every polynomial test function up to degree 2:

  - Derivative block: (xi_alpha / r^3) * dphi_k
  - Energy block:     (1/4) * dphi_k^2 * xi_ell^2 / r^3

The test functions dphi_k in 2D are degree 1+2 monomials in (xi_x, xi_y):
    k=1: x,  k=2: y,  k=3: x^2,  k=4: xy,  k=5: y^2

Derivative block: 2 (alpha) x 5 (k) = 10 rows
Energy block:     2 (ell)   x 5 (k) = 10 rows
Total: 20 operator equations per node.
"""
from __future__ import annotations

import numpy as np

from .moments_2d import disc_moment_over_r3


# 2D test function exponents (dphi_k is a monomial of degree 1 or 2)
_DPHI_EXPONENTS_2D = {
    1: (1, 0),
    2: (0, 1),
    3: (2, 0),
    4: (1, 1),
    5: (0, 2),
}

N_DPHI_2D = 5  # number of test functions


def _dphi_exponents_2d(k: int) -> tuple[int, int]:
    return _DPHI_EXPONENTS_2D[k]


def build_targets_bb_2d(delta: float) -> tuple[np.ndarray, np.ndarray]:
    """2D BB-PD analytical targets (20 = 10 derivative + 10 energy).

    Derivative kernel : (xi_alpha / r^3) * dphi_k
    Energy kernel     : (1/4) * dphi_k^2 * xi_ell^2 / r^3
    """
    n_deriv = 2 * N_DPHI_2D  # = 10
    d_star = np.zeros(n_deriv)
    for alpha in range(2):
        for k in range(1, N_DPHI_2D + 1):
            ax, ay = _dphi_exponents_2d(k)
            deg = ax + ay  # degree of monomial (1 or 2)
            px = ax + (1 if alpha == 0 else 0)
            py = ay + (1 if alpha == 1 else 0)
            # disc_moment_over_r3 requires px+py > 1
            p_total = px + py
            if p_total > 1:
                # delta-scaled dphi: divide by delta^deg
                d_star[alpha * N_DPHI_2D + (k - 1)] = (
                    disc_moment_over_r3(delta, px, py) / delta**deg
                )
            # else: 0 (angular integral vanishes for odd exponents anyway)

    n_energy = 2 * N_DPHI_2D  # = 10
    e_star = np.zeros(n_energy)
    for ell in range(2):
        for k in range(1, N_DPHI_2D + 1):
            ax, ay = _dphi_exponents_2d(k)
            deg = ax + ay  # degree of monomial (1 or 2)
            px = 2 * ax + (2 if ell == 0 else 0)
            py = 2 * ay + (2 if ell == 1 else 0)
            p_total = px + py
            if p_total > 1:
                # delta-scaled dphi squared: divide by delta^(2*deg)
                e_star[ell * N_DPHI_2D + (k - 1)] = (
                    0.25 * disc_moment_over_r3(delta, px, py) / delta**(2 * deg)
                )

    return d_star, e_star

"""1-D analytical operator targets (full-horizon reference values).

For 1-D BB-PD, the surface-correction weights are obtained by enforcing that two
non-local operators evaluated on the truncated horizon equal their full-horizon
analytical values for every polynomial test function up to degree 2:

  - Derivative block: (xi / |xi|) * dphi_k
  - Energy block:     (1/4) * dphi_k^2 * xi^2 / |xi|^3

The test functions dphi_k in 1D are the degree 1+2 monomials in xi:
    k=1: x,  k=2: x^2

Derivative block: 1 (alpha) x 2 (k) = 2 rows
Energy block:     1 (ell)   x 2 (k) = 2 rows
Total: 4 operator equations per node.

Choice of the derivative kernel
-------------------------------
The generic BB derivative kernel xi_alpha/|xi|^3 of eq. (45) is not integrable
over a 1-D horizon: int_{-delta}^{delta} xi^2/|xi|^3 dxi diverges logarithmically.
The paper therefore writes the 1-D BB derivative operator in eq. (24) with the
kernel xi/(delta^2 |xi|); the constant delta^-2 is dropped here because it is
absorbed consistently into both the moment matrix and the target.  The energy
kernel keeps exponent 3 and reduces to (du)^2/|xi| in 1-D, again matching eq. (24).
"""
from __future__ import annotations

import numpy as np

from .moments_1d import segment_moment_over_r, segment_moment_over_r3


# 1D test function exponents (dphi_k is a monomial of degree 1 or 2)
_DPHI_EXPONENTS_1D = {1: 1, 2: 2}

N_DPHI_1D = 2  # number of test functions


def _dphi_exponent_1d(k: int) -> int:
    return _DPHI_EXPONENTS_1D[k]


def build_targets_bb_1d(delta: float) -> tuple[np.ndarray, np.ndarray]:
    """1D BB-PD analytical targets (4 = 2 derivative + 2 energy).

    Derivative kernel : (xi / |xi|) * dphi_k
    Energy kernel     : (1/4) * dphi_k^2 * xi^2 / |xi|^3
    (The micro-modulus c is set to 1: it is a common factor of the energy block
    and its target, so it never influences the weights.)
    """
    d_star = np.zeros(N_DPHI_1D)
    for k in range(1, N_DPHI_1D + 1):
        deg = _dphi_exponent_1d(k)
        p = deg + 1                       # one extra xi from the kernel
        if p > 0:
            # delta-scaled dphi: divide by delta^deg
            d_star[k - 1] = segment_moment_over_r(delta, p) / delta ** deg

    e_star = np.zeros(N_DPHI_1D)
    for k in range(1, N_DPHI_1D + 1):
        deg = _dphi_exponent_1d(k)
        p = 2 * deg + 2                   # dphi_k^2 * xi^2
        # delta-scaled dphi squared: divide by delta^(2*deg)
        e_star[k - 1] = 0.25 * segment_moment_over_r3(delta, p) / delta ** (2 * deg)

    return d_star, e_star

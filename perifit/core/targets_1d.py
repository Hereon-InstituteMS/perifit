"""1-D analytical operator targets for the nodal weights (full-horizon values).

The nodal-weight fit asks the truncated discrete operators to reproduce their
full-horizon values on polynomial test functions dphi_k = (xi/delta)^k:

  Derivative rows (k = 1, 2):  kernel (xi / |xi|) * dphi_k
      The 1-D BB derivative operator of the paper, eq. (24); the generic
      kernel xi/|xi|^3 of eq. (45) is not integrable in 1-D.
  Energy row      (k = 1):     kernel (1/4) * dphi_k^2 * xi^2 / |xi|^3
      The linear test is the homogeneous strain that sets the elastic
      stiffness.  The quadratic test is *not* used: on a fixed-m grid it is
      integrated inconsistently with the linear one (ratio 0.64 vs 0.80 at
      m = 4) and pulls the bulk weight away from its exact value m/(m+1).

Material constants are set to 1: they are common factors of a row and its
target and cancel in the per-row normalisation.
"""
from __future__ import annotations

import numpy as np

from .moments_1d import segment_moment_over_r, segment_moment_over_r3

#: degrees of the test functions used in the derivative and energy rows
DERIV_DEGREES = (1, 2)
ENERGY_DEGREES = (1,)


def build_targets_bb_1d(delta: float) -> tuple[np.ndarray, np.ndarray]:
    """Full-horizon targets ``(d_star (2,), e_star (1,))`` for 1-D bond-based PD."""
    d_star = np.array([segment_moment_over_r(delta, k + 1) / delta ** k
                       for k in DERIV_DEGREES])
    e_star = np.array([0.25 * segment_moment_over_r3(delta, 2 * k + 2) / delta ** (2 * k)
                       for k in ENERGY_DEGREES])
    return d_star, e_star

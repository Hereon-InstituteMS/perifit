"""2-D analytical operator targets for the nodal weights (full-disc values).

The 2-D fit uses exactly the moments that appear in the bond-based operator
itself, with the paper's kernels unchanged:

  Derivative rows: kernel (xi_a / r^3) * dphi_k, quadratic tests
      dphi_k in {x^2, xy, y^2} (scaled by delta^2) for a in {x, y}  -> 6 rows.
      These are third moments int xi_a xi_b xi_c / r^3: the force that an
      affine field produces at a node.  They vanish on a full disc and are
      non-zero on a truncated horizon (the surface-force imbalance), so all
      six targets are zero.
  Energy rows: kernel (1/4) * dphi_k^2 * xi_l^2 / r^3, linear tests
      dphi_k in {x, y} (scaled by delta) for l in {x, y}               -> 4 rows.
      These are the fourth moments that define the elastic constants.

The degree-1 derivative rows (second moments int xi_a xi_b / r^3) are not
used: in 2-D their integrand does not vanish at r = 0, so on a fixed-m grid
the node's own cell carries a fixed share of the integral that no bond weight
can supply, which biases the bulk weight.  The quadratic energy tests are not
used either; with them removed the bulk weight is fixed by the elastic rows.
Row selection was validated on the cantilever benchmark for m = 3 and 4.
"""
from __future__ import annotations

import numpy as np

from .moments_2d import disc_moment_over_r3

#: (x, y) exponents of the test functions
DERIV_TESTS = ((2, 0), (1, 1), (0, 2))     # quadratic
ENERGY_TESTS = ((1, 0), (0, 1))            # linear


def build_targets_bb_2d(delta: float) -> tuple[np.ndarray, np.ndarray]:
    """Full-disc targets ``(d_star (6,), e_star (4,))`` for 2-D bond-based PD.

    Row order: direction-major, i.e. ``index = direction * n_tests + test``.
    """
    d_star = np.array([disc_moment_over_r3(delta, ax + (a == 0), ay + (a == 1)) / delta ** (ax + ay)
                       for a in range(2) for ax, ay in DERIV_TESTS])
    e_star = np.array([0.25 * disc_moment_over_r3(delta, 2 * ax + 2 * (l == 0), 2 * ay + 2 * (l == 1))
                       / delta ** (2 * (ax + ay))
                       for l in range(2) for ax, ay in ENERGY_TESTS])
    return d_star, e_star

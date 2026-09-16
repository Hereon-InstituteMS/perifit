"""Example 1 of the perifit paper (Section 5.1): 1-D bar under uniaxial tension.

Unit bar, L = E = 1, exact solution u_ref(x) = x.  The left end is clamped and a
concentrated traction P = 1 is applied at the right end, both at a single node so
that the boundary effect is isolated.  Standard PD is compared with modified-PD
using the optimised nodal influence weights.

Run with::

    python examples/bar_1d_example1.py
"""
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import spsolve

from perifit import build_families_1d, compute_weights_1d

E, L, P = 1.0, 1.0, 1.0


def bar(dx):
    n = int(round(L / dx))
    return (np.arange(n) + 0.5) * dx, np.full(n, dx)


def solve_bar(x, vols, delta, w=None):
    """Static BB-PD solve, c = 2E/delta^2, midpoint quadrature, no beta factor."""
    n = len(x)
    w = np.ones(n) if w is None else w
    fam = build_families_1d(x, delta)
    c = 2.0 * E / delta ** 2
    A = lil_matrix((n, n))
    for i in range(n):
        j = np.asarray(fam[i])
        k = c * 0.5 * (w[i] + w[j]) * vols[j] / np.abs(x[j] - x[i])
        A[i, j] = k
        A[i, i] = -k.sum()
    rhs = np.zeros(n)
    rhs[-1] = -P / vols[-1]          # concentrated traction at the end node
    A[0, :] = 0.0                    # clamp at the single node x_0
    A[0, 0] = 1.0
    return spsolve(csr_matrix(A), rhs)


def l2_error(u, u_ref):
    return float(np.sqrt(np.sum((u - u_ref) ** 2) / np.sum(u_ref ** 2)))


def main():
    print("delta-convergence (paper, Fig. 6)")
    print(f"  {'m':>3} {'dx':>9} {'N':>5} {'e_u PD':>9} {'e_u modified-PD':>16}")
    for m, dxs in ((4, [0.02, 0.01, 0.005]),
                   (8, [0.02, 0.01, 0.005]),
                   (16, [0.01, 0.005, 0.0025])):
        for dx in dxs:
            x, vols = bar(dx)
            delta = m * dx
            w = compute_weights_1d(x, vols, delta)
            e_pd = l2_error(solve_bar(x, vols, delta), x)
            e_mod = l2_error(solve_bar(x, vols, delta, w), x)
            print(f"  {m:3d} {dx:9g} {len(x):5d} {e_pd:9.5f} {e_mod:16.5f}")

    print("\nnodal influence weights at dx = 0.005 (paper, Fig. 7)")
    print(f"  {'m':>3} {'w at end':>9} {'min w':>8} {'interior w':>11} {'m/(m+1)':>9}")
    dx = 0.005
    for m in (4, 8, 16):
        x, vols = bar(dx)
        w = compute_weights_1d(x, vols, m * dx)
        interior = w[(x > 0.4) & (x < 0.6)].mean()
        print(f"  {m:3d} {w[0]:9.3f} {w.min():8.3f} {interior:11.3f} "
              f"{m / (m + 1):9.3f}")

    x, vols = bar(dx)
    delta = 4 * dx
    w = compute_weights_1d(x, vols, delta)
    print(f"\nu(L) at m = 4, dx = {dx}:  PD {solve_bar(x, vols, delta)[-1]:.4f}  "
          f"modified-PD {solve_bar(x, vols, delta, w)[-1]:.4f}  exact {x[-1]:.4f}")


if __name__ == "__main__":
    main()

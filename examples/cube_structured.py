"""
Example: structured 3-D Cartesian cube mesh.

Reproduces the benchmark from the C++ implementation:
  - Domain  [0, 1]^3
  - dx = 0.05  (20 x 20 x 20 = 8,000 nodes)
  - m  = 3     -> delta = 0.15
  - E  = 1.0,  nu = 0.25

Demonstrates the full perifit workflow:
  1. Build a structured mesh (no external mesh file required).
  2. Compute optimised surface correction weights.
  3. Export to CSV, VTK, and both solver formats.
  4. Print a summary and optionally compute the weight-corrected
     L2 error against the analytical uniaxial-tension solution.
"""

import numpy as np
import sys
import os

# Allow running from the repository root without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from perifit import compute_weights, write_weights


# ---------------------------------------------------------------------------
# 1. Build a structured Cartesian mesh
# ---------------------------------------------------------------------------

def build_structured_mesh(dx: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Return (coords, volumes) for a [0,1]^3 Cartesian grid with spacing dx."""
    xs = np.arange(dx / 2, 1.0, dx)
    N1d = len(xs)
    gx, gy, gz = np.meshgrid(xs, xs, xs, indexing='ij')
    coords  = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
    volumes = np.full(len(coords), dx ** 3, dtype=np.float64)
    return coords, volumes


# ---------------------------------------------------------------------------
# 2. Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    dx    = 0.05
    m     = 3
    delta = m * dx      # horizon = 0.15
    E     = 1.0
    nu    = 0.25

    print("=" * 60)
    print("perifit surface correction  --  structured cube example")
    print(f"  dx={dx}  m={m}  delta={delta}  E={E}  nu={nu}")
    print("=" * 60)

    coords, volumes = build_structured_mesh(dx)
    N = len(coords)
    print(f"Mesh: {N:,d} nodes  ({round(N**(1/3))}^3 grid)")

    # Compute weights
    weights = compute_weights(
        coords, volumes,
        horizon=delta,
        verbose=True,
    )

    # Write outputs
    outdir = './example_output'
    paths = write_weights(
        coords, volumes, weights,
        outdir=outdir,
        stem='cube_structured',
        formats=['csv', 'dat', 'vtk', 'peridigm', 'perilab'],
        E=E, nu=nu, horizon=delta,
    )

    print("\nOutput files written:")
    for fmt, path in paths.items():
        print(f"  [{fmt}]  {path}")

    # ---------------------------------------------------------------------------
    # 3. Optional: verify weights by checking the weighted-volume correction
    #    Interior nodes should have  w_i ≈ 1; boundary nodes w_i > 1.
    # ---------------------------------------------------------------------------
    from scipy.spatial import KDTree

    tree = KDTree(coords)
    pairs = tree.query_ball_tree(tree, r=delta)
    fam = [[j for j in pairs[i] if j != i] for i in range(N)]

    # Surface fraction: nodes whose family is smaller than 90 % of the interior
    # family size (estimated from a node near the domain centre).
    centre_idx = np.argmin(np.sum((coords - 0.5) ** 2, axis=1))
    interior_nF = len(fam[centre_idx])
    n_surface = sum(1 for f in fam if len(f) < 0.9 * interior_nF)
    print(f"\nSurface nodes (< 90% interior family size): "
          f"{n_surface} / {N}  ({100*n_surface/N:.1f}%)")
    print(f"Interior family size: {interior_nF}")
    print(f"Weights — min: {weights.min():.4f}  max: {weights.max():.4f}"
          f"  mean: {weights.mean():.4f}")
    print(f"\nInterior weight (node near centre): {weights[centre_idx]:.6f}"
          f"  (should be ≈ 1.0)")

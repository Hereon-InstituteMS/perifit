"""perifit quick start: surface corrections for a 2-D plate.

Run:  python examples/quickstart.py
"""
import numpy as np

import perifit

# A unit square sampled on a regular grid, horizon delta = 4 dx
dx = 1 / 40
s = np.arange(0.0, 1.0 + dx / 2, dx)
X, Y = np.meshgrid(s, s, indexing="ij")
coords = np.column_stack([X.ravel(), Y.ravel()])
volumes = np.full(len(coords), dx * dx)        # nodal areas (unit thickness)
horizon = 4 * dx

# --- 1. One weight per node --------------------------------------------------
w = perifit.nodal_weights(coords, volumes, horizon)
print(f"nodal weights : {len(w):6d} values, interior {np.median(w):.3f}, corner {w.max():.3f}")

# --- 2. One weight per bond --------------------------------------------------
bonds, beta = perifit.bond_weights(coords, volumes, horizon)
print(f"bond weights  : {len(beta):6d} values, median {np.median(beta):.3f}, max {beta.max():.3f}")

# --- 3. Either one as a multiplier per bond ---------------------------------
for method in ("nodal", "bond"):
    corr = perifit.surface_correction(coords, volumes, horizon, method=method)
    # In a bond-based solver, scale the micromodulus of bond b = corr.bonds[b]:
    #     c_b = c * corr.factor[b]
    print(f"{method:5s} -> {len(corr):6d} bond multipliers in [{corr.factor.min():.2f}, "
          f"{corr.factor.max():.2f}]")

"""
Quick-start test script.

Run this after installing the package:
    pip install perifit
    python run_test_mesh.py

Or from the repository root:
    python3 examples/run_test_mesh.py

Loads the included 1000-node structured cube mesh, computes surface
correction weights, and writes output to ./perifit_output/.
"""

import sys
from pathlib import Path

# Allow running without installation
sys.path.insert(0, str(Path(__file__).parent.parent))

from perifit import load_mesh, compute_weights, write_weights

# -- Parameters --------------------------------------------------------------
MESH_FILE = Path(__file__).parent / "test_mesh_cube_1000.csv"
DELTA     = 0.30   # horizon (m = delta/dx = 3)

# -- Load mesh ---------------------------------------------------------------
print(f"Loading mesh: {MESH_FILE.name}")
coords, volumes = load_mesh(MESH_FILE, horizon=DELTA, vol_col=3)
print(f"  {len(coords):,d} nodes  |  volume per node = {volumes[0]:.4e}")

# -- Compute weights ---------------------------------------------------------
weights = compute_weights(
    coords, volumes,
    horizon=DELTA,
    verbose=True,
)

# -- Write all supported formats ---------------------------------------------
paths = write_weights(
    coords, volumes, weights,
    outdir="./perifit_output",
    stem="cube_1000",
    formats=["csv", "dat", "vtk", "peridigm", "perilab"],
    horizon=DELTA,
)

print("\nOutput files:")
for fmt, p in paths.items():
    print(f"  [{fmt:10s}]  {p}")

print("\nDone.  Open perifit_output/cube_1000.vtk in ParaView to visualise the weights.")

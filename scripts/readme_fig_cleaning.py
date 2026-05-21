#!/usr/bin/env python3
"""
README figure: mesh quality diagnostics on rocker-arm geometry.

Three panels:
  (a) Original mesh colored by weight values
  (b) Flagged nodes highlighted (negative / outlier)
  (c) Cleaned mesh after removing flagged nodes

Uses rocker_arm_res32_mesh.csv for the geometry (1699 nodes, fast).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import Normalize
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from perifit import compute_weights
from perifit.quality.diagnostics import diagnose_weights
from perifit.quality.cleaning import clean_mesh

# ---------------------------------------------------------------------------
# rcParams
# ---------------------------------------------------------------------------
mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'mathtext.fontset': 'cm',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
})

# ---------------------------------------------------------------------------
# Load rocker-arm mesh
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parents[1].parent / "data"
mesh_file = DATA_DIR / "rocker_arm_res32_mesh.csv"
raw = np.loadtxt(mesh_file, delimiter=',', skiprows=1)
coords = raw[:, :3]
volumes = raw[:, 3]
N = len(coords)
print(f"Loaded {N} nodes from {mesh_file.name}")

# ---------------------------------------------------------------------------
# Compute weights
# ---------------------------------------------------------------------------
weights = compute_weights(coords, volumes, m_ratio=3, verbose=True)

# ---------------------------------------------------------------------------
# Diagnose
# ---------------------------------------------------------------------------
diag = diagnose_weights(weights, threshold_sigma=2.0)
print(diag.summary)

# ---------------------------------------------------------------------------
# Clean
# ---------------------------------------------------------------------------
result = clean_mesh(coords, volumes, weights, threshold_sigma=2.0, verbose=True)

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(16, 4.8))

# Common view angle
elev, azim = 25, -60

# Panel (a): Original mesh colored by weight
ax1 = fig.add_subplot(131, projection='3d')
vmin, vmax = 1.0, np.percentile(weights, 97)
norm = Normalize(vmin=vmin, vmax=vmax)
sc1 = ax1.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                  c=weights, cmap='RdYlBu_r', norm=norm,
                  s=4, alpha=0.8, edgecolors='none')
ax1.set_title(f'(a)  Original mesh\n{N:,d} nodes', pad=10)
ax1.view_init(elev=elev, azim=azim)
ax1.set_xlabel('x'); ax1.set_ylabel('y'); ax1.set_zlabel('z')
ax1.tick_params(labelsize=7)
cb1 = fig.colorbar(sc1, ax=ax1, shrink=0.55, pad=0.08, label='Weight $w_i$')
cb1.ax.tick_params(labelsize=8)

# Panel (b): Flagged nodes highlighted
ax2 = fig.add_subplot(132, projection='3d')
flagged = diag.flagged_indices
normal_mask = np.ones(N, dtype=bool)
normal_mask[flagged] = False

ax2.scatter(coords[normal_mask, 0], coords[normal_mask, 1], coords[normal_mask, 2],
            c='#cccccc', s=3, alpha=0.4, edgecolors='none', label='OK')
ax2.scatter(coords[flagged, 0], coords[flagged, 1], coords[flagged, 2],
            c='#CC0000', s=18, alpha=0.9, edgecolors='black', linewidths=0.3,
            marker='x', label=f'Flagged ({len(flagged)})')
ax2.set_title(f'(b)  Flagged nodes\n{len(flagged)} outliers ($\\sigma$-threshold = 2)',
              pad=10)
ax2.view_init(elev=elev, azim=azim)
ax2.set_xlabel('x'); ax2.set_ylabel('y'); ax2.set_zlabel('z')
ax2.tick_params(labelsize=7)
ax2.legend(loc='upper right', fontsize=8, markerscale=1.2)

# Panel (c): Cleaned mesh
ax3 = fig.add_subplot(133, projection='3d')
cc = result.coords
cw = result.weights
norm3 = Normalize(vmin=vmin, vmax=vmax)
sc3 = ax3.scatter(cc[:, 0], cc[:, 1], cc[:, 2],
                  c=cw, cmap='RdYlBu_r', norm=norm3,
                  s=4, alpha=0.8, edgecolors='none')
ax3.set_title(f'(c)  Cleaned mesh\n{result.cleaned_n:,d} nodes'
              f' ({result.original_n - result.cleaned_n} removed)', pad=10)
ax3.view_init(elev=elev, azim=azim)
ax3.set_xlabel('x'); ax3.set_ylabel('y'); ax3.set_zlabel('z')
ax3.tick_params(labelsize=7)
cb3 = fig.colorbar(sc3, ax=ax3, shrink=0.55, pad=0.08, label='Weight $w_i$')
cb3.ax.tick_params(labelsize=8)

plt.tight_layout(w_pad=1.5)

# Save
OUT = ROOT / "docs" / "images"
OUT.mkdir(parents=True, exist_ok=True)
for ext in ['png']:
    fig.savefig(OUT / f'readme_cleaning.{ext}', dpi=200, bbox_inches='tight',
                facecolor='white')
print(f"Saved to {OUT / 'readme_cleaning.png'}")

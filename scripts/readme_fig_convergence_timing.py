#!/usr/bin/env python3
"""
README figures: convergence + timing for the perifit package.

Panel layout: 2 subplots side by side
  (a) Convergence: log(dx) vs log(error), m=3, standard PD vs corrected
  (b) Timing: N vs wall-clock time (log-log), m=3,4,5

Reads: results/data/paper_bar_convergence.csv
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import csv
from pathlib import Path

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
    'legend.fontsize': 9,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.4,
    'lines.markersize': 7,
})


def load_completed_rows(path):
    rows = []
    with open(path, newline='') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        ncols = len(header)
        for raw in reader:
            if len(raw) != ncols:
                continue
            try:
                rows.append([float(v) for v in raw])
            except ValueError:
                continue
    return np.array(rows, dtype=float)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT.parent / "results" / "data"
DATA_PATH = RESULTS / "paper_bar_convergence.csv"
OUT = ROOT / "docs" / "images"
OUT.mkdir(parents=True, exist_ok=True)

data = load_completed_rows(DATA_PATH)
# Columns: 0-nu, 1-m, 2-dx, 3-N, 4-frac_surface, 5-err_PD, 6-err_w,
#           7-cond_weight_sys, 8-time_weights, 9-time_total

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))

# ===== (a) Convergence: m=3, all Poisson ratios =====
nu_vals = sorted(set(data[:, 0]))
colors = {0.10: '#CC0000', 0.25: '#0000CC', 0.30: '#228B22', 0.40: '#FF8C00'}
markers = {0.10: 'o', 0.25: 's', 0.30: '^', 0.40: 'D'}

mr = 3
for nu in nu_vals:
    mask = (data[:, 0] == nu) & (data[:, 1] == mr)
    if not np.any(mask):
        continue
    dx_vals = data[mask, 2]
    err_pd = data[mask, 5]
    err_w = data[mask, 6]
    idx = np.argsort(dx_vals)
    dx_vals, err_pd, err_w = dx_vals[idx], err_pd[idx], err_w[idx]

    eps = 1e-16
    err_pd = np.clip(err_pd, eps, None)
    err_w = np.clip(err_w, eps, None)
    log_dx = np.log(dx_vals)

    c = colors.get(nu, '#333')
    mk = markers.get(nu, 'o')

    # Standard PD (dashed, open)
    ax1.plot(log_dx, np.log(err_pd), '--', marker=mk, color=c,
             markerfacecolor='none', markeredgecolor=c,
             markersize=6, linewidth=1.0,
             label=r'standard, $\nu=%.2f$' % nu)
    # Corrected (solid, filled)
    ax1.plot(log_dx, np.log(err_w), '-', marker=mk, color=c,
             markerfacecolor=c, markeredgecolor=c,
             markersize=5.5, linewidth=1.3,
             label=r'corrected, $\nu=%.2f$' % nu)

# Reference slope
x_ref = np.array([-3.8, -2.5])
y_ref = x_ref - 0.4
ax1.plot(x_ref, y_ref, 'k-', linewidth=0.8, alpha=0.5)
ax1.text(-2.45, y_ref[-1] + 0.12, 'slope 1', fontsize=8, alpha=0.6)

ax1.set_xlabel(r'$\log(\Delta x)$')
ax1.set_ylabel(r'$\log(e_u)$')
ax1.set_title(r'(a)  Displacement error convergence ($m = 3$)')
leg1 = ax1.legend(loc='lower right', frameon=True, fancybox=False,
                  edgecolor='black', framealpha=1.0, ncol=2, fontsize=7.5)
leg1.get_frame().set_linewidth(0.5)

# ===== (b) Timing: all m values =====
nu_ref = 0.25
m_ratios = sorted(set(data[:, 1].astype(int)))
colors_m = {3: '#CC0000', 4: '#0000CC', 5: '#228B22'}
markers_m = {3: 'o', 4: 's', 5: '^'}

for mr in m_ratios:
    mask = (data[:, 0] == nu_ref) & (data[:, 1] == mr) & (data[:, 8] > 0)
    if not np.any(mask):
        continue
    N_vals = data[mask, 3]
    t_w = data[mask, 8]
    idx = np.argsort(N_vals)
    N_vals, t_w = N_vals[idx], t_w[idx]

    c = colors_m.get(mr, '#333')
    mk = markers_m.get(mr, 'o')
    ax2.loglog(N_vals, t_w, '-', marker=mk, color=c,
               markerfacecolor=c, markeredgecolor=c,
               markersize=6, linewidth=1.3,
               label=r'$m = %d$' % mr)

    # Fit scaling exponent
    if len(N_vals) >= 3:
        coeffs = np.polyfit(np.log(N_vals), np.log(t_w), 1)
        alpha = coeffs[0]
        N_fit = np.logspace(np.log10(N_vals.min()), np.log10(N_vals.max()), 50)
        t_fit = np.exp(coeffs[1]) * N_fit ** alpha
        ax2.loglog(N_fit, t_fit, '--', color=c, linewidth=0.8, alpha=0.5)
        ax2.text(N_vals[-1] * 1.15, t_w[-1],
                 r'$\mathcal{O}(N^{%.2f})$' % alpha, fontsize=8.5, color=c,
                 va='center')

ax2.set_xlabel(r'Number of nodes $N$')
ax2.set_ylabel(r'Weight computation time [s]')
ax2.set_title('(b)  Preprocessing time')
leg2 = ax2.legend(loc='upper left', frameon=True, fancybox=False,
                  edgecolor='black', framealpha=1.0)
leg2.get_frame().set_linewidth(0.5)

plt.tight_layout(w_pad=2.5)

for ext in ['png']:
    fig.savefig(OUT / f'readme_convergence_timing.{ext}',
                dpi=200, bbox_inches='tight', facecolor='white')
print(f"Saved to {OUT / 'readme_convergence_timing.png'}")

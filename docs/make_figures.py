"""Figures of the README (light and dark variants).

    python docs/make_figures.py              # all figures, both themes
    python docs/make_figures.py hero setups  # selected figures

Weight fields are computed with the public API; convergence data are read
from docs/data/*.csv, written by examples/benchmarks.py.  Style: one palette
per theme (THEMES), one style per series (SERIES), minimal text.
"""
from __future__ import annotations

import csv
import sys
from functools import lru_cache
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Rectangle, Wedge

import perifit
from perifit.benchmarks import cantilever_exact, cylinder

DOCS = Path(__file__).resolve().parent
IMAGES = DOCS / "images"
DATA = DOCS / "data"

THEMES = {
    "light": dict(bg="#FFFFFF", pd="#7A7A7A", nodal="#0072B2", bond="#D55E00", text="#1F2328",
                  muted="#59636E", grid="#E6E8EB", node="#B8BEC6", body="#F3F5F7",
                  field="magma_r", transparent=False),
    "dark": dict(bg="#0D1117", pd="#9E9E9E", nodal="#56B4E9", bond="#FF8A3D", text="#E6EDF3",
                 muted="#9198A1", grid="#30363D", node="#5A636E", body="#161B22",
                 field="magma", transparent=True),
}


#: series -> (label, marker, line style, line width); colours come from the theme
SERIES = {
    "pd": ("standard PD", "o", (0, (4, 2.5)), 1.6),
    "nodal": ("nodal weights", "s", "-", 2.0),
    "bond": ("bond weights", "D", "-", 2.0),
}


def plot_series(ax, key, x, y, t, plot="loglog"):
    """Plot one series in its house style (open markers for standard PD)."""
    _, marker, ls, lw = SERIES[key]
    hollow = key == "pd"
    getattr(ax, plot)(x, y, ls=ls, marker=marker, lw=lw, color=t[key],
                      mfc=t["bg"] if hollow else t[key], mec=t[key] if hollow else t["bg"],
                      mew=1.0, ms=6, zorder=3)


def spread(values, min_ratio=1.35, step=6.0):
    """Vertical label offsets (points) that keep labels of nearby log-values apart."""
    order = np.argsort(values)
    offsets = np.zeros(len(values))
    for a, b in zip(order[:-1], order[1:]):
        if values[b] / values[a] < min_ratio:
            offsets[a] -= step
            offsets[b] += step
    return offsets


def first_font(names):
    available = {f.name for f in mpl.font_manager.fontManager.ttflist}
    return next((n for n in names if n in available), "DejaVu Sans")


FONT = first_font(["Inter", "Source Sans 3", "Lato", "DejaVu Sans"])


def style(t):
    plt.rcdefaults()
    plt.rcParams.update({
        "font.family": FONT, "font.size": 10.5, "text.color": t["text"],
        "axes.titlesize": 12, "axes.titleweight": "semibold", "axes.titlelocation": "left",
        "axes.titlepad": 8, "axes.titlecolor": t["text"],
        "axes.labelsize": 10, "axes.labelcolor": t["muted"], "axes.edgecolor": t["muted"],
        "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.facecolor": "none", "figure.facecolor": t["bg"],
        "axes.grid": True, "axes.axisbelow": True, "grid.color": t["grid"], "grid.linewidth": 0.6,
        "xtick.color": t["muted"], "ytick.color": t["muted"], "xtick.labelsize": 9,
        "ytick.labelsize": 9, "xtick.major.size": 3, "ytick.major.size": 3,
        "xtick.minor.size": 0, "ytick.minor.size": 0,
        "lines.linewidth": 2.0, "lines.markersize": 6, "lines.solid_capstyle": "round",
        "mathtext.fontset": "custom", "mathtext.rm": FONT, "mathtext.it": f"{FONT}:italic",
        "mathtext.bf": f"{FONT}:bold", "legend.frameon": False,
    })


def save(fig, name, t, theme):
    IMAGES.mkdir(parents=True, exist_ok=True)
    suffix = "" if theme == "light" else "_dark"
    fig.savefig(IMAGES / f"{name}{suffix}.png", dpi=200, bbox_inches="tight", pad_inches=0.12,
                facecolor="none" if t["transparent"] else t["bg"], transparent=t["transparent"])
    plt.close(fig)


def field_cmap(t):
    base = plt.get_cmap(t["field"])
    return mcolors.ListedColormap(base(np.linspace(0.04, 0.9, 256)))


def plate(n):
    """Unit square with (n+1)^2 nodes; returns (nodes, volumes, horizon = 4 dx)."""
    s = np.linspace(0.0, 1.0, n + 1)
    X, Y = np.meshgrid(s, s, indexing="ij")
    return np.column_stack([X.ravel(), Y.ravel()]), np.full((n + 1) ** 2, (1.0 / n) ** 2), 4.0 / n * 1.0001


def colorbar(fig, ax, mappable, t, ticks, rect=(0.2, -0.085, 0.6, 0.032)):
    """Thin horizontal colour bar below an axes, labelled at the given ticks."""
    cb = fig.colorbar(mappable, cax=ax.inset_axes(list(rect)), orientation="horizontal",
                      ticks=ticks, extend="max")
    cb.ax.set_xticklabels([f"{v:.3g}" for v in ticks])
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, labelsize=9, colors=t["muted"], pad=3)
    return cb


# ---------------------------------------------------------------------------
# Hero: truncated horizons | nodal weights | bond weights
# ---------------------------------------------------------------------------
def hero(t, theme):
    fig = plt.figure(figsize=(12.4, 4.6))
    gs = fig.add_gridspec(1, 3, wspace=0.10)
    axs = [fig.add_subplot(gs[k]) for k in range(3)]
    for ax in axs:
        ax.set_aspect("equal"); ax.axis("off")

    # (a) truncated horizons on a node grid
    ax = axs[0]
    nodes, _, delta = plate(16)
    ax.add_patch(Rectangle((0, 0), 1, 1, fc=t["body"], ec=t["muted"], lw=0.9, zorder=0))
    ax.scatter(*nodes.T, s=6, c=t["node"], lw=0, zorder=2)
    for centre, colour in (((0.5, 0.5), t["nodal"]), ((0.0, 0.0), t["bond"])):
        c = np.array(centre)
        dist = np.linalg.norm(nodes - c, axis=1)
        nb = nodes[(dist <= delta) & (dist > 0)]
        ax.add_collection(LineCollection([[c, p] for p in nb], colors=colour, lw=0.6,
                                         alpha=0.55, zorder=3))
        ax.scatter(*nb.T, s=12, c=colour, lw=0, zorder=4)
        ax.scatter(*c, s=40, c=colour, ec=t["bg"], lw=1.0, zorder=5)
        if centre == (0.5, 0.5):
            ax.add_patch(Circle(c, delta, fc=colour, alpha=0.10, ec="none", zorder=1))
            ax.add_patch(Circle(c, delta, fc="none", ec=colour, lw=1.1, zorder=3))
        else:
            ax.add_patch(Wedge(c, delta, 0, 90, fc=colour, alpha=0.13, ec="none", zorder=1))
            ax.add_patch(Wedge(c, delta, 0, 90, fc="none", ec=colour, lw=1.1, zorder=3))
            ax.add_patch(Wedge(c, delta, 90, 360, fc="none", ec=colour, lw=0.9, ls=(0, (3, 2.5)),
                               alpha=0.8, zorder=3))
    ax.annotate("", xy=(0.5 + delta, 0.5), xytext=(0.5, 0.5),
                arrowprops=dict(arrowstyle="-|>", color=t["text"], lw=0.9, shrinkA=0, shrinkB=0,
                                mutation_scale=8), zorder=6)
    ax.text(0.5 + delta / 2, 0.53, r"$\delta$", ha="center", va="bottom", fontsize=12, color=t["text"])
    ax.set_xlim(-0.3, 1.05); ax.set_ylim(-0.3, 1.05)
    ax.set_title("Truncated horizons", loc="center")

    # (b) nodal-weight field
    ax = axs[1]
    n = 40
    nodes, V, delta = plate(n)
    w = perifit.nodal_weights(nodes, V, delta).reshape(n + 1, n + 1)
    lo, hi = float(np.median(w)), float(np.percentile(w, 99))
    edges = np.linspace(-0.5 / n, 1 + 0.5 / n, n + 2)
    pm = ax.pcolormesh(edges, edges, w.T, cmap=field_cmap(t), vmin=lo, vmax=hi,
                       shading="flat", rasterized=True)
    ax.add_patch(Rectangle((-0.5 / n, -0.5 / n), 1 + 1 / n, 1 + 1 / n, fc="none", ec=t["muted"], lw=0.9))
    ax.set_xlim(-0.04, 1.04); ax.set_ylim(-0.04, 1.04)
    ax.set_title(r"Nodal weights  $w_i$", loc="center")
    colorbar(fig, ax, pm, t, [lo, hi])

    # (c) bond weights: the bonds of a corner, an edge and an interior node
    ax = axs[2]
    n = 16
    nodes, V, delta = plate(n)
    bonds, beta = perifit.bond_weights(nodes, V, delta)
    ax.add_patch(Rectangle((0, 0), 1, 1, fc=t["body"], ec=t["muted"], lw=0.9, zorder=0))
    ax.scatter(*nodes.T, s=6, c=t["node"], lw=0, zorder=2)
    cmap = plt.get_cmap("RdBu_r" if theme == "light" else "berlin")
    norm = mcolors.TwoSlopeNorm(vmin=0.0, vcenter=1.0, vmax=2.5)
    for centre in ((0.0, 0.0), (0.5, 0.0), (0.5, 0.5)):
        i = int(np.argmin(np.linalg.norm(nodes - centre, axis=1)))
        mine = np.flatnonzero((bonds[:, 0] == i) | (bonds[:, 1] == i))
        order = mine[np.argsort(np.abs(beta[mine] - 1.0))]
        lc = LineCollection(nodes[bonds[order]], array=beta[order], cmap=cmap, norm=norm,
                            lw=2.0, capstyle="round", zorder=3)
        ax.add_collection(lc)
        ax.scatter(*nodes[i], s=40, c=t["text"], ec=t["bg"], lw=1.0, zorder=5)
    ax.set_xlim(-0.3, 1.05); ax.set_ylim(-0.3, 1.05)
    ax.set_title(r"Bond weights  $\beta_b$", loc="center")
    colorbar(fig, ax, lc, t, [0, 1, 2.5])
    save(fig, "hero", t, theme)


# ---------------------------------------------------------------------------
# Setups of the three benchmarks
# ---------------------------------------------------------------------------
def _hatch_wall(ax, x, y0, y1, t, depth=0.05, n=7):
    """Clamped edge: a solid line with hatching on its left."""
    ax.plot([x, x], [y0, y1], color=t["text"], lw=1.4, solid_capstyle="butt", zorder=4)
    for yy in np.linspace(y0, y1, n):
        ax.plot([x, x - depth], [yy, yy - depth], color=t["muted"], lw=0.8, zorder=4)


def _arrow(ax, start, end, t, colour=None, lw=1.2, ms=9):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=ms, lw=lw,
                                 color=colour or t["text"], shrinkA=0, shrinkB=0, zorder=5))


def setups(t, theme):
    fig = plt.figure(figsize=(12.4, 3.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.0, 1.25], wspace=0.08, bottom=0.12)
    axs = [fig.add_subplot(gs[k]) for k in range(3)]
    for ax in axs:
        ax.set_aspect("equal"); ax.axis("off")

    # (a) 1-D bar
    ax = axs[0]
    x = (np.arange(16) + 0.5) / 16
    ax.plot([0, 1], [0, 0], color=t["muted"], lw=1.0, zorder=1)
    ax.scatter(x, np.zeros_like(x), s=22, c=t["node"], ec=t["muted"], lw=0.5, zorder=3)
    i = len(x) - 1                                    # truncated horizon of the end node
    ax.add_patch(Rectangle((x[i] - 4 / 16, -0.06), 4 / 16, 0.12, fc=t["bond"], alpha=0.14,
                           ec=t["bond"], lw=0.9))
    ax.add_patch(Rectangle((x[i], -0.06), 4 / 16, 0.12, fc="none", ec=t["bond"], lw=0.9,
                           ls=(0, (3, 2))))
    ax.scatter(x[i], 0, s=36, c=t["bond"], ec=t["bg"], lw=0.8, zorder=4)
    _hatch_wall(ax, 0.0, -0.12, 0.12, t)
    _arrow(ax, (x[-1] + 0.02, 0), (x[-1] + 0.2, 0), t)
    ax.text(x[-1] + 0.11, -0.12, "P", fontsize=11, color=t["text"], style="italic", ha="center")
    ax.set_xlim(-0.12, 1.32); ax.set_ylim(-0.3, 0.3)

    # (b) 2-D cantilever with parabolic end shear
    ax = axs[1]
    nodes, _, _ = plate(10)
    ax.add_patch(Rectangle((0, 0), 1, 1, fc=t["body"], ec=t["muted"], lw=0.9, zorder=0))
    ax.scatter(*nodes.T, s=9, c=t["node"], lw=0, zorder=2)
    s = np.linspace(0, 1, 120)                        # boundary, counter-clockwise
    edge = np.concatenate([np.column_stack([s, 0 * s]), np.column_stack([1 + 0 * s, s]),
                           np.column_stack([s[::-1], 1 + 0 * s]), np.column_stack([0 * s, s[::-1]])])
    outline = edge + 0.022 * cantilever_exact(*edge.T)  # exact deformed shape, scaled
    ax.add_patch(Polygon(outline, closed=True, fc="none", ec=t["nodal"], lw=1.0, ls=(0, (4, 2.5)),
                         alpha=0.9, zorder=3))
    _hatch_wall(ax, 0.0, 0.0, 1.0, t, n=9)
    for yv in np.linspace(0.1, 0.9, 6):              # parabolic end shear (arrow length ~ t_y)
        L = 0.13 * 4 * yv * (1 - yv) + 0.02
        _arrow(ax, (1.08, yv + L / 2), (1.08, yv - L / 2), t, lw=1.1, ms=7)
    ax.set_xlim(-0.12, 1.25); ax.set_ylim(-0.16, 1.08)

    # (c) 3-D circular cantilever (oblique sketch)
    ax = axs[2]
    R, Lc, k = 0.42, 1.25, 0.38              # radius, length, foreshortening of the end faces
    th = np.linspace(0, 2 * np.pi, 200)
    ell = lambda x0: (x0 + k * R * np.cos(th), R * np.sin(th))
    ax.fill_between([0, Lc], [-R, -R], [R, R], color=t["body"], lw=0, zorder=0)
    ax.plot([0, Lc], [R, R], color=t["muted"], lw=0.9)
    ax.plot([0, Lc], [-R, -R], color=t["muted"], lw=0.9)
    back = (th > np.pi / 2) & (th < 3 * np.pi / 2)            # clamped face: visible half solid
    ex, ey = ell(0.0)
    ax.plot(ex[back], ey[back], color=t["text"], lw=1.4, zorder=1)
    ax.plot(ex[~back], ey[~back], color=t["muted"], lw=0.8, ls=(0, (3, 2.5)), zorder=1)
    for a in np.linspace(np.pi / 2 + 0.15, 3 * np.pi / 2 - 0.15, 8):   # clamp hatching
        px, py = k * R * np.cos(a), R * np.sin(a)
        ax.plot([px, px - 0.07], [py, py - 0.05], color=t["muted"], lw=0.8, zorder=1)
    ex, ey = ell(Lc)
    ax.fill(ex, ey, color=t["bg"], ec=t["muted"], lw=0.9, zorder=2)
    g = np.linspace(-R, R, 9)                                 # node sample on the end face
    GY, GZ = np.meshgrid(g, g)
    inside = GY ** 2 + GZ ** 2 <= R ** 2 * 1.0001
    ax.scatter(Lc + k * GZ[inside], GY[inside], s=8, c=t["node"], lw=0, zorder=3)
    for zz in (-0.6, -0.2, 0.2, 0.6):
        for yv in (-0.45, 0.0, 0.45):
            ya, za = yv * R, zz * R
            if ya ** 2 + za ** 2 < R ** 2:
                L = 0.13 * (1 - (ya ** 2 + 0.3 * za ** 2) / R ** 2)
                _arrow(ax, (Lc + k * za, ya + L / 2), (Lc + k * za, ya - L / 2), t, lw=0.9, ms=6)
    ax.set_xlim(-0.25, Lc + 0.3); ax.set_ylim(-0.62, 0.62)
    for ax, label in zip(axs, ("1-D bar under tension", "2-D cantilever, end shear",
                               "3-D circular cantilever")):             # common caption baseline
        box = ax.get_position()
        fig.text((box.x0 + box.x1) / 2, 0.02, label, ha="center", fontsize=10.5,
                 color=t["text"], fontweight="semibold")
    save(fig, "setups", t, theme)


# ---------------------------------------------------------------------------
# Convergence
# ---------------------------------------------------------------------------
def read_csv(name):
    with (DATA / name).open() as fh:
        return list(csv.DictReader(fh))


def convergence(t, theme):
    rows = read_csv("convergence.csv")
    fig, axs = plt.subplots(1, 3, figsize=(12.4, 4.0), sharey=True)
    titles = {"bar": "1-D bar", "cantilever": "2-D cantilever", "cylinder": "3-D circular cantilever"}
    for ax, prob in zip(axs, titles):
        sel = [r for r in rows if r["problem"] == prob and float(r["m"]) == 4.0]
        dx = np.array([float(r["dx"]) for r in sel])
        finest = {}
        for key in SERIES:
            e = 100 * np.array([float(r[key]) for r in sel])
            plot_series(ax, key, dx, e, t)
            finest[key] = e[-1]
        # label every line at its finest grid (full names on the first panel only)
        offsets = spread(np.array(list(finest.values())))
        for (key, v), dy in zip(finest.items(), offsets):
            txt = f"{SERIES[key][0]}  {v:.2g}%" if prob == "bar" else f"{v:.2g}%"
            ax.annotate(txt, (dx[-1], v), xytext=(7, dy), textcoords="offset points",
                        va="center", fontsize=9, color=t[key], fontweight="semibold")
        ax.set_xlim(dx.max() * 1.25, dx.min() / (3.2 if prob == "bar" else 2.2))
        ax.set_title(titles[prob])
        ax.set_xlabel(r"grid spacing  $\Delta x$")
        ax.set_xticks(dx)
        ax.set_xticklabels([f"1/{round(1 / d)}" for d in dx])
        ax.minorticks_off()
    axs[0].set_ylim(0.06, 150)
    axs[0].set_yticks([0.1, 1, 10, 100])
    axs[0].set_yticklabels(["0.1%", "1%", "10%", "100%"])
    axs[0].set_ylabel("displacement error")
    # slope-1 guide on the 2-D panel
    ax = axs[1]
    x0, x1, y0 = 0.02, 0.01, 0.55
    ax.plot([x0, x1, x0, x0], [y0, y0 * x1 / x0, y0 * x1 / x0, y0], color=t["muted"], lw=0.8)
    ax.text(x0 * 1.06, y0 * 0.7, "1", fontsize=8.5, color=t["muted"], ha="right", va="center")
    fig.subplots_adjust(wspace=0.12)
    save(fig, "convergence", t, theme)


# ---------------------------------------------------------------------------
# Horizon ratio (2-D)
# ---------------------------------------------------------------------------
def horizon_ratio(t, theme):
    rows = read_csv("horizon_ratio.csv") + read_csv("convergence.csv")
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    ms = np.array([3.0, 4.0, 5.0])
    ax.axvspan(2.8, 4.2, color=t["nodal"], alpha=0.06, lw=0)
    ax.text(3.5, 75, "nodal weights validated", ha="center", fontsize=8.5, color=t["nodal"])
    finest = {float(r["m"]): r for r in rows
              if r["problem"] == "cantilever" and abs(float(r["dx"]) - 0.005) < 1e-12}
    for key in SERIES:
        e = [100 * float(finest[m][key]) for m in ms]
        plot_series(ax, key, ms, e, t, plot="semilogy")
        ax.annotate(SERIES[key][0], (ms[0], e[0]), xytext=(-14, 0), textcoords="offset points",
                    ha="right", va="center", fontsize=9, color=t[key], fontweight="semibold")
        for m, v in zip(ms, e):
            ax.annotate(f"{v:.2g}%", (m, v), xytext=(0, 9 if key != "bond" else -10),
                        textcoords="offset points", ha="center",
                        va="bottom" if key != "bond" else "top", fontsize=8.5, color=t[key],
                        fontweight="semibold",
                        bbox=dict(boxstyle="round,pad=0.12", fc=t["bg"], ec="none", alpha=0.85))
    ax.set_xticks(ms); ax.set_xticklabels(["m = 3", "m = 4", "m = 5"])
    ax.set_xlim(2.05, 5.3); ax.set_ylim(0.15, 120)
    ax.set_yticks([1, 10, 100]); ax.set_yticklabels(["1%", "10%", "100%"]); ax.minorticks_off()
    ax.set_title(r"2-D cantilever,  $\Delta x$ = 1/200")
    save(fig, "horizon_ratio", t, theme)


# ---------------------------------------------------------------------------
# 3-D weights on the circular bar: longitudinal section + cross-section
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _cylinder_weights(n=30):
    """Nodal weights and per-node mean bond weight on the circular bar (computed once)."""
    prob = cylinder(1 / n)
    bonds = prob.bonds
    corr_n = perifit.surface_correction(prob.nodes, prob.volumes, prob.horizon,
                                        method="nodal", bonds=bonds)
    corr_b = perifit.surface_correction(prob.nodes, prob.volumes, prob.horizon,
                                        method="bond", bonds=bonds)
    N = len(prob.nodes)
    count = np.bincount(bonds.ravel(), minlength=N)
    mean_beta = (np.bincount(bonds[:, 0], corr_b.factor, N)
                 + np.bincount(bonds[:, 1], corr_b.factor, N)) / np.maximum(count, 1)
    return prob.nodes, corr_n.node_weights, mean_beta


def weights_3d(t, theme):
    n = 30
    nodes, w, mean_beta = _cylinder_weights(n)
    h = 1 / n
    ij = np.rint(nodes / h).astype(int)                 # lattice indices (x: 0..n, y/z: -n/2..n/2)
    half = n // 2

    def section(vals, axis_fixed, index):
        """Field on the lattice plane where coordinate `axis_fixed` has lattice index `index`."""
        sel = ij[:, axis_fixed] == index
        a, b = [k for k in range(3) if k != axis_fixed]
        A = np.full((ij[:, a].max() - ij[:, a].min() + 1, 2 * half + 1), np.nan)
        A[ij[sel, a] - ij[:, a].min(), ij[sel, b] + half] = vals[sel]
        return A

    fig = plt.figure(figsize=(12.4, 3.5))
    gs = fig.add_gridspec(1, 5, width_ratios=[2.0, 1.0, 0.25, 2.0, 1.0], wspace=0.06)
    panels = [(w, r"Nodal weights  $w_i$", gs[0], gs[1]),
              (mean_beta, r"Bond weights  (mean $\beta_b$ at each node)", gs[3], gs[4])]
    for vals, title, g_long, g_cross in panels:
        lo, hi = float(np.median(vals)), float(np.percentile(vals, 99))
        kw = dict(cmap=field_cmap(t), vmin=lo, vmax=hi, shading="flat", rasterized=True)
        ax = fig.add_subplot(g_long)
        L = section(vals, 2, 0)                         # z = 0: x along, y across
        xe = (np.arange(L.shape[0] + 1) - 0.5) * h
        ye = (np.arange(-half, half + 2) - 0.5) * h
        pm = ax.pcolormesh(xe, ye, np.ma.masked_invalid(L).T, **kw)
        ax.add_patch(Rectangle((-h / 2, -0.5 - h / 2), 1 + h, 1 + h, fc="none", ec=t["muted"], lw=0.8))
        ax.axvline(0.5, color=t["text"], lw=0.8, ls=(0, (3, 2)))
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(title, loc="left")
        ax2 = fig.add_subplot(g_cross)
        C = section(vals, 0, half)                      # x = L/2: y across, z up
        ax2.pcolormesh(ye, ye, np.ma.masked_invalid(C).T, **kw)
        ax2.add_patch(Circle((0, 0), 0.5 + h / 2, fc="none", ec=t["muted"], lw=0.8))
        ax2.set_aspect("equal"); ax2.axis("off")
        colorbar(fig, ax, pm, t, [lo, hi], rect=(0.25, -0.12, 0.5, 0.045))
    save(fig, "weights_3d", t, theme)


FIGURES = {"hero": hero, "setups": setups, "convergence": convergence,
           "horizon_ratio": horizon_ratio, "weights_3d": weights_3d}

if __name__ == "__main__":
    names = sys.argv[1:] or list(FIGURES)
    for theme, t in THEMES.items():
        style(t)
        for name in names:
            FIGURES[name](t, theme)
            print(f"{name} ({theme})")

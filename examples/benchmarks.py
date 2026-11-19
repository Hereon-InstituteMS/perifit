"""Benchmarks: standard PD vs nodal weights vs bond weights (1-D, 2-D, 3-D).

Runs the three problems of :mod:`perifit.benchmarks` (1-D bar, 2-D cantilever,
3-D circular cantilever) on a sequence of grids and writes the relative L2
displacement errors to a CSV file, which docs/make_figures.py turns into the
README figures.

Usage
-----
    python examples/benchmarks.py                     # all problems -> docs/data/convergence.csv
    python examples/benchmarks.py --quick             # two coarsest grids only (about a minute)
    python examples/benchmarks.py --problem cantilever --m 3 5 --out docs/data/horizon_ratio.csv
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

import perifit
from perifit.benchmarks import bar, cantilever, cylinder

ROOT = Path(__file__).resolve().parents[1]

#: problem -> (constructor, grid spacings)
PROBLEMS = {
    "bar": (bar, [1 / 25, 1 / 50, 1 / 100, 1 / 200, 1 / 400]),
    "cantilever": (cantilever, [1 / 25, 1 / 50, 1 / 100, 1 / 200]),
    "cylinder": (cylinder, [1 / 20, 1 / 30, 1 / 40, 1 / 50]),
}


def run(name: str, dx: float, m: float = 4.0) -> dict:
    """Errors of standard PD, nodal weights and bond weights on one grid (plus timings)."""
    prob = PROBLEMS[name][0](dx, m)
    bonds = prob.bonds
    row = {"problem": name, "m": m, "dx": dx, "nodes": len(prob.nodes), "bonds": len(bonds),
           "pd": prob.error(bonds, np.ones(len(bonds)))}
    for method in ("nodal", "bond"):
        t0 = time.perf_counter()
        corr = perifit.surface_correction(prob.nodes, prob.volumes, prob.horizon,
                                          method=method, bonds=bonds)
        row[f"t_{method}"] = time.perf_counter() - t0
        row[method] = prob.error(corr.bonds, corr.factor)
    return row


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--problem", choices=list(PROBLEMS), nargs="*", default=list(PROBLEMS))
    ap.add_argument("--m", type=float, nargs="*", default=[4.0], help="horizon ratios delta/dx")
    ap.add_argument("--quick", action="store_true", help="two coarsest grids only")
    ap.add_argument("--out", default=str(ROOT / "docs" / "data" / "convergence.csv"))
    args = ap.parse_args(argv)

    rows = []
    for name in args.problem:
        grids = PROBLEMS[name][1][:2] if args.quick else PROBLEMS[name][1]
        for m in args.m:
            for dx in grids:
                r = run(name, dx, m)
                rows.append(r)
                print(f"{name:10s} m={m:g} dx=1/{round(1 / dx):<4d} N={r['nodes']:7d}   "
                      f"PD {r['pd']:.3e}   nodal {r['nodal']:.3e}   bond {r['bond']:.3e}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

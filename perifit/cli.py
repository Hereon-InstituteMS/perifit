"""
Command-line interface for perifit.

Usage
-----
    # Minimal: auto-infer dx and horizon (m=3 default)
    perifit --mesh cube.csv

    # Explicit horizon-to-spacing ratio
    perifit --mesh cube.csv --m 4

    # Explicit horizon
    perifit --mesh cube.csv --horizon 0.06 --output csv,vtk

    # Bond-based PD model
    perifit --mesh cube.csv --model bb

    # Exodus mesh for Peridigm
    perifit --mesh solid.exo --output peridigm --outdir ./peridigm_run

    # Diagnose mesh quality
    perifit --mesh cube.csv --diagnose

    # Clean mesh (remove flagged nodes)
    perifit --mesh cube.csv --clean --threshold-sigma 2.0

    # Built-in demo (no mesh needed)
    perifit --demo

    perifit --help
"""

from __future__ import annotations

import argparse


# Map CLI model names to the core API model codes
_MODEL_MAP = {
    'bb': 'bb',
    'osb': 'osb',
    'bb_pd': 'bb',
    'osb_pd': 'osb',
    'bond_based': 'bb',
    'state_based': 'osb',
}


def _infer_dx(coords):
    """Estimate nodal spacing as the median nearest-neighbour distance."""
    from scipy.spatial import KDTree
    import numpy as np
    tree = KDTree(coords)
    dists, _ = tree.query(coords, k=2)
    return float(np.median(dists[:, 1]))


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='perifit',
        description=(
            'Compute peridynamic surface correction weights for any 3-D mesh '
            'and export to CSV, DAT, VTK, Peridigm, or PeriLab format.\n\n'
            'Supports bond-based (BB) and ordinary state-based (OSB) PD models.\n\n'
            'Nodal spacing (dx) and volumes are inferred automatically from '
            'the median nearest-neighbour distance (assumes a regular grid). '
            'The horizon is delta = m * dx when --horizon is not given.'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  # Minimal -- auto-infer everything (m=3 default)
  perifit --mesh cube.csv

  # Bond-based PD
  perifit --mesh cube.csv --model bb

  # Specify horizon-to-spacing ratio
  perifit --mesh cube.csv --m-ratio 4 --output csv,vtk

  # Provide explicit horizon
  perifit --mesh solid.exo --horizon 0.005 --output peridigm

  # Gmsh mesh (requires meshio) for PeriLab
  perifit --mesh part.msh --output perilab,csv --stem part_weights

  # Diagnose mesh quality after computing weights
  perifit --mesh cube.csv --diagnose

  # Clean mesh (remove negative/outlier weight nodes)
  perifit --mesh cube.csv --clean --threshold-sigma 2.0

  # Built-in 1000-node cube demo
  perifit --demo
""",
    )

    # Input
    p.add_argument('--mesh', required=False, default=None,
                   help='Path to the mesh file. Supported extensions: '
                        '.csv, .dat, .txt, .npy, .npz, .vtk, .vtu, '
                        '.exo, .g, .msh, .inp')

    # Model selection
    p.add_argument('--model', default='osb',
                   choices=['osb', 'bb', 'osb_pd', 'bb_pd',
                            'state_based', 'bond_based'],
                   help='PD model: osb (LPS/state-based), bb (bond-based). '
                        'Legacy aliases: osb_pd, bb_pd, state_based, bond_based. '
                        'Default: osb.')

    # Horizon / discretisation
    g = p.add_argument_group('horizon / discretisation')
    g.add_argument('--horizon', type=float, default=None,
                   help='Peridynamic horizon radius delta. '
                        'Auto-computed as m*dx when not given.')
    g.add_argument('--m-ratio', type=float, default=3.0, dest='m_ratio',
                   help='Horizon-to-spacing ratio  m = delta/dx. '
                        'Used only when --horizon is not given. Default: 3.')

    # Column overrides for text-based meshes
    g2 = p.add_argument_group('text-format column mapping (CSV/DAT/TXT only)')
    g2.add_argument('--x-col',    type=int, default=0,
                    help='Column index for x coordinate (default: 0).')
    g2.add_argument('--y-col',    type=int, default=1,
                    help='Column index for y coordinate (default: 1).')
    g2.add_argument('--z-col',    type=int, default=2,
                    help='Column index for z coordinate (default: 2).')
    g2.add_argument('--delimiter', default=None,
                    help='Field delimiter for text files (default: auto).')
    g2.add_argument('--comment',   default='#',
                    help='Comment character for text files (default: #).')

    # Output
    g3 = p.add_argument_group('output')
    g3.add_argument('--output', default='csv',
                    help='Comma-separated list of output formats: '
                         'csv, dat, vtk, peridigm, perilab  (default: csv).')
    g3.add_argument('--outdir', default='./perifit_output',
                    help='Output directory (created if needed).')
    g3.add_argument('--stem', default='perifit_weights',
                    help='Base filename stem (without extension).')

    # Solver
    g4 = p.add_argument_group('solver')
    g4.add_argument('--tol',      type=float, default=1e-10,
                    help='BiCGSTAB relative tolerance (default: 1e-10).')
    g4.add_argument('--max-iter', type=int,   default=5000,
                    help='Maximum BiCGSTAB iterations (default: 5000).')

    # Quality / cleaning
    g5 = p.add_argument_group('mesh quality')
    g5.add_argument('--diagnose', action='store_true',
                    help='Print weight diagnostics (negative/outlier detection).')
    g5.add_argument('--clean', action='store_true',
                    help='Remove flagged nodes and output a cleaned mesh.')
    g5.add_argument('--threshold-sigma', type=float, default=2.0,
                    dest='threshold_sigma',
                    help='Outlier detection threshold in standard deviations '
                         '(default: 2.0).')

    p.add_argument('--quiet', action='store_true',
                   help='Suppress progress output.')

    p.add_argument('--demo', action='store_true',
                   help='Run a self-contained demo on the bundled 1000-node '
                        'cube mesh and exit.  No other arguments needed.')

    p.add_argument('--version', action='store_true',
                   help='Print the version banner and exit.')

    return p


def main(argv=None):
    import numpy as np

    parser = _build_parser()
    args = parser.parse_args(argv)

    from perifit import load_mesh, write_weights, get_example_mesh
    from perifit.core.weights import compute_weights, build_families

    # Resolve model name to core API code
    model = _MODEL_MAP.get(args.model, args.model)

    # --- Version mode ---
    if args.version:
        from perifit import show_version
        show_version()
        return

    # --- Demo mode ---
    if args.demo:
        model_label = 'BB-PD' if model == 'bb' else 'OSB-PD'
        print("=" * 60)
        print("perifit  --  built-in demo (1000-node cube)")
        print("  Mesh   : structured [0,1]^3, dx=0.1, 1000 nodes")
        print("  Horizon: delta = 0.30  (m = 3, auto-inferred)")
        print(f"  Model  : {model_label}")
        print("=" * 60)
        mesh_path = get_example_mesh()
        coords, volumes = load_mesh(mesh_path)
        dx = _infer_dx(coords)
        horizon = args.m_ratio * dx
        weights = compute_weights(coords, volumes, horizon,
                                  model=model, verbose=not args.quiet)
        outdir = args.outdir
        paths = write_weights(
            coords, volumes, weights,
            outdir=outdir,
            stem="demo_cube",
            formats=["csv", "dat", "vtk"],
        )
        print("\nOutput files:")
        for fmt, path in paths.items():
            print(f"  [{fmt:6s}]  {path}")
        print(f"\nOpen {outdir}/demo_cube.vtk in ParaView to visualise the weights.")
        return

    if args.mesh is None:
        parser.error("--mesh is required (or use --demo for the built-in example)")

    # --- Load mesh ---
    print(f"Loading mesh: {args.mesh}")
    coords, volumes = load_mesh(
        args.mesh,
        x_col=args.x_col,
        y_col=args.y_col,
        z_col=args.z_col,
        delimiter=args.delimiter,
        comment=args.comment,
        horizon=args.horizon,
    )
    print(f"  {len(coords):,d} nodes loaded.")

    # --- Infer horizon if not given ---
    if args.horizon is not None:
        horizon = args.horizon
    else:
        dx = _infer_dx(coords)
        horizon = args.m_ratio * dx
        if not args.quiet:
            print(f"  Auto-inferred dx={dx:.4g}, horizon={horizon:.4g} (m={args.m_ratio})")

    # --- Compute weights ---
    weights = compute_weights(
        coords, volumes, horizon,
        model=model,
        tol=args.tol,
        max_iter=args.max_iter,
        verbose=not args.quiet,
    )

    # --- Diagnose ---
    if args.diagnose:
        from perifit.quality.diagnostics import diagnose_weights
        diag = diagnose_weights(weights, threshold_sigma=args.threshold_sigma)
        print(f"\n{diag.summary}")

    # --- Clean ---
    if args.clean:
        from perifit.quality.cleaning import clean_mesh
        result = clean_mesh(coords, volumes, weights,
                            threshold_sigma=args.threshold_sigma)
        print(f"\nCleaning: {result.original_n} -> {result.cleaned_n} nodes "
              f"({result.original_n - result.cleaned_n} removed)")
        coords = result.coords
        volumes = result.volumes
        weights = result.weights

    # --- Write output ---
    formats = [f.strip() for f in args.output.split(',')]
    paths = write_weights(
        coords, volumes, weights,
        outdir=args.outdir,
        stem=args.stem,
        formats=formats,
        horizon=weights.mean(),  # informational only
    )

    print("\nOutput files:")
    for fmt, path in paths.items():
        print(f"  [{fmt:10s}]  {path}")


if __name__ == '__main__':
    main()

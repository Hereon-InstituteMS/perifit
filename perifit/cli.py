"""
Command-line interface for perifit.

Usage
-----
    # Choose the method explicitly: nodal weights or bond weights
    perifit --mesh cube.csv --method nodal
    perifit --mesh cube.csv --method bond --output csv,npz

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
    perifit --demo --method nodal

    perifit --help
"""

from __future__ import annotations

import argparse

import numpy as np


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
            'Compute peridynamic surface corrections for a mesh and export them.\n\n'
            'Two methods (choose one with --method):\n'
            '  nodal : one weight per node (BB or OSB model); CSV, DAT, VTK, Peridigm, PeriLab\n'
            '  bond  : one weight per bond (BB model); CSV (i,j,beta) or NPZ\n\n'
            'Nodal spacing (dx) and volumes are inferred automatically from '
            'the median nearest-neighbour distance (assumes a regular grid). '
            'The horizon is delta = m * dx when --horizon is not given.'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  # Nodal weights (one per node), auto-inferred dx and horizon (m=3 default)
  perifit --mesh cube.csv --method nodal

  # Bond weights (one per bond), bond-based PD
  perifit --mesh cube.csv --method bond --output csv,npz

  # A planar mesh stored as (x, y, 0)
  perifit --mesh plate.csv --method bond --dim 2

  # Specify horizon-to-spacing ratio
  perifit --mesh cube.csv --method nodal --m-ratio 4 --output csv,vtk

  # Provide explicit horizon
  perifit --mesh solid.exo --method nodal --horizon 0.005 --output peridigm

  # Gmsh mesh (requires meshio) for PeriLab
  perifit --mesh part.msh --method nodal --output perilab,csv --stem part_weights

  # Diagnose mesh quality after computing weights
  perifit --mesh cube.csv --method nodal --diagnose

  # Clean mesh (remove negative/outlier weight nodes)
  perifit --mesh cube.csv --method nodal --clean --threshold-sigma 2.0

  # Built-in 1000-node cube demo
  perifit --demo --method bond
""",
    )

    # Input
    p.add_argument('--mesh', required=False, default=None,
                   help='Path to the mesh file. Supported extensions: '
                        '.csv, .dat, .txt, .npy, .npz, .vtk, .vtu, '
                        '.exo, .g, .msh, .inp')

    # Method and model selection
    p.add_argument('--method', choices=['nodal', 'bond'], default=None,
                   help='Surface correction (required): nodal = one weight per node, '
                        'bond = one weight per bond.')
    p.add_argument('--model', default=None,
                   choices=['osb', 'bb', 'osb_pd', 'bb_pd',
                            'state_based', 'bond_based'],
                   help='PD model: osb (LPS/state-based, nodal only), bb (bond-based). '
                        'Default: osb for --method nodal, bb for --method bond.')
    p.add_argument('--dim', type=int, choices=[1, 2, 3], default=None,
                   help='Spatial dimension of the mesh; needed when a planar mesh is '
                        'stored with a constant z column.')

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
                    help='Comma-separated list of output formats. nodal: csv, dat, vtk, '
                         'peridigm, perilab; bond: csv, npz  (default: csv).')
    g3.add_argument('--outdir', default='./perifit_output',
                    help='Output directory (created if needed).')
    g3.add_argument('--stem', default=None,
                    help='Base filename stem (default: perifit_weights or perifit_bond_weights).')

    # Solver
    g4 = p.add_argument_group('solver')
    g4.add_argument('--tol',      type=float, default=1e-10,
                    help='BiCGSTAB relative tolerance (default: 1e-10).')
    g4.add_argument('--max-iter', type=int,   default=5000,
                    help='Maximum BiCGSTAB iterations (default: 5000).')

    # Quality / cleaning
    g5 = p.add_argument_group('mesh quality')
    g5.add_argument('--diagnose', action='store_true',
                    help='Print weight diagnostics (nodal weights only).')
    g5.add_argument('--clean', action='store_true',
                    help='Remove flagged nodes and recompute the weights (nodal weights only).')
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


def _write(corr, coords, volumes, horizon, outdir, stem, formats):
    """Write nodal weights (per-node formats) or bond weights (CSV/NPZ)."""
    from perifit import write_bond_weights, write_weights
    if corr.method == 'nodal':
        return write_weights(coords, volumes, corr.node_weights, outdir=outdir, stem=stem,
                             formats=formats, horizon=horizon)
    return write_bond_weights(coords, volumes, corr.bonds, corr.factor, outdir=outdir, stem=stem,
                              formats=formats, horizon=horizon)


def _report(corr, quiet):
    if not quiet and corr.method == 'bond':
        b = corr.factor
        print(f"  {len(b):,d} bond weights: min={b.min():.4f}  max={b.max():.4f}  "
              f"median={np.median(b):.4f}")


def main(argv=None):
    from perifit import get_example_mesh, load_mesh, surface_correction
    from perifit.api import METHOD_HELP, canonical_model

    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.version:
        from perifit import show_version
        show_version()
        return
    if args.method is None:
        parser.error(METHOD_HELP.replace("  nodal :", "  --method nodal :")
                     .replace("  bond  :", "  --method bond  :"))
    method = args.method
    if method == 'bond' and (args.diagnose or args.clean):
        parser.error("--diagnose and --clean apply to nodal weights only")
    if method == 'bond' and args.model and canonical_model(args.model) != 'bb':
        parser.error("bond weights are defined for the bond-based model only (use --model bb)")
    stem = args.stem or ('perifit_weights' if method == 'nodal' else 'perifit_bond_weights')
    formats = [f.strip() for f in args.output.split(',')]

    def model_for(dim):
        """--model, else state-based for 3-D nodal weights and bond-based otherwise."""
        if args.model:
            return canonical_model(args.model)
        return 'osb' if (method == 'nodal' and dim == 3) else 'bb'

    def correct(coords, volumes, horizon, dim):
        corr = surface_correction(coords, volumes, horizon, method=method,
                                  model=model_for(dim or coords.shape[1]),
                                  dim=dim, tol=args.tol, max_iter=args.max_iter,
                                  verbose=not args.quiet)
        _report(corr, args.quiet)
        return corr

    # --- Demo mode ---
    if args.demo:
        label = {'nodal': 'nodal weights', 'bond': 'bond weights'}[method]
        print("=" * 60)
        print("perifit  --  built-in demo (1000-node cube)")
        print("  Mesh   : structured [0,1]^3, dx=0.1, 1000 nodes")
        print(f"  Horizon: delta = {args.m_ratio * 0.1:.2f}  (m = {args.m_ratio:g})")
        print(f"  Method : {label}  ({'BB-PD' if model_for(3) == 'bb' else 'OSB-PD'})")
        print("=" * 60)
        coords, volumes = load_mesh(get_example_mesh())
        horizon = args.m_ratio * _infer_dx(coords)
        corr = correct(coords, volumes, horizon, None)
        demo_formats = ["csv", "dat", "vtk"] if method == 'nodal' else ["csv", "npz"]
        paths = _write(corr, coords, volumes, horizon, args.outdir, "demo_cube", demo_formats)
        print("\nOutput files:")
        for fmt, path in paths.items():
            print(f"  [{fmt:6s}]  {path}")
        return

    if args.mesh is None:
        parser.error("--mesh is required (or use --demo for the built-in example)")

    # --- Load mesh ---
    print(f"Loading mesh: {args.mesh}")
    coords, volumes = load_mesh(
        args.mesh, x_col=args.x_col, y_col=args.y_col, z_col=args.z_col,
        delimiter=args.delimiter, comment=args.comment, horizon=args.horizon,
    )
    print(f"  {len(coords):,d} nodes loaded.")

    if args.horizon is not None:
        horizon = args.horizon
    else:
        dx = _infer_dx(coords)
        horizon = args.m_ratio * dx
        if not args.quiet:
            print(f"  Auto-inferred dx={dx:.4g}, horizon={horizon:.4g} (m={args.m_ratio})")

    # --- Compute ---
    try:
        corr = correct(coords, volumes, horizon, args.dim)
    except ValueError as err:
        parser.error(f"{err}  (command line: --dim N)")

    if args.diagnose:
        from perifit.quality.diagnostics import diagnose_weights
        print(f"\n{diagnose_weights(corr.node_weights, threshold_sigma=args.threshold_sigma).summary}")

    if args.clean:
        from perifit.quality.cleaning import clean_mesh
        result = clean_mesh(coords, volumes, corr.node_weights, threshold_sigma=args.threshold_sigma)
        print(f"\nCleaning: {result.original_n} -> {result.cleaned_n} nodes "
              f"({result.original_n - result.cleaned_n} removed); recomputing weights")
        coords, volumes = result.coords, result.volumes
        corr = correct(coords, volumes, horizon, args.dim)

    # --- Write ---
    paths = _write(corr, coords, volumes, horizon, args.outdir, stem, formats)
    print("\nOutput files:")
    for fmt, path in paths.items():
        print(f"  [{fmt:10s}]  {path}")


if __name__ == '__main__':
    main()

"""
Output writers for peridynamic surface correction weights.

Supported output formats
------------------------
  csv      — plain CSV (node_id, x, y, z, volume, weight)
  dat      — space-delimited text, suitable for most in-house codes
  vtk      — legacy ASCII VTK (ParaView / VisIt compatible)
  peridigm — Peridigm-compatible nodal-data CSV + YAML input snippet
  perilab  — PeriLab HDF5 mesh file (requires h5py)

Usage
-----
    from perifit.io.writers import write_weights
    write_weights(coords, volumes, weights, outdir='./output',
                  formats=['csv', 'vtk', 'peridigm'])
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Master dispatcher
# ---------------------------------------------------------------------------

def write_weights(
    coords: np.ndarray,
    volumes: np.ndarray,
    weights: np.ndarray,
    outdir: str | Path = '.',
    stem: str = 'perifit_weights',
    formats: list[str] | str = 'csv',
    *,
    E: float | None = None,
    nu: float | None = None,
    horizon: float | None = None,
    block_id: int = 1,
) -> dict[str, Path]:
    """
    Write surface correction weights to one or more output formats.

    Parameters
    ----------
    coords   : (N, 3) float64 — nodal coordinates
    volumes  : (N,)   float64 — nodal volumes
    weights  : (N,)   float64 — optimised influence weights
    outdir   : output directory (created if necessary)
    stem     : base filename stem (without extension)
    formats  : one or more of: 'csv', 'dat', 'vtk', 'peridigm', 'perilab'
    E, nu, horizon : material / discretisation parameters (included in
                     file headers / metadata where applicable)
    block_id : material block ID (used in peridigm and perilab outputs)

    Returns
    -------
    paths : dict mapping format name → Path of the written file
    """
    if isinstance(formats, str):
        formats = [f.strip() for f in formats.split(',')]

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    coords  = np.asarray(coords,  dtype=np.float64)
    volumes = np.asarray(volumes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    N = len(coords)

    meta = dict(E=E, nu=nu, horizon=horizon, N=N)
    paths = {}

    for fmt in formats:
        fmt = fmt.lower().strip()
        if fmt == 'csv':
            p = _write_csv(coords, volumes, weights, outdir, stem, meta)
        elif fmt in ('dat', 'txt'):
            p = _write_dat(coords, volumes, weights, outdir, stem, meta)
        elif fmt == 'vtk':
            p = _write_vtk(coords, volumes, weights, outdir, stem, meta)
        elif fmt == 'peridigm':
            p = _write_peridigm(coords, volumes, weights, outdir, stem, meta, block_id)
        elif fmt == 'perilab':
            p = _write_perilab(coords, volumes, weights, outdir, stem, meta, block_id)
        else:
            warnings.warn(f"Unknown output format '{fmt}' — skipped.", UserWarning)
            continue
        paths[fmt] = p

    return paths


# ---------------------------------------------------------------------------
# Header builder
# ---------------------------------------------------------------------------

def _header(meta: dict) -> str:
    parts = ["Peridynamic surface correction weights (optimised nodal influence)"]
    if meta.get('N') is not None:
        parts.append(f"N = {meta['N']}")
    if meta.get('horizon') is not None:
        parts.append(f"delta = {meta['horizon']:.6g}")
    if meta.get('E') is not None:
        parts.append(f"E = {meta['E']:.6g}")
    if meta.get('nu') is not None:
        parts.append(f"nu = {meta['nu']:.6g}")
    return "  ".join(parts)


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------

def _write_csv(
    coords: np.ndarray,
    volumes: np.ndarray,
    weights: np.ndarray,
    outdir: Path,
    stem: str,
    meta: dict,
) -> Path:
    p = outdir / f"{stem}.csv"
    N = len(coords)
    lines = [
        f"# {_header(meta)}",
        "node_id,x,y,z,volume,weight",
    ]
    for i in range(N):
        lines.append(
            f"{i},{coords[i,0]:.10e},{coords[i,1]:.10e},{coords[i,2]:.10e},"
            f"{volumes[i]:.10e},{weights[i]:.10e}"
        )
    p.write_text("\n".join(lines) + "\n")
    return p


# ---------------------------------------------------------------------------
# DAT / plain-text writer
# ---------------------------------------------------------------------------

def _write_dat(
    coords: np.ndarray,
    volumes: np.ndarray,
    weights: np.ndarray,
    outdir: Path,
    stem: str,
    meta: dict,
) -> Path:
    p = outdir / f"{stem}.dat"
    N = len(coords)
    lines = [
        f"# {_header(meta)}",
        "# Columns: node_id  x  y  z  volume  weight",
    ]
    for i in range(N):
        lines.append(
            f"{i:8d}  "
            f"{coords[i,0]:18.10e}  "
            f"{coords[i,1]:18.10e}  "
            f"{coords[i,2]:18.10e}  "
            f"{volumes[i]:18.10e}  "
            f"{weights[i]:18.10e}"
        )
    p.write_text("\n".join(lines) + "\n")
    return p


# ---------------------------------------------------------------------------
# Legacy VTK writer (ASCII, no external dependency)
# ---------------------------------------------------------------------------

def _write_vtk(
    coords: np.ndarray,
    volumes: np.ndarray,
    weights: np.ndarray,
    outdir: Path,
    stem: str,
    meta: dict,
) -> Path:
    p = outdir / f"{stem}.vtk"
    N = len(coords)

    lines = [
        "# vtk DataFile Version 2.0",
        _header(meta),
        "ASCII",
        "DATASET UNSTRUCTURED_GRID",
        f"POINTS {N} double",
    ]
    for i in range(N):
        lines.append(f"{coords[i,0]:.10e} {coords[i,1]:.10e} {coords[i,2]:.10e}")

    # Write each node as a vertex cell (cell type 1)
    lines.append(f"\nCELLS {N} {2*N}")
    for i in range(N):
        lines.append(f"1 {i}")
    lines.append(f"\nCELL_TYPES {N}")
    for _ in range(N):
        lines.append("1")  # VTK_VERTEX

    lines.append(f"\nPOINT_DATA {N}")

    # Weight field
    lines.append("SCALARS weight double 1")
    lines.append("LOOKUP_TABLE default")
    for w in weights:
        lines.append(f"{w:.10e}")

    # Volume field
    lines.append("\nSCALARS volume double 1")
    lines.append("LOOKUP_TABLE default")
    for v in volumes:
        lines.append(f"{v:.10e}")

    p.write_text("\n".join(lines) + "\n")
    return p


# ---------------------------------------------------------------------------
# Peridigm writer
# ---------------------------------------------------------------------------
#
# Peridigm uses Exodus II mesh files as primary input.  When the user has
# netCDF4 available, we write a minimal Exodus-compatible mesh with the
# weight stored as a nodal variable.  Otherwise we produce a plain CSV that
# can be loaded via Peridigm's CSV discretization and a YAML template.

def _write_peridigm(
    coords: np.ndarray,
    volumes: np.ndarray,
    weights: np.ndarray,
    outdir: Path,
    stem: str,
    meta: dict,
    block_id: int,
) -> Path:
    # Attempt Exodus output first
    try:
        return _write_peridigm_exodus(coords, volumes, weights, outdir, stem, meta, block_id)
    except ImportError:
        pass

    # Fallback: CSV + YAML snippet
    return _write_peridigm_csv(coords, volumes, weights, outdir, stem, meta, block_id)


def _write_peridigm_exodus(
    coords, volumes, weights, outdir, stem, meta, block_id
):
    """Write an Exodus II file with weights as a nodal variable."""
    import netCDF4 as nc4

    p = outdir / f"{stem}_peridigm.exo"
    N = len(coords)

    ds = nc4.Dataset(str(p), 'w', format='NETCDF3_64BIT_OFFSET')
    ds.title = _header(meta)
    ds.version = np.float32(5.1)
    ds.api_version = np.float32(5.1)
    ds.floating_point_word_size = np.int32(8)
    ds.file_size = np.int32(1)

    # Dimensions
    ds.createDimension('len_string', 33)
    ds.createDimension('num_dim', 3)
    ds.createDimension('num_nodes', N)
    ds.createDimension('num_elem', N)       # one "sphere" element per node
    ds.createDimension('num_el_blk', 1)
    ds.createDimension('num_nod_var', 2)    # weight + volume
    ds.createDimension('num_el_in_blk1', N)
    ds.createDimension('num_nod_per_el1', 1)
    ds.createDimension('time_step', 1)

    # Coordinates
    cx = ds.createVariable('coordx', 'f8', ('num_nodes',))
    cy = ds.createVariable('coordy', 'f8', ('num_nodes',))
    cz = ds.createVariable('coordz', 'f8', ('num_nodes',))
    cx[:] = coords[:, 0]
    cy[:] = coords[:, 1]
    cz[:] = coords[:, 2]

    # Element connectivity (each node is its own sphere element)
    conn = ds.createVariable('connect1', 'i4', ('num_el_in_blk1', 'num_nod_per_el1'))
    conn[:, 0] = np.arange(1, N + 1)
    blk_type = ds.createVariable('elem_type1', 'S1', ('len_string',))
    blk_type._Attrib = 'SPHERE'

    # Element block attributes (radius placeholder)
    att = ds.createVariable('attrib1', 'f8', ('num_el_in_blk1',))
    att[:] = np.ones(N)

    # Nodal variable names
    nvar_names = ds.createVariable('name_nod_var', 'S1', ('num_nod_var', 'len_string'))
    for s, name in enumerate(['weight          ', 'volume          ']):
        for c, ch in enumerate(name[:33]):
            nvar_names[s, c] = ch

    # Time
    tv = ds.createVariable('time_whole', 'f8', ('time_step',))
    tv[0] = 0.0

    # Nodal variable data
    wvar = ds.createVariable('vals_nod_var1', 'f8', ('time_step', 'num_nodes'))
    wvar[0, :] = weights
    vvar = ds.createVariable('vals_nod_var2', 'f8', ('time_step', 'num_nodes'))
    vvar[0, :] = volumes

    ds.close()

    # Also write YAML template
    _write_peridigm_yaml(outdir, stem, p.name, meta, block_id)
    return p


def _write_peridigm_csv(
    coords, volumes, weights, outdir, stem, meta, block_id
):
    """
    Write a Peridigm text-mesh CSV and a YAML input-deck template.

    Peridigm's "Text File" discretization reads a file with columns:
        x  y  z  block_id  volume
    The weight field is stored separately in a companion file.
    """
    # Primary mesh file (Peridigm text format)
    p_mesh = outdir / f"{stem}_peridigm_mesh.txt"
    N = len(coords)
    lines = [
        "# Peridigm text mesh",
        f"# {_header(meta)}",
        "# x   y   z   block_id   volume",
    ]
    for i in range(N):
        lines.append(
            f"{coords[i,0]:.10e}  {coords[i,1]:.10e}  {coords[i,2]:.10e}"
            f"  {block_id}  {volumes[i]:.10e}"
        )
    p_mesh.write_text("\n".join(lines) + "\n")

    # Companion weight file
    p_w = outdir / f"{stem}_peridigm_weights.csv"
    wlines = [f"# {_header(meta)}", "node_id,x,y,z,volume,weight"]
    for i in range(N):
        wlines.append(
            f"{i},{coords[i,0]:.10e},{coords[i,1]:.10e},{coords[i,2]:.10e},"
            f"{volumes[i]:.10e},{weights[i]:.10e}"
        )
    p_w.write_text("\n".join(wlines) + "\n")

    # YAML template
    _write_peridigm_yaml(outdir, stem, p_mesh.name, meta, block_id)
    return p_w


def _write_peridigm_yaml(outdir, stem, mesh_filename, meta, block_id):
    """Write a Peridigm YAML input-deck snippet."""
    delta_str = f"{meta['horizon']:.6g}" if meta.get('horizon') else "FILL_IN"
    E_str  = f"{meta['E']:.6g}"  if meta.get('E')  else "FILL_IN"
    nu_str = f"{meta['nu']:.6g}" if meta.get('nu') else "FILL_IN"

    yaml_text = f"""\
# -----------------------------------------------------------------------
# Peridigm input deck template — generated by perifit
# Load the weight file and apply to the LPS material model.
# -----------------------------------------------------------------------

Verbose: false

Discretization:
  Type: "Text File"
  Input Mesh File: "{mesh_filename}"

Materials:
  LPS Material:
    Material Model: "Linear LPS Peridynamic Solid"
    Apply Automatic Differentiation Jacobian: false
    Bulk Modulus: {E_str}          # Replace with K = E/(3*(1-2*nu))
    Shear Modulus: {E_str}         # Replace with G = E/(2*(1+nu))
    Horizon: {delta_str}
    # Surface correction: load pre-computed nodal weights from companion CSV
    # and set  w_bar_ij = (w_i + w_j) / 2  inside the material routine.

Blocks:
  My Block:
    Block Names: "block_{block_id}"
    Material: "LPS Material"
    Horizon: {delta_str}

# -----------------------------------------------------------------------
# Usage note
# ---------
# The companion file '{stem}_peridigm_weights.csv' contains one row per
# node:  node_id, x, y, z, volume, weight.
# Apply the bond-averaged weight  w_bar = (w_i + w_j)/2  uniformly to all
# influence-function calls in your material model.
# -----------------------------------------------------------------------
"""
    (outdir / f"{stem}_peridigm_input.yaml").write_text(yaml_text)


# ---------------------------------------------------------------------------
# PeriLab writer
# ---------------------------------------------------------------------------
#
# PeriLab (Julia) reads HDF5 mesh files with the following structure:
#   /coordinates   (N, 3) float64
#   /volume        (N,)   float64
#   /block_id      (N,)   int32
#   /fieldnames    string dataset listing extra fields
#   /Nodal_Surface_Correction  (N,) float64

def _write_perilab(
    coords: np.ndarray,
    volumes: np.ndarray,
    weights: np.ndarray,
    outdir: Path,
    stem: str,
    meta: dict,
    block_id: int,
) -> Path:
    try:
        return _write_perilab_hdf5(coords, volumes, weights, outdir, stem, meta, block_id)
    except ImportError:
        warnings.warn(
            "h5py not found; falling back to CSV for PeriLab output.  "
            "Install with:  pip install h5py",
            UserWarning,
            stacklevel=3,
        )
        return _write_perilab_csv(coords, volumes, weights, outdir, stem, meta, block_id)


def _write_perilab_hdf5(
    coords, volumes, weights, outdir, stem, meta, block_id
):
    """Write a PeriLab-compatible HDF5 mesh file."""
    import h5py

    p = outdir / f"{stem}_perilab.h5"
    N = len(coords)

    with h5py.File(str(p), 'w') as f:
        f.attrs['description'] = _header(meta)
        if meta.get('horizon'):
            f.attrs['horizon'] = meta['horizon']
        if meta.get('E'):
            f.attrs['E'] = meta['E']
        if meta.get('nu'):
            f.attrs['nu'] = meta['nu']

        f.create_dataset('coordinates', data=coords, dtype='float64')
        f.create_dataset('volume',      data=volumes, dtype='float64')
        f.create_dataset('block_id',    data=np.full(N, block_id, dtype=np.int32))

        # Surface correction weights as a nodal field
        f.create_dataset('Nodal_Surface_Correction', data=weights, dtype='float64')

        # Fieldnames list (PeriLab convention)
        f.create_dataset('fieldnames',
                         data=np.array(['Nodal_Surface_Correction'], dtype='S64'))

    # Write a companion YAML snippet
    _write_perilab_yaml(outdir, stem, p.name, meta, block_id)
    return p


def _write_perilab_csv(
    coords, volumes, weights, outdir, stem, meta, block_id
):
    """CSV fallback for PeriLab when h5py is unavailable."""
    p = outdir / f"{stem}_perilab.csv"
    N = len(coords)
    plines = [
        f"# {_header(meta)}",
        "node_id,x,y,z,volume,block_id,Nodal_Surface_Correction",
    ]
    for i in range(N):
        plines.append(
            f"{i},{coords[i,0]:.10e},{coords[i,1]:.10e},{coords[i,2]:.10e},"
            f"{volumes[i]:.10e},{block_id},{weights[i]:.10e}"
        )
    p.write_text("\n".join(plines) + "\n")
    _write_perilab_yaml(outdir, stem, p.name, meta, block_id)
    return p


def _write_perilab_yaml(outdir, stem, mesh_filename, meta, block_id):
    """Write a PeriLab YAML input-deck snippet."""
    delta_str = f"{meta['horizon']:.6g}" if meta.get('horizon') else "FILL_IN"
    E_str  = f"{meta['E']:.6g}"  if meta.get('E')  else "FILL_IN"
    nu_str = f"{meta['nu']:.6g}" if meta.get('nu') else "FILL_IN"

    yaml_text = f"""\
# -----------------------------------------------------------------------
# PeriLab input snippet — generated by perifit
# -----------------------------------------------------------------------

PeriLab:
  Models:
    Material Models:
      LPS Elastic:
        Material Model: "LPS"
        Young's Modulus: {E_str}
        Poisson's Ratio: {nu_str}
        Horizon: {delta_str}
        Surface Correction: Nodal_Surface_Correction   # field from mesh file

  Discretization:
    Type: HDF5          # or CSV if using the fallback
    Input Mesh File: "{mesh_filename}"
    Node Sets:
      Nodeset 1: "FILL_IN"

  Blocks:
    block_{block_id}:
      Block ID: {block_id}
      Material Model: LPS Elastic
      Horizon: {delta_str}

# -----------------------------------------------------------------------
# Usage note
# ---------
# The field 'Nodal_Surface_Correction' in {mesh_filename} stores w_i for
# each node.  PeriLab applies the bond-averaged weight
#   w_bar_ij = (w_i + w_j) / 2
# internally when Surface Correction is set to this field name.
# -----------------------------------------------------------------------
"""
    (outdir / f"{stem}_perilab_input.yaml").write_text(yaml_text)

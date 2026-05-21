"""
Flexible mesh readers for the perifit surface correction package.

Supported formats
-----------------
  .csv / .txt / .dat   — plain text with columns (x, y, z[, vol])
  .npy                 — NumPy array (N, 3) or (N, 4)
  .npz                 — NumPy archive with keys 'coords'[, 'volumes']
  .vtk                 — legacy ASCII VTK (DATASET UNSTRUCTURED_GRID or
                         STRUCTURED_POINTS / RECTILINEAR_GRID)
  .vtu                 — VTK XML unstructured; requires pyvista or meshio
  .exo / .g            — Exodus II; requires netCDF4
  .msh                 — Gmsh v2/v4; requires meshio
  .inp                 — Abaqus input; requires meshio

The function ``load_mesh`` dispatches to the appropriate reader based on the
file extension.  It always returns ``(coords, volumes)`` as NumPy float64
arrays.  If the volume information is absent from the file, volumes are
estimated uniformly from the domain bounding box divided by N.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def load_mesh(
    path: str | Path,
    *,
    x_col: int = 0,
    y_col: int = 1,
    z_col: int = 2,
    vol_col: int | None = None,
    delimiter: str = None,
    comment: str = '#',
    horizon: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a mesh from *path* and return ``(coords, volumes)``.

    Parameters
    ----------
    path      : str or Path
        Path to the mesh file.
    x_col, y_col, z_col : int
        Column indices for x, y, z coordinates in text-based formats.
    vol_col   : int or None
        Column index for the nodal volume.  If None the volume is
        estimated automatically.
    delimiter : str or None
        Field delimiter for text-based formats (None → any whitespace).
    comment   : str
        Comment character for text-based formats.
    horizon   : float or None
        Horizon radius used only for the volume estimation fallback
        (uniform cubic volume = horizon^3 is used when the file does
        not contain volumes).

    Returns
    -------
    coords  : (N, 3) float64
    volumes : (N,)   float64
    """
    p = Path(path)
    suffix = p.suffix.lower()

    if suffix in ('.csv', '.txt', '.dat'):
        # Default delimiter: comma for .csv, whitespace for .txt/.dat
        if delimiter is None and suffix == '.csv':
            delimiter = ','
        return _read_text(p, x_col, y_col, z_col, vol_col, delimiter, comment, horizon)
    elif suffix == '.npy':
        return _read_npy(p, horizon)
    elif suffix == '.npz':
        return _read_npz(p, horizon)
    elif suffix == '.vtk':
        return _read_legacy_vtk(p, horizon)
    elif suffix == '.vtu':
        return _read_vtu(p, horizon)
    elif suffix in ('.exo', '.g', '.e'):
        return _read_exodus(p, horizon)
    elif suffix == '.msh':
        return _read_gmsh(p, horizon)
    elif suffix == '.inp':
        return _read_abaqus(p, horizon)
    else:
        raise ValueError(
            f"Unrecognised file extension '{suffix}'.  "
            "Supported: .csv, .txt, .dat, .npy, .npz, .vtk, .vtu, .exo, .g, .msh, .inp"
        )


# ---------------------------------------------------------------------------
# Volume estimation helper
# ---------------------------------------------------------------------------

def _estimate_volumes(coords: np.ndarray, horizon: float | None) -> np.ndarray:
    """
    Estimate uniform nodal volumes when not provided by the file.

    If *horizon* is given, the cubic voxel edge is estimated as the
    median nearest-neighbour distance (cheap O(N) approximation via
    bounding-box).  Otherwise the total bounding-box volume is divided
    equally among the N nodes.
    """
    N = len(coords)
    lo = coords.min(axis=0)
    hi = coords.max(axis=0)
    bbox_vol = float(np.prod(np.maximum(hi - lo, 1e-30)))
    uniform_vol = bbox_vol / N
    warnings.warn(
        f"No nodal volumes found in the mesh file.  "
        f"Assigning uniform volume = {uniform_vol:.4g} (bbox_vol/N).",
        UserWarning,
        stacklevel=3,
    )
    return np.full(N, uniform_vol, dtype=np.float64)


# ---------------------------------------------------------------------------
# Text-based reader (csv / txt / dat)
# ---------------------------------------------------------------------------

def _read_text(
    path: Path,
    x_col: int,
    y_col: int,
    z_col: int,
    vol_col: int | None,
    delimiter,
    comment: str,
    horizon: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    # Auto-detect and skip a string header row (e.g. "X,Y,Z" or "x,y,z,volume")
    skip = 0
    with open(path) as fh:
        first = fh.readline().strip()
        if first and not first.startswith(comment):
            try:
                [float(v) for v in (first.split(delimiter) if delimiter else first.split())]
            except ValueError:
                skip = 1   # first row is a text header
    data = np.loadtxt(path, delimiter=delimiter, comments=comment, skiprows=skip)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    coords = data[:, [x_col, y_col, z_col]].astype(np.float64)
    if vol_col is not None and vol_col < data.shape[1]:
        volumes = data[:, vol_col].astype(np.float64)
    else:
        volumes = _estimate_volumes(coords, horizon)
    return coords, volumes


# ---------------------------------------------------------------------------
# NumPy formats
# ---------------------------------------------------------------------------

def _read_npy(path: Path, horizon: float | None) -> tuple[np.ndarray, np.ndarray]:
    arr = np.load(path)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(
            f".npy file must be a 2-D array with at least 3 columns (x,y,z); "
            f"got shape {arr.shape}"
        )
    coords = arr[:, :3].astype(np.float64)
    if arr.shape[1] >= 4:
        volumes = arr[:, 3].astype(np.float64)
    else:
        volumes = _estimate_volumes(coords, horizon)
    return coords, volumes


def _read_npz(path: Path, horizon: float | None) -> tuple[np.ndarray, np.ndarray]:
    arc = np.load(path)
    # Accept several common key names
    for key in ('coords', 'coordinates', 'nodes', 'points'):
        if key in arc:
            coords = arc[key].astype(np.float64)
            break
    else:
        raise KeyError(
            f"Cannot find coordinate data in {path}.  "
            "Expected one of: 'coords', 'coordinates', 'nodes', 'points'."
        )
    for key in ('volumes', 'volume', 'vol'):
        if key in arc:
            volumes = arc[key].astype(np.float64)
            break
    else:
        volumes = _estimate_volumes(coords, horizon)
    return coords, volumes


# ---------------------------------------------------------------------------
# Legacy ASCII VTK reader (no external dependency)
# ---------------------------------------------------------------------------

def _read_legacy_vtk(path: Path, horizon: float | None) -> tuple[np.ndarray, np.ndarray]:
    """
    Minimal parser for legacy ASCII VTK files.  Handles
    DATASET UNSTRUCTURED_GRID (extracts POINTS) and
    DATASET STRUCTURED_POINTS / RECTILINEAR_GRID (reconstructs nodal coords).
    """
    text = path.read_text()
    lines = text.splitlines()

    dataset_type = None
    coords = None
    volumes = None

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line.upper().startswith('DATASET'):
            dataset_type = line.split()[1].upper()

        elif line.upper().startswith('POINTS'):
            parts = line.split()
            n_pts = int(parts[1])
            pts_lines = []
            i += 1
            while len(pts_lines) < n_pts * 3:
                pts_lines.extend(lines[i].split())
                i += 1
            coords = np.array(pts_lines[:n_pts * 3], dtype=np.float64).reshape(n_pts, 3)
            i -= 1  # will be incremented below

        elif line.upper().startswith('DIMENSIONS'):
            dims = list(map(int, line.split()[1:4]))

        elif line.upper().startswith('ORIGIN'):
            origin = list(map(float, line.split()[1:4]))

        elif line.upper().startswith('SPACING'):
            spacing = list(map(float, line.split()[1:4]))
            # Build coords from structured grid
            nx, ny, nz = dims
            xs = origin[0] + np.arange(nx) * spacing[0]
            ys = origin[1] + np.arange(ny) * spacing[1]
            zs = origin[2] + np.arange(nz) * spacing[2]
            gx, gy, gz = np.meshgrid(xs, ys, zs, indexing='ij')
            coords = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
            cell_vol = spacing[0] * spacing[1] * spacing[2]
            volumes = np.full(len(coords), cell_vol, dtype=np.float64)

        elif line.upper().startswith('SCALARS') and 'VOL' in line.upper():
            # Skip LOOKUP_TABLE line
            i += 1
            while not lines[i].strip().upper().startswith('LOOKUP_TABLE'):
                i += 1
            i += 1
            vol_data = []
            while i < len(lines) and len(vol_data) < len(coords):
                vol_data.extend(lines[i].split())
                i += 1
            volumes = np.array(vol_data[:len(coords)], dtype=np.float64)
            i -= 1

        i += 1

    if coords is None:
        raise RuntimeError(f"Could not parse nodal coordinates from {path}")
    if volumes is None:
        volumes = _estimate_volumes(coords, horizon)

    return coords.astype(np.float64), volumes.astype(np.float64)


# ---------------------------------------------------------------------------
# Formats requiring optional dependencies
# ---------------------------------------------------------------------------

def _read_vtu(path: Path, horizon: float | None) -> tuple[np.ndarray, np.ndarray]:
    """Read a VTK XML unstructured grid (.vtu). Tries pyvista then meshio."""
    try:
        import pyvista as pv
        mesh = pv.read(str(path))
        coords = np.asarray(mesh.points, dtype=np.float64)
        if 'volume' in mesh.point_data:
            volumes = np.asarray(mesh.point_data['volume'], dtype=np.float64)
        elif 'vol' in mesh.point_data:
            volumes = np.asarray(mesh.point_data['vol'], dtype=np.float64)
        else:
            volumes = _estimate_volumes(coords, horizon)
        return coords, volumes
    except ImportError:
        pass

    try:
        import meshio
        mesh = meshio.read(str(path))
        coords = np.asarray(mesh.points, dtype=np.float64)
        if mesh.point_data and 'volume' in mesh.point_data:
            volumes = np.asarray(mesh.point_data['volume'], dtype=np.float64)
        else:
            volumes = _estimate_volumes(coords, horizon)
        return coords, volumes
    except ImportError:
        raise ImportError(
            "Reading .vtu files requires pyvista or meshio.  "
            "Install with:  pip install pyvista  or  pip install meshio"
        )


def _read_exodus(path: Path, horizon: float | None) -> tuple[np.ndarray, np.ndarray]:
    """
    Read an Exodus II mesh file (.exo, .g).

    Tries (in order): exodus Python bindings, netCDF4, meshio.
    The nodal volume field is read from a nodal variable named
    'volume', 'vol', or 'nodal_volume' if present.
    """
    # --- Try netCDF4 (most reliable for Exodus) ---
    try:
        import netCDF4 as nc4
        ds = nc4.Dataset(str(path), 'r')
        xc = np.asarray(ds.variables['coordx'][:], dtype=np.float64)
        yc = np.asarray(ds.variables['coordy'][:], dtype=np.float64)
        zc = np.asarray(ds.variables.get('coordz', np.zeros_like(xc))[:], dtype=np.float64)
        coords = np.column_stack([xc, yc, zc])

        volumes = None
        nvar_names = [
            ds.variables.get('name_nod_var', None)
        ]
        for key in ('volume', 'vol', 'nodal_volume'):
            if key in ds.variables:
                volumes = np.asarray(ds.variables[key][-1], dtype=np.float64)
                break
        ds.close()
        if volumes is None:
            volumes = _estimate_volumes(coords, horizon)
        return coords, volumes
    except ImportError:
        pass

    # --- Try meshio fallback ---
    try:
        import meshio
        mesh = meshio.read(str(path))
        coords = np.asarray(mesh.points, dtype=np.float64)
        volumes = None
        if mesh.point_data:
            for key in ('volume', 'vol', 'nodal_volume'):
                if key in mesh.point_data:
                    volumes = np.asarray(mesh.point_data[key], dtype=np.float64)
                    break
        if volumes is None:
            volumes = _estimate_volumes(coords, horizon)
        return coords, volumes
    except ImportError:
        raise ImportError(
            "Reading Exodus files requires netCDF4 or meshio.  "
            "Install with:  pip install netCDF4  or  pip install meshio"
        )


def _read_gmsh(path: Path, horizon: float | None) -> tuple[np.ndarray, np.ndarray]:
    """Read a Gmsh mesh file (.msh v2 or v4). Requires meshio."""
    try:
        import meshio
        mesh = meshio.read(str(path))
        coords = np.asarray(mesh.points, dtype=np.float64)
        if mesh.point_data and 'volume' in mesh.point_data:
            volumes = np.asarray(mesh.point_data['volume'], dtype=np.float64)
        else:
            volumes = _estimate_volumes(coords, horizon)
        return coords, volumes
    except ImportError:
        raise ImportError(
            "Reading Gmsh files requires meshio.  "
            "Install with:  pip install meshio"
        )


def _read_abaqus(path: Path, horizon: float | None) -> tuple[np.ndarray, np.ndarray]:
    """Read an Abaqus input file (.inp). Requires meshio."""
    try:
        import meshio
        mesh = meshio.read(str(path))
        coords = np.asarray(mesh.points, dtype=np.float64)
        if mesh.point_data and 'volume' in mesh.point_data:
            volumes = np.asarray(mesh.point_data['volume'], dtype=np.float64)
        else:
            volumes = _estimate_volumes(coords, horizon)
        return coords, volumes
    except ImportError:
        raise ImportError(
            "Reading Abaqus .inp files requires meshio.  "
            "Install with:  pip install meshio"
        )

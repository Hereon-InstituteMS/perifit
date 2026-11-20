"""
perifit -- surface corrections for peridynamics.
================================================

Nodes near free surfaces, cracks and voids have truncated horizons, which
makes peridynamic models artificially soft there and spoils convergence.
perifit computes, once and purely from the geometry, corrections that restore
full-horizon behaviour.  Two methods are offered; the user picks one:

* **nodal weights** -- one weight per node; a bond (i, j) is scaled by
  ``(w_i + w_j) / 2``.  Compact; works with per-node solvers.
* **bond weights** -- one weight per bond.  More accurate and robust; breaking
  a bond is just setting its weight to zero.

Quick start
-----------
    import perifit

    # one multiplier per bond, whichever method is chosen
    corr = perifit.surface_correction(coords, volumes, horizon, method="bond")
    for (i, j), f in zip(corr.bonds, corr.factor):
        ...  # multiply the stiffness / force of bond (i, j) by f

    w = perifit.nodal_weights(coords, volumes, horizon)          # (N,)
    bonds, beta = perifit.bond_weights(coords, volumes, horizon)  # (Nb, 2), (Nb,)

Authors:  Arman Shojaei & Alexander Hermann
          Institute of Material Systems Modeling,
          Helmholtz-Zentrum Hereon, Geesthacht, Germany.
"""
from __future__ import annotations

import functools as _functools
import warnings as _warnings

__version__ = "3.0.1"

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
from .api import SurfaceCorrection, bond_weights, nodal_weights, surface_correction
from .core.bond_weights import BondWeightInfo
from .core.families import build_bonds, build_families, families_from_bonds

# Closed-form horizon moments (1-D segment, 2-D disc, 3-D ball)
from .core.moments import (
    ball_moment,
    ball_moment_over_r2,
    ball_moment_over_r3,
    full_ball_shape_tensor_diag,
    full_ball_volume,
    full_ball_weighted_volume,
)
from .core.moments_1d import (
    full_segment_length,
    full_segment_weighted_volume,
    segment_moment,
    segment_moment_over_r,
    segment_moment_over_r3,
)
from .core.moments_2d import (
    disc_moment,
    disc_moment_over_r2,
    disc_moment_over_r3,
    full_disc_area,
    full_disc_shape_tensor_diag,
    full_disc_weighted_volume,
)

# Mesh I/O, configuration and weight-quality tools
from .config import CorrectionConfig
from .io.writers import write_bond_weights, write_weights
from .mesh.readers import load_mesh
from .quality.cleaning import CleaningResult, clean_mesh
from .quality.diagnostics import WeightDiagnostics, diagnose_weights


# ---------------------------------------------------------------------------
# Deprecated v2 entry points (removed in v4)
# ---------------------------------------------------------------------------
_REMOVED_KWARGS = {
    "row_norm": "the per-row normalised 1-D fit is now the only one (it is unit independent)",
    "row_normalize": "the per-row normalised fit is the only one",
    "c_bond": "material constants are not needed; the weights are purely geometric",
}


def _deprecated(func, replacement):
    @_functools.wraps(func)
    def wrapper(*args, **kwargs):
        for name in _REMOVED_KWARGS.keys() & kwargs.keys():
            raise TypeError(f"perifit.{func.__name__}({name}=...) was removed in perifit 3.0: "
                            f"{_REMOVED_KWARGS[name]}.")
        _warnings.warn(f"perifit.{func.__name__}() is deprecated and will be removed in "
                       f"perifit 4.0; use {replacement} instead.  Note that 1-D and 2-D weights "
                       "changed in 3.0 (see CHANGELOG.md).",
                       DeprecationWarning, stacklevel=2)
        return func(*args, **kwargs)
    return wrapper


from .core.weights import compute_weights as _cw3
from .core.weights_1d import compute_weights_1d as _cw1
from .core.weights_2d import compute_weights_2d as _cw2

compute_weights = _deprecated(_cw3, "perifit.nodal_weights(coords, volumes, horizon, model=...)")
compute_weights_2d = _deprecated(_cw2, "perifit.nodal_weights(coords, volumes, horizon)")
compute_weights_1d = _deprecated(_cw1, "perifit.nodal_weights(coords, volumes, horizon)")


def build_families_1d(coords, horizon, tol=None):
    """Deprecated alias of :func:`build_families`."""
    return _deprecated(build_families, "perifit.build_families")(coords, horizon, tol)


def build_families_2d(coords, horizon, tol=None):
    """Deprecated alias of :func:`build_families`."""
    return _deprecated(build_families, "perifit.build_families")(coords, horizon, tol)


def get_example_mesh() -> "pathlib.Path":
    """Path of the bundled example mesh: structured cube [0, 1]^3, 1000 nodes, dx = 0.1."""
    import pathlib
    return pathlib.Path(__file__).parent / "data" / "test_mesh_cube_1000.csv"


_LOGO = r"""
 ____           _ _____ _ _
|  _ \ ___ _ __(_)  ___(_) |_
| |_) / _ \ '__| | |_  | | __|
|  __/  __/ |  | |  _| | | |_
|_|   \___|_|  |_|_|   |_|\__|
"""


def show_version() -> None:
    """Print the perifit version banner."""
    print(_LOGO.lstrip("\n"))
    print(f"perifit v{__version__}")
    print("Surface corrections for peridynamics: nodal weights and bond weights")
    print()
    print("Developers:  Arman Shojaei & Alexander Hermann")
    print("             Helmholtz-Zentrum Hereon, Germany")


__all__ = [
    # surface corrections
    "surface_correction", "SurfaceCorrection", "nodal_weights", "bond_weights", "BondWeightInfo",
    # neighbourhoods
    "build_bonds", "build_families", "families_from_bonds",
    # horizon moments
    "segment_moment", "segment_moment_over_r", "segment_moment_over_r3",
    "full_segment_length", "full_segment_weighted_volume",
    "disc_moment", "disc_moment_over_r2", "disc_moment_over_r3",
    "full_disc_area", "full_disc_weighted_volume", "full_disc_shape_tensor_diag",
    "ball_moment", "ball_moment_over_r2", "ball_moment_over_r3",
    "full_ball_volume", "full_ball_weighted_volume", "full_ball_shape_tensor_diag",
    # I/O, configuration, quality
    "load_mesh", "write_weights", "write_bond_weights", "get_example_mesh", "CorrectionConfig",
    "diagnose_weights", "WeightDiagnostics", "clean_mesh", "CleaningResult",
    # deprecated (v2)
    "compute_weights", "compute_weights_2d", "compute_weights_1d",
    "build_families_2d", "build_families_1d",
    "show_version", "__version__",
]

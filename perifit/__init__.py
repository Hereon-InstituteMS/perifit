"""
perifit -- Peridynamic surface correction via optimised nodal influence weights.
==============================================================================

Authors:  Arman Shojaei & Alexander Hermann
          Institute of Material Systems Modeling,
          Helmholtz-Zentrum Hereon, Geesthacht, Germany.

Purpose
-------
Compute per-node scalar influence weights w_i that restore derivative and
energy consistency of the discrete PD operators near free surfaces, cracks,
and voids.  Each bond (i, j) is then scaled by the symmetric factor
(w_i + w_j) / 2, preserving pairwise reciprocity and global stiffness symmetry.

Supported models (3-D):
    - "bb"  : Bond-Based PD (PMB / linearised BB), kernel exponent q = 3
    - "osb" : Ordinary State-Based PD (LPS), kernel exponent q = 2

Supported models (2-D):
    - BB-PD via compute_weights_2d

Usage
-----
    import perifit

    # 1. Build neighbour families (or supply your own)
    families = perifit.build_families(coords, horizon)

    # 2. Compute weights (one-shot, load-independent)
    w = perifit.compute_weights(coords, volumes, horizon,
                                model="bb", families=families)

    # 3. Apply in solver: bond weight = (w[i] + w[j]) / 2

Input:
    coords   -- (N, 3) array of nodal coordinates
    volumes  -- (N,)   array of cell volumes
    horizon  -- scalar, peridynamic horizon delta

Output:
    w        -- (N,)   array of nodal influence weights

The weights are purely geometric: material constants (micro-modulus c,
LPS parameters a, b) cancel after per-row normalisation and need not be
specified.
"""

# ---------------------------------------------------------------------------
# 3-D public API
# ---------------------------------------------------------------------------
from .core.moments import (
    ball_moment,
    ball_moment_over_r2,
    ball_moment_over_r3,
    full_ball_volume,
    full_ball_weighted_volume,
    full_ball_shape_tensor_diag,
)
from .core.targets import (
    build_targets_bb,
    build_targets_osb,
    build_targets_nosb,
)
from .core.local_system import (
    build_local_system_bb,
    build_local_system_osb,
    build_local_system_nosb,
)
from .core.weights import compute_weights, build_families

# ---------------------------------------------------------------------------
# 2-D public API
# ---------------------------------------------------------------------------
from .core.moments_2d import (
    disc_moment,
    disc_moment_over_r2,
    disc_moment_over_r3,
    full_disc_area,
    full_disc_weighted_volume,
    full_disc_shape_tensor_diag,
)
from .core.targets_2d import build_targets_bb_2d
from .core.local_system_2d import build_local_system_bb_2d
from .core.weights_2d import compute_weights_2d, build_families_2d

# ---------------------------------------------------------------------------
# Existing package modules (mesh I/O, quality diagnostics, etc.)
# ---------------------------------------------------------------------------
from .mesh.readers import load_mesh
from .io.writers import write_weights
from .config import CorrectionConfig
from .quality.diagnostics import diagnose_weights, WeightDiagnostics
from .quality.cleaning import clean_mesh, CleaningResult


def get_example_mesh() -> "pathlib.Path":
    """
    Return the path to the bundled test mesh file.

    The file is a structured Cartesian cube [0,1]^3 with 1000 nodes
    (10x10x10, dx=0.1).  Recommended horizon: delta=0.3 (m=3).

    Example
    -------
    >>> from perifit import get_example_mesh, load_mesh, compute_weights
    >>> coords, volumes = load_mesh(get_example_mesh())
    >>> weights = compute_weights(coords, volumes, 0.3, model="bb")
    """
    import pathlib
    return pathlib.Path(__file__).parent / "data" / "test_mesh_cube_1000.csv"


# ---------------------------------------------------------------------------
# Branding
# ---------------------------------------------------------------------------

_LOGO = r"""
 ____           _ _____ _ _
|  _ \ ___ _ __(_)  ___(_) |_
| |_) / _ \ '__| | |_  | | __|
|  __/  __/ |  | |  _| | | |_
|_|   \___|_|  |_|_|   |_|\__|
"""


def show_version() -> None:
    """Print the perifit version banner with ASCII logo and credits."""
    print(_LOGO.lstrip("\n"))
    print(f"perifit v{__version__}")
    print("Peridynamic Surface Correction via Optimised Nodal Influence Weights")
    print()
    print("Developers:  Arman Shojaei & Alexander Hermann")
    print("             Helmholtz-Zentrum Hereon, Germany")


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    # Main 3-D API
    "compute_weights",
    "build_families",
    # 3-D targets
    "build_targets_bb",
    "build_targets_osb",
    "build_targets_nosb",
    # 3-D local systems
    "build_local_system_bb",
    "build_local_system_osb",
    "build_local_system_nosb",
    # 3-D moment utilities
    "ball_moment",
    "ball_moment_over_r2",
    "ball_moment_over_r3",
    "full_ball_volume",
    "full_ball_weighted_volume",
    "full_ball_shape_tensor_diag",
    # 2-D API
    "compute_weights_2d",
    "build_families_2d",
    "build_targets_bb_2d",
    "build_local_system_bb_2d",
    "disc_moment",
    "disc_moment_over_r2",
    "disc_moment_over_r3",
    "full_disc_area",
    "full_disc_weighted_volume",
    "full_disc_shape_tensor_diag",
    # I/O and utilities
    "load_mesh",
    "write_weights",
    "get_example_mesh",
    "CorrectionConfig",
    "diagnose_weights",
    "WeightDiagnostics",
    "clean_mesh",
    "CleaningResult",
    # Branding
    "show_version",
]

__version__ = "2.0.1"

"""
perifit.core -- the computational building blocks.

Neighbourhoods
    families        build_bonds, build_families, families_from_bonds

Nodal weights (one weight per node)
    weights_1d      compute_weights_1d   1-D bond-based
    weights_2d      compute_weights_2d   2-D bond-based
    weights         compute_weights      3-D bond-based and state-based (LPS)
    local_system*   per-node least-squares systems (collocation + operator rows)
    targets*        full-horizon values of the operator rows
    _lsq            shared local solve (normalisation, regularisation, round-off)
    _assembly       global sparse system K_w w = F_w

Bond weights (one weight per bond)
    bond_weights    compute_bond_weights  1-D/2-D/3-D bond-based

Closed-form horizon moments
    moments_1d      segment moments
    moments_2d      disc moments
    moments         ball moments

Most users need only the public API in :mod:`perifit` (``nodal_weights``,
``bond_weights``, ``surface_correction``).
"""
from .bond_weights import BondWeightInfo, compute_bond_weights
from .families import build_bonds, build_families, families_from_bonds
from .local_system import build_local_system_bb, build_local_system_nosb, build_local_system_osb
from .local_system_1d import build_local_system_bb_1d
from .local_system_2d import build_local_system_bb_2d
from .moments import (
    ball_moment,
    ball_moment_over_r2,
    ball_moment_over_r3,
    full_ball_shape_tensor_diag,
    full_ball_volume,
    full_ball_weighted_volume,
)
from .moments_1d import (
    full_segment_length,
    full_segment_weighted_volume,
    segment_moment,
    segment_moment_over_r,
    segment_moment_over_r3,
)
from .moments_2d import (
    disc_moment,
    disc_moment_over_r2,
    disc_moment_over_r3,
    full_disc_area,
    full_disc_shape_tensor_diag,
    full_disc_weighted_volume,
)
from .targets import build_targets_bb, build_targets_nosb, build_targets_osb
from .targets_1d import build_targets_bb_1d
from .targets_2d import build_targets_bb_2d
from .weights import compute_weights
from .weights_1d import compute_weights_1d
from .weights_2d import compute_weights_2d

__all__ = [
    "build_bonds", "build_families", "families_from_bonds",
    "compute_weights", "compute_weights_1d", "compute_weights_2d",
    "compute_bond_weights", "BondWeightInfo",
    "build_local_system_bb", "build_local_system_osb", "build_local_system_nosb",
    "build_local_system_bb_1d", "build_local_system_bb_2d",
    "build_targets_bb", "build_targets_osb", "build_targets_nosb",
    "build_targets_bb_1d", "build_targets_bb_2d",
    "segment_moment", "segment_moment_over_r", "segment_moment_over_r3",
    "full_segment_length", "full_segment_weighted_volume",
    "disc_moment", "disc_moment_over_r2", "disc_moment_over_r3",
    "full_disc_area", "full_disc_weighted_volume", "full_disc_shape_tensor_diag",
    "ball_moment", "ball_moment_over_r2", "ball_moment_over_r3",
    "full_ball_volume", "full_ball_weighted_volume", "full_ball_shape_tensor_diag",
]

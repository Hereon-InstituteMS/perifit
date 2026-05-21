"""
perifit.core -- Core computation modules for peridynamic surface correction.

3-D modules:
    moments, targets, local_system, weights

2-D modules:
    moments_2d, targets_2d, local_system_2d, weights_2d
"""

# 3-D moments
from .moments import (
    ball_moment,
    ball_moment_over_r2,
    ball_moment_over_r3,
    full_ball_volume,
    full_ball_weighted_volume,
    full_ball_shape_tensor_diag,
)

# 3-D targets
from .targets import (
    build_targets_bb,
    build_targets_osb,
    build_targets_nosb,
)

# 3-D local systems
from .local_system import (
    build_local_system_bb,
    build_local_system_osb,
    build_local_system_nosb,
)

# 3-D weights (public API)
from .weights import compute_weights, build_families

# 2-D moments
from .moments_2d import (
    disc_moment,
    disc_moment_over_r2,
    disc_moment_over_r3,
    full_disc_area,
    full_disc_weighted_volume,
    full_disc_shape_tensor_diag,
)

# 2-D targets
from .targets_2d import build_targets_bb_2d

# 2-D local systems
from .local_system_2d import build_local_system_bb_2d

# 2-D weights
from .weights_2d import compute_weights_2d, build_families_2d

__all__ = [
    # 3-D main API
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
    # 3-D moments
    "ball_moment",
    "ball_moment_over_r2",
    "ball_moment_over_r3",
    "full_ball_volume",
    "full_ball_weighted_volume",
    "full_ball_shape_tensor_diag",
    # 2-D main API
    "compute_weights_2d",
    "build_families_2d",
    # 2-D targets
    "build_targets_bb_2d",
    # 2-D local systems
    "build_local_system_bb_2d",
    # 2-D moments
    "disc_moment",
    "disc_moment_over_r2",
    "disc_moment_over_r3",
    "full_disc_area",
    "full_disc_weighted_volume",
    "full_disc_shape_tensor_diag",
]

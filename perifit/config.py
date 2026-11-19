"""
Configuration dataclass for the peridynamic surface correction computation.

This consolidates all tunable parameters in one place and provides
human-readable defaults with brief explanations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class CorrectionConfig:
    """
    All parameters for a peridynamic surface correction run.

    Note: The correction weights are purely geometric — independent of
    material constants E and nu.  Only the mesh geometry and horizon
    (or equivalently, the horizon-to-spacing ratio m) are needed.

    Method and model
    ----------------
    method  : surface correction: 'nodal' (one weight per node) or 'bond'
              (one weight per bond).  Must be chosen explicitly.
    model   : PD model: 'bb' / 'bond_based' or 'osb' / 'state_based'
              (state-based is available for nodal weights in 3-D only).

    Discretisation
    ---------------
    horizon : Peridynamic horizon radius  delta.  If None, it is
              computed as  m_ratio * dx  where dx is the inferred
              median nearest-neighbour nodal spacing.
    m_ratio : Horizon-to-spacing ratio  m = delta / dx.  Default: 3.
              Used only when horizon is None.

    Solver parameters
    -----------------
    tol      : Relative residual tolerance for the global BiCGSTAB solve.
    max_iter : Maximum BiCGSTAB iterations.

    Regularisation (local pseudoinverse)
    ------------------------------------
    tikhonov_rel : Regularisation factor relative to max(diag(M^T M)).
    tikhonov_abs : Absolute floor for the regularisation parameter.

    Output
    ------
    formats  : List of output format strings. Supported: 'csv', 'dat',
               'vtk', 'peridigm', 'perilab'.
    outdir   : Directory where output files are written.
    stem     : Base filename stem (without extension).

    Miscellaneous
    -------------
    verbose  : Print progress information.
    """

    # Method and model
    method: Optional[str] = None
    model: str = 'bb'

    # Discretisation
    horizon: float | None = None
    m_ratio: float = 3.0

    # Solver
    tol:      float = 1e-10
    max_iter: int   = 5000

    # Regularisation
    tikhonov_rel: float = 1.0e-8
    tikhonov_abs: float = 1.0e-12

    # Output
    formats: List[str] = field(default_factory=lambda: ['csv'])
    outdir:  str       = './perifit_output'
    stem:    str       = 'perifit_weights'

    # Misc
    verbose: bool = True

    def __post_init__(self):
        if self.horizon is not None and self.horizon <= 0.0:
            raise ValueError(f"Horizon delta={self.horizon} must be positive.")
        if self.m_ratio <= 0.0:
            raise ValueError(f"m_ratio={self.m_ratio} must be positive.")
        if self.method not in (None, 'nodal', 'bond'):
            raise ValueError(f"method={self.method!r} must be 'nodal' or 'bond'.")
        aliases = {'bond_based': 'bb', 'state_based': 'osb'}
        self.model = aliases.get(self.model, self.model)
        if self.model not in ('bb', 'osb'):
            raise ValueError(f"model={self.model!r} must be 'bb' or 'osb'.")
        if self.method == 'bond' and self.model != 'bb':
            raise ValueError("bond weights are defined for the bond-based model ('bb') only.")

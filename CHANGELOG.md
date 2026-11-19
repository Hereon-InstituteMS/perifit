# Changelog

## 3.0.0

Two surface corrections, chosen explicitly by the user, each with one
validated calibration per dimension.

### New
- **Bond weights** — one weight per bond for bond-based PD in 1-D, 2-D and 3-D
  (`perifit.bond_weights`). The weights are the smallest change from 1 for which
  every node reproduces the full-horizon energy moments; `β ≥ 0` is enforced.
  Most accurate in all benchmarks and robust for horizon ratios m = 3–5; a broken
  bond is simply `β_b = 0`.
- **One front door**: `perifit.surface_correction(coords, volumes, horizon, method="nodal" | "bond")`
  returns one multiplier per bond for either method; `method` has no default.
- `perifit.nodal_weights` dispatches on the dimension of the coordinates;
  `dim=` drops constant columns of planar meshes stored in 3 columns.
- `perifit.build_bonds`, `perifit.families_from_bonds`, `perifit.write_bond_weights`.
- CLI: `--method nodal|bond` (required), `--dim`; bond weights are written as CSV or NPZ.
- `examples/benchmarks.py` (1-D bar, 2-D cantilever, 3-D circular cantilever with exact
  solutions) and `docs/make_figures.py` reproduce every number and figure of the README.

### Changed — nodal weights
- **1-D**: the energy row of the quadratic test is no longer fitted; the bulk weight is
  now exactly m/(m+1). First-order δ-convergence on the bar (0.17 % at Δx = 1/400,
  previously a 4.8 % floor). The unit-dependent default `row_norm="none"` and the
  option itself are removed.
- **2-D**: only the moments of the bond-based operator are fitted — third moments
  (quadratic-test derivative rows, kernel ξ/r³) and fourth moments (linear-test energy
  rows). 2-D cantilever: 1.5 % at Δx = 1/200 with slope ≈ 1 (previously 6.7 %, slope → 0.1).
  Validated for m = 3–4.
- **3-D (BB and OSB)**: operator rows that vanish by symmetry are recognised within groups
  of equal block and test degree instead of with an absolute threshold, and the neighbour
  cutoff is relative (`horizon · (1 + 1e-10)`). Weights no longer depend on the
  floating-point summation order or on the length unit; on an 11³ cube (m = 3) they move
  by up to 0.10 (BB) and 0.08 (OSB) compared with 2.0.1.

### Fixed
- 2-D: `tikhonov_rel`/`tikhonov_abs` were passed into the wrong arguments (regularisation
  10⁴× too small).
- The global weight system is assembled vectorised instead of entry by entry (large meshes).
- CLI: the writer no longer receives the mean weight as the horizon; `--clean` recomputes
  the weights of the cleaned mesh.

### Removed / deprecated
- `compute_weights`, `compute_weights_2d`, `compute_weights_1d` at package level are
  deprecated aliases of `nodal_weights` (removal in 4.0).
- The correspondence (NOSB) model is no longer advertised as a public model.
- Python ≥ 3.10 and SciPy ≥ 1.12 are required (Python 3.9 reached end-of-life in October 2025
  and is no longer available on GitHub's CI runners).

To reproduce the figures of the IJNME 2026 paper exactly, use `perifit==2.0.1`.

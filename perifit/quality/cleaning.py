from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .diagnostics import diagnose_weights, WeightDiagnostics


@dataclass
class CleaningResult:
    original_n: int
    cleaned_n: int
    removed_indices: np.ndarray
    coords: np.ndarray
    volumes: np.ndarray
    weights: np.ndarray | None
    diagnostics: WeightDiagnostics


def clean_mesh(
    coords: np.ndarray,
    volumes: np.ndarray,
    weights: np.ndarray,
    *,
    threshold_sigma: float = 2.0,
    remove_negative: bool = True,
    remove_outliers: bool = True,
    verbose: bool = True,
) -> CleaningResult:
    """Remove flagged nodes, return cleaned arrays."""
    diag = diagnose_weights(weights, threshold_sigma=threshold_sigma,
                           flag_negative=remove_negative)

    # Determine which nodes to remove
    remove = np.array([], dtype=int)
    if remove_negative:
        remove = np.union1d(remove, diag.negative_indices)
    if remove_outliers:
        remove = np.union1d(remove, np.union1d(diag.outlier_high_indices, diag.outlier_low_indices))

    # Keep mask
    keep = np.ones(len(weights), dtype=bool)
    keep[remove] = False

    cleaned_coords = coords[keep]
    cleaned_volumes = volumes[keep]
    cleaned_weights = weights[keep]

    if verbose:
        print(f"  Removed {len(remove)} of {len(weights)} nodes")

    return CleaningResult(
        original_n=len(weights),
        cleaned_n=int(keep.sum()),
        removed_indices=remove,
        coords=cleaned_coords,
        volumes=cleaned_volumes,
        weights=cleaned_weights,
        diagnostics=diag,
    )

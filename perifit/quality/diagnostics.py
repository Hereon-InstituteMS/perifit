from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class WeightDiagnostics:
    n_nodes: int
    n_negative: int
    n_outlier_high: int
    n_outlier_low: int
    negative_indices: np.ndarray
    outlier_high_indices: np.ndarray
    outlier_low_indices: np.ndarray
    flagged_indices: np.ndarray
    mean_weight: float
    std_weight: float
    threshold_sigma: float
    summary: str


def diagnose_weights(weights: np.ndarray, *, threshold_sigma: float = 2.0, flag_negative: bool = True) -> WeightDiagnostics:
    """Flag negative weights and outliers beyond mean +/- k*sigma."""
    weights = np.asarray(weights, dtype=np.float64)
    n = len(weights)
    mean_w = float(np.mean(weights))
    std_w = float(np.std(weights))

    # Negative weights (physically meaningless)
    if flag_negative:
        negative_idx = np.where(weights < 0)[0]
    else:
        negative_idx = np.array([], dtype=int)

    # Outliers: beyond mean +/- k*sigma
    lo = mean_w - threshold_sigma * std_w
    hi = mean_w + threshold_sigma * std_w
    outlier_high_idx = np.where(weights > hi)[0]
    outlier_low_idx = np.where(weights < lo)[0]

    # Union of all flagged
    flagged = np.union1d(negative_idx, np.union1d(outlier_high_idx, outlier_low_idx))

    summary_lines = [
        f"Weight diagnostics  ({n} nodes, threshold_sigma={threshold_sigma})",
        f"  mean={mean_w:.4f}  std={std_w:.4f}",
        f"  range=[{weights.min():.4f}, {weights.max():.4f}]",
        f"  negative: {len(negative_idx)}",
        f"  outlier high (>{hi:.4f}): {len(outlier_high_idx)}",
        f"  outlier low  (<{lo:.4f}): {len(outlier_low_idx)}",
        f"  total flagged: {len(flagged)}",
    ]

    return WeightDiagnostics(
        n_nodes=n,
        n_negative=len(negative_idx),
        n_outlier_high=len(outlier_high_idx),
        n_outlier_low=len(outlier_low_idx),
        negative_indices=negative_idx,
        outlier_high_indices=outlier_high_idx,
        outlier_low_indices=outlier_low_idx,
        flagged_indices=flagged,
        mean_weight=mean_w,
        std_weight=std_w,
        threshold_sigma=threshold_sigma,
        summary="\n".join(summary_lines),
    )

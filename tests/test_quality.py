"""Tests for mesh quality diagnostics and cleaning."""

import numpy as np
import pytest
from perifit.quality.diagnostics import diagnose_weights, WeightDiagnostics
from perifit.quality.cleaning import clean_mesh, CleaningResult


class TestDiagnoseWeights:
    def test_all_ones_nothing_flagged(self):
        """All-ones weights should have zero flagged (no outliers, no negatives)."""
        w = np.ones(100)
        diag = diagnose_weights(w)
        assert diag.n_negative == 0
        assert diag.n_outlier_high == 0
        assert diag.n_outlier_low == 0
        assert len(diag.flagged_indices) == 0

    def test_negative_flagged(self):
        """Injected negative weights should be flagged."""
        w = np.ones(100)
        w[5] = -0.5
        w[10] = -1.0
        diag = diagnose_weights(w)
        assert diag.n_negative == 2
        assert 5 in diag.negative_indices
        assert 10 in diag.negative_indices
        assert 5 in diag.flagged_indices
        assert 10 in diag.flagged_indices

    def test_outlier_high_flagged(self):
        """Extreme high values should be flagged as outliers."""
        w = np.ones(100)
        w[42] = 100.0  # way above mean + 2*sigma
        diag = diagnose_weights(w, threshold_sigma=2.0)
        assert diag.n_outlier_high >= 1
        assert 42 in diag.outlier_high_indices

    def test_outlier_low_flagged(self):
        """Extreme low values should be flagged as outliers."""
        w = np.ones(100) * 5.0
        w[7] = -10.0
        diag = diagnose_weights(w, threshold_sigma=2.0)
        assert 7 in diag.flagged_indices

    def test_threshold_sensitivity(self):
        """Tighter sigma should flag more nodes."""
        rng = np.random.default_rng(42)
        w = 1.0 + 0.5 * rng.standard_normal(200)
        diag_tight = diagnose_weights(w, threshold_sigma=1.0)
        diag_loose = diagnose_weights(w, threshold_sigma=3.0)
        assert len(diag_tight.flagged_indices) >= len(diag_loose.flagged_indices)

    def test_flag_negative_disabled(self):
        """When flag_negative=False, negatives are not flagged."""
        w = np.ones(50)
        w[3] = -0.1
        diag = diagnose_weights(w, flag_negative=False)
        assert diag.n_negative == 0

    def test_summary_is_string(self):
        diag = diagnose_weights(np.ones(10))
        assert isinstance(diag.summary, str)
        assert "diagnostics" in diag.summary.lower()

    def test_dataclass_fields(self):
        diag = diagnose_weights(np.ones(10))
        assert isinstance(diag, WeightDiagnostics)
        assert diag.n_nodes == 10
        assert diag.mean_weight == pytest.approx(1.0)
        assert diag.std_weight == pytest.approx(0.0)


class TestCleanMesh:
    def test_no_removal_needed(self):
        """All healthy weights -> nothing removed."""
        coords = np.random.randn(50, 3)
        volumes = np.ones(50) * 0.001
        weights = np.ones(50)
        result = clean_mesh(coords, volumes, weights, verbose=False)
        assert result.original_n == 50
        assert result.cleaned_n == 50
        assert len(result.removed_indices) == 0

    def test_negative_removed(self):
        """Negative weights should be removed."""
        coords = np.random.randn(50, 3)
        volumes = np.ones(50) * 0.001
        weights = np.ones(50)
        weights[3] = -0.5
        weights[7] = -1.0
        result = clean_mesh(coords, volumes, weights, verbose=False)
        assert result.cleaned_n < result.original_n
        assert 3 in result.removed_indices
        assert 7 in result.removed_indices

    def test_output_arrays_consistent(self):
        """Cleaned arrays should have matching lengths."""
        coords = np.random.randn(30, 3)
        volumes = np.ones(30) * 0.001
        weights = np.ones(30)
        weights[0] = -1.0
        result = clean_mesh(coords, volumes, weights, verbose=False)
        assert len(result.coords) == result.cleaned_n
        assert len(result.volumes) == result.cleaned_n
        assert len(result.weights) == result.cleaned_n

    def test_remove_only_negative(self):
        """With remove_outliers=False, only negatives are removed."""
        rng = np.random.default_rng(42)
        coords = rng.standard_normal((100, 3))
        volumes = np.ones(100) * 0.001
        weights = 1.0 + 0.5 * rng.standard_normal(100)
        weights[0] = -1.0  # negative
        weights[1] = 50.0   # outlier

        result_both = clean_mesh(coords, volumes, weights,
                                 remove_outliers=True, verbose=False)
        result_neg = clean_mesh(coords, volumes, weights,
                                remove_outliers=False, verbose=False)
        # With outlier removal off, only the negative is removed
        assert result_neg.cleaned_n >= result_both.cleaned_n

    def test_dataclass_type(self):
        result = clean_mesh(np.random.randn(10, 3), np.ones(10),
                           np.ones(10), verbose=False)
        assert isinstance(result, CleaningResult)
        assert isinstance(result.diagnostics, WeightDiagnostics)

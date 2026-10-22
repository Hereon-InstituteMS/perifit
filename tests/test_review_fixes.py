"""Tests for the review fixes: NOSB through compute_weights and the CLI, the horizon written by the CLI,
recomputation of the weights after mesh cleaning, and partial-volume (bond volume factor) support."""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from perifit import compute_weights, partial_volume_factor, get_example_mesh
from perifit.cli import main
from perifit.quality.cleaning import clean_mesh


def _lattice(n, dx=1.0):
    g = (np.arange(n) + 0.5) * dx
    x, y, z = np.meshgrid(g, g, g, indexing="ij")
    coords = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    return coords, np.full(len(coords), dx ** 3)


def _centre(coords):
    return int(np.argmin(np.linalg.norm(coords - coords.mean(axis=0), axis=1)))


def test_nosb_through_compute_weights():
    coords, vol = _lattice(7)
    w = compute_weights(coords, vol, 3.0, model="nosb")
    assert w.shape == (len(coords),)
    assert np.all(np.isfinite(w))


def test_partial_volume_factor_values():
    f = partial_volume_factor(3.0, 1.0)
    r = np.array([1.0, 2.5, 3.0, 3.25, 3.5, 4.0])
    np.testing.assert_allclose(f(r), [1.0, 1.0, 0.5, 0.25, 0.0, 0.0])


def test_bond_volume_factor_changes_the_operator():
    """With partial volumes the discrete bulk sum is smaller, so the bulk weight is larger
    than with full volumes (full volumes overestimate the ball integral for delta = 3 dx)."""
    coords, vol = _lattice(11)
    delta = 3.0 * (1 + 1e-7)
    c = _centre(coords)
    w_full = compute_weights(coords, vol, delta, model="bb")
    w_part = compute_weights(coords, vol, delta, model="bb",
                             bond_volume_factor=partial_volume_factor(delta, 1.0))
    assert w_full[c] < 0.95
    assert w_part[c] > 1.05


def test_cli_writes_true_horizon(tmp_path):
    main(["--mesh", str(get_example_mesh()), "--model", "bb", "--output", "dat",
          "--outdir", str(tmp_path), "--stem", "w", "--quiet"])
    header = (tmp_path / "w.dat").read_text().splitlines()[0]
    assert "delta = 0.3" in header, header


def test_cli_accepts_nosb(tmp_path):
    main(["--mesh", str(get_example_mesh()), "--model", "nosb", "--output", "csv",
          "--outdir", str(tmp_path), "--stem", "w", "--quiet"])
    assert (tmp_path / "w.csv").exists()


def test_clean_mesh_recomputes_weights():
    coords, vol = _lattice(7, dx=0.1)
    delta = 0.3
    w = compute_weights(coords, vol, delta, model="bb")
    res = clean_mesh(coords, vol, w, threshold_sigma=1.0, verbose=False, horizon=delta, model="bb")
    assert res.cleaned_n < res.original_n
    np.testing.assert_allclose(res.weights, compute_weights(res.coords, res.volumes, delta, model="bb"))


def test_clean_mesh_warns_without_horizon():
    coords, vol = _lattice(7, dx=0.1)
    w = compute_weights(coords, vol, 0.3, model="bb")
    with pytest.warns(RuntimeWarning, match="not recomputed"):
        clean_mesh(coords, vol, w, threshold_sigma=1.0, verbose=False)

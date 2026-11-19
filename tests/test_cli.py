"""Tests for the perifit command-line interface."""
import subprocess
import sys

import numpy as np
import pytest


def _run_perifit(*args, timeout=180):
    return subprocess.run([sys.executable, "-m", "perifit.cli", *args],
                          capture_output=True, text=True, timeout=timeout)


class TestCLI:
    def test_help(self):
        r = _run_perifit("--help")
        assert r.returncode == 0
        assert "--method" in r.stdout

    def test_method_is_required(self):
        r = _run_perifit("--demo")
        assert r.returncode != 0
        assert "--method nodal" in r.stderr and "--method bond" in r.stderr

    def test_demo_nodal_state_based(self, tmp_path):
        r = _run_perifit("--demo", "--method", "nodal", "--outdir", str(tmp_path))
        assert r.returncode == 0, r.stderr
        assert "OSB-PD" in r.stdout
        assert (tmp_path / "demo_cube.csv").exists() and (tmp_path / "demo_cube.vtk").exists()

    def test_demo_nodal_bond_based(self, tmp_path):
        r = _run_perifit("--demo", "--method", "nodal", "--model", "bb", "--outdir", str(tmp_path))
        assert r.returncode == 0, r.stderr
        assert "BB-PD" in r.stdout

    def test_demo_bond(self, tmp_path):
        r = _run_perifit("--demo", "--method", "bond", "--outdir", str(tmp_path))
        assert r.returncode == 0, r.stderr
        data = np.load(tmp_path / "demo_cube.npz")
        assert data["bonds"].shape[1] == 2 and len(data["beta"]) == len(data["bonds"])
        rows = np.loadtxt(tmp_path / "demo_cube.csv", delimiter=",")
        assert rows.shape == (len(data["beta"]), 3)

    def test_bond_rejects_state_based_model(self):
        r = _run_perifit("--demo", "--method", "bond", "--model", "osb")
        assert r.returncode != 0 and "bond-based" in r.stderr

    def test_bond_rejects_clean(self):
        r = _run_perifit("--demo", "--method", "bond", "--clean")
        assert r.returncode != 0

    def test_mesh_required_without_demo(self):
        assert _run_perifit("--method", "nodal").returncode != 0

    def test_mesh_file(self, tmp_path):
        from perifit import get_example_mesh
        r = _run_perifit("--mesh", str(get_example_mesh()), "--method", "nodal",
                         "--outdir", str(tmp_path), "--output", "csv", "--stem", "test")
        assert r.returncode == 0, r.stderr
        assert (tmp_path / "test.csv").exists()

    def test_planar_mesh_needs_dim(self, tmp_path):
        s = np.arange(12) * 0.1
        X, Y = np.meshgrid(s, s, indexing="ij")
        mesh = tmp_path / "plate.csv"
        np.savetxt(mesh, np.column_stack([X.ravel(), Y.ravel(), np.zeros(X.size)]), delimiter=",")
        r = _run_perifit("--mesh", str(mesh), "--method", "bond", "--outdir", str(tmp_path))
        assert r.returncode != 0 and "dim=2" in r.stderr
        r = _run_perifit("--mesh", str(mesh), "--method", "bond", "--dim", "2",
                         "--horizon", "0.3", "--outdir", str(tmp_path))
        assert r.returncode == 0, r.stderr

"""Tests for the perifit CLI."""

import subprocess
import sys
import pytest


def _run_perifit(*args, timeout=120):
    """Run perifit CLI and return CompletedProcess."""
    return subprocess.run(
        [sys.executable, '-m', 'perifit.cli', *args],
        capture_output=True, text=True, timeout=timeout,
    )


class TestCLI:
    def test_help(self):
        r = _run_perifit('--help')
        assert r.returncode == 0
        assert 'perifit' in r.stdout.lower()

    def test_demo_state_based(self, tmp_path):
        r = _run_perifit('--demo', '--outdir', str(tmp_path))
        assert r.returncode == 0
        assert 'perifit' in r.stdout.lower()
        # Check output files exist
        assert (tmp_path / 'demo_cube.csv').exists()
        assert (tmp_path / 'demo_cube.vtk').exists()

    def test_demo_bond_based(self, tmp_path):
        r = _run_perifit('--demo', '--model', 'bb', '--outdir', str(tmp_path))
        assert r.returncode == 0
        assert 'BB-PD' in r.stdout or 'bb' in r.stdout.lower()

    def test_mesh_required_without_demo(self):
        r = _run_perifit()
        assert r.returncode != 0

    def test_mesh_file(self, tmp_path):
        """Run on the bundled test mesh."""
        from perifit import get_example_mesh
        mesh = str(get_example_mesh())
        outdir = str(tmp_path)
        r = _run_perifit('--mesh', mesh, '--outdir', outdir,
                         '--output', 'csv', '--stem', 'test')
        assert r.returncode == 0
        assert (tmp_path / 'test.csv').exists()

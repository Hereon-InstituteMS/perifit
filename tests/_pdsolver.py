"""Benchmark problems for the tests, built on :mod:`perifit.benchmarks`.

Each helper returns ``(nodes, volumes, horizon, bonds, error)`` where
``error(factor)`` solves the static PD problem with one multiplier per bond
and returns the relative L2 displacement error.
"""
import numpy as np

from perifit.benchmarks import bar, cantilever


def _unpack(prob):
    return prob.nodes, prob.volumes, prob.horizon, prob.bonds, lambda f: prob.error(prob.bonds, f)


def bar_problem(dx, m=4):
    """Example 1 of the IJNME paper (nodes as a 1-D array)."""
    nodes, V, delta, bonds, error = _unpack(bar(dx, m))
    return nodes[:, 0], V, delta, bonds, error


def cantilever_problem(dx, m=4):
    """Example 2 of the IJNME paper."""
    return _unpack(cantilever(dx, m))


def nodal_factor(w, bonds):
    """Bond multipliers of nodal weights, (w_i + w_j) / 2."""
    return 0.5 * (w[bonds[:, 0]] + w[bonds[:, 1]])

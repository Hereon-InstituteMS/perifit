"""Closed-form 1-D segment moments for the weight-fitting analytical targets.

For the 1-D horizon [-delta, delta] and a non-negative integer exponent p:

    segment_moment(delta, p)         = int x^p dx
    segment_moment_over_r(delta, p)  = int x^p / |x| dx
    segment_moment_over_r3(delta, p) = int x^p / |x|^3 dx

All three vanish unless p is even, which is the 1-D case of the general d-ball
moment of eq. (63): the angular factor on S^0 is 2 for even exponents and 0 for
odd ones.
"""
from __future__ import annotations


def _parity(p: int) -> float:
    """Angular factor on S^0: 2 for even exponents, 0 for odd ones."""
    return 0.0 if p % 2 else 2.0


def segment_moment(delta: float, p: int) -> float:
    """int_{-delta}^{delta} x^p dx."""
    ang = _parity(p)
    if ang == 0.0:
        return 0.0
    return ang * delta ** (p + 1) / (p + 1)


def segment_moment_over_r(delta: float, p: int) -> float:
    """int_{-delta}^{delta} x^p / |x| dx.  Requires p > 0."""
    ang = _parity(p)
    if ang == 0.0:
        return 0.0
    if p == 0:
        raise ValueError("segment_moment_over_r is singular for p=0")
    return ang * delta ** p / p


def segment_moment_over_r3(delta: float, p: int) -> float:
    """int_{-delta}^{delta} x^p / |x|^3 dx.  Requires p > 2."""
    ang = _parity(p)
    if ang == 0.0:
        return 0.0
    if p <= 2:
        raise ValueError("segment_moment_over_r3 is singular for p<=2")
    return ang * delta ** (p - 2) / (p - 2)


# Convenient analytical full-horizon quantities ------------------------------
def full_segment_length(delta: float) -> float:
    return 2.0 * delta


def full_segment_weighted_volume(delta: float) -> float:
    """m_a = int_{-delta}^{delta} |xi|^2 dxi = 2*delta^3/3 (paper, eq. (25))."""
    return 2.0 * delta ** 3 / 3.0

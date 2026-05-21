"""Closed-form 2-D disc moments for the weight-fitting analytical targets.

For a 2-D disc D_delta of radius delta centred at the origin and
non-negative integer exponents (px, py):

    disc_moment(delta, px, py)         = int_{D_delta} x^px y^py dA
    disc_moment_over_r2(delta, px, py) = int_{D_delta} x^px y^py / r^2 dA
    disc_moment_over_r3(delta, px, py) = int_{D_delta} x^px y^py / r^3 dA

All three vanish unless every exponent is even.
"""
from __future__ import annotations

import math


def _angular_2d(px: int, py: int) -> float:
    """Angular integral on the unit circle: int_0^{2pi} cos^px sin^py dtheta.

    Returns 0 if any exponent is odd.
    """
    if (px % 2) or (py % 2):
        return 0.0
    # For px=2a, py=2b:
    # integral = 2 * B(a+1/2, b+1/2) = 2 * Gamma(a+1/2)*Gamma(b+1/2) / Gamma(a+b+1)
    a = px / 2.0
    b = py / 2.0
    return 2.0 * math.gamma(a + 0.5) * math.gamma(b + 0.5) / math.gamma(a + b + 1.0)


def disc_moment(delta: float, px: int, py: int) -> float:
    """int_{D_delta} x^px y^py dA."""
    ang = _angular_2d(px, py)
    if ang == 0.0:
        return 0.0
    p = px + py
    # radial: int_0^delta r^{p+1} dr = delta^{p+2}/(p+2)
    return ang * delta ** (p + 2) / (p + 2)


def disc_moment_over_r2(delta: float, px: int, py: int) -> float:
    """int_{D_delta} x^px y^py / r^2 dA.  Requires px+py > 0."""
    ang = _angular_2d(px, py)
    if ang == 0.0:
        return 0.0
    p = px + py
    if p == 0:
        raise ValueError("disc_moment_over_r2 is singular for p=0")
    # radial: int_0^delta r^{p-1} dr = delta^p / p
    return ang * delta ** p / p


def disc_moment_over_r3(delta: float, px: int, py: int) -> float:
    """int_{D_delta} x^px y^py / r^3 dA.  Requires px+py > 1."""
    ang = _angular_2d(px, py)
    if ang == 0.0:
        return 0.0
    p = px + py
    if p <= 1:
        raise ValueError("disc_moment_over_r3 is singular for p<=1")
    # radial: int_0^delta r^{p-2} dr = delta^{p-1} / (p-1)
    return ang * delta ** (p - 1) / (p - 1)


# Convenient analytical full-disc quantities --------------------------------
def full_disc_area(delta: float) -> float:
    return math.pi * delta ** 2


def full_disc_weighted_volume(delta: float) -> float:
    """m_a = int_{D_delta} |xi|^2 dA = pi*delta^4/2."""
    return math.pi * delta ** 4 / 2.0


def full_disc_shape_tensor_diag(delta: float) -> float:
    """K_xx = K_yy = int_{D_delta} xi_x^2 dA = pi*delta^4/4."""
    return math.pi * delta ** 4 / 4.0

"""Closed-form 3-D ball moments for the weight-fitting analytical targets.

For a ball B_delta of radius delta centred at the origin, with non-negative
integer exponents r = (r1, r2, r3), the general ball moment is:

    I_q(r) = int_{B_delta} xi_1^r1 * xi_2^r2 * xi_3^r3 / ||xi||^q  dV

Three cases are used:
    ball_moment(delta, r1, r2, r3)         ->  q = 0
    ball_moment_over_r2(delta, r1, r2, r3) ->  q = 2  (OSB kernel)
    ball_moment_over_r3(delta, r1, r2, r3) ->  q = 3  (BB kernel)

All three vanish unless every exponent is even.  When all exponents are even,
the angular factor reduces to a product of Gamma functions:

    angular = 2 * Gamma((r1+1)/2) * Gamma((r2+1)/2) * Gamma((r3+1)/2)
                / Gamma((r1+r2+r3+3)/2)

Authors: Arman Shojaei & Alexander Hermann
         Helmholtz-Zentrum Hereon, Geesthacht, Germany.
"""
from __future__ import annotations

import math


def _angular_part(px: int, py: int, pz: int) -> float:
    """Solid-angle integral of x^px y^py z^pz on the unit sphere.

    Returns int_{S^2} (x/r)^px (y/r)^py (z/r)^pz dS.
    Zero unless all exponents are even.
    """
    if (px % 2) or (py % 2) or (pz % 2):
        return 0.0
    a = (px + 1) / 2.0
    b = (py + 1) / 2.0
    c = (pz + 1) / 2.0
    # Surface integral on unit sphere of x^px y^py z^pz =
    # 2 * Gamma(a)*Gamma(b)*Gamma(c) / Gamma(a+b+c)
    return 2.0 * math.gamma(a) * math.gamma(b) * math.gamma(c) / math.gamma(a + b + c)


def ball_moment(delta: float, px: int, py: int, pz: int) -> float:
    """int_{B_delta} x^px y^py z^pz dV."""
    ang = _angular_part(px, py, pz)
    if ang == 0.0:
        return 0.0
    p = px + py + pz
    return ang * delta ** (p + 3) / (p + 3)


def ball_moment_over_r2(delta: float, px: int, py: int, pz: int) -> float:
    """int_{B_delta} x^px y^py z^pz / r^2 dV."""
    ang = _angular_part(px, py, pz)
    if ang == 0.0:
        return 0.0
    p = px + py + pz
    return ang * delta ** (p + 1) / (p + 1)


def ball_moment_over_r3(delta: float, px: int, py: int, pz: int) -> float:
    """int_{B_delta} x^px y^py z^pz / r^3 dV.  (p+px+py+pz must be > 0.)"""
    ang = _angular_part(px, py, pz)
    if ang == 0.0:
        return 0.0
    p = px + py + pz
    if p == 0:
        raise ValueError("ball_moment_over_r3 is singular for p=0")
    return ang * delta ** p / p


# Convenient analytical full-ball quantities -------------------------------
def full_ball_volume(delta: float) -> float:
    return 4.0 * math.pi * delta ** 3 / 3.0


def full_ball_weighted_volume(delta: float) -> float:
    """m_a = int_{B_delta} |xi|^2 dV = 4 pi delta^5 / 5."""
    return 4.0 * math.pi * delta ** 5 / 5.0


def full_ball_shape_tensor_diag(delta: float) -> float:
    """K_xx = K_yy = K_zz = int_{B_delta} xi_x^2 dV = 4 pi delta^5 / 15."""
    return 4.0 * math.pi * delta ** 5 / 15.0

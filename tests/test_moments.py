"""Tests for analytical ball moment functions."""
import math
import pytest
from perifit.core.moments import (
    ball_moment, ball_moment_over_r2, full_ball_weighted_volume,
)

DELTA = 0.15


def test_full_ball_weighted_volume():
    m_a = full_ball_weighted_volume(DELTA)
    assert abs(m_a - 4 * math.pi * DELTA ** 5 / 5) < 1e-15


def test_ball_moment_known_values():
    # M_0(2,0,0) = (4pi/15) * delta^5
    assert abs(ball_moment(DELTA, 2, 0, 0) - (4 * math.pi / 15) * DELTA ** 5) < 1e-12
    # M_0(0,2,0) == M_0(2,0,0) by symmetry
    assert abs(ball_moment(DELTA, 0, 2, 0) - ball_moment(DELTA, 2, 0, 0)) < 1e-15
    # M_0(0,0,2) == M_0(2,0,0) by symmetry
    assert abs(ball_moment(DELTA, 0, 0, 2) - ball_moment(DELTA, 2, 0, 0)) < 1e-15
    # M_0(4,0,0) = (4pi/35)*delta^7
    assert abs(ball_moment(DELTA, 4, 0, 0) - (4 * math.pi / 35) * DELTA ** 7) < 1e-12
    # Sum M_0(2,0,0) * 3 = m_a
    assert abs(3 * ball_moment(DELTA, 2, 0, 0) - full_ball_weighted_volume(DELTA)) < 1e-12


def test_ball_moment_over_r2_known_values():
    # M_2(2,0,0) = (4pi/9)*delta^3
    assert abs(ball_moment_over_r2(DELTA, 2, 0, 0) - (4 * math.pi / 9) * DELTA ** 3) < 1e-12
    # M_2(4,0,0) = (4pi/25)*delta^5
    assert abs(ball_moment_over_r2(DELTA, 4, 0, 0) - (4 * math.pi / 25) * DELTA ** 5) < 1e-12


def test_odd_exponents_zero():
    for delta in (0.1, 0.3, 1.0):
        for (p, q, r) in [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 2, 0), (3, 1, 2)]:
            assert ball_moment(delta, p, q, r) == 0.0
            assert ball_moment_over_r2(delta, p, q, r) == 0.0


def test_ball_moment_over_r2_zero_total():
    # P=0 case: ball_moment_over_r2 returns 0 (angular_part gives 4*pi
    # but the radial integral diverges; the reference code returns 0 for P<=0)
    # The reference moments.py uses Gamma-based angular_part which gives nonzero
    # for (0,0,0), but ball_moment_over_r2 has p+1 in denominator -> diverges at p=0
    # Actually in the reference code, ball_moment_over_r2(delta, 0, 0, 0):
    #   ang = _angular_part(0, 0, 0) -> 2*Gamma(0.5)^3/Gamma(1.5) = 4*pi
    #   p = 0 -> delta^1 / 1 * 4*pi = 4*pi*delta
    # Wait, let me check: the reference has no special case for P=0 in ball_moment_over_r2
    # p+1 = 1, so it's delta^1/1 * angular = fine.
    # But the old test expected 0.0. Let's verify the new behaviour.
    result = ball_moment_over_r2(DELTA, 0, 0, 0)
    # In the new reference code, ball_moment_over_r2(delta, 0, 0, 0):
    # angular = 2*Gamma(0.5)*Gamma(0.5)*Gamma(0.5)/Gamma(1.5)
    # = 2*pi^(3/2)/(sqrt(pi)/2) = 4*pi
    # result = 4*pi * delta^1 / 1 = 4*pi*delta
    expected = 4 * math.pi * DELTA
    assert abs(result - expected) < 1e-12

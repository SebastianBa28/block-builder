"""Trajectory interpolation utilities using polynomial splines.

Provides cubic and quintic spline functions for smooth joint-space
trajectory generation with configurable boundary conditions on
position, velocity, and acceleration.
"""

import numpy as np


# ── Basic Trajectories ───────────────────────────────────────────────

def hold(p0):
    """Hold constant position. Returns (position, zero velocity)."""
    return (p0, 0 * p0)


def interpolate(t, T, p0, pf):
    """Linear interpolation from p0 to pf over duration T.

    Arguments
    ---------
    t : float
        Current time within the segment.
    T : float
        Total segment duration.
    p0 : np.ndarray
        Start position.
    pf : np.ndarray
        Final position.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (position, velocity) at time t.
    """
    p = p0 + (pf - p0) / T * t
    v = (pf - p0) / T
    return (p, v)


# ── Cubic Splines ────────────────────────────────────────────────────

def goto(t, T, p0, pf):
    """Cubic spline from p0 to pf with zero velocity at both endpoints.

    Arguments
    ---------
    t : float
        Current time within the segment.
    T : float
        Total segment duration.
    p0 : np.ndarray
        Start position.
    pf : np.ndarray
        Final position.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (position, velocity) at time t.
    """
    p = p0 + (pf - p0) * (3 * (t / T) ** 2 - 2 * (t / T) ** 3)
    v = (pf - p0) / T * (6 * (t / T) - 6 * (t / T) ** 2)
    return (p, v)


def spline(t, T, p0, pf, v0, vf):
    """Cubic spline with specified endpoint velocities.

    Arguments
    ---------
    t : float
        Current time within the segment.
    T : float
        Total segment duration.
    p0 : np.ndarray
        Start position.
    pf : np.ndarray
        Final position.
    v0 : np.ndarray
        Start velocity.
    vf : np.ndarray
        Final velocity.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (position, velocity) at time t.
    """
    a = p0
    b = v0
    c = 3 * (pf - p0) / T ** 2 - vf / T - 2 * v0 / T
    d = -2 * (pf - p0) / T ** 3 + vf / T ** 2 + v0 / T ** 2
    p = a + b * t + c * t ** 2 + d * t ** 3
    v = b + 2 * c * t + 3 * d * t ** 2
    return (p, v)


# ── Quintic Splines ──────────────────────────────────────────────────

def goto5(t, T, p0, pf):
    """Quintic spline from p0 to pf with zero velocity and acceleration.

    Arguments
    ---------
    t : float
        Current time within the segment.
    T : float
        Total segment duration.
    p0 : np.ndarray
        Start position.
    pf : np.ndarray
        Final position.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        (position, velocity, acceleration) at time t.
    """
    p = p0 + (pf - p0) * (10 * (t / T) ** 3 - 15 * (t / T) ** 4 + 6 * (t / T) ** 5)
    v = (pf - p0) / T * (30 * (t / T) ** 2 - 60 * (t / T) ** 3 + 30 * (t / T) ** 4)
    a = (pf - p0) / (T ** 2) * (60 * (t / T) - 180 * (t / T) ** 2 + 120 * (t / T) ** 3)
    return (p, v, a)


def spline5(t, T, p0, pf, v0, vf, a0, af):
    """Quintic spline with specified endpoint velocities and accelerations.

    Arguments
    ---------
    t : float
        Current time within the segment.
    T : float
        Total segment duration.
    p0 : np.ndarray
        Start position.
    pf : np.ndarray
        Final position.
    v0 : np.ndarray
        Start velocity.
    vf : np.ndarray
        Final velocity.
    a0 : np.ndarray
        Start acceleration.
    af : np.ndarray
        Final acceleration.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (position, velocity) at time t.
    """
    a = p0
    b = v0
    c = a0
    d = (10 * (pf - p0) / T ** 3 - 6 * v0 / T ** 2
         - 3 * a0 / T - 4 * vf / T ** 2 + 0.5 * af / T)
    e = (-15 * (pf - p0) / T ** 4 + 8 * v0 / T ** 3
         + 3 * a0 / T ** 2 + 7 * vf / T ** 3 - af / T ** 2)
    f = (6 * (pf - p0) / T ** 5 - 3 * v0 / T ** 4
         - a0 / T ** 3 - 3 * vf / T ** 4 + 0.5 * af / T ** 3)
    p = a + b * t + c * t ** 2 + d * t ** 3 + e * t ** 4 + f * t ** 5
    v = b + 2 * c * t + 3 * d * t ** 2 + 4 * e * t ** 3 + 5 * f * t ** 4
    return (p, v)


# ── Geometry Intersections ───────────────────────────────────────────

def solve_line_ellipsoid_interception(
    p_ball_start, v_ball_start, p_ellipsoid_center, axes, eps=0.5
):
    """Solve line-ellipsoid intersection for ball catching.

    Transforms the problem into unit-sphere space and solves the
    resulting quadratic.  Returns the earliest positive intersection
    time plus an offset eps, or the closest-approach point if the
    line misses the ellipsoid entirely.

    Arguments
    ---------
    p_ball_start : np.ndarray
        Starting position [x, y, z] of the ball.
    v_ball_start : np.ndarray
        Velocity [vx, vy, vz] of the ball.
    p_ellipsoid_center : np.ndarray
        Center [x, y, z] of the ellipsoid.
    axes : tuple[float, float, float]
        Semi-axis lengths (a, b, c) of the ellipsoid.
    eps : float
        Time offset added to the intersection time to allow the arm
        to arrive slightly after the ball crosses the boundary.

    Returns
    -------
    tuple[float, np.ndarray]
        (t, p) -- interception time and position.
    """
    a, b, c = axes

    # Transform to unit sphere space
    rel_pos = p_ball_start - p_ellipsoid_center
    p_scaled = rel_pos / np.array([a, b, c])
    v_scaled = v_ball_start / np.array([a, b, c])

    quad_a = np.dot(v_scaled, v_scaled)
    quad_b = 2 * np.dot(p_scaled, v_scaled)
    quad_c = np.dot(p_scaled, p_scaled) - 1.0

    delta = quad_b ** 2 - 4 * quad_a * quad_c

    if delta < 0:
        # Ball misses ellipsoid -- return closest approach
        t_closest = -quad_b / (2 * quad_a)
        if t_closest < 0:
            return 0.0, p_ball_start
        return t_closest, p_ball_start + v_ball_start * t_closest

    sqrt_delta = np.sqrt(delta)
    t1 = (-quad_b - sqrt_delta) / (2 * quad_a)
    t2 = (-quad_b + sqrt_delta) / (2 * quad_a)

    if t1 > 0:
        return t1 + eps, p_ball_start + v_ball_start * t1
    if t2 > 0:
        return t2 + eps, p_ball_start + v_ball_start * t2

    return 0.0, p_ball_start

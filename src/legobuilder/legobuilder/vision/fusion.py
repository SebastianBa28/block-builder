"""Overhead camera contour-to-pointcloud conversion.

Fills each detected block's rectangular footprint with points at voxel
resolution so every overlapping voxel gets at least one observation.
Points are in world frame at a known Z height (block surface).
"""

from __future__ import annotations

import numpy as np
from numpy import ndarray as arr

from legobuilder.schemas import Color


# ── Color Mapping ──────────────────────────────────────────────────────

COLOR_RGB: dict[Color, arr] = {
    Color.ORANGE: np.array([1.0, 0.5, 0.0], dtype=np.float32),
    Color.YELLOW: np.array([1.0, 1.0, 0.0], dtype=np.float32),
    Color.BLUE:   np.array([0.0, 0.0, 1.0], dtype=np.float32),
    Color.GREEN:  np.array([0.0, 1.0, 0.0], dtype=np.float32),
    Color.RED:    np.array([1.0, 0.0, 0.0], dtype=np.float32),
}


def contours_to_pointcloud(
    contour_infos,
    z_height: float,
    voxel_resolution: float = 0.005,
) -> tuple[arr, arr]:
    """Fill detected block rectangles with points at voxel resolution.

    For each contour with corner_xys, the rectangular footprint is filled
    via bilinear interpolation across the quad so that every voxel the
    rectangle overlaps receives at least one point.  Contours without
    corner_xys are skipped.

    Arguments
    ---------
    contour_infos : list[ContourInfo]
        Overhead camera contour detections with world-frame coordinates.
    z_height : float
        Z coordinate in meters for all projected points (block surface).
    voxel_resolution : float
        Spacing between fill points in meters.  Should match the voxel
        grid resolution so every covered voxel gets a hit.

    Returns
    -------
    tuple[arr, arr]
        points: (N, 3) float64 XYZ in world frame.
        colors: (N, 3) float32 RGB in [0, 1].
    """
    all_points = []
    all_colors = []

    for ci in contour_infos:
        if not (hasattr(ci, 'corner_xys') and ci.corner_xys):
            continue
        corners = ci.corner_xys
        if len(corners) < 4:
            continue

        rgb = COLOR_RGB.get(
            ci.color,
            np.array([0.5, 0.5, 0.5], dtype=np.float32),
        )

        # Corners as numpy: c0--c1 and c0--c3 define two edges.
        c0 = np.array(corners[0], dtype=np.float64)
        c1 = np.array(corners[1], dtype=np.float64)
        c3 = np.array(corners[3], dtype=np.float64)
        c2 = np.array(corners[2], dtype=np.float64)

        # Edge lengths determine step counts.
        len_u = max(np.linalg.norm(c1 - c0), 1e-9)
        len_v = max(np.linalg.norm(c3 - c0), 1e-9)
        steps_u = max(int(np.ceil(len_u / voxel_resolution)), 1) + 1
        steps_v = max(int(np.ceil(len_v / voxel_resolution)), 1) + 1

        # Bilinear fill: p(u,v) = (1-v)*((1-u)*c0 + u*c1)
        #                        +   v *((1-u)*c3 + u*c2)
        us = np.linspace(0.0, 1.0, steps_u)
        vs = np.linspace(0.0, 1.0, steps_v)
        uu, vv = np.meshgrid(us, vs)
        uu = uu.ravel()
        vv = vv.ravel()

        xs = ((1 - vv) * ((1 - uu) * c0[0] + uu * c1[0])
              + vv * ((1 - uu) * c3[0] + uu * c2[0]))
        ys = ((1 - vv) * ((1 - uu) * c0[1] + uu * c1[1])
              + vv * ((1 - uu) * c3[1] + uu * c2[1]))
        zs = np.full_like(xs, z_height)

        pts = np.column_stack([xs, ys, zs])
        all_points.append(pts)
        all_colors.append(np.tile(rgb, (len(pts), 1)))

    if not all_points:
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 3), dtype=np.float32),
        )

    return (
        np.concatenate(all_points, axis=0),
        np.concatenate(all_colors, axis=0).astype(np.float32),
    )

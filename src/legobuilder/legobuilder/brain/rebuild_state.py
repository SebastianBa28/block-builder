"""Stateless placed_grid reconstruction from detections."""

from __future__ import annotations

import numpy as np

from legobuilder.brain.block_structure import BlockStructure
from legobuilder.config import BLOCK_SIZE
from legobuilder.schemas import Object

XY_TOL = BLOCK_SIZE / 2   # ~8.5 mm
Z_TOL = BLOCK_SIZE / 2    # ~17 mm


def rebuild_placed_grid(
    detections: list[Object],
    structure: BlockStructure,
    logger=None,
) -> tuple[np.ndarray, list[Object]]:
    """Rebuild placed_grid from detections alone (stateless).

    Matches each detection to the closest candidate cell in
    ``structure.block_grid`` using 3-D tolerance, then fills the
    rebuilt grid.  Blocks below a matched cell that exist in
    ``block_grid`` are assumed placed (invisible-block inference).

    Parameters
    ----------
    detections : list[Object]
        Detected objects from a scan of the structure area.
    structure : BlockStructure
        Provides ``block_grid`` (target design) and
        ``get_world_position`` for coordinate conversion.
    logger : optional
        ROS-compatible logger for debug output.

    Returns
    -------
    placed_grid : np.ndarray
        Same shape as ``structure.block_grid``.  Matched cells store the
        *detected* ``ObjectType.value`` (not the expected type).
        Inferred invisible cells store the expected type from
        ``block_grid``.
    unassigned : list[Object]
        Detections not assigned to any grid cell (misplaced blocks,
        available for re-grasping).
    """
    block_grid = structure.block_grid

    if not detections:
        return np.zeros_like(block_grid, dtype=int), []

    # 1. Candidates: every non-zero cell in block_grid.
    candidate_indices = np.argwhere(block_grid != 0)
    candidates: dict[tuple[int, int, int], int] = {}
    for idx in candidate_indices:
        cell = (int(idx[0]), int(idx[1]), int(idx[2]))
        candidates[cell] = int(block_grid[cell])

    # 2. Pre-compute world positions.
    cell_positions = {
        cell: structure.get_world_position(*cell)
        for cell in candidates
    }

    # 3. Score (detection, cell) pairs within tolerance.
    pairs: list[tuple[tuple[int, int, int], int, bool, float]] = []
    for det_idx, det in enumerate(detections):
        dx, dy, dz = det.center_xyz
        for cell, pos in cell_positions.items():
            ex, ey, ez = float(pos[0]), float(pos[1]), float(pos[2])
            if (abs(dx - ex) < XY_TOL
                    and abs(dy - ey) < XY_TOL
                    and abs(dz - ez) < Z_TOL):
                color_match = det.obj_type.value == candidates[cell]
                dist = (
                    (dx - ex) ** 2 + (dy - ey) ** 2 + (dz - ez) ** 2
                ) ** 0.5
                pairs.append((cell, det_idx, color_match, dist))

    # Sort: colour match first (True > False), then closest distance.
    pairs.sort(key=lambda p: (not p[2], p[3]))

    # 4. Greedy assignment.
    assigned_cells: dict[tuple[int, int, int], Object] = {}
    assigned_dets: set[int] = set()
    for cell, det_idx, _cm, _d in pairs:
        if cell not in assigned_cells and det_idx not in assigned_dets:
            assigned_cells[cell] = detections[det_idx]
            assigned_dets.add(det_idx)

    # 4b. Remove lower-layer matches where a higher-layer match exists
    # at the same (r, c) — lower layers are inferred as invisible.
    rc_max_layer: dict[tuple[int, int], int] = {}
    for cell in assigned_cells:
        r, c, layer = cell
        rc = (r, c)
        if rc not in rc_max_layer or layer > rc_max_layer[rc]:
            rc_max_layer[rc] = layer

    cells_to_remove = []
    for cell in assigned_cells:
        r, c, layer = cell
        if layer < rc_max_layer[(r, c)]:
            cells_to_remove.append(cell)

    for cell in cells_to_remove:
        det = assigned_cells.pop(cell)
        det_idx = detections.index(det)
        assigned_dets.discard(det_idx)

    unassigned = [d for i, d in enumerate(detections)
                  if i not in assigned_dets]

    if logger:
        for cell, det in assigned_cells.items():
            pos = cell_positions[cell]
            dx = det.center_xyz[0] - float(pos[0])
            dy = det.center_xyz[1] - float(pos[1])
            dz = det.center_xyz[2] - float(pos[2])
            logger.debug(
                f"rebuild: matched (r,c,l)={cell} <- {det.obj_type.name} "
                f"[{det.center_xyz[0]:.4f}, {det.center_xyz[1]:.4f}, "
                f"{det.center_xyz[2]:.4f}] "
                f"\u0394(x={dx:+.4f}, y={dy:+.4f}, z={dz:+.4f})"
            )
        for det in unassigned:
            # Find nearest candidate cell for diagnostic info
            nearest_cell = None
            nearest_dist = float('inf')
            for cell, pos in cell_positions.items():
                d = ((det.center_xyz[0] - float(pos[0])) ** 2
                     + (det.center_xyz[1] - float(pos[1])) ** 2
                     + (det.center_xyz[2] - float(pos[2])) ** 2) ** 0.5
                if d < nearest_dist:
                    nearest_dist = d
                    nearest_cell = cell
            suffix = ""
            if nearest_cell is not None:
                pos = cell_positions[nearest_cell]
                dx = det.center_xyz[0] - float(pos[0])
                dy = det.center_xyz[1] - float(pos[1])
                dz = det.center_xyz[2] - float(pos[2])
                suffix = (
                    f" nearest={nearest_cell} "
                    f"\u0394(x={dx:+.4f}, y={dy:+.4f}, z={dz:+.4f})"
                )
            logger.info(
                f"rebuild: unassigned {det.obj_type.name} "
                f"[{det.center_xyz[0]:.4f}, {det.center_xyz[1]:.4f}, "
                f"{det.center_xyz[2]:.4f}]{suffix}"
            )

    # 5. Build placed_grid.
    placed_grid = np.zeros_like(block_grid, dtype=int)
    for cell, det in assigned_cells.items():
        r, c, layer = cell
        # Store detected type (not expected) to expose wrong-color placements.
        placed_grid[r, c, layer] = det.obj_type.value
        # Infer invisible blocks below from block_grid.
        for below in range(layer):
            if block_grid[r, c, below] != 0:
                placed_grid[r, c, below] = int(block_grid[r, c, below])

    return placed_grid, unassigned

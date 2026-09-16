"""Pure reachability analysis functions for grid-based block picking.

Determines which blocks on the placed structure grid are physically
ungrippable (face-blocked on both grip axes) or unreachable (blocked
by overhang in the gripper footprint). These functions operate on the
placed_grid numpy array and grid-cell coordinates only -- no ROS
dependencies.
"""

from __future__ import annotations

import numpy as np

from legobuilder.config import BLOCK_SIZE

# Grid offsets for each grip axis.
# Axis A: gripper clamps col- and col+ faces (left/right).
AXIS_A_FACE_OFFSETS = [(0, -1, 0), (0, 1, 0)]   # (dr, dc, dl)
AXIS_A_FOOTPRINT_OFFSETS = [(0, -1), (0, 1)]     # (dr, dc)

# Axis B: gripper clamps row- and row+ faces (front/back).
AXIS_B_FACE_OFFSETS = [(-1, 0, 0), (1, 0, 0)]
AXIS_B_FOOTPRINT_OFFSETS = [(-1, 0), (1, 0)]

GRIP_AXES = [
    ('axis_a', AXIS_A_FACE_OFFSETS, AXIS_A_FOOTPRINT_OFFSETS),
    ('axis_b', AXIS_B_FACE_OFFSETS, AXIS_B_FOOTPRINT_OFFSETS),
]


def is_axis_grippable(
    placed_grid: np.ndarray,
    row: int, col: int, layer: int,
    face_offsets: list[tuple[int, int, int]],
    occupied_neighbors: set[tuple[int, int, int]] | None = None,
) -> bool:
    """Check if both faces on a grip axis are clear for gripping.

    Parameters
    ----------
    placed_grid : np.ndarray
        The (rows, cols, layers) grid of placed block types.
    row, col, layer : int
        Grid coordinates of the target block.
    face_offsets : list of (dr, dc, dl)
        The two face neighbor offsets for this axis.
    occupied_neighbors : set of (row, col, layer) or None
        Additional occupied cells from unassigned blocks mapped to grid.

    Returns
    -------
    bool
        True if neither face on this axis is blocked.
    """
    if occupied_neighbors is None:
        occupied_neighbors = set()

    for dr, dc, dl in face_offsets:
        nr, nc, nl = row + dr, col + dc, layer + dl
        # Check placed_grid bounds -- out-of-bounds is implicitly clear
        if (0 <= nr < placed_grid.shape[0]
                and 0 <= nc < placed_grid.shape[1]
                and 0 <= nl < placed_grid.shape[2]):
            if placed_grid[nr, nc, nl] != 0:
                return False
        # Check unassigned block neighbors
        if (nr, nc, nl) in occupied_neighbors:
            return False
    return True


def has_overhang_on_axis(
    placed_grid: np.ndarray,
    row: int, col: int, layer: int,
    footprint_offsets: list[tuple[int, int]],
) -> bool:
    """Check if any column in the gripper footprint has blocks above target.

    Checks the target column itself plus the adjacent columns for this
    axis. Any placed block in those columns at layers above the target
    constitutes an overhang.

    Parameters
    ----------
    placed_grid : np.ndarray
        The (rows, cols, layers) grid.
    row, col, layer : int
        Grid coordinates of the target block.
    footprint_offsets : list of (dr, dc)
        Column offsets to check (relative to target).

    Returns
    -------
    bool
        True if an overhang exists (axis is blocked from above).
    """
    num_layers = placed_grid.shape[2]

    # Check all footprint columns (target + adjacent) at layers above target
    columns_to_check = [(row, col)] + [(row + dr, col + dc) for dr, dc in footprint_offsets]
    for r, c in columns_to_check:
        if not (0 <= r < placed_grid.shape[0] and 0 <= c < placed_grid.shape[1]):
            continue
        for lyr in range(layer + 1, num_layers):
            if placed_grid[r, c, lyr] != 0:
                return True
    return False


def map_unassigned_to_grid(
    unassigned: list,
    structure,
) -> dict[tuple[int, int, int], int]:
    """Map unassigned detected blocks to their actual grid cells.

    Uses the inverse world-to-grid transform so that misplaced blocks
    are mapped to the cell they physically occupy, even if that cell
    has no target block in block_grid.

    Parameters
    ----------
    unassigned : list of Object
        Detected objects not assigned to any grid cell.
    structure : BlockStructure
        Provides world_to_grid and get_world_position for matching.

    Returns
    -------
    dict mapping (row, col, layer) to ObjectType.value
        Grid cells occupied by unassigned blocks with their detected color.
    """
    occupied = {}
    grid_shape = structure.block_grid.shape

    for obj in unassigned:
        if not obj.obj_type.is_block():
            continue
        ox, oy, oz = obj.center_xyz
        r, c, lyr = structure.world_to_grid(ox, oy, oz)
        # Must be within grid bounds
        if not (0 <= r < grid_shape[0] and 0 <= c < grid_shape[1]
                and 0 <= lyr < grid_shape[2]):
            continue
        # Sanity check: detected position must be close to cell center
        wp = structure.get_world_position(r, c, lyr)
        dist = ((ox - wp[0])**2 + (oy - wp[1])**2 + (oz - wp[2])**2) ** 0.5
        if dist < BLOCK_SIZE:
            occupied[(r, c, lyr)] = obj.obj_type.value
    return occupied


def get_unreachable_cells(
    placed_grid: np.ndarray,
    unassigned: list | None = None,
    structure=None,
) -> dict[tuple[int, int, int], tuple[str, int]]:
    """Identify all unreachable cells in the placed grid.

    A cell is unreachable when no grip axis provides both clear faces
    and clear overhang. The reason is classified as 'ungrippable' (both
    axes face-blocked), 'overhang' (faces clear but overhang blocks all
    axes), or 'both'.

    Parameters
    ----------
    placed_grid : np.ndarray
        The (rows, cols, layers) grid of placed block types.
    unassigned : list of Object or None
        Detected objects not assigned to any grid cell.
    structure : BlockStructure or None
        Required if unassigned is provided.

    Returns
    -------
    dict mapping (row, col, layer) to (reason, block_type)
        Reason is 'ungrippable', 'overhang', or 'both'.
        block_type is the ObjectType.value of the detected block.
    """
    occupied_neighbors: dict[tuple[int, int, int], int] = {}
    if unassigned and structure:
        occupied_neighbors = map_unassigned_to_grid(unassigned, structure)

    # is_axis_grippable expects a set for occupied_neighbors
    occupied_set = set(occupied_neighbors.keys())

    unreachable = {}
    placed_cells = np.argwhere(placed_grid != 0)

    for idx in placed_cells:
        r, c, lyr = int(idx[0]), int(idx[1]), int(idx[2])
        any_axis_clear = False

        for _axis_name, face_offsets, footprint_offsets in GRIP_AXES:
            grippable = is_axis_grippable(
                placed_grid, r, c, lyr, face_offsets, occupied_set
            )
            overhang = has_overhang_on_axis(
                placed_grid, r, c, lyr, footprint_offsets
            )
            if grippable and not overhang:
                any_axis_clear = True
                break

        if not any_axis_clear:
            # Determine reason by checking each axis independently
            a_grip = is_axis_grippable(
                placed_grid, r, c, lyr, AXIS_A_FACE_OFFSETS, occupied_set
            )
            b_grip = is_axis_grippable(
                placed_grid, r, c, lyr, AXIS_B_FACE_OFFSETS, occupied_set
            )
            a_oh = has_overhang_on_axis(
                placed_grid, r, c, lyr, AXIS_A_FOOTPRINT_OFFSETS
            )
            b_oh = has_overhang_on_axis(
                placed_grid, r, c, lyr, AXIS_B_FOOTPRINT_OFFSETS
            )

            grip_blocked = not a_grip and not b_grip
            oh_blocked = a_oh or b_oh

            if grip_blocked and oh_blocked:
                reason = 'both'
            elif grip_blocked:
                reason = 'ungrippable'
            else:
                reason = 'overhang'
            unreachable[(r, c, lyr)] = (reason, int(placed_grid[r, c, lyr]))

    # Also check unassigned block positions for grippability
    for (r, c, lyr), block_type in occupied_neighbors.items():
        any_axis_clear = False
        for _axis_name, face_offsets, footprint_offsets in GRIP_AXES:
            grippable = is_axis_grippable(
                placed_grid, r, c, lyr, face_offsets, occupied_set
            )
            overhang = has_overhang_on_axis(
                placed_grid, r, c, lyr, footprint_offsets
            )
            if grippable and not overhang:
                any_axis_clear = True
                break

        if not any_axis_clear:
            a_grip = is_axis_grippable(
                placed_grid, r, c, lyr, AXIS_A_FACE_OFFSETS, occupied_set
            )
            b_grip = is_axis_grippable(
                placed_grid, r, c, lyr, AXIS_B_FACE_OFFSETS, occupied_set
            )
            a_oh = has_overhang_on_axis(
                placed_grid, r, c, lyr, AXIS_A_FOOTPRINT_OFFSETS
            )
            b_oh = has_overhang_on_axis(
                placed_grid, r, c, lyr, AXIS_B_FOOTPRINT_OFFSETS
            )

            grip_blocked = not a_grip and not b_grip
            oh_blocked = a_oh or b_oh

            if grip_blocked and oh_blocked:
                reason = 'both'
            elif grip_blocked:
                reason = 'ungrippable'
            else:
                reason = 'overhang'
            unreachable[(r, c, lyr)] = (reason, block_type)

    return unreachable

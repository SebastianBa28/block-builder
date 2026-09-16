"""Smart block placement ordering: center-outward, bottom-up, color-aware."""

from __future__ import annotations

import logging
import numpy as np

from legobuilder.schemas import Color, ObjectType

_NSEW_OFFSETS = ((-1, 0), (1, 0), (0, -1), (0, 1))

_log = logging.getLogger(__name__)


def _layer_center(block_grid: np.ndarray, layer: int):
    """Return (center_row, center_col) for occupied cells on *layer*."""
    layer_occupied = np.argwhere(block_grid[:, :, layer] != 0)
    if layer_occupied.size > 0:
        return layer_occupied[:, 0].mean(), layer_occupied[:, 1].mean()
    occupied = np.argwhere(block_grid != 0)
    return occupied[:, 0].mean(), occupied[:, 1].mean()


def _is_placeable(block_grid, placed_grid, r, c, layer):
    """True if all required support blocks below (r, c, layer) are placed."""
    if layer == 0:
        return True
    for l in range(layer):
        if block_grid[r, c, l] != 0 and placed_grid[r, c, l] == 0:
            return False
    return True


def _sort_cells_center_outward(cells, block_grid, target_layer):
    """Sort *cells* by distance-from-center then support score (descending)."""
    center_row, center_col = _layer_center(block_grid, target_layer)
    dists = np.sqrt(
        (cells[:, 0] - center_row) ** 2
        + (cells[:, 1] - center_col) ** 2
    )
    num_layers = block_grid.shape[2]
    support_scores = np.array([
        sum(
            block_grid[cell[0], cell[1], l] != 0
            for l in range(target_layer + 1, num_layers)
        )
        for cell in cells
    ])
    order = np.lexsort((-support_scores, dists))
    return cells[order]


def _count_same_layer_placed_connections(
    block_grid: np.ndarray,
    placed_grid: np.ndarray,
    r: int,
    c: int,
    layer: int,
) -> int:
    """Count NSEW neighbors on *layer* that are already correctly placed."""
    rows, cols = block_grid.shape[0], block_grid.shape[1]
    count = 0
    for dr, dc in _NSEW_OFFSETS:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            if block_grid[nr, nc, layer] != 0 and placed_grid[nr, nc, layer] == block_grid[nr, nc, layer]:
                count += 1
    return count


def _count_theoretical_connections(
    block_grid: np.ndarray,
    r: int,
    c: int,
    layer: int,
) -> int:
    """Count NSEW neighbors on *layer* that exist in the full block_grid design."""
    rows, cols = block_grid.shape[0], block_grid.shape[1]
    count = 0
    for dr, dc in _NSEW_OFFSETS:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            if block_grid[nr, nc, layer] != 0:
                count += 1
    return count


def _smooth_connection_order(
    block_grid: np.ndarray,
    placed_grid: np.ndarray,
    candidate_cell: tuple[int, int, int],
    candidate_type,
    candidates: list,
) -> tuple:
    """Optionally swap *candidate_cell* for an adjacent neighbor that reduces connection-spike loss.

    Only swaps to a neighbor N when:
    - N is unplaced, in *candidates* (color available), and adjacent to candidate
    - theoretical_connections(N) >= theoretical_connections(candidate)
    - Placing N first has lower sum-exp(c+1) loss than placing candidate first

    Loss formulas (C = candidate, N = neighbor; they are adjacent so placing one
    adds 1 to the other's connection count):
      loss_A (C first, N second) = exp(c_C + 1) + exp(c_N + 2)
      loss_B (N first, C second) = exp(c_N + 1) + exp(c_C + 2)
    Swap when loss_B < loss_A (equivalent to c_N > c_C).
    """
    r_c, col_c, layer = candidate_cell
    candidate_cell_set = {cell for cell, _ in candidates}
    candidate_type_map = {cell: ot for cell, ot in candidates}

    rows, cols = block_grid.shape[0], block_grid.shape[1]
    c_C = _count_same_layer_placed_connections(block_grid, placed_grid, r_c, col_c, layer)
    theo_C = _count_theoretical_connections(block_grid, r_c, col_c, layer)

    best_cell = candidate_cell
    best_type = candidate_type
    best_loss_B = None

    for dr, dc in _NSEW_OFFSETS:
        nr, nc = r_c + dr, col_c + dc
        if not (0 <= nr < rows and 0 <= nc < cols):
            continue
        n_cell = (nr, nc, layer)
        if block_grid[nr, nc, layer] == 0:
            continue
        if placed_grid[nr, nc, layer] == block_grid[nr, nc, layer]:
            continue  # already placed
        if n_cell not in candidate_cell_set:
            continue  # color not available
        theo_N = _count_theoretical_connections(block_grid, nr, nc, layer)
        if theo_N < theo_C:
            continue  # gate: N must have >= theoretical connections as C
        c_N = _count_same_layer_placed_connections(block_grid, placed_grid, nr, nc, layer)
        loss_A = np.exp(c_C + 1) + np.exp(c_N + 2)
        loss_B = np.exp(c_N + 1) + np.exp(c_C + 2)
        if loss_B < loss_A:
            if best_loss_B is None or loss_B < best_loss_B:
                best_loss_B = loss_B
                best_cell = n_cell
                best_type = candidate_type_map[n_cell]

    return best_cell, best_type


def _next_target_for_color(
    color_value: int,
    block_grid: np.ndarray,
    placed_grid: np.ndarray,
    unassigned_types: list[ObjectType] | None = None,
    available_blocks: list[ObjectType] | None = None,
) -> tuple[
    tuple[int, int, int] | None,
    ObjectType | None,
    tuple[int, int, int] | None,
]:
    """Find the next target cell for a single color, checking for mistakes.

    Parameters
    ----------
    color_value : int
        The Color.value (or ObjectType.value) to process.
    block_grid, placed_grid : np.ndarray
        Target and current-placement grids.
    unassigned_types, available_blocks :
        Same semantics as in `next_target_block`.

    Returns
    -------
    (target_cell, block_type, mistake_cell)
        *mistake_cell* is the first cell where this color is placed but does
        not belong (``None`` if no mistakes).  When a mistake exists,
        *target_cell* and *block_type* are ``None`` — the mistake must be
        resolved first.  Otherwise *target_cell* / *block_type* give the
        next cell needing this color (both ``None`` when the color is done).
    """
    obj_type = ObjectType(color_value)

    # --- Mistake check: this color is placed where it should not be --------
    placed_here = np.argwhere(placed_grid == color_value)
    for cell in placed_here:
        r, c, l = int(cell[0]), int(cell[1]), int(cell[2])
        if block_grid[r, c, l] != color_value:
            return None, None, (r, c, l)

    # --- Unfilled cells that need this color --------------------------------
    needs_color = np.argwhere(
        (block_grid == color_value) & (placed_grid != color_value)
    )
    if needs_color.size == 0:
        return None, None, None  # color complete

    # Filter to placeable cells.
    placeable_mask = np.array([
        _is_placeable(block_grid, placed_grid, c[0], c[1], c[2])
        for c in needs_color
    ])
    placeable = needs_color[placeable_mask]
    if placeable.size == 0:
        return None, None, None  # remaining cells blocked by unsupported layers

    # Lowest placeable layer, then center-outward ordering.
    target_layer = int(placeable[:, 2].min())
    layer_cells = placeable[placeable[:, 2] == target_layer]
    layer_cells = _sort_cells_center_outward(layer_cells, block_grid, target_layer)

    def _cell_tuple(cell):
        return (int(cell[0]), int(cell[1]), int(cell[2]))

    # Priority 1: unassigned blocks matching this color.
    if unassigned_types is not None:
        if obj_type in unassigned_types:
            return _cell_tuple(layer_cells[0]), obj_type, None

    # Priority 2: available blocks filter.
    if available_blocks is not None:
        if obj_type in available_blocks:
            return _cell_tuple(layer_cells[0]), obj_type, None
        return None, None, None  # this color not available

    # No filters — return closest to center.
    return _cell_tuple(layer_cells[0]), obj_type, None


def next_target_block(
    block_grid: np.ndarray,
    placed_grid: np.ndarray,
    unassigned_types: list[ObjectType] | None = None,
    available_blocks: list[ObjectType] | None = None,
    skip_cells: set | None = None,
) -> tuple[tuple[int, int, int] | None, ObjectType | None, bool]:
    """Return (grid_index, block_type, is_mistake) for the next block action.

    Iterates over each color independently: first checks that every placed
    block of that color is in a correct position, then selects the next
    cell to fill using a center-outward, bottom-up strategy.

    Parameters
    ----------
    block_grid : np.ndarray
        Target grid of shape (rows, cols, layers) with ObjectType.value ints.
    placed_grid : np.ndarray
        Current placement grid, same shape as *block_grid*.
    unassigned_types : list[ObjectType] or None
        Block types of unassigned (misplaced) blocks on/near the structure.
        Prioritized over *available_blocks*.
    available_blocks : list[ObjectType] or None
        Block types available in the pile. ``None`` means all types available.
    skip_cells : set or None
        Set of (row, col, layer) tuples to exclude from target selection.
        Cells in this set are treated as unavailable (e.g. unreachable).
        None means no cells are skipped.

    Returns
    -------
    tuple[tuple[int, int, int] | None, ObjectType | None, bool]
        ``((row, col, layer), ObjectType, is_mistake)``.
        When *is_mistake* is ``True``, the cell contains a wrong-color block
        that must be removed before building can continue.  *ObjectType* is
        the type of the block currently occupying the cell (what needs to be
        picked up).
        When *is_mistake* is ``False``, this is a normal placement target.
        Returns ``(None, None, False)`` when done or no placeable cell
        matches available blocks.
    """
    # Gather per-color ObjectType values present in the target grid.
    unique_values = set(np.unique(block_grid)) | set(np.unique(placed_grid))
    unique_values.discard(0)
    color_values = sorted(unique_values)

    # --- Pass 1: detect mistakes across all colors --------------------------
    for cv in color_values:
        _, _, mistake = _next_target_for_color(
            cv, block_grid, placed_grid, unassigned_types, available_blocks,
        )
        if mistake is not None:
            if skip_cells and mistake in skip_cells:
                continue  # unreachable — nod gesture handles it
            # Return the mistake cell — block_type is what is *currently*
            # placed there (the wrong block that needs to be removed).
            wrong_val = placed_grid[mistake]
            wrong_type = ObjectType(wrong_val) if wrong_val != 0 else None
            return mistake, wrong_type, True

    # --- Determine global target layer (lowest unfilled, placeable) ---------
    # This enforces strict bottom-up ordering across all colors: we never
    # place on a higher layer while a lower layer still has unfilled cells.
    all_unfilled = np.argwhere(
        (block_grid != 0) & (block_grid != placed_grid)
    )
    if all_unfilled.size == 0:
        return None, None, False

    # Filter out skipped cells (e.g. unreachable) from unfilled candidates
    if skip_cells:
        keep_mask = np.array([
            (int(c[0]), int(c[1]), int(c[2])) not in skip_cells
            for c in all_unfilled
        ])
        all_unfilled = all_unfilled[keep_mask]
        if all_unfilled.size == 0:
            return None, None, False

    placeable_mask = np.array([
        _is_placeable(block_grid, placed_grid, c[0], c[1], c[2])
        for c in all_unfilled
    ])
    all_placeable = all_unfilled[placeable_mask]
    if all_placeable.size == 0:
        return None, None, False

    global_target_layer = int(all_placeable[:, 2].min())

    # --- Pass 2: collect candidate targets per color on target layer -------
    candidates = []
    for cv in color_values:
        cell, obj_type, _ = _next_target_for_color(
            cv, block_grid, placed_grid, unassigned_types, available_blocks,
        )
        if cell is not None and cell[2] == global_target_layer:
            if skip_cells and cell in skip_cells:
                continue
            candidates.append((cell, obj_type))

    if not candidates:
        return None, None, False

    # Sort candidates: closest to layer center → highest support score.
    num_layers = block_grid.shape[2]

    def _sort_key(item):
        cell, _ = item
        r, c, layer = cell
        center_row, center_col = _layer_center(block_grid, layer)
        dist = np.sqrt((r - center_row) ** 2 + (c - center_col) ** 2)
        support = sum(
            block_grid[r, c, l] != 0
            for l in range(layer + 1, num_layers)
        )
        return (dist, -support)

    def _is_already_placed(cell):
        """Guard: True if cell is already correctly placed (should not be targeted)."""
        r, c, l = cell
        bg = block_grid[r, c, l]
        pg = placed_grid[r, c, l]
        if bg != 0 and bg == pg:
            _log.warning(
                f"next_target_block: skipping already-placed cell {cell} "
                f"(block_grid={bg}, placed_grid={pg}, "
                f"bg_dtype={block_grid.dtype}, pg_dtype={placed_grid.dtype})"
            )
            return True
        return False

    # Priority 1: unassigned types — pick the best candidate that matches.
    if unassigned_types is not None:
        unassigned_set = set(unassigned_types)
        unassigned_candidates = [
            (cell, ot) for cell, ot in candidates if ot in unassigned_set
        ]
        if unassigned_candidates:
            unassigned_candidates.sort(key=_sort_key)
            for cell, ot in unassigned_candidates:
                if not _is_already_placed(cell):
                    cell, ot = _smooth_connection_order(
                        block_grid, placed_grid, cell, ot, unassigned_candidates
                    )
                    return cell, ot, False
            # All unassigned candidates already placed — fall through.

    # Sort all candidates and return the best non-placed one.
    candidates.sort(key=_sort_key)
    for cell, ot in candidates:
        if not _is_already_placed(cell):
            cell, ot = _smooth_connection_order(
                block_grid, placed_grid, cell, ot, candidates
            )
            return cell, ot, False
    return None, None, False

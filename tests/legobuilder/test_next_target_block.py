"""Tests for next_target_block smart placement ordering."""

import numpy as np
import pytest

from legobuilder.brain.next_target_block import (
    next_target_block,
    _count_same_layer_placed_connections,
    _count_theoretical_connections,
)
from legobuilder.schemas import ObjectType

Y = ObjectType.YELLOW_BLOCK.value
B = ObjectType.BLUE_BLOCK.value
G = ObjectType.GREEN_BLOCK.value
R = ObjectType.RED_BLOCK.value


def _grid(*layers):
    """Build a (rows, cols, layers) grid from 2-D layer arrays."""
    return np.stack(layers, axis=-1)


# ---------------------------------------------------------------------------
# Original tests (updated for new return type)
# ---------------------------------------------------------------------------

# 1. Empty delta — all placed
def test_empty_delta():
    grid = _grid([[Y, B], [G, R]])
    idx, btype, is_mistake = next_target_block(grid, grid.copy())
    assert idx is None
    assert btype is None


# 2. Single cell
def test_single_cell():
    block = _grid([[Y]])
    placed = _grid([[0]])
    idx, btype, is_mistake = next_target_block(block, placed)
    assert idx == (0, 0, 0)
    assert btype is ObjectType.YELLOW_BLOCK


# 3. Bottom layer first
def test_bottom_layer_first():
    block = _grid([[Y]], [[B]])
    placed = _grid([[0]], [[0]])
    idx, btype, is_mistake = next_target_block(block, placed)
    assert idx[2] == 0  # layer 0
    assert btype is ObjectType.YELLOW_BLOCK


# 4. Center outward — closest to center returned first
def test_center_outward():
    block = _grid([[Y, Y, Y]])
    placed = _grid([[0, 0, 0]])
    idx, btype, is_mistake = next_target_block(block, placed)
    # Center col = 1.0 → (0,1,0) is closest
    assert idx == (0, 1, 0)
    assert btype is ObjectType.YELLOW_BLOCK


# 5. Unassigned match
def test_in_hand_match():
    block = _grid([[Y, B]])
    placed = _grid([[0, 0]])
    idx, btype, is_mistake = next_target_block(block, placed, unassigned_types=[ObjectType.BLUE_BLOCK])
    assert idx == (0, 1, 0)
    assert btype is ObjectType.BLUE_BLOCK


# 6. Unassigned match, pick closest to center
def test_in_hand_match_closest_to_center():
    block = _grid([[B, Y, B]])
    placed = _grid([[0, 0, 0]])
    idx, btype, is_mistake = next_target_block(block, placed, unassigned_types=[ObjectType.BLUE_BLOCK])
    assert btype is ObjectType.BLUE_BLOCK
    assert block[idx] == B


# 7. Unassigned mismatch — falls through, returns closest unfilled
def test_in_hand_mismatch():
    block = _grid([[Y, Y]])
    placed = _grid([[0, 0]])
    idx, btype, is_mistake = next_target_block(block, placed, unassigned_types=[ObjectType.BLUE_BLOCK])
    assert btype is ObjectType.YELLOW_BLOCK
    assert idx is not None


# 8. In-hand None
def test_in_hand_none():
    block = _grid([[Y, B]])
    placed = _grid([[0, 0]])
    idx, btype, is_mistake = next_target_block(block, placed, None)
    assert idx is not None
    assert btype is not None


# 9. Layer fully placed — returns next layer
def test_layer_fully_placed():
    block = _grid([[Y]], [[B]])
    placed = _grid([[Y]], [[0]])
    idx, btype, is_mistake = next_target_block(block, placed)
    assert idx == (0, 0, 1)
    assert btype is ObjectType.BLUE_BLOCK


# 10. Asymmetric grid (3x5)
def test_asymmetric_grid():
    layer = np.array([
        [Y, Y, Y, Y, Y],
        [Y, Y, Y, Y, Y],
        [Y, Y, Y, Y, Y],
    ])
    block = _grid(layer)
    placed = np.zeros_like(block)
    idx, btype, is_mistake = next_target_block(block, placed)
    # Center at (1.0, 2.0) → (1,2,0)
    assert idx == (1, 2, 0)
    assert btype is ObjectType.YELLOW_BLOCK


# 11. Corner design — center at corner
def test_corner_design():
    layer = np.array([
        [Y, Y, 0, 0],
        [Y, 0, 0, 0],
        [0, 0, 0, 0],
    ])
    block = _grid(layer)
    placed = np.zeros_like(block)
    idx, btype, is_mistake = next_target_block(block, placed)
    assert idx == (0, 0, 0)
    assert btype is ObjectType.YELLOW_BLOCK


# 12. All same color, unassigned matches
def test_all_same_color_in_hand_matches():
    block = _grid([[Y, Y, Y]])
    placed = _grid([[0, 0, 0]])
    idx, btype, is_mistake = next_target_block(block, placed, unassigned_types=[ObjectType.YELLOW_BLOCK])
    assert btype is ObjectType.YELLOW_BLOCK
    assert idx == (0, 1, 0)


# 13. Multiple colors on layer — unassigned only matches its own color
def test_multiple_colors_in_hand_filters():
    block = _grid([[R, Y, G, B, R]])
    placed = _grid([[0, 0, 0, 0, 0]])
    idx, btype, is_mistake = next_target_block(block, placed, unassigned_types=[ObjectType.RED_BLOCK])
    assert btype is ObjectType.RED_BLOCK
    assert block[idx] == R
    assert idx[0] == 0 and idx[2] == 0
    assert idx[1] in (0, 4)


# ---------------------------------------------------------------------------
# New tests: available_blocks
# ---------------------------------------------------------------------------

# 14. available_blocks filters out type
def test_available_blocks_filters_out():
    block = _grid([[Y]])
    placed = _grid([[0]])
    idx, btype, is_mistake = next_target_block(block, placed, available_blocks=[ObjectType.BLUE_BLOCK])
    assert idx is None
    assert btype is None


# 15. available_blocks allows type
def test_available_blocks_allows():
    block = _grid([[Y]])
    placed = _grid([[0]])
    idx, btype, is_mistake = next_target_block(block, placed, available_blocks=[ObjectType.YELLOW_BLOCK])
    assert idx == (0, 0, 0)
    assert btype is ObjectType.YELLOW_BLOCK


# 16. available_blocks skips to available type
def test_available_blocks_skips_to_available():
    block = _grid([[Y, B, Y]])
    placed = _grid([[0, 0, 0]])
    # Center col=1.0. Y cells at col 0,2 (dist 1). B at col 1 (dist 0).
    # Only BLUE available → returns center B cell.
    idx, btype, is_mistake = next_target_block(block, placed, available_blocks=[ObjectType.BLUE_BLOCK])
    assert idx == (0, 1, 0)
    assert btype is ObjectType.BLUE_BLOCK


# 17. available_blocks None means all available
def test_available_blocks_none():
    block = _grid([[Y]])
    placed = _grid([[0]])
    idx, btype, is_mistake = next_target_block(block, placed, available_blocks=None)
    assert idx == (0, 0, 0)
    assert btype is ObjectType.YELLOW_BLOCK


# ---------------------------------------------------------------------------
# New tests: support constraint
# ---------------------------------------------------------------------------

# 18. Unsupported cell skipped — layer 1 not placeable without layer 0
def test_support_unsupported_skipped():
    block = _grid([[Y]], [[B]])
    placed = _grid([[0]], [[0]])
    idx, btype, is_mistake = next_target_block(block, placed)
    # Layer 1 not placeable (layer 0 empty), so returns layer 0.
    assert idx == (0, 0, 0)
    assert idx[2] == 0


# 19. Supported cell on layer 1
def test_support_layer1_supported():
    block = _grid([[Y]], [[B]])
    placed = _grid([[Y]], [[0]])
    idx, btype, is_mistake = next_target_block(block, placed)
    assert idx == (0, 0, 1)
    assert btype is ObjectType.BLUE_BLOCK


# 20. Support allows skipping layer — partially placed layer 0
def test_support_allows_higher_layer():
    # Layer 0: only (0,1) needs Y. Layer 1: (0,0) needs B.
    # (0,0) on layer 0 is already placed → (0,0,1) is supported.
    # available_blocks=[BLUE] → can't place Y on layer 0, but can place B on layer 1.
    block = _grid([[Y, Y]], [[B, 0]])
    placed = _grid([[Y, 0]], [[0, 0]])
    idx, btype, is_mistake = next_target_block(
        block, placed, available_blocks=[ObjectType.BLUE_BLOCK]
    )
    # Layer 0 has (0,1) needing Y but BLUE not available there.
    # Layer 1 has (0,0) needing B and it's supported. Should return it.
    # Both layers have placeable cells; lowest layer (0) checked first.
    # But available_blocks=[BLUE] filters out Y on layer 0 → (None, None)?
    # Actually per algorithm: target_layer = min placeable layer = 0,
    # filter to available → no match on layer 0 → return (None, None).
    # This is correct: we don't skip layers, we pick lowest first.
    assert idx is None
    assert btype is None


# 21. unassigned prioritized over available_blocks
def test_in_hand_prioritized_over_available():
    block = _grid([[Y, B]])
    placed = _grid([[0, 0]])
    idx, btype, is_mistake = next_target_block(
        block, placed,
        unassigned_types=[ObjectType.BLUE_BLOCK],
        available_blocks=[ObjectType.YELLOW_BLOCK],
    )
    assert idx == (0, 1, 0)
    assert btype is ObjectType.BLUE_BLOCK


# 22. unassigned mismatch falls through to available
def test_in_hand_mismatch_falls_to_available():
    block = _grid([[Y, B]])
    placed = _grid([[0, 0]])
    idx, btype, is_mistake = next_target_block(
        block, placed,
        unassigned_types=[ObjectType.RED_BLOCK],
        available_blocks=[ObjectType.YELLOW_BLOCK],
    )
    # RED not on layer → falls through. YELLOW available, Y at col 0.
    assert btype is ObjectType.YELLOW_BLOCK
    assert idx == (0, 0, 0)


# 23. Layer 1 block with empty layer 0 — now placeable (sits on build plate)
def test_no_layer0_block_is_placeable():
    block = _grid([[0]], [[B]])
    placed = _grid([[0]], [[0]])
    idx, btype, is_mistake = next_target_block(block, placed)
    assert idx == (0, 0, 1)
    assert btype is ObjectType.BLUE_BLOCK


# ---------------------------------------------------------------------------
# New tests: multiple unassigned types, support-aware tiebreaking, layer order
# ---------------------------------------------------------------------------

# 24. Multiple unassigned types — matches closest cell of either type
def test_multiple_unassigned_types():
    block = _grid([[R, Y, B]])
    placed = _grid([[0, 0, 0]])
    idx, btype, is_mistake = next_target_block(
        block, placed,
        unassigned_types=[ObjectType.RED_BLOCK, ObjectType.BLUE_BLOCK],
    )
    # Center col=1.0. R at col0 (dist 1), B at col2 (dist 1). Either valid.
    assert btype in (ObjectType.RED_BLOCK, ObjectType.BLUE_BLOCK)
    assert block[idx] == btype.value


# 25. Support-aware tiebreaking — cell with more dependents above placed first
def test_support_aware_tiebreak():
    block = _grid([[Y, Y]], [[B, 0]])
    placed = np.zeros_like(_grid([[Y, Y]], [[B, 0]]))
    # Center = (0, 0.5). Cells (0,0,0) and (0,1,0) equidistant (0.5 each).
    # (0,0) has B above in block_grid layer1 → support_count=1.
    # (0,1) has 0 above → support_count=0.
    idx, btype, is_mistake = next_target_block(block, placed)
    assert idx == (0, 0, 0)


# 26. Layer complete before upper — lower layer unfilled cells first
def test_layer_complete_before_upper_unassigned():
    block = _grid([[Y, B]], [[R, 0]])
    placed = _grid([[Y, 0]], [[0, 0]])
    # Layer 0 has unfilled cell (0,1). unassigned_types=[RED] matches layer 1
    # cell, but layer 0 is target_layer. Must return (0,1,0) = BLUE.
    idx, btype, is_mistake = next_target_block(
        block, placed,
        unassigned_types=[ObjectType.RED_BLOCK],
    )
    assert idx == (0, 1, 0)
    assert btype is ObjectType.BLUE_BLOCK


# ---------------------------------------------------------------------------
# New tests: structures starting above layer 0, per-layer center
# ---------------------------------------------------------------------------

# 27. Structure starting at layer 1 (no layer 0 blocks) — layer 1 is placeable
def test_structure_starting_at_layer1():
    grid = np.zeros((3, 3, 3), dtype=int)
    grid[1, 1, 1] = Y
    grid[0, 1, 1] = G
    grid[2, 1, 1] = G
    grid[1, 0, 1] = G
    grid[1, 2, 1] = G
    placed = np.zeros_like(grid)
    idx, btype, is_mistake = next_target_block(grid, placed)
    assert idx is not None
    assert idx[2] == 1  # must return layer 1, not None


# 28. Flower-like structure — layer 1 complete before layer 2
def test_flower_layer1_before_layer2():
    grid = np.zeros((5, 5, 5), dtype=int)
    grid[2, 2, 1] = G
    grid[1, 2, 1] = G
    grid[3, 2, 1] = G
    grid[2, 1, 1] = G
    grid[2, 3, 1] = G
    grid[2, 2, 2] = G
    placed = np.zeros_like(grid)
    placed[2, 2, 1] = G  # center of layer 1 placed
    idx, btype, is_mistake = next_target_block(grid, placed)
    # Must return a layer 1 cell (4 remaining), not layer 2
    assert idx[2] == 1


# 29. Multi-layer air gap — blocks at layer 2 with nothing at 0 or 1
def test_multi_layer_air_gap():
    grid = np.zeros((3, 3, 4), dtype=int)
    grid[1, 1, 2] = Y
    placed = np.zeros_like(grid)
    idx, btype, is_mistake = next_target_block(grid, placed)
    assert idx == (1, 1, 2)
    assert btype is ObjectType.YELLOW_BLOCK


# 30. Per-layer center — asymmetric layers use per-layer center
def test_per_layer_center():
    grid = np.zeros((1, 5, 2), dtype=int)
    grid[0, 0, 0] = Y
    grid[0, 1, 0] = Y
    grid[0, 2, 0] = Y
    grid[0, 3, 0] = Y
    grid[0, 4, 0] = Y
    grid[0, 0, 1] = B
    placed = np.zeros_like(grid)
    idx, btype, is_mistake = next_target_block(grid, placed)
    # Per-layer center for layer 0 = (0, 2.0) → closest is (0,2,0)
    assert idx == (0, 2, 0)


# ---------------------------------------------------------------------------
# Wrong-color (mistake) detection tests
# ---------------------------------------------------------------------------

# 31. Wrong color placed — returns mistake cell with wrong block's type
def test_wrong_color_detected_as_mistake():
    """A BLUE block placed where YELLOW belongs → returns that cell with BLUE (to remove)."""
    block = _grid([[Y, B]])
    placed = _grid([[B, 0]])  # BLUE placed at (0,0) but YELLOW expected
    idx, btype, is_mistake = next_target_block(block, placed)
    assert idx == (0, 0, 0)
    assert is_mistake is True
    # btype is the wrong block currently there (what to pick up)
    assert btype is ObjectType.BLUE_BLOCK


# 32. Wrong color on one cell, correct on another — mistake takes priority
def test_wrong_color_prioritized_over_unfilled():
    """Mistake cell returned even when other cells are simply unfilled."""
    block = _grid([[Y, B, G]])
    placed = _grid([[R, B, 0]])  # (0,0) has RED but needs YELLOW; (0,2) unfilled
    idx, btype, is_mistake = next_target_block(block, placed)
    # Mistake at (0,0): RED placed, YELLOW expected
    assert idx == (0, 0, 0)
    assert is_mistake is True
    assert btype is ObjectType.RED_BLOCK


# 33. Two colors swapped — first mistake found is returned
def test_swapped_colors_detected():
    """YELLOW and BLUE swapped: both cells are mistakes."""
    block = _grid([[Y, B]])
    placed = _grid([[B, Y]])  # swapped
    idx, btype, is_mistake = next_target_block(block, placed)
    # Both (0,0) and (0,1) are mistakes. Either is valid.
    assert idx in ((0, 0, 0), (0, 1, 0))
    assert is_mistake is True
    # btype is whatever is currently at that cell (the wrong block)
    assert btype is ObjectType(placed[idx])


# 34. Wrong color placed in empty target cell — block where nothing belongs
def test_wrong_color_in_empty_target_cell():
    """A block placed where the target grid is empty (0) → mistake."""
    block = _grid([[Y, 0]])
    placed = _grid([[Y, B]])  # BLUE placed at (0,1) but target is empty
    idx, btype, is_mistake = next_target_block(block, placed)
    assert idx == (0, 1, 0)
    assert is_mistake is True
    # btype is the wrong block there (BLUE)
    assert btype is ObjectType.BLUE_BLOCK


# 35. Correct colors placed — no mistake, returns next unfilled
def test_correct_colors_no_mistake():
    """All placed blocks match → normal unfilled-cell logic."""
    block = _grid([[Y, B, G]])
    placed = _grid([[Y, 0, 0]])
    idx, btype, is_mistake = next_target_block(block, placed)
    # No mistakes. Center col = 1.0. B at col 1 (dist 0), G at col 2 (dist 1).
    assert idx == (0, 1, 0)
    assert btype is ObjectType.BLUE_BLOCK
    assert is_mistake is False


# 36. Wrong color on upper layer — mistake detected even above layer 0
def test_wrong_color_on_upper_layer():
    """Mistake on layer 1: RED placed where BLUE expected."""
    block = _grid([[Y]], [[B]])
    placed = _grid([[Y]], [[R]])  # layer 1 has RED instead of BLUE
    idx, btype, is_mistake = next_target_block(block, placed)
    assert idx == (0, 0, 1)
    assert is_mistake is True
    assert btype is ObjectType.RED_BLOCK


# 37. Wrong color with unassigned_types — mistake still takes priority
def test_wrong_color_overrides_unassigned():
    """Mistakes are checked before unassigned/available logic."""
    block = _grid([[Y, B]])
    placed = _grid([[G, 0]])  # GREEN at (0,0) but YELLOW expected
    idx, btype, is_mistake = next_target_block(
        block, placed, unassigned_types=[ObjectType.BLUE_BLOCK]
    )
    # Mistake at (0,0) takes priority over the unassigned BLUE block
    assert idx == (0, 0, 0)
    assert is_mistake is True
    assert btype is ObjectType.GREEN_BLOCK


# 38. All correct, multiple colors — returns proper center-outward target
def test_multicolor_all_correct_center_outward():
    """No mistakes across multiple colors → normal ordering."""
    block = _grid([[R, Y, B, Y, G]])
    placed = _grid([[0, 0, 0, 0, 0]])
    idx, btype, is_mistake = next_target_block(block, placed)
    # Center col = 2.0. B at col 2 (dist 0) is closest.
    assert idx == (0, 2, 0)
    assert btype is ObjectType.BLUE_BLOCK
    assert is_mistake is False


# ---------------------------------------------------------------------------
# Connection-smoothing tests
# ---------------------------------------------------------------------------
# Note: the swap only fires for different-color adjacent candidates. For
# same-color rows, _next_target_for_color returns ONE candidate per color so
# same-color neighbors are never both in `candidates`; center-outward already
# picks the central block first in those cases.

# 39. Same color placed in multiple wrong spots
def test_multiple_wrong_placements_same_color():
    """Two RED blocks placed where YELLOW and BLUE belong."""
    block = _grid([[Y, B, G]])
    placed = _grid([[R, R, 0]])  # RED at (0,0) and (0,1) — both wrong
    idx, btype, is_mistake = next_target_block(block, placed)
    # RED doesn't belong at (0,0) or (0,1). Mistake detected.
    assert idx in ((0, 0, 0), (0, 1, 0))
    assert is_mistake is True
    # btype is RED (the wrong block currently placed there)
    assert btype is ObjectType.RED_BLOCK


# ---------------------------------------------------------------------------
# Connection-smoothing tests (tests 40-45)
# ---------------------------------------------------------------------------

# 40. Unit — theoretical connection count
def test_count_theoretical_connections():
    block = _grid([[G, Y, B, G]])
    # col 1 (Y): neighbors G at col0 and B at col2 → theo=2
    assert _count_theoretical_connections(block, 0, 1, 0) == 2
    # col 0 (G): only Y at col1 → theo=1
    assert _count_theoretical_connections(block, 0, 0, 0) == 1
    # col 3 (G): only B at col2 → theo=1
    assert _count_theoretical_connections(block, 0, 3, 0) == 1


# 41. Unit — placed connection count
def test_count_same_layer_placed_connections():
    block = _grid([[G, Y, B, G]])
    placed = _grid([[0, 0, 0, G]])  # only rightmost G placed
    # col 2 (B): right neighbor G is placed → c=1
    assert _count_same_layer_placed_connections(block, placed, 0, 2, 0) == 1
    # col 1 (Y): no placed neighbors → c=0
    assert _count_same_layer_placed_connections(block, placed, 0, 1, 0) == 0


# 42. Swap triggered — B has 1 placed connection, Y has 0
def test_smooth_swap_to_higher_placed_connections():
    """B (col2) has 1 placed connection (G at col3), Y (col1) has 0.
    Without smoothing: Y wins the col1/col2 tie (lower color value, stable sort).
    With smoothing: c_N(B)=1 > c_C(Y)=0 → swap to B.
    """
    block = _grid([[G, Y, B, G]])
    placed = _grid([[0, 0, 0, G]])  # right G placed
    idx, btype, is_mistake = next_target_block(block, placed)
    assert btype is ObjectType.BLUE_BLOCK
    assert idx == (0, 2, 0)


# 43. No swap when no placed connections exist
def test_smooth_no_swap_nothing_placed():
    """All c=0 everywhere → c_N == c_C, strict inequality not met, no swap."""
    block = _grid([[G, Y, B, G]])
    placed = _grid([[0, 0, 0, 0]])
    # Y (col1) wins the col1/col2 distance tie by color-value order.
    idx, btype, is_mistake = next_target_block(block, placed)
    assert btype is ObjectType.YELLOW_BLOCK
    assert idx == (0, 1, 0)


# 44. No swap when candidate already has more placed connections than neighbor
def test_smooth_no_swap_candidate_already_higher():
    """Y (col1) has c=1 (left G placed), B (col2) has c=0 → c_N < c_C, no swap."""
    block = _grid([[G, Y, B, G]])
    placed = _grid([[G, 0, 0, 0]])  # left G placed
    idx, btype, is_mistake = next_target_block(block, placed)
    assert btype is ObjectType.YELLOW_BLOCK
    assert idx == (0, 1, 0)


# 45. Full sequence — user's 3-block example in multi-color form
def test_smooth_full_sequence_balanced_connections():
    """Grid [G, Y, B, G] with right G already placed.

    Without smoothing the sequence would be Y(c=0), B(c=2), G(c=1) — a spike
    at B.  With smoothing the sequence is B(c=1), Y(c=1), G(c=1) — balanced.
    """
    block = _grid([[G, Y, B, G]])
    placed = _grid([[0, 0, 0, G]])

    # Step 1: smoothing swaps from Y to B (B has more placed connections)
    idx1, btype1, _ = next_target_block(block, placed)
    assert btype1 is ObjectType.BLUE_BLOCK
    assert idx1 == (0, 2, 0)

    # Step 2: after placing B, Y now has 1 placed neighbor (B at col2)
    placed2 = placed.copy()
    placed2[0, 2, 0] = B
    idx2, btype2, _ = next_target_block(block, placed2)
    assert btype2 is ObjectType.YELLOW_BLOCK
    assert idx2 == (0, 1, 0)

    # Step 3: after placing Y, G(col0) has 1 placed neighbor (Y at col1)
    placed3 = placed2.copy()
    placed3[0, 1, 0] = Y
    idx3, btype3, _ = next_target_block(block, placed3)
    assert btype3 is ObjectType.GREEN_BLOCK
    assert idx3 == (0, 0, 0)

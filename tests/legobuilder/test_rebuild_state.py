"""Tests for stateless rebuild_placed_grid."""

import logging

import numpy as np
import pytest

from legobuilder.brain.block_structure import BlockStructure
from legobuilder.brain.rebuild_state import rebuild_placed_grid
from legobuilder.config import BLOCK_SIZE
from legobuilder.schemas import Object, ObjectType

BS = BLOCK_SIZE

# Shortcuts for ObjectType values.
Y = ObjectType.YELLOW_BLOCK.value
B = ObjectType.BLUE_BLOCK.value
G = ObjectType.GREEN_BLOCK.value
R = ObjectType.RED_BLOCK.value


# ── Helpers ────────────────────────────────────────────────────────────

def _make_structure(block_grid: np.ndarray) -> BlockStructure:
    """Create a minimal BlockStructure with known origin and roll=0.

    Origin = [0, 0, BLOCK_SIZE] so that:
        cell (r, c, L) → world (c*BS + BS/2, r*BS + BS/2, L*BS + BS/2)
    """
    logger = logging.getLogger("test")
    rows, cols, layers = block_grid.shape
    s = BlockStructure(
        logger=logger,
        max_length=rows,
        max_width=cols,
        max_height=layers,
        roll=0.0,
        origin=np.array([0.0, 0.0, BS]),
    )
    s.block_grid = block_grid.copy()
    return s


def _cell_world_pos(r: int, c: int, layer: int) -> tuple[float, float, float]:
    """Expected world position for a cell with origin=[0,0,BS], roll=0."""
    return (
        c * BS + BS / 2,
        r * BS + BS / 2,
        layer * BS + BS / 2,
    )


def _make_det(
    obj_type: ObjectType,
    r: int,
    c: int,
    layer: int,
    *,
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Object:
    """Create a mock detection at the world position of cell (r, c, layer)."""
    wx, wy, wz = _cell_world_pos(r, c, layer)
    xyz = (wx + offset[0], wy + offset[1], wz + offset[2])
    return Object(
        t=0.0,
        frame_idx=0,
        obj_type=obj_type,
        center_uv=(0, 0),
        center_xyz=xyz,
        corner_xys=[(0.0, 0.0)] * 4,
    )


def _grid(*layers_2d) -> np.ndarray:
    """Stack 2-D arrays into (rows, cols, layers)."""
    return np.stack(layers_2d, axis=-1)


# ── Tests ──────────────────────────────────────────────────────────────

# 1. No detections → all-zero placed_grid, empty unassigned
def test_no_detections():
    bg = _grid([[Y, B], [G, R]])
    s = _make_structure(bg)
    pg, unassigned = rebuild_placed_grid([], s)
    assert pg.shape == bg.shape
    assert np.all(pg == 0)
    assert unassigned == []


# 2. Single detection matches single cell
def test_single_match():
    bg = _grid([[Y]])
    s = _make_structure(bg)
    det = _make_det(ObjectType.YELLOW_BLOCK, 0, 0, 0)
    pg, unassigned = rebuild_placed_grid([det], s)
    assert pg[0, 0, 0] == Y
    assert unassigned == []


# 3. Detection wrong color → stores detected type
def test_wrong_color_stores_detected():
    bg = _grid([[Y]])  # expects YELLOW
    s = _make_structure(bg)
    det = _make_det(ObjectType.RED_BLOCK, 0, 0, 0)  # detected RED
    pg, unassigned = rebuild_placed_grid([det], s)
    assert pg[0, 0, 0] == R  # detected type, not expected
    assert unassigned == []


# 4. Invisible block inference — detection on layer 1 infers layer 0
def test_invisible_inference_layer1():
    bg = _grid([[Y]], [[B]])  # layer 0: Y, layer 1: B
    s = _make_structure(bg)
    det = _make_det(ObjectType.BLUE_BLOCK, 0, 0, 1)  # only see layer 1
    pg, unassigned = rebuild_placed_grid([det], s)
    assert pg[0, 0, 1] == B  # detected
    assert pg[0, 0, 0] == Y  # inferred from block_grid


# 5. Multi-layer inference — detection on layer 2 infers layers 0 and 1
def test_multi_layer_inference():
    bg = _grid([[Y]], [[B]], [[G]])
    s = _make_structure(bg)
    det = _make_det(ObjectType.GREEN_BLOCK, 0, 0, 2)
    pg, unassigned = rebuild_placed_grid([det], s)
    assert pg[0, 0, 2] == G
    assert pg[0, 0, 1] == B  # inferred
    assert pg[0, 0, 0] == Y  # inferred


# 6. Inference skips empty cells in block_grid
def test_inference_skips_empty():
    # block_grid: layer 0 empty, layer 1 has block
    bg = np.zeros((1, 1, 2), dtype=int)
    bg[0, 0, 1] = B
    s = _make_structure(bg)
    det = _make_det(ObjectType.BLUE_BLOCK, 0, 0, 1)
    pg, unassigned = rebuild_placed_grid([det], s)
    assert pg[0, 0, 1] == B
    assert pg[0, 0, 0] == 0  # no block expected below, stays 0


# 7. Unassigned detection — not near any grid cell
def test_unassigned_detection():
    bg = _grid([[Y]])
    s = _make_structure(bg)
    # Detection far away from any cell
    far_det = Object(
        t=0.0,
        frame_idx=0,
        obj_type=ObjectType.RED_BLOCK,
        center_uv=(0, 0),
        center_xyz=(10.0, 10.0, 0.0),
        corner_xys=[(0.0, 0.0)] * 4,
    )
    pg, unassigned = rebuild_placed_grid([far_det], s)
    assert np.all(pg == 0)
    assert len(unassigned) == 1
    assert unassigned[0].obj_type is ObjectType.RED_BLOCK


# 8. Multiple detections, multiple cells — color-match prioritized
def test_multiple_matches_color_priority():
    bg = _grid([[Y, B]])
    s = _make_structure(bg)
    det_y = _make_det(ObjectType.YELLOW_BLOCK, 0, 0, 0)
    det_b = _make_det(ObjectType.BLUE_BLOCK, 0, 1, 0)
    pg, unassigned = rebuild_placed_grid([det_y, det_b], s)
    assert pg[0, 0, 0] == Y
    assert pg[0, 1, 0] == B
    assert unassigned == []


# 9. Ambiguous detection resolved by distance
def test_ambiguous_resolved_by_distance():
    # Two cells at (0,0,0) and (0,1,0), both YELLOW.
    # Detection closer to (0,0,0).
    bg = _grid([[Y, Y]])
    s = _make_structure(bg)
    # Slightly offset toward (0,0,0)
    det = _make_det(ObjectType.YELLOW_BLOCK, 0, 0, 0, offset=(0.001, 0.0, 0.0))
    pg, unassigned = rebuild_placed_grid([det], s)
    assert pg[0, 0, 0] == Y
    assert pg[0, 1, 0] == 0  # not matched
    assert unassigned == []


# 10. All cells matched — full structure detected
def test_all_cells_matched():
    bg = _grid([[Y, B], [G, R]])
    s = _make_structure(bg)
    dets = [
        _make_det(ObjectType.YELLOW_BLOCK, 0, 0, 0),
        _make_det(ObjectType.BLUE_BLOCK, 0, 1, 0),
        _make_det(ObjectType.GREEN_BLOCK, 1, 0, 0),
        _make_det(ObjectType.RED_BLOCK, 1, 1, 0),
    ]
    pg, unassigned = rebuild_placed_grid(dets, s)
    assert pg[0, 0, 0] == Y
    assert pg[0, 1, 0] == B
    assert pg[1, 0, 0] == G
    assert pg[1, 1, 0] == R
    assert unassigned == []


# 11. Partial structure — only some cells detected
def test_partial_structure():
    bg = _grid([[Y, B, G]])
    s = _make_structure(bg)
    # Only detect first and last cell
    dets = [
        _make_det(ObjectType.YELLOW_BLOCK, 0, 0, 0),
        _make_det(ObjectType.GREEN_BLOCK, 0, 2, 0),
    ]
    pg, unassigned = rebuild_placed_grid(dets, s)
    assert pg[0, 0, 0] == Y
    assert pg[0, 1, 0] == 0  # not detected
    assert pg[0, 2, 0] == G
    assert unassigned == []

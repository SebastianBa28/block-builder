"""Tests for StructureScanner FSM."""

import logging
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from legobuilder.brain.block_structure import BlockStructure
from legobuilder.brain.structure_scanner import StructureScanner, ScanResult
from legobuilder.config import BLOCK_SIZE
from legobuilder.schemas import Object, ObjectType

BS = BLOCK_SIZE

Y = ObjectType.YELLOW_BLOCK.value
B = ObjectType.BLUE_BLOCK.value
G = ObjectType.GREEN_BLOCK.value
R = ObjectType.RED_BLOCK.value


# ── Helpers ────────────────────────────────────────────────────────────

def _make_structure(block_grid: np.ndarray) -> BlockStructure:
    """Create a minimal BlockStructure with known origin and roll=0."""
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


class MockTimer:
    """Minimal timer mock that records the callback."""

    def __init__(self, period, callback):
        self.period = period
        self.callback = callback
        self.cancelled = False

    def cancel(self):
        self.cancelled = True


def _make_scanner(block_grid: np.ndarray) -> tuple[StructureScanner, dict]:
    """Create a StructureScanner with mock callbacks, return (scanner, mocks)."""
    structure = _make_structure(block_grid)
    logger = logging.getLogger("test_scanner")

    timers = []

    def mock_create_timer(period, callback):
        t = MockTimer(period, callback)
        timers.append(t)
        return t

    def mock_destroy_timer(timer):
        pass

    mocks = {
        'send_ik_request': MagicMock(return_value='ik_1'),
        'publish_trajectory': MagicMock(),
        'request_scan': MagicMock(),
        'create_timer': mock_create_timer,
        'destroy_timer': mock_destroy_timer,
        'get_t': MagicMock(return_value=0.0),
        'timers': timers,
    }

    scanner = StructureScanner(
        send_ik_request_fn=mocks['send_ik_request'],
        publish_trajectory_fn=mocks['publish_trajectory'],
        request_scan_fn=mocks['request_scan'],
        create_timer_fn=mocks['create_timer'],
        destroy_timer_fn=mocks['destroy_timer'],
        get_t_fn=mocks['get_t'],
        structure=structure,
        logger=logger,
    )

    return scanner, mocks


def _advance_to_detect(scanner, mocks):
    """Advance scanner from IDLE through IK + trajectory to DETECT state.

    Simulates the full IK→trajectory→detect flow so that
    on_worldmap_detections can be called.
    """
    # start_scan → MOVE_TO_SCAN (sends IK request)
    # Simulate IK response with dummy solutions
    ik_id = mocks['send_ik_request'].return_value
    q_solutions = [0.0] * 6  # 6-DOF joint solutions
    with patch('legobuilder.brain.ros_bridge.apply_ik_solutions'):
        scanner.on_ik_response(ik_id, True, q_solutions)

    # Now trajectory is published, simulate trajectory complete
    # For VERIFY_SCAN_SOURCE == 1, this goes straight to DETECT
    # For VERIFY_SCAN_SOURCE == 0, this goes to DWELL_SCAN first
    scanner.on_trajectory_complete()

    # If in DWELL_SCAN, fire the dwell-complete timer to get to DETECT
    if scanner._state == StructureScanner.DWELL_SCAN:
        scanner._on_dwell_complete()

    assert scanner._state == StructureScanner.DETECT


# ── Tests ──────────────────────────────────────────────────────────────

# 1. Scan completes with detections — correct ScanResult
def test_scan_with_detections():
    bg = _grid([[Y, B], [G, R]])
    scanner, mocks = _make_scanner(bg)

    result_holder = []
    scanner.start_scan(on_result=result_holder.append, skip_orbit=True)
    _advance_to_detect(scanner, mocks)

    # Feed detections for all 4 cells
    dets = [
        _make_det(ObjectType.YELLOW_BLOCK, 0, 0, 0),
        _make_det(ObjectType.BLUE_BLOCK, 0, 1, 0),
        _make_det(ObjectType.GREEN_BLOCK, 1, 0, 0),
        _make_det(ObjectType.RED_BLOCK, 1, 1, 0),
    ]
    scanner.on_worldmap_detections(dets)

    assert len(result_holder) == 1
    result = result_holder[0]
    assert isinstance(result, ScanResult)
    assert result.placed_grid[0, 0, 0] == Y
    assert result.placed_grid[0, 1, 0] == B
    assert result.placed_grid[1, 0, 0] == G
    assert result.placed_grid[1, 1, 0] == R
    # All cells filled, build complete
    assert result.target_cell is None
    assert result.block_type is None
    assert result.unassigned == []
    assert result.in_hand_block is None
    assert not scanner.is_active


# 2. Empty scan — zero grid, target is first cell
def test_empty_scan():
    bg = _grid([[Y]])
    scanner, mocks = _make_scanner(bg)

    result_holder = []
    scanner.start_scan(on_result=result_holder.append, skip_orbit=True)
    _advance_to_detect(scanner, mocks)

    scanner.on_worldmap_detections([])

    assert len(result_holder) == 1
    result = result_holder[0]
    assert np.all(result.placed_grid == 0)
    assert result.target_cell == (0, 0, 0)
    assert result.block_type == ObjectType.YELLOW_BLOCK
    assert not scanner.is_active


# 3. Build complete — all cells detected
def test_build_complete():
    bg = _grid([[Y]])
    scanner, mocks = _make_scanner(bg)

    result_holder = []
    scanner.start_scan(on_result=result_holder.append, skip_orbit=True)
    _advance_to_detect(scanner, mocks)

    dets = [_make_det(ObjectType.YELLOW_BLOCK, 0, 0, 0)]
    scanner.on_worldmap_detections(dets)

    result = result_holder[0]
    assert result.target_cell is None
    assert result.block_type is None


# 4. Unassigned detection → in_hand_block
def test_unassigned_sets_in_hand_block():
    bg = _grid([[Y]])
    scanner, mocks = _make_scanner(bg)

    result_holder = []
    scanner.start_scan(on_result=result_holder.append, skip_orbit=True)
    _advance_to_detect(scanner, mocks)

    # Detection at cell position + extra unassigned far away
    det_match = _make_det(ObjectType.YELLOW_BLOCK, 0, 0, 0)
    det_extra = Object(
        t=0.0, frame_idx=0,
        obj_type=ObjectType.RED_BLOCK,
        center_uv=(0, 0),
        center_xyz=(10.0, 10.0, 0.0),
        corner_xys=[(0.0, 0.0)] * 4,
    )
    scanner.on_worldmap_detections([det_match, det_extra])

    result = result_holder[0]
    assert len(result.unassigned) == 1
    assert result.unassigned[0].obj_type is ObjectType.RED_BLOCK
    # Build is complete (all cells filled), so block_type is None →
    # in_hand_block is None (no target to match unassigned against).
    assert result.in_hand_block is None


# 4b. Unassigned type does NOT match target → in_hand_block is None
def test_unassigned_mismatch_target():
    bg = _grid([[Y, B]])
    scanner, mocks = _make_scanner(bg)

    result_holder = []
    scanner.start_scan(on_result=result_holder.append, skip_orbit=True)
    _advance_to_detect(scanner, mocks)

    # Detect Y at (0,0), leave (0,1) unfilled. Extra RED unassigned.
    det_match = _make_det(ObjectType.YELLOW_BLOCK, 0, 0, 0)
    det_extra = Object(
        t=0.0, frame_idx=0,
        obj_type=ObjectType.RED_BLOCK,
        center_uv=(0, 0),
        center_xyz=(10.0, 10.0, 0.0),
        corner_xys=[(0.0, 0.0)] * 4,
    )
    scanner.on_worldmap_detections([det_match, det_extra])

    result = result_holder[0]
    assert result.target_cell == (0, 1, 0)
    assert result.block_type is ObjectType.BLUE_BLOCK
    # RED != BLUE → in_hand_block should be None
    assert result.in_hand_block is None


# 4c. Unassigned type DOES match target → in_hand_block set
def test_unassigned_match_target():
    bg = _grid([[Y, B]])
    scanner, mocks = _make_scanner(bg)

    result_holder = []
    scanner.start_scan(on_result=result_holder.append, skip_orbit=True)
    _advance_to_detect(scanner, mocks)

    # Detect Y at (0,0), leave (0,1) unfilled. Extra BLUE unassigned.
    det_match = _make_det(ObjectType.YELLOW_BLOCK, 0, 0, 0)
    det_extra = Object(
        t=0.0, frame_idx=0,
        obj_type=ObjectType.BLUE_BLOCK,
        center_uv=(0, 0),
        center_xyz=(10.0, 10.0, 0.0),
        corner_xys=[(0.0, 0.0)] * 4,
    )
    scanner.on_worldmap_detections([det_match, det_extra])

    result = result_holder[0]
    assert result.target_cell == (0, 1, 0)
    assert result.block_type is ObjectType.BLUE_BLOCK
    # BLUE == BLUE → in_hand_block should be set
    assert result.in_hand_block is ObjectType.BLUE_BLOCK


# 5. Reset while active clears state to IDLE
def test_reset_while_active():
    bg = _grid([[Y, B]])
    scanner, mocks = _make_scanner(bg)

    scanner.start_scan(on_result=lambda r: None, skip_orbit=True)
    assert scanner.is_active
    assert scanner._state == StructureScanner.MOVE_TO_SCAN

    scanner.reset()
    assert not scanner.is_active
    assert scanner._state == StructureScanner.IDLE


# 6. State transitions: IDLE → MOVE_TO_SCAN → DWELL/DETECT
def test_state_transitions():
    bg = _grid([[Y]])
    scanner, mocks = _make_scanner(bg)

    assert scanner._state == StructureScanner.IDLE

    scanner.start_scan(on_result=lambda r: None)
    assert scanner._state == StructureScanner.MOVE_TO_SCAN

    # Simulate IK response
    ik_id = mocks['send_ik_request'].return_value
    with patch('legobuilder.brain.ros_bridge.apply_ik_solutions'):
        scanner.on_ik_response(ik_id, True, [0.0] * 6)

    # Trajectory complete
    scanner.on_trajectory_complete()
    # Should be in either DWELL_SCAN or DETECT depending on VERIFY_SCAN_SOURCE
    assert scanner._state in (
        StructureScanner.DWELL_SCAN,
        StructureScanner.DETECT,
    )


# 7. IK failure finishes with empty result
def test_ik_failure_finishes_empty():
    bg = _grid([[Y, B]])
    scanner, mocks = _make_scanner(bg)

    result_holder = []
    scanner.start_scan(on_result=result_holder.append, skip_orbit=True)

    ik_id = mocks['send_ik_request'].return_value
    scanner.on_ik_response(ik_id, False, [])

    assert len(result_holder) == 1
    result = result_holder[0]
    assert np.all(result.placed_grid == 0)
    assert not scanner.is_active


# 8. Partial detection — some cells missing, returns correct target
def test_partial_detection_returns_target():
    bg = _grid([[Y, B, G]])
    scanner, mocks = _make_scanner(bg)

    result_holder = []
    scanner.start_scan(on_result=result_holder.append, skip_orbit=True)
    _advance_to_detect(scanner, mocks)

    # Only detect first cell
    dets = [_make_det(ObjectType.YELLOW_BLOCK, 0, 0, 0)]
    scanner.on_worldmap_detections(dets)

    result = result_holder[0]
    assert result.placed_grid[0, 0, 0] == Y
    assert result.placed_grid[0, 1, 0] == 0
    assert result.placed_grid[0, 2, 0] == 0
    assert result.target_cell is not None
    assert result.block_type is not None


# 9. is_active property
def test_is_active_property():
    bg = _grid([[Y]])
    scanner, mocks = _make_scanner(bg)

    assert not scanner.is_active

    scanner.start_scan(on_result=lambda r: None)
    assert scanner.is_active

    scanner.reset()
    assert not scanner.is_active


# 10. Detect timeout with skip_orbit finishes empty
def test_detect_timeout_skip_orbit():
    bg = _grid([[Y]])
    scanner, mocks = _make_scanner(bg)

    result_holder = []
    scanner.start_scan(on_result=result_holder.append, skip_orbit=True)
    _advance_to_detect(scanner, mocks)

    assert scanner._state == StructureScanner.DETECT

    # Simulate detect timeout
    scanner._on_detect_timeout()

    assert len(result_holder) == 1
    result = result_holder[0]
    assert np.all(result.placed_grid == 0)
    assert not scanner.is_active


# 11. Stale IK response is ignored
def test_stale_ik_response_ignored():
    bg = _grid([[Y]])
    scanner, mocks = _make_scanner(bg)

    scanner.start_scan(on_result=lambda r: None)
    assert scanner._state == StructureScanner.MOVE_TO_SCAN

    # Send stale IK response with wrong ID
    scanner.on_ik_response('wrong_id', True, [0.0] * 6)
    # State should not change
    assert scanner._state == StructureScanner.MOVE_TO_SCAN
    assert scanner._waiting_for_ik  # Still waiting


# 12. Detections in wrong state are ignored
def test_detections_in_wrong_state_ignored():
    bg = _grid([[Y]])
    scanner, mocks = _make_scanner(bg)

    scanner.start_scan(on_result=lambda r: None)
    # In MOVE_TO_SCAN, not DETECT — detections should be ignored
    dets = [_make_det(ObjectType.YELLOW_BLOCK, 0, 0, 0)]
    scanner.on_worldmap_detections(dets)
    # Should still be active, no result produced
    assert scanner.is_active

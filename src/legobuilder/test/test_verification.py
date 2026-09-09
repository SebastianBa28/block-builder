"""Unit tests for PlacementVerifier scan/orbit and match_detections_to_grid.

Validates that verification scan and orbit waypoints are centered on
GRID_CENTER_XY (the 5x5 base centroid) rather than the individual
target cell position.  Also validates the standalone grid-matching
function used for detection-to-cell assignment.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from legobuilder.config import (
    BLOCK_SIZE,
    GRID_CENTER_XY,
    ORBIT_SCAN_HEIGHT,
    ORBIT_SCAN_POSITIONS,
    ORBIT_SCAN_RADIUS,
    VERIFY_SCAN_HEIGHT,
    grip_check_roll
)
from legobuilder.brain.verification import (
    MAX_RECOVERY_ATTEMPTS,
    PlacementVerifier,
    identify_misplaced_block,
    match_detections_to_grid,
)
from legobuilder.schemas import Object, ObjectType


@pytest.fixture
def verifier():
    """Create a PlacementVerifier with mocked callbacks and structure.

    The mock structure's get_world_position returns a position that is
    intentionally DIFFERENT from GRID_CENTER_XY so tests can distinguish
    between grid-center usage and target-cell usage.
    """
    v = PlacementVerifier(
        send_ik_request_fn=MagicMock(),
        publish_trajectory_fn=MagicMock(),
        request_scan_fn=MagicMock(),
        create_timer_fn=MagicMock(),
        destroy_timer_fn=MagicMock(),
        get_t_fn=MagicMock(return_value=0.0),
        structure=MagicMock(),
        logger=MagicMock(),
    )
    # Return a position far from grid center so misuse is detectable
    v.structure.get_world_position.return_value = np.array([0.4, 0.35, 0.0])
    v._target_cell = (0, 0, 1)
    return v


@patch(
    'legobuilder.brain.verification.is_point_reachable',
    return_value=True,
)
@patch(
    'legobuilder.brain.verification.grip_check_roll',
    return_value=0.0,
)
def test_scan_uses_grid_center(mock_roll, mock_reach, verifier):
    """Primary scan position XY must match GRID_CENTER_XY, not target cell."""
    states = verifier._plan_verify_scan_trajectory()
    assert states is not None, "Expected a trajectory, got None"
    scan_pos = states[0].final_state.p
    assert abs(scan_pos[0] - GRID_CENTER_XY[0]) < 1e-6, (
        f"Scan X={scan_pos[0]:.4f} != GRID_CENTER_XY[0]={GRID_CENTER_XY[0]:.4f}"
    )
    assert abs(scan_pos[1] - GRID_CENTER_XY[1]) < 1e-6, (
        f"Scan Y={scan_pos[1]:.4f} != GRID_CENTER_XY[1]={GRID_CENTER_XY[1]:.4f}"
    )


@patch(
    'legobuilder.brain.verification.is_point_reachable',
    return_value=True,
)
@patch(
    'legobuilder.brain.verification.grip_check_roll',
    return_value=0.0,
)
def test_orbit_uses_grid_center(mock_roll, mock_reach, verifier):
    """Orbit waypoints must be centered on GRID_CENTER_XY, not target cell."""
    waypoints = verifier._generate_orbit_waypoints()
    assert len(waypoints) == ORBIT_SCAN_POSITIONS

    # Average of all waypoint XY should be GRID_CENTER_XY
    xs = [wp[0][0] for wp in waypoints]
    ys = [wp[0][1] for wp in waypoints]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    assert abs(mean_x - GRID_CENTER_XY[0]) < 1e-6, (
        f"Orbit mean X={mean_x:.4f} != GRID_CENTER_XY[0]={GRID_CENTER_XY[0]:.4f}"
    )
    assert abs(mean_y - GRID_CENTER_XY[1]) < 1e-6, (
        f"Orbit mean Y={mean_y:.4f} != GRID_CENTER_XY[1]={GRID_CENTER_XY[1]:.4f}"
    )


@patch(
    'legobuilder.brain.verification.is_point_reachable',
    return_value=True,
)
@patch(
    'legobuilder.brain.verification.grip_check_roll',
    return_value=0.0,
)
def test_scan_height_unchanged(mock_roll, mock_reach, verifier):
    """Scan Z must be VERIFY_SCAN_HEIGHT (absolute), not relative to layer."""
    states = verifier._plan_verify_scan_trajectory()
    assert states is not None
    scan_pos = states[0].final_state.p
    assert abs(scan_pos[2] - VERIFY_SCAN_HEIGHT) < 1e-6, (
        f"Scan Z={scan_pos[2]:.4f} != VERIFY_SCAN_HEIGHT={VERIFY_SCAN_HEIGHT}"
    )


@patch(
    'legobuilder.brain.verification.is_point_reachable',
    return_value=True,
)
@patch(
    'legobuilder.brain.verification.grip_check_roll',
    return_value=0.0,
)
def test_orbit_params_unchanged(mock_roll, mock_reach, verifier):
    """Orbit must use 5 positions, 0.1m radius, 0.25m height."""
    waypoints = verifier._generate_orbit_waypoints()
    assert len(waypoints) == 5, f"Expected 5 orbit positions, got {len(waypoints)}"

    # Check radius: distance from GRID_CENTER_XY in XY plane
    for pos, _ in waypoints:
        dx = pos[0] - GRID_CENTER_XY[0]
        dy = pos[1] - GRID_CENTER_XY[1]
        r = (dx**2 + dy**2) ** 0.5
        assert abs(r - ORBIT_SCAN_RADIUS) < 1e-6, (
            f"Orbit radius={r:.4f} != {ORBIT_SCAN_RADIUS}"
        )
        assert abs(pos[2] - ORBIT_SCAN_HEIGHT) < 1e-6, (
            f"Orbit Z={pos[2]:.4f} != {ORBIT_SCAN_HEIGHT}"
        )


# ── Grid-matching tests ──────────────────────────────────────────────


def _make_object(obj_type, center_xyz):
    """Create a minimal Object for grid-matching tests."""
    return Object(
        t=0.0,
        frame_idx=0,
        obj_type=obj_type,
        center_uv=(0, 0),
        center_xyz=center_xyz,
        angle=0.0,
        corner_xys=[],
        corner_uvs=[],
        face_conns_xyzs=[],
    )


class TestMatchDetectionsToGrid:
    """Tests for the standalone match_detections_to_grid function.

    All tests use layer=1 (1-based, matching JSON convention where z=1
    is ground level) with origin z=BLOCK_SIZE (matching production).
    """

    def _make_structure(self, make_block_structure, target_cell, target_type,
                        placed_cells=None):
        """Build a BlockStructure with target in block_grid and optional placed cells.

        Arguments
        ---------
        make_block_structure : fixture
            Factory from conftest.
        target_cell : tuple
            (row, col, layer) of the target.
        target_type : ObjectType
            Type to write into block_grid at target cell.
        placed_cells : list of (row, col, layer, ObjectType) or None
            Previously-placed cells to write into both block_grid and placed_grid.
        """
        bs = make_block_structure(rows=5, cols=5, layers=3)
        r, c, l = target_cell
        bs.block_grid[r, c, l] = target_type.value
        if placed_cells:
            for pr, pc, pl, ptype in placed_cells:
                bs.block_grid[pr, pc, pl] = ptype.value
                bs.placed_grid[pr, pc, pl] = ptype.value
        return bs

    # ── Empty / no detections ────────────────────────────────────────

    def test_empty_detections_returns_retry(self, make_block_structure):
        """Empty detection list must return 'retry'."""
        bs = self._make_structure(
            make_block_structure, (0, 0, 1), ObjectType.YELLOW_BLOCK,
        )
        result = match_detections_to_grid([], bs, (0, 0, 1))
        assert result['result'] == 'retry'
        assert result['target_match'] is None
        assert result['unassigned'] == []
        assert result['cell_matches'] == {}

    # ── Single detection at target, correct color ────────────────────

    def test_single_detection_correct_color_verified(self, make_block_structure):
        """Detection at target cell with matching color returns 'verified'."""
        target = (0, 0, 1)
        bs = self._make_structure(
            make_block_structure, target, ObjectType.YELLOW_BLOCK,
        )
        world_pos = bs.get_world_position(*target)
        det = _make_object(ObjectType.YELLOW_BLOCK, tuple(world_pos))
        result = match_detections_to_grid([det], bs, target)
        assert result['result'] == 'verified'
        assert result['target_match'] is det

    # ── Single detection at target, wrong color ──────────────────────

    def test_single_detection_wrong_color_failed(self, make_block_structure):
        """Detection at target cell with wrong color returns 'failed'."""
        target = (0, 0, 1)
        bs = self._make_structure(
            make_block_structure, target, ObjectType.YELLOW_BLOCK,
        )
        world_pos = bs.get_world_position(*target)
        det = _make_object(ObjectType.BLUE_BLOCK, tuple(world_pos))
        result = match_detections_to_grid([det], bs, target)
        assert result['result'] == 'failed'
        assert result['target_match'] is det

    # ── Detection outside tolerance ──────────────────────────────────

    def test_detection_outside_tolerance_retry(self, make_block_structure):
        """Detection far from all cells returns 'retry' and is unassigned."""
        target = (0, 0, 1)
        bs = self._make_structure(
            make_block_structure, target, ObjectType.YELLOW_BLOCK,
        )
        det = _make_object(ObjectType.YELLOW_BLOCK, (0.0, 0.0, 0.0))
        result = match_detections_to_grid([det], bs, target)
        assert result['result'] == 'retry'
        assert result['target_match'] is None
        assert det in result['unassigned']

    # ── XY boundary: just inside ─────────────────────────────────────

    def test_xy_just_inside_tolerance_assigned(self, make_block_structure):
        """Detection at BLOCK_SIZE/4 - epsilon from cell center is assigned."""
        target = (0, 0, 1)
        bs = self._make_structure(
            make_block_structure, target, ObjectType.YELLOW_BLOCK,
        )
        world_pos = bs.get_world_position(*target)
        eps = 1e-6
        xy_tol = BLOCK_SIZE / 4
        shifted = (
            world_pos[0] + xy_tol - eps,
            world_pos[1],
            world_pos[2],
        )
        det = _make_object(ObjectType.YELLOW_BLOCK, shifted)
        result = match_detections_to_grid([det], bs, target)
        assert result['result'] == 'verified'

    # ── XY boundary: just outside ────────────────────────────────────

    def test_xy_just_outside_tolerance_unassigned(self, make_block_structure):
        """Detection at BLOCK_SIZE/4 + epsilon from cell center is unassigned."""
        target = (0, 0, 1)
        bs = self._make_structure(
            make_block_structure, target, ObjectType.YELLOW_BLOCK,
        )
        world_pos = bs.get_world_position(*target)
        eps = 1e-6
        xy_tol = BLOCK_SIZE / 4
        shifted = (
            world_pos[0] + xy_tol + eps,
            world_pos[1],
            world_pos[2],
        )
        det = _make_object(ObjectType.YELLOW_BLOCK, shifted)
        result = match_detections_to_grid([det], bs, target)
        assert result['result'] == 'retry'

    # ── Z boundary: just inside ──────────────────────────────────────

    def test_z_just_inside_tolerance_assigned(self, make_block_structure):
        """Detection at BLOCK_SIZE/2 - epsilon Z offset is assigned."""
        target = (0, 0, 1)
        bs = self._make_structure(
            make_block_structure, target, ObjectType.YELLOW_BLOCK,
        )
        world_pos = bs.get_world_position(*target)
        eps = 1e-6
        z_tol = BLOCK_SIZE / 2
        shifted = (world_pos[0], world_pos[1], world_pos[2] + z_tol - eps)
        det = _make_object(ObjectType.YELLOW_BLOCK, shifted)
        result = match_detections_to_grid([det], bs, target)
        assert result['result'] == 'verified'

    # ── Z boundary: just outside ─────────────────────────────────────

    def test_z_just_outside_tolerance_unassigned(self, make_block_structure):
        """Detection at BLOCK_SIZE/2 + epsilon Z offset is unassigned."""
        target = (0, 0, 1)
        bs = self._make_structure(
            make_block_structure, target, ObjectType.YELLOW_BLOCK,
        )
        world_pos = bs.get_world_position(*target)
        eps = 1e-6
        z_tol = BLOCK_SIZE / 2
        shifted = (world_pos[0], world_pos[1], world_pos[2] + z_tol + eps)
        det = _make_object(ObjectType.YELLOW_BLOCK, shifted)
        result = match_detections_to_grid([det], bs, target)
        assert result['result'] == 'retry'

    # ── First block (empty placed_grid, no anchors) ──────────────────

    def test_first_block_no_anchors_verified(self, make_block_structure):
        """First block with empty placed_grid still verifies via target cell."""
        target = (2, 2, 1)
        bs = self._make_structure(
            make_block_structure, target, ObjectType.GREEN_BLOCK,
        )
        # placed_grid is all zeros — no anchors
        assert not np.any(bs.placed_grid)
        world_pos = bs.get_world_position(*target)
        det = _make_object(ObjectType.GREEN_BLOCK, tuple(world_pos))
        result = match_detections_to_grid([det], bs, target)
        assert result['result'] == 'verified'

    # ── Multiple detections same cell: color match wins ──────────────

    def test_multiple_detections_color_match_wins(self, make_block_structure):
        """When two detections are near the same cell, correct color wins."""
        target = (0, 0, 1)
        bs = self._make_structure(
            make_block_structure, target, ObjectType.YELLOW_BLOCK,
        )
        world_pos = bs.get_world_position(*target)
        # Wrong color but closer
        det_wrong = _make_object(
            ObjectType.BLUE_BLOCK,
            tuple(world_pos),
        )
        # Correct color, slightly offset
        det_right = _make_object(
            ObjectType.YELLOW_BLOCK,
            (world_pos[0] + 0.002, world_pos[1], world_pos[2]),
        )
        result = match_detections_to_grid(
            [det_wrong, det_right], bs, target,
        )
        assert result['result'] == 'verified'
        assert result['target_match'] is det_right

    # ── Multiple detections same cell, same color: closest wins ──────

    def test_multiple_detections_same_color_closest_wins(
        self, make_block_structure,
    ):
        """When two correct-color detections compete, closest wins."""
        target = (0, 0, 1)
        bs = self._make_structure(
            make_block_structure, target, ObjectType.YELLOW_BLOCK,
        )
        world_pos = bs.get_world_position(*target)
        det_far = _make_object(
            ObjectType.YELLOW_BLOCK,
            (world_pos[0] + 0.005, world_pos[1], world_pos[2]),
        )
        det_close = _make_object(
            ObjectType.YELLOW_BLOCK,
            (world_pos[0] + 0.001, world_pos[1], world_pos[2]),
        )
        result = match_detections_to_grid(
            [det_far, det_close], bs, target,
        )
        assert result['result'] == 'verified'
        assert result['target_match'] is det_close

    # ── Anchor cells matched, target unoccupied ──────────────────────

    def test_anchors_matched_target_empty_retry(self, make_block_structure):
        """Anchors match but target has no detection returns 'retry'."""
        target = (1, 1, 1)
        anchor_cell = (0, 0, 1)
        bs = self._make_structure(
            make_block_structure, target, ObjectType.BLUE_BLOCK,
            placed_cells=[(0, 0, 1, ObjectType.YELLOW_BLOCK)],
        )
        # Detection at anchor only
        anchor_pos = bs.get_world_position(*anchor_cell)
        det = _make_object(ObjectType.YELLOW_BLOCK, tuple(anchor_pos))
        result = match_detections_to_grid([det], bs, target)
        assert result['result'] == 'retry'
        assert result['target_match'] is None

    # ── Missing anchor detection does not cause failure ──────────────

    def test_missing_anchor_no_failure(self, make_block_structure):
        """Missing anchor detection does not fail verification."""
        target = (1, 1, 1)
        bs = self._make_structure(
            make_block_structure, target, ObjectType.BLUE_BLOCK,
            placed_cells=[(0, 0, 1, ObjectType.YELLOW_BLOCK)],
        )
        # Only detection at target, none at anchor
        target_pos = bs.get_world_position(*target)
        det = _make_object(ObjectType.BLUE_BLOCK, tuple(target_pos))
        result = match_detections_to_grid([det], bs, target)
        assert result['result'] == 'verified'

    # ── Unassigned detections returned ───────────────────────────────

    def test_unassigned_detections_returned(self, make_block_structure):
        """Detections not matching any cell appear in unassigned list."""
        target = (0, 0, 1)
        bs = self._make_structure(
            make_block_structure, target, ObjectType.YELLOW_BLOCK,
        )
        world_pos = bs.get_world_position(*target)
        det_good = _make_object(ObjectType.YELLOW_BLOCK, tuple(world_pos))
        det_stray = _make_object(ObjectType.RED_BLOCK, (0.0, 0.0, 0.0))
        result = match_detections_to_grid(
            [det_good, det_stray], bs, target,
        )
        assert result['result'] == 'verified'
        assert det_stray in result['unassigned']
        assert det_good not in result['unassigned']


class TestIdentifyMisplacedBlock:
    """Tests for the standalone identify_misplaced_block function."""

    # ── Empty unassigned list ────────────────────────────────────────

    def test_empty_unassigned_returns_none(self):
        """Empty unassigned list returns None."""
        result = identify_misplaced_block(
            unassigned=[],
            expected_type_val=ObjectType.YELLOW_BLOCK.value,
            target_world_pos=np.array([0.4, 0.4, 0.0]),
        )
        assert result is None

    # ── No color match ───────────────────────────────────────────────

    def test_no_color_match_returns_none(self):
        """Unassigned detections with no matching color returns None."""
        det_blue = _make_object(ObjectType.BLUE_BLOCK, (0.4, 0.4, 0.0))
        det_red = _make_object(ObjectType.RED_BLOCK, (0.41, 0.4, 0.0))
        result = identify_misplaced_block(
            unassigned=[det_blue, det_red],
            expected_type_val=ObjectType.YELLOW_BLOCK.value,
            target_world_pos=np.array([0.4, 0.4, 0.0]),
        )
        assert result is None

    # ── Single color-matched detection ───────────────────────────────

    def test_single_color_match_returned(self):
        """Single color-matched detection is returned."""
        det = _make_object(ObjectType.YELLOW_BLOCK, (0.5, 0.5, 0.0))
        result = identify_misplaced_block(
            unassigned=[det],
            expected_type_val=ObjectType.YELLOW_BLOCK.value,
            target_world_pos=np.array([0.4, 0.4, 0.0]),
        )
        assert result is det

    # ── Multiple color matches: closest wins ─────────────────────────

    def test_multiple_color_matches_closest_wins(self):
        """Among multiple color-matched detections, closest to target wins."""
        det_far = _make_object(ObjectType.GREEN_BLOCK, (0.5, 0.5, 0.0))
        det_close = _make_object(ObjectType.GREEN_BLOCK, (0.41, 0.41, 0.0))
        result = identify_misplaced_block(
            unassigned=[det_far, det_close],
            expected_type_val=ObjectType.GREEN_BLOCK.value,
            target_world_pos=np.array([0.4, 0.4, 0.0]),
        )
        assert result is det_close

    # ── Mixed color detections: only color-matched considered ────────

    def test_mixed_colors_only_matched_considered(self):
        """Non-matching colors are ignored even if closer to target."""
        # Wrong color but very close
        det_wrong_close = _make_object(ObjectType.BLUE_BLOCK, (0.401, 0.4, 0.0))
        # Right color but farther
        det_right_far = _make_object(ObjectType.YELLOW_BLOCK, (0.5, 0.5, 0.0))
        result = identify_misplaced_block(
            unassigned=[det_wrong_close, det_right_far],
            expected_type_val=ObjectType.YELLOW_BLOCK.value,
            target_world_pos=np.array([0.4, 0.4, 0.0]),
        )
        assert result is det_right_far

    # ── No distance threshold: far detection still accepted ──────────

    def test_no_distance_threshold(self):
        """Color-matched detection is accepted regardless of distance."""
        det_very_far = _make_object(ObjectType.RED_BLOCK, (10.0, 10.0, 10.0))
        result = identify_misplaced_block(
            unassigned=[det_very_far],
            expected_type_val=ObjectType.RED_BLOCK.value,
            target_world_pos=np.array([0.0, 0.0, 0.0]),
        )
        assert result is det_very_far


# ── Verification flow tests ─────────────────────────────────────────


@patch(
    'legobuilder.brain.verification.is_point_reachable',
    return_value=True,
)
@patch(
    'legobuilder.brain.verification.grip_check_roll',
    return_value=0.0,
)
def test_start_verification_does_primary_scan_first(
    mock_roll, mock_reach, verifier,
):
    """start_verification must transition to VERIFY_MOVE_TO_SCAN (primary scan)."""
    verifier.start_verification(
        target_cell=(0, 0, 1),
        pickup_context={'pickup_roll': 0.0},
        on_result=MagicMock(),
    )
    assert verifier._state == PlacementVerifier.VERIFY_MOVE_TO_SCAN
    verifier._send_ik_request.assert_called()


@patch('legobuilder.brain.verification.VERIFY_SCAN_SOURCE', 0)
def test_on_trajectory_complete_scan_source_0_does_dwell(verifier):
    """VERIFY_SCAN_SOURCE=0: trajectory complete in VERIFY_MOVE_TO_SCAN does dwell."""
    verifier._state = PlacementVerifier.VERIFY_MOVE_TO_SCAN
    verifier._waiting_for_ik = False
    verifier.on_trajectory_complete()
    assert verifier._state == PlacementVerifier.VERIFY_DWELL_SCAN


@patch('legobuilder.brain.verification.VERIFY_SCAN_SOURCE', 1)
def test_on_trajectory_complete_scan_source_1_skips_dwell(verifier):
    """VERIFY_SCAN_SOURCE=1: trajectory complete in VERIFY_MOVE_TO_SCAN skips to detect."""
    verifier._state = PlacementVerifier.VERIFY_MOVE_TO_SCAN
    verifier._waiting_for_ik = False
    verifier.on_trajectory_complete()
    assert verifier._state == PlacementVerifier.VERIFY_DETECT


def test_primary_scan_retry_transitions_to_orbit(verifier):
    """Primary scan 'retry' result triggers orbit fallback."""
    verifier._state = PlacementVerifier.VERIFY_DETECT
    verifier._target_cell = (0, 0, 1)
    verifier._is_orbit_fallback = False
    # Structure must support match_detections_to_grid calls
    verifier.structure.get_simulated_detection_grid.return_value = np.zeros(
        (5, 5, 3), dtype=int,
    )
    verifier.structure.block_grid = np.zeros((5, 5, 3), dtype=int)
    verifier.structure.block_grid[0, 0, 1] = ObjectType.YELLOW_BLOCK.value
    verifier.structure.get_world_position.return_value = np.array([0.4, 0.4, 0.0])

    # Empty detections -> retry from match_detections_to_grid
    with patch(
        'legobuilder.brain.verification.is_point_reachable',
        return_value=True,
    ), patch(
        'legobuilder.brain.verification.grip_check_roll',
        return_value=0.0,
    ):
        verifier.on_worldmap_detections([])
    assert verifier._state == PlacementVerifier.RECOVERY_ORBIT
    assert verifier._is_orbit_fallback is True


def test_orbit_fallback_retry_triggers_recovery_not_immediate_fail(verifier):
    """After orbit fallback, 'retry' triggers _attempt_recovery (not _finish('failed')).

    When no color-matched unassigned detection exists, _attempt_recovery
    ultimately finishes 'failed', but the path goes through recovery logic.
    """
    on_result = MagicMock()
    verifier._state = PlacementVerifier.VERIFY_DETECT
    verifier._target_cell = (0, 0, 1)
    verifier._is_orbit_fallback = True
    verifier._on_result = on_result
    verifier._recovery_attempt_count = 0
    # Structure setup for match_detections_to_grid
    verifier.structure.get_simulated_detection_grid.return_value = np.zeros(
        (5, 5, 3), dtype=int,
    )
    verifier.structure.block_grid = np.zeros((5, 5, 3), dtype=int)
    verifier.structure.block_grid[0, 0, 1] = ObjectType.YELLOW_BLOCK.value
    verifier.structure.get_world_position.return_value = np.array([0.4, 0.4, 0.0])

    # Empty detections -> retry, orbit fallback -> _attempt_recovery
    # No unassigned detections with matching color -> still fails, but via recovery path
    verifier.on_worldmap_detections([])
    # Recovery attempted (count incremented) then failed due to no color match
    on_result.assert_called_once_with('failed')


def test_last_verification_result_stored(verifier, make_block_structure):
    """_check_placement stores structured result on _last_verification_result."""
    target = (0, 0, 1)
    bs = make_block_structure(rows=5, cols=5, layers=3)
    bs.block_grid[0, 0, 1] = ObjectType.YELLOW_BLOCK.value

    verifier.structure = bs
    verifier._target_cell = target

    world_pos = bs.get_world_position(*target)
    det = _make_object(ObjectType.YELLOW_BLOCK, tuple(world_pos))

    # Call _check_placement directly to inspect result before _finish clears it
    result_str = verifier._check_placement([det], 'primary')
    assert result_str == 'verified'

    lvr = verifier._last_verification_result
    assert lvr is not None
    assert lvr['result'] == 'verified'
    assert lvr['target_cell'] == target
    assert lvr['target_match'] is det
    assert lvr['num_detections'] == 1
    assert lvr['scan_type'] == 'primary'
    assert 'matched_cells' in lvr
    assert 'unassigned_detections' in lvr


# ── Recovery wiring tests ──────────────────────────────────────────


def test_orbit_retry_triggers_recovery(verifier):
    """When _is_orbit_fallback is True and result is 'retry', _attempt_recovery is called."""
    on_result = MagicMock()
    verifier._state = PlacementVerifier.VERIFY_DETECT
    verifier._target_cell = (0, 0, 1)
    verifier._is_orbit_fallback = True
    verifier._on_result = on_result
    verifier._recovery_attempt_count = 0
    verifier._pickup_context = {'pickup_roll': 0.0}
    # Structure setup
    verifier.structure.get_simulated_detection_grid.return_value = np.zeros(
        (5, 5, 3), dtype=int,
    )
    verifier.structure.block_grid = np.zeros((5, 5, 3), dtype=int)
    verifier.structure.block_grid[0, 0, 1] = ObjectType.YELLOW_BLOCK.value
    target_pos = np.array([0.4, 0.4, 0.0])
    verifier.structure.get_world_position.return_value = target_pos

    # Provide a color-matched unassigned detection so recovery proceeds
    misplaced = _make_object(
        ObjectType.YELLOW_BLOCK, (0.42, 0.42, 0.01),
    )
    with patch(
        'legobuilder.brain.verification.is_point_reachable',
        return_value=True,
    ):
        verifier.on_worldmap_detections([misplaced])

    # Recovery should have identified the misplaced block and called _plan_recovery_grasp
    # (which sends an IK request)
    assert verifier._send_ik_request.called
    # on_result should NOT have been called yet (recovery in progress)
    on_result.assert_not_called()


@patch(
    'legobuilder.brain.verification.is_point_reachable',
    return_value=True,
)
def test_recovery_grasp_uses_detected_z(mock_reach, verifier):
    """_plan_recovery_grasp uses obj.center_xyz[2] for p_pick Z, not PICK_Z."""
    detected_z = 0.025  # Block on grid surface
    verifier._misplaced_obj = _make_object(
        ObjectType.YELLOW_BLOCK, (0.42, 0.42, detected_z),
    )
    verifier._pickup_context = {'pickup_roll': 0.0}
    verifier.structure.PICK_Z = 0.099  # Different from detected_z

    verifier._plan_recovery_grasp()

    # Extract the states from the IK request call
    call_args = verifier._send_ik_request.call_args
    states = call_args[0][0]
    # Second state is the descend-to-pick state
    p_pick = states[1].final_state.p
    assert abs(p_pick[2] - detected_z) < 1e-6, (
        f"Recovery grasp Z={p_pick[2]:.4f} should be detected Z={detected_z}, "
        f"not PICK_Z={verifier.structure.PICK_Z}"
    )


def test_recovery_attempt_limit(verifier):
    """After MAX_RECOVERY_ATTEMPTS recovery cycles, verifier reports 'failed'."""
    on_result = MagicMock()
    verifier._on_result = on_result
    verifier._target_cell = (0, 0, 1)
    verifier._recovery_attempt_count = MAX_RECOVERY_ATTEMPTS
    # Set up a valid _last_verification_result with unassigned
    verifier._last_verification_result = {
        'result': 'retry',
        'unassigned_detections': [
            _make_object(ObjectType.YELLOW_BLOCK, (0.42, 0.42, 0.0)),
        ],
    }
    verifier.structure.block_grid = np.zeros((5, 5, 3), dtype=int)
    verifier.structure.block_grid[0, 0, 1] = ObjectType.YELLOW_BLOCK.value
    verifier.structure.get_world_position.return_value = np.array([0.4, 0.4, 0.0])

    verifier._attempt_recovery()

    on_result.assert_called_once_with('failed')


def test_recovery_place_triggers_reverify(verifier):
    """_on_recovery_place_complete calls _transition_to_move_to_scan."""
    with patch(
        'legobuilder.brain.verification.is_point_reachable',
        return_value=True,
    ), patch(
        'legobuilder.brain.verification.grip_check_roll',
        return_value=0.0,
    ):
        verifier._on_recovery_place_complete()
    assert verifier._state == PlacementVerifier.VERIFY_MOVE_TO_SCAN


def test_recovery_resets_orbit_fallback(verifier):
    """_transition_to_move_to_scan resets _is_orbit_fallback to False."""
    verifier._is_orbit_fallback = True
    with patch(
        'legobuilder.brain.verification.is_point_reachable',
        return_value=True,
    ), patch(
        'legobuilder.brain.verification.grip_check_roll',
        return_value=0.0,
    ):
        verifier._transition_to_move_to_scan()
    assert verifier._is_orbit_fallback is False


@patch(
    'legobuilder.brain.verification.is_point_reachable',
    return_value=True,
)
def test_grip_retry_lift_uses_detected_z(mock_reach, verifier):
    """_plan_grip_retry_lift uses obj.center_xyz[2] not PICK_Z."""
    detected_z = 0.025
    verifier._misplaced_obj = _make_object(
        ObjectType.YELLOW_BLOCK, (0.42, 0.42, detected_z),
    )
    verifier.structure.PICK_Z = 0.099  # Should NOT be used

    verifier._plan_grip_retry_lift()

    call_args = verifier._send_ik_request.call_args
    states = call_args[0][0]
    # First state: open gripper at pick Z
    p_open = states[0].final_state.p
    assert abs(p_open[2] - detected_z) < 1e-6, (
        f"Grip retry open Z={p_open[2]:.4f} should be {detected_z}"
    )
    # Second state: lift above
    p_above = states[1].final_state.p
    expected_above_z = detected_z + verifier.APPROACH_HEIGHT
    assert abs(p_above[2] - expected_above_z) < 1e-6, (
        f"Grip retry lift Z={p_above[2]:.4f} should be {expected_above_z}"
    )

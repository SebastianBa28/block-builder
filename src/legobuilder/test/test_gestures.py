"""Unit tests for GestureManager idle scan-chomp cycle and pipeline integration.

Tests the 5-state sequential state machine using mocked callbacks
(no ROS dependencies). Covers GEST-01, GEST-02, and GEST-05.
"""

import pytest
from unittest.mock import MagicMock, PropertyMock, call
import numpy as np

from legobuilder.brain.gestures import GestureManager
from legobuilder.config import (
    Q_CHOMP, Q_SCAN, Q_READY, DURATION,
    IDLE_CHOMP_BURST_COUNT, IDLE_CHOMP_DURATION,
    IDLE_CHOMP_PAUSE, IDLE_CHOMP_CLOSED_RAD,
    IDLE_CHOMP_INITIAL_DELAY,
    GRIPPER_JOINT_INDEX, GRIPPER_MOTOR_OPEN_RAD,
)


@pytest.fixture
def gesture_deps():
    """Create mocked dependencies for GestureManager."""
    scanner = MagicMock()
    type(scanner).is_active = PropertyMock(return_value=False)
    return {
        'publish_trajectory_fn': MagicMock(),
        'create_timer_fn': MagicMock(return_value=MagicMock()),
        'destroy_timer_fn': MagicMock(),
        'get_t_fn': MagicMock(return_value=0.0),
        'scanner': scanner,
        'logger': MagicMock(),
    }


@pytest.fixture
def gm(gesture_deps):
    """Create a GestureManager with mocked dependencies."""
    return GestureManager(**gesture_deps)


# ── GEST-01: Q_CHOMP definition ──────────────────────────────────


def test_q_chomp_definition():
    """Q_CHOMP has base=-0.70, other joints match Q_READY, correct gripper."""
    assert Q_CHOMP[0] == -0.70, f"Q_CHOMP base should be -0.70, got {Q_CHOMP[0]}"
    np.testing.assert_array_almost_equal(
        Q_CHOMP[1:5], Q_READY[1:5],
        err_msg="Q_CHOMP[1:5] should match Q_READY[1:5]",
    )
    assert Q_CHOMP[5] == GRIPPER_MOTOR_OPEN_RAD, (
        f"Q_CHOMP gripper should be {GRIPPER_MOTOR_OPEN_RAD}, got {Q_CHOMP[5]}"
    )


# ── GEST-01: start_idle transitions ─────────────────────────────


def test_start_idle_transitions_to_move_to_scan(gm, gesture_deps):
    """After start_idle(), state is MOVE_TO_SCAN and trajectory published to Q_SCAN."""
    gm.start_idle()

    assert gm.state == GestureManager.MOVE_TO_SCAN

    # Trajectory should be published with Q_SCAN target
    publish = gesture_deps['publish_trajectory_fn']
    assert publish.call_count == 1
    states_arg = publish.call_args[0][0]
    assert len(states_arg) == 1
    np.testing.assert_array_almost_equal(
        states_arg[0].final_state.q, Q_SCAN,
    )


def test_start_idle_noop_when_active(gm, gesture_deps):
    """Calling start_idle() when not INACTIVE does nothing."""
    gm.start_idle()
    publish = gesture_deps['publish_trajectory_fn']
    initial_calls = publish.call_count

    # Call again -- should be no-op
    gm.start_idle()
    assert publish.call_count == initial_calls


# ── GEST-01: Chomp trajectory at Q_CHOMP ─────────────────────────


def test_chomp_trajectory_at_q_chomp(gm, gesture_deps):
    """Chomp burst uses Q_CHOMP (base=-0.70) with correct gripper alternation."""
    # Advance to CHOMPING by firing timer callbacks
    gm.start_idle()
    assert gm.state == GestureManager.MOVE_TO_SCAN

    # Fire _on_at_scan_pose timer callback
    scan_timer_cb = gesture_deps['create_timer_fn'].call_args[0][1]
    scan_timer_cb()
    assert gm.state == GestureManager.SCANNING

    # Fire scan result callback
    scan_call = gesture_deps['scanner'].start_scan.call_args
    on_result = scan_call[1]['on_result']
    on_result(MagicMock())
    assert gm.state == GestureManager.MOVE_TO_CHOMP

    # Fire _on_at_chomp_pose timer callback
    chomp_timer_cb = gesture_deps['create_timer_fn'].call_args[0][1]
    chomp_timer_cb()
    assert gm.state == GestureManager.CHOMPING

    # Check the chomp burst trajectory
    publish = gesture_deps['publish_trajectory_fn']
    # Last publish call should be the chomp burst
    chomp_call = publish.call_args_list[-1]
    burst_states = chomp_call[0][0]

    assert len(burst_states) == IDLE_CHOMP_BURST_COUNT * 2

    for i, s in enumerate(burst_states):
        assert s.final_state.q[0] == -0.70, (
            f"Burst state {i} base should be -0.70"
        )
        if i % 2 == 0:
            # Closed
            assert s.final_state.q[GRIPPER_JOINT_INDEX] == IDLE_CHOMP_CLOSED_RAD
            assert s.final_state.gripper_open is False
        else:
            # Open
            assert s.final_state.q[GRIPPER_JOINT_INDEX] == GRIPPER_MOTOR_OPEN_RAD
            assert s.final_state.gripper_open is True


# ── GEST-05: Sequential scan-then-chomp ──────────────────────────


def test_sequential_scan_then_chomp(gm, gesture_deps):
    """Full cycle: INACTIVE -> MOVE_TO_SCAN -> SCANNING -> MOVE_TO_CHOMP -> CHOMPING -> MOVE_TO_SCAN."""
    assert gm.state == GestureManager.INACTIVE

    gm.start_idle()
    assert gm.state == GestureManager.MOVE_TO_SCAN

    # Fire scan arrival timer
    cb = gesture_deps['create_timer_fn'].call_args[0][1]
    cb()
    assert gm.state == GestureManager.SCANNING

    # Fire scan result
    on_result = gesture_deps['scanner'].start_scan.call_args[1]['on_result']
    on_result(MagicMock())
    assert gm.state == GestureManager.MOVE_TO_CHOMP

    # Fire chomp arrival timer
    cb = gesture_deps['create_timer_fn'].call_args[0][1]
    cb()
    assert gm.state == GestureManager.CHOMPING

    # Fire chomp complete timer -- should cycle back to MOVE_TO_SCAN
    cb = gesture_deps['create_timer_fn'].call_args[0][1]
    cb()
    assert gm.state == GestureManager.MOVE_TO_SCAN


# ── GEST-02: stop_idle cancellation ──────────────────────────────


def test_stop_idle_cancels_all(gm, gesture_deps):
    """stop_idle() from active state sets INACTIVE and destroys timer."""
    gm.start_idle()
    assert gm.state == GestureManager.MOVE_TO_SCAN

    gm.stop_idle()
    assert gm.state == GestureManager.INACTIVE
    gesture_deps['destroy_timer_fn'].assert_called()


def test_stop_idle_from_chomping(gm, gesture_deps):
    """stop_idle() works from CHOMPING state."""
    # Advance to CHOMPING
    gm.start_idle()
    cb = gesture_deps['create_timer_fn'].call_args[0][1]
    cb()  # -> SCANNING
    on_result = gesture_deps['scanner'].start_scan.call_args[1]['on_result']
    on_result(MagicMock())  # -> MOVE_TO_CHOMP
    cb = gesture_deps['create_timer_fn'].call_args[0][1]
    cb()  # -> CHOMPING
    assert gm.state == GestureManager.CHOMPING

    gm.stop_idle()
    assert gm.state == GestureManager.INACTIVE


# ── GEST-05: stop_idle resets scanner ────────────────────────────


def test_stop_idle_resets_scanner(gm, gesture_deps):
    """stop_idle() resets scanner if it is active."""
    scanner = gesture_deps['scanner']
    type(scanner).is_active = PropertyMock(return_value=True)

    gm.start_idle()
    gm.stop_idle()

    scanner.reset.assert_called_once()


def test_stop_idle_does_not_reset_inactive_scanner(gm, gesture_deps):
    """stop_idle() does not reset scanner if not active."""
    scanner = gesture_deps['scanner']
    type(scanner).is_active = PropertyMock(return_value=False)

    gm.start_idle()
    gm.stop_idle()

    scanner.reset.assert_not_called()


# ── Scan result callback ─────────────────────────────────────────


def test_scan_result_callback_invoked(gesture_deps):
    """scan_result_callback is called with result before chomp transition."""
    result_cb = MagicMock()
    gm = GestureManager(**gesture_deps, scan_result_callback=result_cb)

    gm.start_idle()
    # -> MOVE_TO_SCAN, fire timer
    cb = gesture_deps['create_timer_fn'].call_args[0][1]
    cb()  # -> SCANNING
    # Fire scan result
    mock_result = MagicMock()
    on_result = gesture_deps['scanner'].start_scan.call_args[1]['on_result']
    on_result(mock_result)

    result_cb.assert_called_once_with(mock_result)
    assert gm.state == GestureManager.MOVE_TO_CHOMP


# ── Stale callback guard ─────────────────────────────────────────


def test_stale_callback_guard(gm, gesture_deps):
    """If stop_idle() during SCANNING, subsequent scan result does not transition."""
    gm.start_idle()
    # -> MOVE_TO_SCAN, fire timer
    cb = gesture_deps['create_timer_fn'].call_args[0][1]
    cb()  # -> SCANNING
    assert gm.state == GestureManager.SCANNING

    # Capture the scan result callback
    on_result = gesture_deps['scanner'].start_scan.call_args[1]['on_result']

    # Stop idle
    type(gesture_deps['scanner']).is_active = PropertyMock(return_value=True)
    gm.stop_idle()
    assert gm.state == GestureManager.INACTIVE

    # Fire stale scan result -- should NOT change state
    on_result(MagicMock())
    assert gm.state == GestureManager.INACTIVE


# ── Pipeline integration tests ──────────────────────────────────


def _make_pipeline():
    """Create a BuildPipeline with all mocked dependencies."""
    from legobuilder.brain.build_pipeline import BuildPipeline
    scanner = MagicMock()
    type(scanner).is_active = PropertyMock(return_value=False)
    pipeline = BuildPipeline(
        send_ik_request_fn=MagicMock(),
        publish_trajectory_fn=MagicMock(),
        request_grip_check_fn=MagicMock(),
        create_timer_fn=MagicMock(return_value=MagicMock()),
        destroy_timer_fn=MagicMock(),
        get_t_fn=MagicMock(return_value=0.0),
        fkin_all_fn=MagicMock(),
        request_connection_check_fn=MagicMock(),
        structure=MagicMock(),
        logger=MagicMock(),
        precompute_manager=MagicMock(),
        scanner=scanner,
    )
    return pipeline


def test_clear_build_state_stops_idle():
    """Verify clear_build_state() calls gesture_manager.stop_idle() -- root cause fix."""
    pipeline = _make_pipeline()
    pipeline._gesture_manager.stop_idle = MagicMock()
    pipeline.clear_build_state()
    pipeline._gesture_manager.stop_idle.assert_called_once()


def test_mark_idle_calls_start_idle():
    """Verify _mark_idle() calls gesture_manager.start_idle() after delay."""
    pipeline = _make_pipeline()
    pipeline._gesture_manager.start_idle = MagicMock()
    # First call sets _idle_since
    pipeline._mark_idle()
    pipeline._gesture_manager.start_idle.assert_not_called()
    # Set time past delay
    pipeline._get_t = MagicMock(return_value=IDLE_CHOMP_INITIAL_DELAY + 1.0)
    pipeline._mark_idle()
    pipeline._gesture_manager.start_idle.assert_called_once()

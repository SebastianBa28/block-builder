"""Idle gesture management -- scan/chomp/nod cycle coordination.

Manages the idle scan-then-chomp animation cycle as a sequential
state machine, extended with a help-me nod gesture for unreachable
blocks.  Follows the callback-injection pattern: receives injected
functions for ROS operations -- no ROS imports.

States: INACTIVE -> MOVE_TO_SCAN -> SCANNING -> [MOVE_TO_CHOMP -> CHOMPING | MOVE_TO_NOD -> NODDING -> POST_NOD_PAUSE]
        (CHOMPING loops back to MOVE_TO_SCAN; POST_NOD_PAUSE either exits or loops; stop_idle() from any state -> INACTIVE)
"""

from math import atan2
from typing import Callable, Optional

import numpy as np

from legobuilder.kinematics.trajectory import TrajectoryState
from legobuilder.kinematics.block_manipulator import ManipulatorState
from legobuilder.config import (
    Q_SCAN, Q_CHOMP, DURATION, BLOCK_SIZE, BASE_MOTOR_POS,
    IDLE_CHOMP_BURST_COUNT, IDLE_CHOMP_DURATION,
    IDLE_CHOMP_PAUSE, IDLE_CHOMP_CLOSED_RAD,
    GRIPPER_JOINT_INDEX, GRIPPER_MOTOR_OPEN_RAD, GRIPPER_MOTOR_CLOSED_RAD
)
from legobuilder_interfaces.msg import TrajectoryCommandMsg


class GestureManager:
    """Sequential state machine for idle scan-chomp gesture cycle.

    Replaces the dual-timer idle approach with a clean sequential flow:
    scan at Q_SCAN, move to Q_CHOMP, burst chomp, repeat.

    Parameters
    ----------
    publish_trajectory_fn : Callable
        Publishes a list of TrajectoryState with a command type.
    create_timer_fn : Callable
        Creates a one-shot timer (delay_seconds, callback) -> timer handle.
    destroy_timer_fn : Callable
        Destroys a timer by handle.
    get_t_fn : Callable
        Returns current time in seconds.
    scanner : StructureScanner or None
        Scanner instance for idle structure scans.
    logger
        Logger instance for status messages.
    scan_result_callback : Callable or None
        Optional callback invoked with scan result before transitioning
        to chomp. Allows BuildPipeline to process scan results (update
        placed_grid, find unassigned blocks). If the callback calls
        stop_idle(), the subsequent chomp transition is guarded.
    """

    # States
    INACTIVE = 'inactive'
    MOVE_TO_SCAN = 'move_to_scan'
    SCANNING = 'scanning'
    MOVE_TO_CHOMP = 'move_to_chomp'
    CHOMPING = 'chomping'
    MOVE_TO_NOD = 'move_to_nod'
    NODDING = 'nodding'
    POST_NOD_PAUSE = 'post_nod_pause'

    def __init__(
        self,
        publish_trajectory_fn: Callable,
        create_timer_fn: Callable,
        destroy_timer_fn: Callable,
        get_t_fn: Callable,
        scanner,
        logger,
        scan_result_callback: Optional[Callable] = None,
        unreachable_tracker=None,
        structure=None,
        on_nod_complete_callback: Optional[Callable] = None,
        post_event_fn: Optional[Callable] = None,
    ):
        self._publish_trajectory = publish_trajectory_fn
        self._create_timer = create_timer_fn
        self._destroy_timer = destroy_timer_fn
        self._get_t = get_t_fn
        self._scanner = scanner
        self.logger = logger
        self._scan_result_callback = scan_result_callback
        self._unreachable_tracker = unreachable_tracker
        self._structure = structure
        self._on_nod_complete_callback = on_nod_complete_callback
        self._post_event = post_event_fn

        self._state: str = self.INACTIVE
        self._timer = None
        self._nod_target_cell = None
        self._has_pickable_block = False

    def _emit(self, event_type: str, data: dict) -> None:
        """Post an event to the dashboard server."""
        if self._post_event is None:
            return
        self._post_event({
            'type': event_type,
            'source': 'gestures',
            'data': data,
            'timestamp': self._get_t(),
        })

    def _set_state(self, new_state: str, detail: str = '') -> None:
        """Update state and emit a dashboard event."""
        self._state = new_state
        self._emit('state_change', {'state': new_state, 'detail': detail})

    # ── Public API ────────────────────────────────────────────────────

    @property
    def state(self) -> str:
        """Current state of the gesture manager."""
        return self._state

    def start_idle(self) -> None:
        """Begin idle scan-chomp cycle.

        Only starts if currently INACTIVE. Transitions to MOVE_TO_SCAN
        and publishes trajectory to Q_SCAN.
        """
        if self._state != self.INACTIVE:
            return
        self.logger.info("GestureManager: starting idle cycle")
        self._transition_to_move_to_scan()

    def stop_idle(self) -> None:
        """Immediately cancel all idle activity from any state.

        Cancels all timers, resets scanner if active, and returns to
        INACTIVE. Does NOT publish any trajectory -- lets the pipeline
        take over from the current joint state.
        """
        self._cancel_all_timers()
        if self._scanner is not None and self._scanner.is_active:
            self._scanner.reset()
        self._set_state(self.INACTIVE)
        self._nod_target_cell = None
        self._has_pickable_block = False

    # ── State Transitions (internal) ─────────────────────────────────

    _IDLE_MOVE_DURATION = DURATION / 2

    def _transition_to_move_to_scan(self) -> None:
        """Publish trajectory to Q_SCAN and set timer for arrival."""
        states = [
            TrajectoryState(
                final_state=ManipulatorState(q=Q_SCAN.copy()),
                min_duration=self._IDLE_MOVE_DURATION,
            )
        ]
        self._publish_trajectory(states, TrajectoryCommandMsg.COMMAND_REPLACE)
        self._set_state(self.MOVE_TO_SCAN)
        self._set_timer(self._IDLE_MOVE_DURATION, self._on_at_scan_pose)

    def _on_at_scan_pose(self) -> None:
        """Timer callback: arrived at Q_SCAN, start scanning."""
        self._cancel_all_timers()
        if self._state != self.MOVE_TO_SCAN:
            return  # stale callback guard
        self._set_state(self.SCANNING)
        self.logger.info("GestureManager: at scan pose, starting scan")
        self._scanner.start_scan(
            on_result=self._on_scan_result,
            skip_orbit=True,
        )

    def _on_scan_result(self, result) -> None:
        """Scanner callback: scan complete, relay result and decide next gesture."""
        if self._state != self.SCANNING:
            return  # guard against callback after stop_idle
        self.logger.info("GestureManager: scan complete")
        # Relay result to pipeline before transitioning
        if self._scan_result_callback is not None:
            self._scan_result_callback(result)
        # Check if callback called stop_idle()
        if self._state != self.SCANNING:
            return

        # Store whether a pickable block was found (target_cell is not None)
        self._has_pickable_block = (
            result.target_cell is not None and not result.is_mistake
        )

        # Nod decision: nod at first non-ignored unreachable block
        nod_target = self._pick_nod_target()
        if nod_target is not None:
            self._nod_target_cell = nod_target
            self.logger.info(
                f"GestureManager: nodding at unreachable block {nod_target}"
            )
            self._transition_to_move_to_nod()
        else:
            self._transition_to_move_to_chomp()

    def _transition_to_move_to_chomp(self) -> None:
        """Publish trajectory to Q_CHOMP and set timer for arrival."""
        states = [
            TrajectoryState(
                final_state=ManipulatorState(q=Q_CHOMP.copy()),
                min_duration=self._IDLE_MOVE_DURATION,
            )
        ]
        self._publish_trajectory(states, TrajectoryCommandMsg.COMMAND_REPLACE)
        self._set_state(self.MOVE_TO_CHOMP)
        self._set_timer(self._IDLE_MOVE_DURATION, self._on_at_chomp_pose)

    def _on_at_chomp_pose(self) -> None:
        """Timer callback: arrived at Q_CHOMP, start chomping."""
        self._cancel_all_timers()
        if self._state != self.MOVE_TO_CHOMP:
            return  # stale callback guard
        self._set_state(self.CHOMPING)
        self.logger.info("GestureManager: at chomp pose, starting burst")
        burst = self._build_chomp_burst()
        self._publish_trajectory(burst, TrajectoryCommandMsg.COMMAND_REPLACE)
        # Timer for burst duration + pause before cycling back
        burst_duration = IDLE_CHOMP_BURST_COUNT * 2 * IDLE_CHOMP_DURATION
        total_delay = burst_duration + IDLE_CHOMP_PAUSE
        self._set_timer(total_delay, self._on_chomp_complete)

    def _build_chomp_burst(self) -> list:
        """Build chomp burst trajectory at Q_CHOMP.

        Returns a list of TrajectoryState alternating between closed
        and open gripper positions for IDLE_CHOMP_BURST_COUNT cycles.
        """
        states = []
        for _ in range(IDLE_CHOMP_BURST_COUNT):
            q_closed = Q_CHOMP.copy()
            q_closed[GRIPPER_JOINT_INDEX] = IDLE_CHOMP_CLOSED_RAD
            states.append(TrajectoryState(
                final_state=ManipulatorState(q=q_closed, gripper_open=False),
                min_duration=IDLE_CHOMP_DURATION,
            ))
            q_open = Q_CHOMP.copy()
            q_open[GRIPPER_JOINT_INDEX] = GRIPPER_MOTOR_OPEN_RAD
            states.append(TrajectoryState(
                final_state=ManipulatorState(q=q_open, gripper_open=True),
                min_duration=IDLE_CHOMP_DURATION,
            ))
        return states

    def _on_chomp_complete(self) -> None:
        """Timer callback: chomp burst finished, cycle back to scan."""
        self._cancel_all_timers()
        if self._state != self.CHOMPING:
            return  # stale callback guard
        self.logger.info("GestureManager: chomp complete, cycling back to scan")
        self._transition_to_move_to_scan()

    # ── Nod Gesture ────────────────────────────────────────────────────

    def _pick_nod_target(self):
        """Select one unreachable block to nod at (first non-ignored)."""
        if self._unreachable_tracker is None:
            return None
        statuses = self._unreachable_tracker.get_all_statuses()
        for cell, status in statuses.items():
            if not status.permanently_ignored:
                return cell
        return None

    def _compute_nod_joint_config(self, cell):
        """Compute joint config to hover above an unreachable block.

        Rotates base joint to face the block's world position.
        Uses Q_SCAN shoulder/elbow/wrist angles (a "looking at structure" pose).
        Gripper stays closed to differentiate from chomp.
        """
        row, col, layer = cell
        world_pos = self._structure.get_world_position(row, col, layer)
        # Base angle: account for 180° yaw flip in L-bracket fixed joint
        dx = world_pos[0] - BASE_MOTOR_POS[0]
        dy = world_pos[1] - BASE_MOTOR_POS[1]
        base_angle = atan2(-dx, dy)
        q = Q_SCAN.copy()
        q[0] = base_angle  # rotate base to face block
        q[1] -= 0.2  # have shoulder point toward table a bit
        q[2] += 0.3  # Have elbow slightly bent down
        q[GRIPPER_JOINT_INDEX] = GRIPPER_MOTOR_OPEN_RAD  # will close in trajectory
        return q

    def _transition_to_move_to_nod(self):
        """Publish trajectory to hover above nod target and set arrival timer."""
        hover_q = self._compute_nod_joint_config(self._nod_target_cell)
        states = [
            TrajectoryState(
                final_state=ManipulatorState(q=hover_q, gripper_open=False),
                min_duration=self._IDLE_MOVE_DURATION,
            )
        ]
        self._publish_trajectory(states, TrajectoryCommandMsg.COMMAND_REPLACE)
        self._set_state(self.MOVE_TO_NOD, f'cell={self._nod_target_cell}')
        self._set_timer(self._IDLE_MOVE_DURATION, self._on_at_nod_pose)

    def _on_at_nod_pose(self):
        """Timer callback: arrived above block, start nod gesture."""
        self._cancel_all_timers()
        if self._state != self.MOVE_TO_NOD:
            return  # stale callback guard
        self._set_state(self.NODDING)
        self.logger.info("GestureManager: at nod pose, starting nod")
        hover_q = self._compute_nod_joint_config(self._nod_target_cell)
        burst = self._build_nod_trajectory(hover_q)
        self._publish_trajectory(burst, TrajectoryCommandMsg.COMMAND_REPLACE)
        # Timer for nod duration
        nod_duration = 3 * 2 * 0.8  # 3 cycles * 2 half-cycles * 0.8s each = 4.8s
        self._set_timer(nod_duration, self._on_nod_complete)

    def _build_nod_trajectory(self, hover_q):
        """Build 3-cycle wrist tilt nod at a fixed hover joint configuration.

        Alternates wrist_tilt (joint index 3) between +/- TILT_DELTA
        from the base hover position. Gripper stays closed throughout.
        """
        WRIST_TILT_INDEX = 3
        TILT_DELTA = 0.3  # ~25-26 degrees
        NOD_DURATION = 0.8  # seconds per half-cycle

        states = []
        base_tilt = hover_q[WRIST_TILT_INDEX]
        for _ in range(3):
            q_up = hover_q.copy()
            q_up[WRIST_TILT_INDEX] = base_tilt + TILT_DELTA
            states.append(TrajectoryState(
                final_state=ManipulatorState(q=q_up),
                min_duration=NOD_DURATION,
            ))
            q_down = hover_q.copy()
            q_down[WRIST_TILT_INDEX] = base_tilt - TILT_DELTA
            states.append(TrajectoryState(
                final_state=ManipulatorState(q=q_down),
                min_duration=NOD_DURATION,
            ))
        return states

    def _on_nod_complete(self):
        """Timer callback: nod finished, enter post-nod pause."""
        self._cancel_all_timers()
        if self._state != self.NODDING:
            return  # stale callback guard
        self._set_state(self.POST_NOD_PAUSE)
        self.logger.info("GestureManager: nod complete, pausing")
        # Record the nod attempt (only after full completion)
        if self._unreachable_tracker is not None and self._nod_target_cell is not None:
            self._unreachable_tracker.record_nod_attempt(self._nod_target_cell)
        # 1.5 second pause above the block
        self._set_timer(1.5, self._on_post_nod_pause_complete)

    def _on_post_nod_pause_complete(self):
        """Timer callback: post-nod pause done, decide next action."""
        self._cancel_all_timers()
        if self._state != self.POST_NOD_PAUSE:
            return  # stale callback guard
        self.logger.info(
            f"GestureManager: post-nod pause done, "
            f"has_pickable={self._has_pickable_block}"
        )
        self._nod_target_cell = None
        if self._on_nod_complete_callback is not None:
            self._on_nod_complete_callback(self._has_pickable_block)
        # After callback, if still in POST_NOD_PAUSE (callback didn't stop_idle),
        # cycle back to scan
        if self._state == self.POST_NOD_PAUSE:
            self._transition_to_move_to_scan()

    # ── Timer Management ─────────────────────────────────────────────

    def _set_timer(self, delay: float, callback: Callable) -> None:
        """Create a one-shot timer, cancelling any existing one first."""
        self._cancel_all_timers()
        self._timer = self._create_timer(delay, callback)

    def _cancel_all_timers(self) -> None:
        """Destroy the current timer if one exists."""
        if self._timer is not None:
            self._destroy_timer(self._timer)
            self._timer = None

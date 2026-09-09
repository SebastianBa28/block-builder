"""Trajectory state machine and waypoint queue for motor control.

Manages a deque of TrajectoryState waypoints, each representing
a timed spline segment with optional delays before and after.  The
Trajectory class sequences these segments, providing the 100Hz
control loop with interpolated joint commands via get_update().
"""

from rclpy.node import Node

import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from collections import deque

from legobuilder.kinematics.trajectory_utils import goto5, spline5, spline
from legobuilder.kinematics.kinematic_chain import KinematicChain
from legobuilder.config import (
    Q_READY, LAMBDA,
    GRIPPER_JOINT_INDEX, GRIPPER_MOTOR_CLOSED_RAD, GRIPPER_MOTOR_OPEN_RAD,
    GRIP_FAILURE_EFFORT_THRESHOLD,
)
from legobuilder.kinematics.block_manipulator import ManipulatorState


# ── TrajectoryState ──────────────────────────────────────────────────

@dataclass
class TrajectoryState:
    """Single waypoint in a trajectory queue.

    Represents one spline segment: the robot moves from its current
    configuration to final_state over min_duration seconds,
    with optional delays before and after the movement.

    Attributes
    ----------
    final_state : ManipulatorState
        Desired joint/task-space state at the end of this segment.
    min_duration : float or None
        Active movement duration in seconds.
    delay_before : float
        Hold time before movement begins (seconds).
    delay_after : float
        Hold time after movement completes (seconds).
    t_at_start : float or None
        Absolute time when this segment starts (set by the queue).
    t_at_end : float or None
        Absolute time when this segment ends (set by the queue).
    """

    final_state: ManipulatorState
    min_duration: float = None
    delay_before: float = 0.0
    delay_after: float = 0.0
    t_at_start: float = None
    t_at_end: float = None
    _q_init: np.ndarray = None
    _qd_init: np.ndarray = None


# ── Trajectory ───────────────────────────────────────────────────────

class Trajectory:
    """Trajectory queue that sequences waypoints for motor control.

    Maintains a deque of TrajectoryState segments and provides
    time-indexed joint commands via get_update().  Each segment
    goes through three phases: delay-before, active spline movement,
    and delay-after.

    Attributes
    ----------
    node : Node
        ROS node for logging.
    logger : rclpy.impl.rcutils_logger.RcutilsLogger
        Convenience logger reference.
    trajectory_states : deque[TrajectoryState]
        Queued waypoint segments.
    dt : float
        Control loop timestep (seconds).
    lambda_ : float
        Position error feedback gain (from config).
    q0 : np.ndarray
        Last known stable joint position (used as hold position
        when the queue is empty).
    qd0 : np.ndarray
        Last known stable joint velocity.
    pos_error : np.ndarray
        Accumulated 3D position error for feedback.
    clock : rclpy.clock.Clock
        ROS clock for time queries.
    start_time : rclpy.time.Time
        Reference time for get_t().
    """

    # ── Lifecycle ────────────────────────────────────────────────────

    def __init__(
        self,
        node: Node,
        q0: np.ndarray,
        dt: float,
        clock,
        start_time,
    ):
        """Initialize the trajectory queue.

        Arguments
        ---------
        node : Node
            ROS node for logging.
        q0 : np.ndarray
            Initial joint configuration (used as hold position).
        dt : float
            Control loop timestep in seconds.
        clock : rclpy.clock.Clock
            ROS clock for get_t() computation.
        start_time : rclpy.time.Time
            Reference start time.
        """
        self.node = node
        self.logger = node.get_logger()
        self.trajectory_states: deque[TrajectoryState] = deque([])
        self.dt = dt
        self.lambda_ = LAMBDA
        self.q0 = q0
        self.qd0 = np.zeros_like(q0)
        self.pos_error = np.zeros(3)
        self.clock = clock
        self.start_time = start_time

    # ── Public API ───────────────────────────────────────────────────

    def get_t(self):
        """Return elapsed time in seconds since start_time."""
        now = self.clock.now()
        return (now - self.start_time).nanoseconds * 1e-9

    def add_state(
        self,
        state: TrajectoryState,
        prioritize: bool = False,
    ):
        """Add a single trajectory state to the queue.

        Arguments
        ---------
        state : TrajectoryState
            Waypoint to add.
        prioritize : bool
            If True, insert at the front of the queue.
        """
        if not state:
            return

        if prioritize:
            self.trajectory_states.appendleft(state)
        else:
            self.trajectory_states.append(state)
        self._update_t_start_end(t=self.get_t())

    def add_states(
        self,
        states: list[TrajectoryState],
        prioritize: bool = False,
    ):
        """Add multiple trajectory states to the queue.

        Arguments
        ---------
        states : list[TrajectoryState]
            Waypoints to add.
        prioritize : bool
            If True, insert at the front (in original order).
        """
        iterator = states if not prioritize else reversed(states)
        for state in iterator:
            self.add_state(state, prioritize=prioritize)

    def clear_states(self, q_curr: np.ndarray, qd_curr: np.ndarray):
        """Clear all queued states and hold at q_curr.

        Arguments
        ---------
        q_curr : np.ndarray
            Current joint configuration to hold.
        """
        if len(self.trajectory_states) > 0:
            is_gripper_open = self.trajectory_states[0].final_state.gripper_open
        else:
            is_gripper_open = None
        self.trajectory_states = deque([])
        gripper_state = self.q0[GRIPPER_JOINT_INDEX]
        self.q0 = q_curr.copy()
        if is_gripper_open == False:
            self.q0[GRIPPER_JOINT_INDEX] = GRIPPER_MOTOR_CLOSED_RAD
        elif is_gripper_open == True:
            self.q0[GRIPPER_JOINT_INDEX] = q_curr[GRIPPER_JOINT_INDEX]
        else:
            self.q0[GRIPPER_JOINT_INDEX] = gripper_state
        self.qd0 = qd_curr
        self.logger.debug(
            "Cleared all trajectory states. Holding current position."
        )

    def check_grip_failure(self, actual_q: np.ndarray,
                           actual_tau: np.ndarray) -> bool:
        """Check if gripper should be holding an object but is not.

        Returns True when all of the following are true:
        1. The current segment commands gripper closed.
        2. The segment started with gripper already closed (holding,
           not in the process of closing).
        3. The gripper effort is above the threshold (close to 0),
           meaning no block resistance.

        Arguments
        ---------
        actual_q : np.ndarray
            Current actual joint positions from motor feedback.
        actual_tau : np.ndarray
            Current actual joint efforts from motor feedback.

        Returns
        -------
        bool
            True if a grip failure is detected.
        """
        if not self.trajectory_states:
            return False

        state = self.trajectory_states[0]
        if state.final_state.q is None:
            return False

        midpoint = (
            GRIPPER_MOTOR_OPEN_RAD + GRIPPER_MOTOR_CLOSED_RAD
        ) / 2

        commanded_gripper = state.final_state.q[GRIPPER_JOINT_INDEX]
        if commanded_gripper > midpoint:
            return False

        if state._q_init is None:
            return False
        if state._q_init[GRIPPER_JOINT_INDEX] > midpoint:
            return False

        actual_effort = actual_tau[GRIPPER_JOINT_INDEX]
        is_fail = actual_effort > GRIP_FAILURE_EFFORT_THRESHOLD

        # if is_fail:
        #     self.logger.info(f"grip failed. effort: {actual_effort:.2f} Nm")
        # else:
        #     self.logger.info(f"grip ok. effort: {actual_effort:.2f} Nm")

        return is_fail

    def get_update(
        self,
        t: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return interpolated joint commands for time t.

        Pops completed segments, then evaluates the current segment's
        phase (delay-before, active movement, or delay-after) and
        returns the appropriate (q, qd) pair.

        Arguments
        ---------
        t : float
            Current elapsed time in seconds.

        Returns
        -------
        q : np.ndarray
            Commanded joint positions.
        qd : np.ndarray
            Commanded joint velocities.
        """
        self._pop_completed_states(t)

        if not self.trajectory_states:
            return self.q0, np.zeros_like(self.q0)

        state_curr = self.trajectory_states[0]

        # ── Delay Before ─────────────────────────────────────────
        if t < state_curr.t_at_start + state_curr.delay_before:
            return self.q0, np.zeros_like(self.q0)

        self._ensure_q_init(state_curr)
        self._ensure_qd_init(state_curr)

        # ── Active Movement ──────────────────────────────────────
        if t <= state_curr.t_at_end - state_curr.delay_after:
            qd_final = (
                state_curr.final_state.qd
                if state_curr.final_state.qd is not None
                else np.zeros_like(self.q0)
            )

            q, qd = spline5(
                t - state_curr.t_at_start - state_curr.delay_before,
                state_curr.min_duration,
                state_curr._q_init,
                state_curr.final_state.q,
                state_curr._qd_init,
                qd_final,
                np.zeros_like(qd_final),
                np.zeros_like(qd_final),
            )
            if (t + self.dt
                    >= state_curr.t_at_end - state_curr.delay_after):
                self.q0 = state_curr.final_state.q
                self.qd0 = qd_final
            return q, qd

        # ── Delay After ──────────────────────────────────────────
        else:
            return self.q0, np.zeros_like(self.q0)

    # ── Private Helpers ──────────────────────────────────────────────

    def _update_t_start_end(self, t: float):
        """Recompute absolute start/end times for all queued states.

        Arguments
        ---------
        t : float
            Current elapsed time used as baseline for the first state.
        """
        if not self.trajectory_states:
            return

        initial_t = self.trajectory_states[0].t_at_start
        running_duration = (
            min(t, initial_t) if initial_t is not None else t
        )
        for s in self.trajectory_states:
            s.t_at_start = running_duration
            s.t_at_end = (
                running_duration + s.delay_before
                + s.min_duration + s.delay_after
            )
            running_duration = s.t_at_end

    def _pop_completed_states(self, t: float):
        """Remove completed states from the front of the queue.

        Updates q0 and qd0 from each popped state's final
        configuration so the next segment starts from the right place.

        Arguments
        ---------
        t : float
            Current elapsed time.
        """
        while (self.trajectory_states
               and t >= self.trajectory_states[0].t_at_end):
            popped = self.trajectory_states.popleft()
            self.q0 = (
                popped.final_state.q
                if popped.final_state.q is not None
                else self.q0
            )
            self.qd0 = (
                popped.final_state.qd
                if popped.final_state.qd is not None
                else np.zeros_like(self.q0)
            )
            self.logger.debug("Completed a trajectory state.")

    def _ensure_q_init(self, state: TrajectoryState):
        """Lazily set the initial joint config from q0 if not yet set."""
        if state._q_init is None:
            state._q_init = self.q0

    def _ensure_qd_init(self, state: TrajectoryState):
        """Lazily set the initial joint velocity from qd0 if not set."""
        if state._qd_init is None:
            state._qd_init = self.qd0

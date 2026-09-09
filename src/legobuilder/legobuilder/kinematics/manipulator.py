"""Low-level motor control ROS node operating at 100 Hz.

Receives trajectory commands from the brain, interpolates joint-space
splines via the Trajectory queue, applies gravity compensation,
detects collisions and grip failures, and publishes joint commands
to the hardware drivers.
"""

import rclpy
import numpy as np
import tf2_ros
import sys
from copy import deepcopy

from math import pi, sin, cos, acos, atan2, sqrt, fmod, exp

from asyncio import Future
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, TwistStamped
from geometry_msgs.msg import TransformStamped, Point, Pose
from sensor_msgs.msg import JointState
from std_msgs.msg import Header, Bool
from std_msgs.msg import Float32MultiArray

from legobuilder.kinematics.trajectory_utils import goto5
from legobuilder.kinematics.trajectory import Trajectory, TrajectoryState
from legobuilder.kinematics.gravity import WeightModel
from legobuilder.config import (
    MANIPULATOR_RATE as RATE,
    Q_READY,
    NUM_DOFS,
    JOINT_NAMES,
    DURATION,
    OUTER_RADIUS,
    INNER_RADIUS,
    BASE_MOTOR_POS,
    COLLISION_DETECTION,
    POSITION_ERROR_THRESHOLD,
    COLLISION_WAIT_DURATION,
    VELOCITY_ERROR_THRESHOLD,
    EFFORT_ERROR_THRESHOLD,
    STARTUP_GRAV_DURATION,
    TEST_GRAVITY,
    TEST_GRIPPER,
    TEST_GRIPPER_EFFORT,
    ARC_HEIGHT,
    ARC_SPEED,
    HEARTBEAT_RATE,
    HEARTBEAT_TIMEOUT,
    DISK_DIMS,
    STRIP_DIMS,
    GRIPPER_MOTOR_CLOSED_RAD,
    GRIPPER_MOTOR_OPEN_RAD,
    GRIPPER_PRISMATIC_CLOSED_M,
    GRIPPER_PRISMATIC_OPEN_M,
    GRIPPER_JOINT_INDEX,
    gripper_rad_to_meters,
    JERK_PREVENTION,
    JERK_THRESHOLD_RAD,
    JERK_IGNORE_COUNT,
    JERK_RECOVERY_DURATION,
    LOOP_GAP_RECOVERY,
    GRIP_FAILURE_CONSECUTIVE_FRAMES,
)

from legobuilder_interfaces.msg import (
    TrajectoryStateMsg,
    TrajectoryCommandMsg,
    TrajectoryCompleteMsg,
    ContactMsg,
    GripFailureMsg,
)

from legobuilder.kinematics.block_manipulator import ManipulatorState


class ManipulatorNode(Node):
    """Low-level motor control node for the 6-DOF robot arm.

    Runs at 100 Hz.  Responsibilities:
    - Execute trajectory waypoints via quintic spline interpolation
    - Apply gravity compensation torques (ramped up during startup)
    - Detect collisions via position/velocity/effort error thresholds
    - Detect grip failures (gripper fully closed with no object)
    - Publish trajectory completion, contact, and grip failure events
    - Maintain heartbeat with brain and detector nodes
    - Republish joint states with gripper in prismatic meters for RViz

    Attributes
    ----------
    future : asyncio.Future
        Future object that signals when the node should shut down.
    start_time : rclpy.time.Time
        Node start time for elapsed-time computation.
    logger : rclpy.impl.rcutils_logger.RcutilsLogger
        Convenience logger reference.
    q : np.ndarray
        Current actual joint positions from motor feedback.
    qd : np.ndarray
        Current actual joint velocities.
    tau : np.ndarray
        Current actual joint efforts.
    qc : np.ndarray or None
        Last commanded joint positions.
    qdc : np.ndarray or None
        Last commanded joint velocities.
    tauc : np.ndarray or None
        Last commanded joint efforts (gravity torques).
    trajectory : Trajectory
        Trajectory queue managing waypoint interpolation.
    weight_model : WeightModel
        Gravity compensation model.
    executing_trajectory : bool
        True while the trajectory queue has pending states.
    was_executing : bool
        Tracks state transitions for completion detection.
    states_executed : int
        Counter for trajectory states completed.
    is_brain_online : bool
        Whether the brain node is sending heartbeats.
    is_detector_online : bool
        Whether the detector node is sending heartbeats.
    dt : float
        Control loop timestep (1 / RATE).
    """

    # ── Lifecycle ────────────────────────────────────────────────────

    def __init__(self, name, future):
        """Initialize the manipulator node.

        Reads the initial joint positions, creates the trajectory
        queue with a startup sequence (lift shoulder then go to
        Q_READY), sets up all publishers/subscribers, and starts
        the 100 Hz control timer.

        Arguments
        ---------
        name : str
            ROS node name.
        future : asyncio.Future
            Future to signal shutdown.
        """
        super().__init__(name)
        self.future = future
        self.start_time = self.get_clock().now()
        self.logger = self.get_logger()

        # Grab initial joint position
        self.q0 = self.grabfbk()
        # self.logger.debug("Initial positions: %r" % self.q0)

        # ── State Variables ──────────────────────────────────────
        self.q = self.q0
        self.qd = np.zeros_like(self.q0)
        self.tau = np.zeros_like(self.q0)

        self.qc = None
        self.qdc = None
        self.tauc = None

        # self.logger.debug(f"self.q0: {self.q0}")

        self.trajectory = Trajectory(
            self, self.q0,
            dt=1 / RATE,
            clock=self.get_clock(),
            start_time=self.start_time,
        )
        self.trajectory.add_states([
            TrajectoryState(
                min_duration=4,
                final_state=ManipulatorState(q=np.array([
                    self.q0[0], 0, self.q0[2],
                    self.q0[3], self.q0[4],
                    GRIPPER_MOTOR_CLOSED_RAD,
                ])),
                delay_before=STARTUP_GRAV_DURATION,
            ),
            TrajectoryState(
                min_duration=4,
                final_state=ManipulatorState(q=Q_READY.copy()),
            ),
        ])

        self.weight_model = WeightModel()

        self.executing_trajectory = False
        self.was_executing = False
        self._last_update_t = None

        # ── Jerk prevention state ────────────────────────────────
        self._jerk_bad_count = 0
        self._jerk_latest_target = None

        # ── Publishers ───────────────────────────────────────────
        self.pubcmd = self.create_publisher(
            JointState, '/joint_commands', 10,
        )
        self.pub_joint_states_viz = self.create_publisher(
            JointState, '/joint_states_viz', 10,
        )
        self.pub_trajectory_complete = self.create_publisher(
            TrajectoryCompleteMsg,
            'manipulator/trajectory_complete', 10,
        )
        self.pub_contact = self.create_publisher(
            ContactMsg, 'manipulator/contact', 10,
        )
        self.pub_grip_failure = self.create_publisher(
            GripFailureMsg, 'manipulator/grip_failure', 10,
        )

        self.states_executed = 0
        self._grip_failure_count: int = 0
        self._grip_failure_published: bool = False

        # ── Heartbeats ───────────────────────────────────────────
        self.is_brain_online = False
        self.is_detector_online = False
        self.brain_last_seen = None
        self.detector_last_seen = None

        self.pub_heartbeat = self.create_publisher(
            Bool, name + '/heartbeat', 10,
        )
        self.create_subscription(
            Bool, 'brain/heartbeat',
            self.recv_brain_heartbeat, 10,
        )
        self.create_subscription(
            Bool, 'detector/heartbeat',
            self.recv_detector_heartbeat, 10,
        )
        self.create_timer(
            1.0 / HEARTBEAT_RATE, self.heartbeat_tick,
        )

        # ── Subscribers ──────────────────────────────────────────
        self.q = self.q0.copy()
        self.create_subscription(
            JointState, '/joint_states', self.recvact, 10,
        )
        self.create_subscription(
            TrajectoryCommandMsg,
            'brain/trajectory_command',
            self.recv_trajectory_command, 10,
        )

        self.logger.info(
            "Waiting for a /joint_commands subscriber..."
        )
        while not self.count_subscribers('/joint_commands'):
            pass

        # ── Control Timer ────────────────────────────────────────
        self.dt = 1 / RATE
        self.timer = self.create_timer(self.dt, self.update)
        self.logger.info(
            "Sending commands with dt of %f seconds (%fHz)"
            % (self.timer.timer_period_ns * 1e-9, RATE)
        )

    def shutdown(self):
        """Destroy the control timer and the node."""
        self.timer.destroy()
        self.destroy_node()

    # ── Public API ───────────────────────────────────────────────────

    def grabfbk(self):
        """Grab a single joint-state feedback message.

        Creates a temporary subscription, blocks until one message
        arrives, then destroys the subscription.  Do NOT call this
        repeatedly in a loop.

        Returns
        -------
        list[float]
            Joint positions from the first received message.
        """
        def cb(fbkmsg):
            self.grabpos = list(fbkmsg.position)
            self.grabready = True

        sub = self.create_subscription(
            JointState, '/joint_states', cb, 1,
        )
        self.grabready = False
        while not self.grabready:
            rclpy.spin_once(self)
        self.destroy_subscription(sub)

        return self.grabpos

    def sendcmd(self, pos, vel, eff=[]):
        """Publish a joint command message.

        Arguments
        ---------
        pos : array-like
            Commanded joint positions.
        vel : array-like
            Commanded joint velocities.
        eff : array-like
            Commanded joint efforts (gravity torques).
        """
        cmdmsg = JointState()
        cmdmsg.header.stamp = self.get_clock().now().to_msg()
        cmdmsg.header.frame_id = 'world'
        cmdmsg.name = JOINT_NAMES
        cmdmsg.position = [float(v) for v in pos]
        cmdmsg.velocity = [float(v) for v in vel]
        cmdmsg.effort = [float(v) for v in eff]
        self.pubcmd.publish(cmdmsg)
        self.logger.debug(
            f"[PUB /joint_commands] "
            f"pos={[round(v, 4) for v in pos]} "
            f"vel={[round(v, 4) for v in vel]} "
            f"eff={[round(v, 4) for v in eff]}"
        )

    def get_t(self):
        """Return elapsed time in seconds since node start."""
        now = self.get_clock().now()
        return (now - self.start_time).nanoseconds * 1e-9

    def publish_trajectory_complete(self):
        """Publish trajectory completion event to the brain."""
        msg = TrajectoryCompleteMsg()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'world'
        msg.states_executed = self.states_executed
        self.pub_trajectory_complete.publish(msg)
        self.logger.debug(
            f"Published trajectory complete: "
            f"{self.states_executed} states executed"
        )

    def publish_contact(
        self, contact_type: int = ContactMsg.CONTACT_COLLISION
    ):
        """Publish a contact/collision event to the brain.

        Arguments
        ---------
        contact_type : int
            One of the ContactMsg constants (e.g. CONTACT_COLLISION).
        """
        msg = ContactMsg()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'world'
        msg.contact_type = contact_type

        if self.qc is not None:
            msg.position_error = list(self.qc - self.q)
        else:
            msg.position_error = [0.0] * len(JOINT_NAMES)

        if self.qdc is not None:
            msg.velocity_error = list(self.qdc - self.qd)
        else:
            msg.velocity_error = [0.0] * len(JOINT_NAMES)

        if self.tauc is not None:
            msg.effort_error = list(self.tauc - self.tau)
        else:
            msg.effort_error = [0.0] * len(JOINT_NAMES)

        msg.q_actual = list(self.q)
        msg.qd_actual = list(self.qd)
        msg.states_remaining = len(
            self.trajectory.trajectory_states
        )

        self.pub_contact.publish(msg)
        self.logger.info(
            f"Published contact: type={contact_type}"
        )

    def publish_grip_failure(self):
        """Publish a grip failure event to the brain."""
        msg = GripFailureMsg()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'world'
        msg.gripper_position = float(
            self.q[GRIPPER_JOINT_INDEX]
        )
        msg.states_remaining = len(
            self.trajectory.trajectory_states
        )
        self.pub_grip_failure.publish(msg)
        self.logger.debug(
            f"Published grip failure: "
            f"gripper_pos={msg.gripper_position:.4f} "
            f"states_remaining={msg.states_remaining}"
        )

    # ── Callbacks ────────────────────────────────────────────────────

    def recv_brain_heartbeat(self, msg: Bool):
        """Update brain online status from heartbeat message."""
        was_online = self.is_brain_online
        self.is_brain_online = True
        self.brain_last_seen = self.get_t()
        self.logger.debug(
            f"[RECV brain/heartbeat] t={self.brain_last_seen:.4f}"
        )
        if not was_online:
            self.logger.info("Brain is now online")

    def recv_detector_heartbeat(self, msg: Bool):
        """Update detector online status from heartbeat message."""
        was_online = self.is_detector_online
        self.is_detector_online = True
        self.detector_last_seen = self.get_t()
        self.logger.debug(
            f"[RECV detector/heartbeat] "
            f"t={self.detector_last_seen:.4f}"
        )
        if not was_online:
            self.logger.info("Detector is now online")

    def heartbeat_tick(self):
        """Publish heartbeat and check for timed-out nodes."""
        t = self.get_t()

        self.pub_heartbeat.publish(Bool(data=True))
        self.logger.debug(
            f"[PUB manipulator/heartbeat] t={t:.4f}"
        )

        # Check brain timeout
        if (self.is_brain_online
                and self.brain_last_seen is not None):
            if t - self.brain_last_seen > HEARTBEAT_TIMEOUT:
                self.logger.warn("Brain went offline")
                self.is_brain_online = False

        # Check detector timeout
        if (self.is_detector_online
                and self.detector_last_seen is not None):
            if t - self.detector_last_seen > HEARTBEAT_TIMEOUT:
                self.logger.warn("Detector went offline")
                self.is_detector_online = False

    def recvact(self, msg):
        """Receive actual joint states from hardware.

        Saves positions, velocities, and efforts for collision
        detection.  Also republishes with gripper converted to
        prismatic meters for RViz visualization.

        Arguments
        ---------
        msg : JointState
            Hardware joint state feedback.
        """
        self.logger.debug(
            f"[RECV /joint_states] "
            f"pos={[round(v, 4) for v in msg.position]} "
            f"vel={[round(v, 4) for v in msg.velocity]}"
        )
        self.q = np.array(msg.position)
        self.qd = np.array(msg.velocity)
        self.tau = np.array(msg.effort)
        # self.logger.info(
        #     f"Received joint states: q={self.q}, "
        #     f"qd={self.qd}, tau={self.tau}"
        # )

        viz_msg = JointState()
        viz_msg.header = msg.header
        viz_msg.name = list(msg.name)
        viz_msg.position = list(msg.position)
        viz_msg.velocity = list(msg.velocity)
        viz_msg.effort = list(msg.effort)
        if len(viz_msg.position) > GRIPPER_JOINT_INDEX:
            viz_msg.position[GRIPPER_JOINT_INDEX] = (
                gripper_rad_to_meters(
                    msg.position[GRIPPER_JOINT_INDEX]
                )
            )
        self.pub_joint_states_viz.publish(viz_msg)
        self.logger.debug(
            f"[PUB /joint_states_viz] "
            f"pos={[round(v, 4) for v in viz_msg.position]}"
        )

    def recv_trajectory_command(self, msg: TrajectoryCommandMsg):
        """Receive a trajectory command from the brain.

        Supports three command types:
        - COMMAND_REPLACE: clear queue and add new states
        - COMMAND_APPEND_FRONT: insert states at front of queue
        - COMMAND_APPEND_BACK: append states to end of queue

        Arguments
        ---------
        msg : TrajectoryCommandMsg
            Command containing an array of TrajectoryStateMsg.
        """
        if not msg.states:
            self.logger.warn(
                "Received empty trajectory command"
            )
            return

        # Convert messages to TrajectoryState objects
        states_to_add = []
        for state_msg in msg.states:
            state = TrajectoryState(
                final_state=ManipulatorState(
                    q=np.array(state_msg.q),
                    qd=np.array(state_msg.qd)
                ),
                min_duration=state_msg.min_duration,
                delay_before=state_msg.delay_before,
                delay_after=state_msg.delay_after,
            )
            states_to_add.append(state)
            self.logger.debug(
                f"  state: "
                f"q={[round(v, 4) for v in state_msg.q]} "
                f"dur={state_msg.min_duration} "
                f"delay_before={state_msg.delay_before} "
                f"delay_after={state_msg.delay_after}"
            )

        if msg.command == TrajectoryCommandMsg.COMMAND_REPLACE:
            self.logger.debug(
                f"REPLACE: Clearing queue, "
                f"adding {len(states_to_add)} states"
            )
            self.trajectory.clear_states(self.q, self.qd)
            self.states_executed = 0
            self.trajectory.add_states(
                states_to_add, prioritize=False,
            )

        elif (msg.command \
              == TrajectoryCommandMsg.COMMAND_APPEND_FRONT):
            self.logger.debug(
                f"APPEND_FRONT: Adding "
                f"{len(states_to_add)} states to front"
            )
            self.trajectory.add_states(
                states_to_add, prioritize=True,
            )

        elif (msg.command \
              == TrajectoryCommandMsg.COMMAND_APPEND_BACK):
            self.logger.debug(
                f"APPEND_BACK: Adding "
                f"{len(states_to_add)} states to back"
            )
            self.trajectory.add_states(
                states_to_add, prioritize=False,
            )

        else:
            self.logger.error(
                f"Unknown command type: {msg.command}"
            )
            return

        self.executing_trajectory = True
        self.was_executing = True

    # ── Private Helpers ──────────────────────────────────────────────

    def gravity(self, q: np.ndarray):
        """Compute gravity compensation torques for joint config q.

        Ramps torques up linearly during STARTUP_GRAV_DURATION to
        prevent sudden jolts on startup.

        Arguments
        ---------
        q : np.ndarray
            Current joint positions.

        Returns
        -------
        np.ndarray
            Gravity compensation torques.
        """
        tau = self.weight_model.compute_gravity_torques(q)
        t = self.get_t()
        # Ramp up over startup duration to avoid jolts
        if t < STARTUP_GRAV_DURATION:
            scale = t / STARTUP_GRAV_DURATION
            tau *= scale
        return tau

    def collision_detection(self, t: float):
        """Check for collisions via position/velocity/effort errors.

        Uses dynamic thresholds that scale with joint velocity to
        reduce false positives during fast movements.  On detection,
        clears the trajectory and either notifies the brain or
        performs local recovery.

        Arguments
        ---------
        t : float
            Current elapsed time.
        """
        def _get_error_thresholds():
            """Compute velocity-scaled error thresholds."""
            # return (POSITION_ERROR_THRESHOLD,
            #         VELOCITY_ERROR_THRESHOLD,
            #         EFFORT_ERROR_THRESHOLD)

            vel_norm = np.linalg.norm(self.qd)
            dynamic_position = (
                POSITION_ERROR_THRESHOLD + 0.2 * vel_norm
            )
            dynamic_velocity = (
                VELOCITY_ERROR_THRESHOLD + 0.3 * vel_norm
            )
            dynamic_effort = (
                EFFORT_ERROR_THRESHOLD + 0.75 * vel_norm
            )
            return (
                dynamic_position,
                dynamic_velocity,
                dynamic_effort,
            )

        if (COLLISION_DETECTION
                and self.qc is not None
                and self.qdc is not None
                and self.tauc is not None):
            p_th, v_th, e_th = _get_error_thresholds()
            pos_error = (
                np.linalg.norm(self.qc - self.q) > p_th
            )
            vel_error = (
                np.linalg.norm(self.qdc - self.qd) > v_th
            )
            eff_error = (
                np.linalg.norm(self.tauc - self.tau) > e_th
            )
            # eff_error = False

            # self.logger.info(
            #     f"Position Error: "
            #     f"{np.linalg.norm(self.qc - self.q):.2f} "
            #     f"(th={p_th:.2f}), "
            #     f"Velocity Error: "
            #     f"{np.linalg.norm(self.qdc - self.qd):.2f} "
            #     f"(th={v_th:.2f}), "
            #     f"Effort Error: "
            #     f"{np.linalg.norm(self.tauc - self.tau):.2f} "
            #     f"(th={e_th:.2f})"
            # )
            if pos_error or vel_error or eff_error:
                if pos_error:
                    self.logger.info(
                        "Position error: "
                        f"{np.linalg.norm(self.qc - self.q)}"
                    )
                elif vel_error:
                    self.logger.info(
                        "Velocity error: "
                        f"{np.linalg.norm(self.qdc - self.qd)}"
                    )
                else:
                    self.logger.info(
                        "Effort error: "
                        f"{np.linalg.norm(self.tauc - self.tau)}"
                    )
                self.logger.info(
                    "Collision detected! Clearing trajectory."
                )
                self.trajectory.clear_states(self.q, self.qd)

                if self.is_brain_online:
                    self.publish_contact(
                        ContactMsg.CONTACT_COLLISION
                    )
                else:
                    self.logger.info(
                        "Brain offline, adding recovery "
                        "trajectory locally"
                    )
                    self.trajectory.add_state(
                        TrajectoryState(
                            final_state=ManipulatorState(
                                q=Q_READY.copy()
                            ),
                            delay_before=COLLISION_WAIT_DURATION,
                            min_duration=DURATION,
                        ),
                    )

                self.executing_trajectory = False
                self.was_executing = False

    def update(self):
        """100 Hz control loop tick.

        Detects loop gaps, pops completed trajectory states,
        runs collision detection, interpolates the next joint
        command, checks for grip failure, applies gravity
        compensation, and publishes the final command.
        """
        t = self.get_t()

        # Loop gap detection and recovery
        if self._last_update_t is not None:
            loop_dt = t - self._last_update_t
            if loop_dt > 0.050:  # expected ~0.01 s
                self.logger.debug(
                    f"[LOOP GAP] update() dt={loop_dt:.4f}s "
                    f"(expected ~0.01s)"
                )
                
                if LOOP_GAP_RECOVERY:
                    # Reset trajectory to restart from actual position
                    if self.trajectory.trajectory_states:
                        state = self.trajectory.trajectory_states[0]
                        state._q_init = self.q.copy()
                        state._q_init[-1] = state.final_state.q[-1]  # preserve gripper command
                        state._qd_init = np.zeros_like(self.q)
                        state.t_at_start = None

                        self.trajectory.q0 = self.q.copy()
                        self.trajectory.qd0 = np.zeros_like(self.q)
                        self.trajectory._update_t_start_end(t)

                        self.logger.warn(
                            "[LOOP GAP RECOVERY] Reset trajectory "
                            f"from actual q={[round(v, 3) for v in self.q]}"
                        )

                    # Hold at actual position this tick
                    tau = self.gravity(self.q)
                    self.sendcmd(
                        self.q, np.zeros_like(self.q), tau,
                    )
                    self._last_update_t = t
                    return
        self._last_update_t = t

        # Trajectory completion detection
        if len(self.trajectory.trajectory_states) == 0:
            if self.was_executing:
                self.publish_trajectory_complete()
                self.was_executing = False
            self.executing_trajectory = False

        queue_len_before = len(
            self.trajectory.trajectory_states
        )

        self.collision_detection(t)

        q, qd = self.trajectory.get_update(t)

        # ── Jerk prevention harness ──────────────────────────────
        if JERK_PREVENTION and len(q) > 0:
            cmd_delta = np.linalg.norm(q - self.q)
            if cmd_delta > JERK_THRESHOLD_RAD:
                self._jerk_bad_count += 1
                self._jerk_latest_target = q.copy()
                self.logger.info(
                    f"[JERK PREVENTION] Suppressed command "
                    f"(delta={cmd_delta:.3f} rad, "
                    f"streak={self._jerk_bad_count}/"
                    f"{JERK_IGNORE_COUNT})"
                )

                if self._jerk_bad_count >= JERK_IGNORE_COUNT:
                    self.logger.warn(
                        "[JERK PREVENTION] Creating smooth "
                        "recovery trajectory to q="
                        f"{[round(v,3) for v in self._jerk_latest_target]}"
                    )
                    q_state = self._jerk_latest_target.copy()
                    if self.trajectory.trajectory_states:
                        # Preserve gripper command to hold blocks securely during recovery
                        q_state[GRIPPER_JOINT_INDEX] = self.trajectory.trajectory_states[0].final_state.q[GRIPPER_JOINT_INDEX] 
                    self.trajectory.clear_states(self.q, self.qd)
                    self.trajectory.add_state(
                        TrajectoryState(
                            final_state=ManipulatorState(
                                q=q_state
                            ),
                            min_duration=JERK_RECOVERY_DURATION,
                        ),
                    )
                    self._jerk_bad_count = 0
                    self._jerk_latest_target = None

                # Hold at actual position this tick
                tau = self.gravity(self.q)
                self.sendcmd(
                    self.q, np.zeros_like(self.q), tau,
                )
                self._last_update_t = t
                return
            else:
                self._jerk_bad_count = 0
                self._jerk_latest_target = None

        # Grip failure check (debounced)
        if self.trajectory.check_grip_failure(self.q, self.tau):
            self._grip_failure_count += 1
            # self.logger.info(f"_grip_failure_count: {self._grip_failure_count}")
            if (self._grip_failure_count >= GRIP_FAILURE_CONSECUTIVE_FRAMES
                    and not self._grip_failure_published):
                self.publish_grip_failure()
                self._grip_failure_published = True
                # self.logger.info(f"publishing grip failure")
        else:
            self._grip_failure_count = 0
            self._grip_failure_published = False

        # Count completed states
        queue_len_after = len(
            self.trajectory.trajectory_states
        )
        if queue_len_after < queue_len_before:
            self.states_executed += (
                queue_len_before - queue_len_after
            )

        tau = self.gravity(self.q)

        self.qc = q if len(q) > 0 else None
        self.qdc = qd if len(qd) > 0 else None
        self.tauc = tau if len(tau) > 0 else None

        # Force gripper to be closed
        # q[-1] = -0.5

        # Limp arm for gravity testing
        if TEST_GRAVITY:
            q = np.array([])
            qd = np.array([])

        # Limp arm for gripper testing, hold wrist roll at 0
        if TEST_GRIPPER:
            q = np.full(NUM_DOFS, np.nan)
            qd = np.full(NUM_DOFS, np.nan)
            roll_idx = JOINT_NAMES.index("wrist_roll")
            q[roll_idx] = 0.0
            qd[roll_idx] = 0.0
            tau[GRIPPER_JOINT_INDEX] = TEST_GRIPPER_EFFORT
        self.sendcmd(q, qd, tau)
        self.logger.debug(
            f"[UPDATE t={t:.4f}] "
            f"q_cmd={[round(v, 4) for v in q]} "
            f"queue_len="
            f"{len(self.trajectory.trajectory_states)} "
            f"executing={self.executing_trajectory}"
        )
        
        # result_array = self.q - self.qc
        # formatted_array_str = np.array2string(result_array, precision=2, floatmode='fixed')
        # self.logger.info(f"q_act - q_cmd: {formatted_array_str}")



# ── Entry Point ──────────────────────────────────────────────────────

def main(args=None):
    """Start the manipulator control node."""
    rclpy.init(args=args)

    future = Future()

    node = ManipulatorNode('manipulator', future)

    rclpy.spin_until_future_complete(node, future)

    if future.done():
        node.logger.info("Stopping: " + future.result())
    else:
        node.logger.info("Stopping: Interrupted")

    node.shutdown()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

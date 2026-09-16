"""Async IK solver ROS node for the legobuilder arm.

Runs in a separate process so that IK computation does not block
the brain's ROS executor (which would starve perception callbacks).
Supports preemption: a PRIMARY request aborts any in-progress
PRECOMPUTE solve.
"""

import threading
import csv
import rclpy
import numpy as np
from rclpy.node import Node
import os
from std_msgs.msg import Bool, Empty

from legobuilder_interfaces.msg import IKRequestMsg, IKResponseMsg

from legobuilder.kinematics.kinematic_chain import KinematicChain
from legobuilder.kinematics.block_manipulator import (
    BlockManipulator, ManipulatorState,
)
from legobuilder.config import (
    JOINT_NAMES,
    HEARTBEAT_RATE,
    NUM_DOFS,
    USE_ERROR_MAP,
    BASE_MOTOR_POS,
    CALIBRATE_Z,
    CALIBRATE_XY,
    USE_EXISTING_XY_CALIBRATIONS,
    USE_EXISTING_Z_CALIBRATIONS,
)


# ── IKSolverNode ─────────────────────────────────────────────────────

class IKSolverNode(Node):
    """ROS node that solves IK requests on a background thread.

    Subscribes to brain/ik_request and publishes solved joint
    angles on ik_solver/ik_response.  Each solve runs in a
    daemon thread so the ROS executor remains responsive.  PRIMARY
    requests preempt any in-progress PRECOMPUTE via an abort event.

    Attributes
    ----------
    block_manipulator : BlockManipulator
        Manipulator wrapper used for IK solves.
    sub_request : Subscription
        Subscription to IKRequestMsg.
    pub_response : Publisher
        Publisher for IKResponseMsg.
    pub_heartbeat : Publisher
        Heartbeat publisher (Bool at HEARTBEAT_RATE Hz).
    offsets : dict[tuple[float, float], tuple[float, float, float]]
        Calibration error map loaded from offsets*.csv.
    """

    # ── Lifecycle ────────────────────────────────────────────────────

    def __init__(self):
        """Initialize the IK solver node and kinematic chain."""
        super().__init__('ik_solver')
        self.logger = self.get_logger()

        tip_chain = KinematicChain(self, "world", "tip", JOINT_NAMES)
        self.block_manipulator = BlockManipulator(tip_chain)

        # ── IK request / response ────────────────────────────────
        self.sub_request = self.create_subscription(
            IKRequestMsg, 'brain/ik_request',
            self.recv_ik_request, 10,
        )
        self.pub_response = self.create_publisher(
            IKResponseMsg, 'ik_solver/ik_response', 10,
        )

        # Preemption: abort flag + serialization lock
        self._abort_event = threading.Event()
        self._solve_lock = threading.Lock()

        # ── Heartbeat ────────────────────────────────────────────
        self.pub_heartbeat = self.create_publisher(
            Bool, 'ik_solver/heartbeat', 10,
        )
        self.create_timer(
            1.0 / HEARTBEAT_RATE, self._heartbeat_tick,
        )

        self.logger.info("IK Solver node initialized and running...")

        self.offsets = {}
        self.get_offsets()          # load default XY offsets from offsets_default.csv
        if not CALIBRATE_XY and USE_EXISTING_XY_CALIBRATIONS:
            self.get_offsets(path=self.XY_CALIBRATION_OUTPUT_PATH)

        self.table_heights = {}
        if not CALIBRATE_Z and USE_EXISTING_Z_CALIBRATIONS:
            self.load_z_offsets()

        if CALIBRATE_Z:
            self.create_subscription(
                Empty, 'brain/z_calibration_ready',
                self._recv_z_calibration_ready, 10,
            )
        if CALIBRATE_XY:
            self.create_subscription(
                Empty, 'brain/xy_calibration_ready',
                self._recv_xy_calibration_ready, 10,
            )

    # ── Public API ───────────────────────────────────────────────────

    XY_CALIBRATION_PATH = 'src/legobuilder/offsets/offsets_default.csv'
    XY_CALIBRATION_OUTPUT_PATH = 'src/legobuilder/offsets/xy_calibrations.csv'

    def get_offsets(self, path=None):
        """Load calibration error offsets from offsets CSV.

        Populates self.offsets with a mapping from (x, y) target
        positions to (offset_x, offset_y, offset_z) corrections.
        """
        path = path or self.XY_CALIBRATION_PATH
        if not os.path.exists(path):
            self.logger.info(
                f"[IK] No xy-calibration file at {path}"
            )
            return
        with open(path, 'r') as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if i == 0:
                    continue
                x, y, z, placed_x, placed_y, placed_z = map(float, row)
                offset_x = x - placed_x
                offset_y = y - placed_y
                offset_z = z - placed_z
                self.offsets[(x, y)] = (offset_x, offset_y, offset_z)

    Z_CALIBRATION_PATH = 'src/legobuilder/offsets/z_calibration.csv'

    def load_z_offsets(self):
        """Load z-calibration offsets from CSV written by Calibrator."""
        if not os.path.exists(self.Z_CALIBRATION_PATH):
            self.logger.info(
                f"[IK] No z-calibration file at {self.Z_CALIBRATION_PATH}"
            )
            return
        self.table_heights = {}
        with open(self.Z_CALIBRATION_PATH, 'r') as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if i == 0:
                    continue
                if len(row) < 3:
                    self.logger.warn(
                        f"[IK] Skipping malformed row {i} in z-calibration CSV"
                    )
                    continue
                r, z, table_height = float(row[0]), float(row[1]), float(row[2])
                self.table_heights[(r, z)] = table_height
        self.logger.info(
            f"[IK] Loaded {len(self.table_heights)} z-calibration entries"
        )

    def _recv_z_calibration_ready(self, msg):
        """Reload z-calibration offsets when signaled by brain."""
        self.logger.info("[IK] Received z_calibration_ready — reloading offsets")
        self.load_z_offsets()

    def _recv_xy_calibration_ready(self, msg):
        """Reload xy-calibration offsets when signaled by brain."""
        self.logger.info("[IK] Received xy_calibration_ready — reloading offsets")
        self.get_offsets(path=self.XY_CALIBRATION_OUTPUT_PATH)

    def apply_z_offsets(self, p):
        """Apply distance-weighted z offsets based on (r, z) proximity.

        table_heights maps (r, z) to the measured table height at that
        position.  The correction subtracts the interpolated table height
        so that z=0 corresponds to the true table surface.
        """
        x, y, z = p

        # Compute radial distance from base motor
        rel_x = x - BASE_MOTOR_POS[0]
        rel_y = y - BASE_MOTOR_POS[1]
        r = np.sqrt(rel_x**2 + rel_y**2)

        # Inverse-distance-weighted interpolation over calibration points
        distances = {}
        sum_weights = 0
        for (r_cal, z_cal) in self.table_heights:
            dist = np.sqrt((r - r_cal)**2 + (z - z_cal)**2)
            if dist < 1e-8:
                return np.array([x, y, z - self.table_heights[(r_cal, z_cal)]])
            distances[(r_cal, z_cal)] = dist
            sum_weights += 1 / dist

        z_adj = z
        for key, dist in distances.items():
            weight = (1 / dist) / sum_weights if sum_weights > 0 else 0
            z_adj -= weight * self.table_heights[key]

        return np.array([x, y, z_adj])



    def apply_offsets(self, p):
        """Apply distance-weighted calibration offsets to position p.

        Computes an inverse-distance-weighted average of all known
        offsets and adds the result to the input position.

        Arguments
        ---------
        p : np.ndarray
            Target position [x, y, z].

        Returns
        -------
        np.ndarray
            Adjusted position [x_adj, y_adj, z_adj].
        """
        x, y, z = p
        distances = {}
        sum_weights = 0
        for (ox, oy), _ in self.offsets.items():
            dist = np.sqrt((x - ox) ** 2 + (y - oy) ** 2)
            if dist < 1e-8:
                ox_x, ox_y, ox_z = self.offsets[(ox, oy)]
                return np.array([x + ox_x, y + ox_y, z + ox_z])
            distances[(ox, oy)] = dist
            sum_weights += 1 / dist

        x_adj = x.copy()
        y_adj = y.copy()
        z_adj = z.copy()
        for (ox, oy), dist in distances.items():
            weight = (
                1 / dist / sum_weights if sum_weights > 0 else 0
            )
            offset_x, offset_y, offset_z = self.offsets[(ox, oy)]
            x_adj += weight * offset_x
            y_adj += weight * offset_y
            z_adj += weight * offset_z
        return np.array([x_adj, y_adj, z_adj])

    # ── Callbacks ────────────────────────────────────────────────────

    def recv_ik_request(self, msg: IKRequestMsg):
        """Handle an incoming IK request.

        PRIMARY requests set the abort event to preempt any
        in-progress PRECOMPUTE.  The actual solve runs in a
        background daemon thread.

        Arguments
        ---------
        msg : IKRequestMsg
            Request containing waypoints, target positions, tilts,
            rolls, and gripper states.
        """
        req_type_str = (
            "PRIMARY"
            if msg.request_type == IKRequestMsg.REQUEST_PRIMARY
            else "PRECOMPUTE"
        )
        self.logger.debug(
            f"[RECV ik_request] id={msg.request_id} "
            f"type={req_type_str} "
            f"waypoints={len(msg.needs_ik)}"
        )

        if msg.request_type == IKRequestMsg.REQUEST_PRIMARY:
            self._abort_event.set()

        threading.Thread(
            target=self._solve_ik, args=(msg,), daemon=True
        ).start()

    # ── Private Helpers ──────────────────────────────────────────────

    def _heartbeat_tick(self):
        """Publish a heartbeat message."""
        self.pub_heartbeat.publish(Bool(data=True))

    def _solve_ik(self, msg: IKRequestMsg):
        """Solve IK for all waypoints in the request.

        Runs under _solve_lock to serialize concurrent solves.
        PRECOMPUTE requests check _abort_event between waypoints
        and exit early if a PRIMARY arrives.

        Arguments
        ---------
        msg : IKRequestMsg
            The IK request to solve.
        """
        with self._solve_lock:
            self._abort_event.clear()

            n = len(msg.needs_ik)
            q_solutions = []
            success = True
            q_prev = None

            for i in range(n):
                if (msg.request_type
                        == IKRequestMsg.REQUEST_PRECOMPUTE
                        and self._abort_event.is_set()):
                    self.logger.info(
                        f"Aborted precompute {msg.request_id} "
                        f"at waypoint {i}/{n}"
                    )
                    return

                if msg.needs_ik[i]:
                    p = np.array(
                        msg.positions[i * 3:(i + 1) * 3]
                    )
                    if not msg.in_calibration:
                        if USE_ERROR_MAP:
                            p = self.apply_offsets(p)
                        if self.table_heights:
                            p = self.apply_z_offsets(p)
                    tilt = msg.tilts[i]
                    roll = msg.rolls[i]
                    gripper_open = msg.gripper_open[i]

                    state = ManipulatorState(
                        p=p,
                        o=np.array([tilt, roll]),
                        gripper_open=gripper_open,
                    )
                    q, _ = self.block_manipulator.ikin(
                        state, q_seed=q_prev,
                        visualize_elbow_down=True,
                    )
                    if q is None:
                        self.logger.error(
                            f"IK failed for waypoint {i}"
                        )
                        success = False
                        q_solutions.extend([0.0] * NUM_DOFS)
                        q_prev = None
                    else:
                        q_solutions.extend(q.tolist())
                        q_prev = q
                else:
                    q_preset = msg.q_preset[
                        i * NUM_DOFS:(i + 1) * NUM_DOFS
                    ]
                    q_solutions.extend(q_preset)
                    q_prev = np.array(q_preset)

            response = IKResponseMsg()
            response.header.stamp = (
                self.get_clock().now().to_msg()
            )
            response.request_id = msg.request_id
            response.request_type = msg.request_type
            response.q_solutions = q_solutions
            response.success = success
            self.pub_response.publish(response)

            self.logger.debug(
                f"[PUB ik_response] id={msg.request_id} "
                f"success={success}"
            )


# ── Entry Point ──────────────────────────────────────────────────────

def main(args=None):
    """Start the IK solver node."""
    rclpy.init(args=args)
    node = IKSolverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

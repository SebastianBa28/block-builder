"""Brain node for legobuilder.

Central coordinator between perception (Detector) and action
(Manipulator).  Thin ROS layer: subscriptions, publishers,
heartbeats.  Delegates build logic to BuildPipeline and
serialization to ros_bridge.
"""

from collections import deque

import rclpy
import numpy as np
from rclpy.node import Node

import cv_bridge
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Bool, ColorRGBA, String as StringMsg
from visualization_msgs.msg import Marker, MarkerArray
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point

from legobuilder_interfaces.msg import (
    ContourInfoArray,
    TrajectoryCommandMsg,
    TrajectoryCompleteMsg,
    ContactMsg,
    GripFailureMsg,
    PointCommandMsg,
    IKRequestMsg,
    IKResponseMsg,
    ScanRequestMsg,
    GripCheckRequestMsg,
    GripCheckResponseMsg,
    ConnectionCheckRequestMsg,
    ConnectionCheckResponseMsg,
    TableHeightMsg,
    TableHeightRequestMsg,
    CalibrationXYMsg,
    GridCoordsMsg,
)

from legobuilder.kinematics.trajectory import TrajectoryState
from legobuilder.kinematics.kinematic_chain import KinematicChain
from legobuilder.kinematics.block_manipulator import (
    ManipulatorState, BlockManipulator,
)

from legobuilder.config import (
    Q_READY,
    Q_SCAN,
    JOINT_NAMES,
    HEARTBEAT_RATE,
    HEARTBEAT_TIMEOUT,
    GRIPPER_JOINT_INDEX,
    GRIPPER_MOTOR_CLOSED_RAD,
    STRUCTURE_BUILD_DEMO,
    EE_DEPTH_SCAN,
    PERPETUAL_SCAN_INTERVAL,
    JOINT_STATE_BUFFER_DURATION,
    VISUAL_GRIP_CHECK,
    DIAGONAL_GRIP_MAX_RETRIES,
    ENABLE_IK_PRECOMPUTE,
    PLACEMENT_VERIFICATION,
    GRID_PLACEMENT,
    CALIBRATE_GRID_COORDS,
    BLOCK_SIZE,
    CALIBRATE_Z,
    CALIBRATE_XY,
    GRIP_CONNECTION_DETECTION,
    DASHBOARD_SERVER_URL,
)

from legobuilder.brain.object_proc import ObjectProcessor
from legobuilder.brain.block_structure import BlockStructure
from legobuilder.brain.build_pipeline import BuildPipeline
from legobuilder.brain.recovery import RecoveryHandler
from legobuilder.brain.precompute import PrecomputeManager
from legobuilder.brain.structure_scanner import StructureScanner
from legobuilder.brain.unreachable_tracker import UnreachableTracker
from legobuilder.brain.calibration import Calibrator
from legobuilder.brain import ros_bridge
from legobuilder.brain.dashboard_client import create_post_event_fn


# ── Structure marker colors (ObjectType int → RGBA) ───────────────────
_OBJTYPE_RGBA = {
    1: ColorRGBA(r=1.0, g=1.0, b=0.0, a=0.6),   # YELLOW_BLOCK
    2: ColorRGBA(r=0.0, g=0.0, b=1.0, a=0.6),   # BLUE_BLOCK
    3: ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.6),   # GREEN_BLOCK
    4: ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.6),   # RED_BLOCK
}

_DEFAULT_RGBA = ColorRGBA(r=1.0, g=1.0, b=1.0, a=0.6)


def _wireframe_cube_points(center, size):
    """Return 24 Point objects (12 edges x 2 endpoints) for a wireframe cube.

    Arguments
    ---------
    center : np.ndarray
        [x, y, z] world position of the cube center.
    size : float
        Side length of the cube.

    Returns
    -------
    list[Point]
        24 geometry_msgs/Point objects defining the 12 edges.
    """
    h = size / 2.0
    cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
    # 8 corners: (x +/- h, y +/- h, z +/- h)
    corners = [
        Point(x=cx - h, y=cy - h, z=cz - h),  # 0: bottom-front-left
        Point(x=cx + h, y=cy - h, z=cz - h),  # 1: bottom-front-right
        Point(x=cx + h, y=cy + h, z=cz - h),  # 2: bottom-back-right
        Point(x=cx - h, y=cy + h, z=cz - h),  # 3: bottom-back-left
        Point(x=cx - h, y=cy - h, z=cz + h),  # 4: top-front-left
        Point(x=cx + h, y=cy - h, z=cz + h),  # 5: top-front-right
        Point(x=cx + h, y=cy + h, z=cz + h),  # 6: top-back-right
        Point(x=cx - h, y=cy + h, z=cz + h),  # 7: top-back-left
    ]
    # 12 edges: 4 bottom, 4 top, 4 vertical
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # bottom
        (4, 5), (5, 6), (6, 7), (7, 4),  # top
        (0, 4), (1, 5), (2, 6), (3, 7),  # vertical
    ]
    points = []
    for a, b in edges:
        points.append(corners[a])
        points.append(corners[b])
    return points


class BrainNode(Node):
    """Central orchestrator ROS node for the legobuilder system.

    Connects detection, IK solving, and trajectory execution through
    ROS subscriptions and publishers.  All build logic is delegated
    to BuildPipeline; all message serialization is delegated to
    ros_bridge.

    Attributes
    ----------
    logger
        ROS logger instance.
    clock
        ROS clock.
    block_manipulator : BlockManipulator
        FK/IK wrapper for the arm and camera chains.
    object_processor : ObjectProcessor
        Handles buffering, clustering, and temporal stability.
    pipeline : BuildPipeline
        3-phase build state machine (grasp -> approach -> place).
    is_manipulator_online : bool
        True if manipulator heartbeat is recent.
    is_detector_online : bool
        True if detector heartbeat is recent.
    is_ik_solver_online : bool
        True if IK solver heartbeat is recent.
    """

    # ── Lifecycle ─────────────────────────────────────────────────────

    def __init__(self, name):
        """Initialize the brain node.

        Sets up kinematic chains, perception processor, block
        structure, build pipeline, heartbeats, publishers,
        subscribers, and optional depth scanning.

        Arguments
        ---------
        name : str
            ROS node name.
        """
        super().__init__(name)

        self.logger = self.get_logger()
        self.clock = self.get_clock()
        self.start_time = self.clock.now()
        self.table_height = 0 # Offset of table from world in z

        # ── Kinematic chains ──────────────────────────────────────────
        tip_chain = KinematicChain(
            self, "world", "tip", JOINT_NAMES,
        )
        self.block_manipulator = BlockManipulator(tip_chain)

        # ── Dashboard callback ──────────────────────────────────────────
        self._post_event = create_post_event_fn(DASHBOARD_SERVER_URL)

        # ── Object processor ──────────────────────────────────────────
        self.object_processor = ObjectProcessor(self.logger, self.get_t)

        # ── Block structure ───────────────────────────────────────────
        structure = None
        if STRUCTURE_BUILD_DEMO:
            if GRID_PLACEMENT:
                structure = BlockStructure.from_json(
                    filename='src/legobuilder/assemblies/castle.json',
                    logger=self.logger,
                )
            else:
                structure = BlockStructure.from_json(
                    filename='src/legobuilder/assemblies/castle.json',
                    origin=np.array([0.3, 0.30, 0.0]),
                    roll=0.0,
                    logger=self.logger,
                )

        # ── Recovery handler ────────────────────────────────────────────
        self.recovery_handler = RecoveryHandler(
            send_ik_request_fn=self._send_ik_request,
            structure=structure,
            logger=self.logger,
            max_retries=DIAGONAL_GRIP_MAX_RETRIES,
            post_event_fn=self._post_event,
        )

        # ── Precompute manager ─────────────────────────────────────────
        self.precompute_manager = PrecomputeManager(
            send_ik_request_fn=self._send_ik_request,
            structure=structure,
            logger=self.logger,
            enable_precompute=ENABLE_IK_PRECOMPUTE,
        )

        # ── Unreachable tracker ───────────────────────────────────────────
        self._unreachable_tracker = UnreachableTracker()

        # ── Structure scanner ───────────────────────────────────────────
        self.structure_scanner = None
        if PLACEMENT_VERIFICATION and structure is not None:
            self.structure_scanner = StructureScanner(
                send_ik_request_fn=self._send_ik_request,
                publish_trajectory_fn=self._publish_trajectory,
                request_scan_fn=self._request_scan,
                create_timer_fn=self.create_timer,
                destroy_timer_fn=self.destroy_timer,
                get_t_fn=self.get_t,
                structure=structure,
                logger=self.logger,
                post_event_fn=self._post_event,
                unreachable_tracker=self._unreachable_tracker,
            )

        # ── Node online status ────────────────────────────────────────
        self.is_manipulator_online = False
        self.is_detector_online = False
        self.is_ik_solver_online = False
        self.manipulator_last_seen = None
        self.detector_last_seen = None
        self.ik_solver_last_seen = None

        # ── Joint state ───────────────────────────────────────────────
        self._current_q = None
        self._joint_history = deque()
        self.create_subscription(
            JointState, '/joint_states',
            self._recv_joint_states, 1,
        )

        # ── Heartbeats ────────────────────────────────────────────────
        self.pub_heartbeat = self.create_publisher(
            Bool, name + '/heartbeat', 10,
        )
        self.create_subscription(
            Bool, 'manipulator/heartbeat',
            self.recv_manipulator_heartbeat, 10,
        )
        self.create_subscription(
            Bool, 'detector/heartbeat',
            self.recv_detector_heartbeat, 10,
        )
        self.create_subscription(
            Bool, 'ik_solver/heartbeat',
            self.recv_ik_solver_heartbeat, 10,
        )
        self.create_timer(
            1.0 / HEARTBEAT_RATE, self.heartbeat_tick,
        )

        # ── Publishers ────────────────────────────────────────────────
        self.pub_trajectory_command = self.create_publisher(
            TrajectoryCommandMsg,
            name + '/trajectory_command', 10,
        )
        self.pub_ik_request = self.create_publisher(
            IKRequestMsg, 'brain/ik_request', 10,
        )

        self.pub_table_height_req = self.create_publisher(
            TableHeightRequestMsg, 'brain/table_height_request', 10,
        )

        from std_msgs.msg import Empty
        if CALIBRATE_Z:
            self.pub_z_calibration_ready = self.create_publisher(
                Empty, 'brain/z_calibration_ready', 10,
            )
        if CALIBRATE_XY:
            self.pub_xy_calibration_ready = self.create_publisher(
                Empty, 'brain/xy_calibration_ready', 10,
            )

        # ── Visual grip check (conditional) ───────────────────────────
        self.pub_grip_check_request = None
        if VISUAL_GRIP_CHECK:
            self.pub_grip_check_request = self.create_publisher(
                GripCheckRequestMsg,
                name + '/grip_check_request', 10,
            )
            self.create_subscription(
                GripCheckResponseMsg,
                'detector/grip_check_response',
                self._recv_grip_check_response, 10,
            )

        # ── Connection check (conditional) ────────────────────────────
        self.pub_connection_check_request = None
        if GRIP_CONNECTION_DETECTION:
            self.pub_connection_check_request = self.create_publisher(
                ConnectionCheckRequestMsg,
                name + '/connection_check_request', 10,
            )
            self.create_subscription(
                ConnectionCheckResponseMsg,
                'detector/connection_check_response',
                self._recv_connection_check_response, 10,
            )

        # ── Subscribers ───────────────────────────────────────────────
        self.create_subscription(
            IKResponseMsg, 'ik_solver/ik_response',
            self.recv_ik_response, 10,
        )
        self.create_subscription(
            ContourInfoArray, 'detector/contour_info',
            self.recv_contours, 10,
        )
        self.create_subscription(
            TrajectoryCompleteMsg,
            'manipulator/trajectory_complete',
            self.recv_trajectory_complete, 10,
        )
        self.create_subscription(
            ContactMsg, 'manipulator/contact',
            self.recv_contact, 10,
        )
        self.create_subscription(
            GripFailureMsg, 'manipulator/grip_failure',
            self.recv_grip_failure, 10,
        )
        self.create_subscription(
            PointCommandMsg, '/point',
            self.recvpoint, 10,
        )

        if CALIBRATE_Z:
            self.create_subscription(
                TableHeightMsg, '/table_height',
                self._recv_table_height, 10,
            )
        if CALIBRATE_XY:
            self.create_subscription(
                CalibrationXYMsg, '/calibration_xy',
                self._recv_calibration_xy, 10,
            )
        if CALIBRATE_GRID_COORDS:
            self.create_subscription(
                GridCoordsMsg, '/grid_coords',
                self._recv_grid_coords, 10,
            )
        self.create_subscription(
            StringMsg, '/structure_json_path',
            self._recv_structure_json_path, 1
        )

        # ── Detections visualization ──────────────────────────────────
        self.bridge = cv_bridge.CvBridge()
        self.pub_detections = self.create_publisher(
            Image, name + '/detections', 3,
        )

        # ── Structure markers (placed blocks + target cell) ────────────
        self.pub_structure_markers = self.create_publisher(
            MarkerArray, 'brain/structure_markers', 1,
        )
        self.create_timer(1.0, self._publish_structure_markers)

        # ── 3D World Mapping (scan triggering) ────────────────────────
        from std_msgs.msg import Bool as BoolMsg
        self.pub_pointcloud_rebuild = self.create_publisher(
            BoolMsg, name + '/pointcloud_rebuild', 10,
        )
        if EE_DEPTH_SCAN:
            self._scan_count = 0
            self.pub_scan_request = self.create_publisher(
                ScanRequestMsg, name + '/worldmap_scan_request', 10,
            )
            if PERPETUAL_SCAN_INTERVAL is not None:
                self.create_timer(PERPETUAL_SCAN_INTERVAL, self._worldmap_scan_callback)

        # ── Grid calibrator (conditional) ─────────────────────────────
        self.calibrator = None
        if CALIBRATE_Z or CALIBRATE_XY or CALIBRATE_GRID_COORDS:
            self.calibrator = Calibrator(
                send_ik_request_fn=self._send_ik_request,
                publish_trajectory_fn=self._publish_trajectory,
                request_scan_fn=self._request_scan,
                request_height_fn=self._request_table_height,
                create_timer_fn=self.create_timer,
                destroy_timer_fn=self.destroy_timer,
                get_t_fn=self.get_t,
                logger=self.logger,
                fkin_fn=lambda q: self.block_manipulator.fkin(q)[0],
                get_current_q_fn=lambda: self._current_q,
                post_event_fn=self._post_event,
            )

        # ── Worldmap contour detections (for structure scanning) ────────
        if PLACEMENT_VERIFICATION and EE_DEPTH_SCAN:
            self.create_subscription(
                ContourInfoArray,
                'detector/worldmap_contours',
                self._recv_worldmap_contours, 10,
            )

        # ── Build pipeline (state machine) ────────────────────────────
        self._ik_request_counter = 0
        self.pipeline = BuildPipeline(
            send_ik_request_fn=self._send_ik_request,
            publish_trajectory_fn=self._publish_trajectory,
            request_grip_check_fn=self._request_grip_check,
            create_timer_fn=self.create_timer,
            destroy_timer_fn=self.destroy_timer,
            get_t_fn=self.get_t,
            fkin_all_fn=lambda q: (
                self.block_manipulator.fkin_all(q, recompute=True)
            ),
            request_connection_check_fn=self._request_connection_check,
            structure=structure,
            logger=self.logger,
            recovery_handler=self.recovery_handler,
            precompute_manager=self.precompute_manager,
            scanner=self.structure_scanner,
            post_event_fn=self._post_event,
        )

        if structure is not None:
            self._post_event({
                'type': 'structure',
                'source': 'brain',
                'data': {
                    'block_grid': structure.block_grid.tolist(),
                    'placed_grid': structure.placed_grid.tolist(),
                },
                'timestamp': self.get_t(),
            })

        self.logger.info("Brain node initialized and running...")

    def shutdown(self):
        """Release resources and destroy the ROS node."""
        self.destroy_node()

    # ── Time helper ───────────────────────────────────────────────────

    def get_t(self):
        """Return seconds elapsed since node start."""
        now = self.clock.now()
        return (now - self.start_time).nanoseconds * 1e-9

    # ── ROS bridge wrappers ───────────────────────────────────────────

    def _send_ik_request(
        self, trajectory_states, request_type, in_calibration=False,
    ):
        """Serialize and publish an IK request via ros_bridge."""
        self._ik_request_counter += 1
        request_id = f"ik_{self._ik_request_counter}"
        msg = ros_bridge.build_ik_request(
            trajectory_states, request_type, request_id,
            self.clock.now().to_msg(),
            in_calibration=in_calibration,
        )
        self.pub_ik_request.publish(msg)
        return request_id

    def _publish_trajectory(self, states, command):
        """Serialize and publish a trajectory command via ros_bridge."""
        msg = ros_bridge.build_trajectory_command(
            states, command, self.clock.now().to_msg(),
        )
        cmd_name = {
            0: 'APPEND_FRONT', 1: 'APPEND_BACK', 2: 'REPLACE',
        }.get(command, f'UNKNOWN({command})')
        self.logger.debug(
            f"[PUB brain/trajectory_command] "
            f"cmd={cmd_name} n_states={len(states)}"
        )
        self.pub_trajectory_command.publish(msg)

    def _request_grip_check(self):
        """Send a grip check request.

        Returns
        -------
        str or None
            Request ID, or None if publisher is unavailable.
        """
        if (self.pub_grip_check_request is None
                or not self.is_detector_online):
            return None
        self.pipeline._grip_check_counter += 1
        request_id = f"grip_{self.pipeline._grip_check_counter}"
        msg = ros_bridge.build_grip_check_request(
            request_id, self.clock.now().to_msg(),
        )
        self.pub_grip_check_request.publish(msg)
        return request_id

    def _request_connection_check(self):
        """Send a connection check request.

        Returns
        -------
        str or None
            Request ID, or None if publisher is unavailable.
        """
        if (self.pub_connection_check_request is None
                or not self.is_detector_online):
            return None
        self.pipeline._connection_check_counter += 1
        request_id = (
            f"conn_{self.pipeline._connection_check_counter}")
        msg = ros_bridge.build_connection_check_request(
            request_id, self.clock.now().to_msg(),
        )
        self.pub_connection_check_request.publish(msg)
        return request_id

    # ── Heartbeats ────────────────────────────────────────────────────

    def recv_manipulator_heartbeat(self, msg: Bool):
        """Update manipulator online status from heartbeat."""
        was_online = self.is_manipulator_online
        self.is_manipulator_online = True
        self.manipulator_last_seen = self.get_t()
        if not was_online:
            self.logger.info("Manipulator is now online")

    def recv_detector_heartbeat(self, msg: Bool):
        """Update detector online status from heartbeat."""
        was_online = self.is_detector_online
        self.is_detector_online = True
        self.detector_last_seen = self.get_t()
        if not was_online:
            self.logger.info("Detector is now online")

    def recv_ik_solver_heartbeat(self, msg: Bool):
        """Update IK solver online status from heartbeat."""
        was_online = self.is_ik_solver_online
        self.is_ik_solver_online = True
        self.ik_solver_last_seen = self.get_t()
        if not was_online:
            self.logger.info("IK Solver is now online")

    def heartbeat_tick(self):
        """Publish own heartbeat and check peer timeouts."""
        t = self.get_t()
        self.pub_heartbeat.publish(Bool(data=True))
        if (self.is_manipulator_online
                and self.manipulator_last_seen is not None):
            if t - self.manipulator_last_seen > HEARTBEAT_TIMEOUT:
                self.logger.warn("Manipulator went offline")
                self.is_manipulator_online = False
        if (self.is_detector_online
                and self.detector_last_seen is not None):
            if t - self.detector_last_seen > HEARTBEAT_TIMEOUT:
                self.logger.warn("Detector went offline")
                self.is_detector_online = False
        if (self.is_ik_solver_online
                and self.ik_solver_last_seen is not None):
            if t - self.ik_solver_last_seen > HEARTBEAT_TIMEOUT:
                self.logger.warn("IK Solver went offline")
                self.is_ik_solver_online = False

        self._post_event({
            'type': 'health',
            'source': 'brain',
            'data': {
                'manipulator': self.is_manipulator_online,
                'detector': self.is_detector_online,
                'ik_solver': self.is_ik_solver_online,
            },
            'timestamp': t,
        })

        # Trigger grid calibration once all nodes are online
        if (self.calibrator is not None
                and not self.calibrator.is_active
                and not self.calibrator.is_done
                and self.is_manipulator_online
                and self.is_detector_online
                and self.is_ik_solver_online):
            self.logger.info("All nodes online — starting grid calibration")
            self.calibrator.start_calibration(
                on_complete=self._on_calibration_complete,
            )

    def _on_calibration_complete(self):
        """Return arm to Q_SCAN after calibration and signal IK solver."""
        self.logger.info("Calibration complete — returning to Q_SCAN")
        from std_msgs.msg import Empty
        if CALIBRATE_Z and hasattr(self, 'pub_z_calibration_ready'):
            self.pub_z_calibration_ready.publish(Empty())
            self.logger.info("Published z_calibration_ready signal to IK solver")
        if CALIBRATE_XY and hasattr(self, 'pub_xy_calibration_ready'):
            self.pub_xy_calibration_ready.publish(Empty())
            self.logger.info("Published xy_calibration_ready signal to IK solver")
        self._publish_trajectory(
            [
                TrajectoryState(
                    final_state=ManipulatorState(
                        q=Q_SCAN.copy(),
                        qd=np.zeros(len(JOINT_NAMES)),
                    ),
                    min_duration=3.0,
                ),
            ],
            TrajectoryCommandMsg.COMMAND_REPLACE,
        )

    # ── Joint state ───────────────────────────────────────────────────

    def _recv_joint_states(self, msg: JointState):
        """Cache current joint angles and forward to pipeline."""
        self._current_q = np.array(msg.position)
        self.pipeline._last_q = self._current_q
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        qd = np.array(msg.velocity)
        self._joint_history.append((stamp, self._current_q, qd))
        cutoff = stamp - JOINT_STATE_BUFFER_DURATION
        while self._joint_history and self._joint_history[0][0] < cutoff:
            self._joint_history.popleft()

    # ── 3D World Mapping ──────────────────────────────────────────────

    def _signal_pointcloud_rebuild(self):
        """Signal the detector to rebuild its cached PointCloud."""
        from std_msgs.msg import Bool as BoolMsg
        msg = BoolMsg()
        msg.data = True
        self.pub_pointcloud_rebuild.publish(msg)

    def _worldmap_scan_callback(self):
        """Periodically trigger the detector to integrate a depth scan."""
        if self.pipeline.executing_trajectory:
            return
        # Don't auto-scan during scanning (scanner has its own timer)
        if (self.pipeline._build_phase == 'scan'
                and self.structure_scanner is not None
                and self.structure_scanner.is_active):
            return
        # Don't auto-scan during calibration
        if (self.calibrator is not None
                and self.calibrator.is_active):
            return
        self._request_scan(ScanRequestMsg.SIMPLY_ACCUMULATE)
        self._signal_pointcloud_rebuild()

    def _request_scan(self, request_type=ScanRequestMsg.SIMPLY_ACCUMULATE):
        """Publish a unified scan request (injectable into verifier/calibrator)."""
        if not hasattr(self, 'pub_scan_request'):
            return
        msg = ScanRequestMsg()
        msg.request_id = str(self._scan_count)
        msg.request_type = request_type
        msg.header.stamp = self.clock.now().to_msg()
        self.pub_scan_request.publish(msg)
        self._scan_count += 1

    def _recv_worldmap_contours(self, msg: ContourInfoArray):
        """Forward worldmap contour detections to the scanner."""
        if (self.structure_scanner is None
                or not self.structure_scanner.is_active):
            return

        objects = ros_bridge.convert_contour_msgs_to_objects(
            msg.contours, self.get_t(), self.logger,
        )
        self.structure_scanner.on_worldmap_detections(objects)

    def _request_table_height(self):
        """Publish a table height request to the detector."""
        msg = TableHeightRequestMsg()
        self.pub_table_height_req.publish(msg)

    def _recv_table_height(self, msg: TableHeightMsg):
        """Receive detected table height and forward to calibrator if active."""
        self.table_height = msg.table_height
        if (self.calibrator is not None
                and self.calibrator.is_active):
            self.calibrator.on_height_recieved(msg.table_height)

    def _recv_calibration_xy(self, msg: CalibrationXYMsg):
        """Receive detected blue circle position and forward to calibrator."""
        if (self.calibrator is not None
                and self.calibrator.is_active):
            if msg.success:
                self.calibrator.on_xy_position_received(msg.x, msg.y, msg.true_x, msg.true_y)
            else:
                self.calibrator.on_xy_detection_failed()

    def _recv_grid_coords(self, msg: GridCoordsMsg):
        """Receive detected grid corners, update structure and forward to calibrator."""
        if not msg.success:
            self.logger.warning("[GRID-COORDS] Detection failed")
            return
        corners = np.array(msg.corners).reshape(4, 2)
        self.logger.info(f"[GRID-COORDS] Received grid corners from detector")
        if self.pipeline.structure is not None:
            self.pipeline.structure.update_grid_coords(corners)
        if (self.calibrator is not None
                and self.calibrator.is_active):
            self.calibrator.on_grid_coords_received(corners)

    # ── Pipeline callbacks ────────────────────────────────────────────

    def recv_trajectory_complete(self, msg: TrajectoryCompleteMsg):
        """Forward trajectory completion to the pipeline or calibrator."""
        if (self.calibrator is not None
                and self.calibrator.is_active):
            self.calibrator.on_trajectory_complete()
            return
        self.pipeline.on_trajectory_complete(msg.states_executed)
        self._signal_pointcloud_rebuild()

    def recv_contours(self, msg: ContourInfoArray):
        """Run perception, publish visualization, delegate to pipeline."""
        # Don't feed detections to pipeline during calibration
        if (self.calibrator is not None
                and not self.calibrator.is_done):
            return

        objects = ros_bridge.convert_contour_msgs_to_objects(
            msg.contours, self.get_t(), self.logger
        )
        detected_objects = self.object_processor.process_contours(objects)

        det_img = self.object_processor.render_detections(
            detected_objects,
        )
        self.pub_detections.publish(
            self.bridge.cv2_to_imgmsg(det_img, 'rgb8'),
        )

        self.pipeline.on_new_detections(
            detected_objects,
            self.is_manipulator_online and self.is_ik_solver_online,
        )

    def recv_ik_response(self, msg: IKResponseMsg):
        """Forward IK response to the calibrator or pipeline."""
        if (self.calibrator is not None
                and self.calibrator.is_active):
            self.calibrator.on_ik_response(
                msg.request_id, msg.success, msg.q_solutions,
            )
            return
        self.pipeline.on_ik_response(
            msg.request_id, msg.request_type,
            msg.success, msg.q_solutions,
        )

    def recv_contact(self, msg: ContactMsg):
        """Forward contact event to the pipeline."""
        self.logger.info(
            f"Contact detected: type={msg.contact_type}"
        )
        self.logger.info(
            f"Position error: {list(msg.position_error)}"
        )
        self.pipeline.on_contact(self._current_q)

    def recv_grip_failure(self, msg: GripFailureMsg):
        """Forward grip failure to the pipeline."""
        # return  # TEMP: ignore grip failures in sim
        self.logger.debug(
            f"Grip failure: "
            f"gripper_pos={msg.gripper_position:.4f}"
            f" states_remaining={msg.states_remaining}"
        )
        self.pipeline.on_grip_failure(
            msg.gripper_position, self._current_q,
        )

    def _recv_grip_check_response(self, msg: GripCheckResponseMsg):
        """Forward visual grip check response to the pipeline."""
        self.pipeline.on_grip_check_response(
            quality=msg.grip_quality,
            diagonal_score=msg.diagonal_score,
            height_score=msg.height_score,
            GRIP_NO_FRAME=GripCheckResponseMsg.GRIP_NO_FRAME,
            GRIP_DIAGONAL=GripCheckResponseMsg.GRIP_DIAGONAL,
            GRIP_PARTIAL=GripCheckResponseMsg.GRIP_PARTIAL,
            GRIP_LOW=GripCheckResponseMsg.GRIP_LOW,
            GRIP_HIGH=GripCheckResponseMsg.GRIP_HIGH,
            request_id=msg.request_id,
        )

    def _recv_connection_check_response(
        self, msg: ConnectionCheckResponseMsg,
    ):
        """Forward connection check response to the pipeline."""
        self.pipeline.on_connection_check_response(
            request_id=msg.request_id,
            nearby_count=msg.nearby_count,
            detected_color=msg.detected_color,
        )

    # ── Structure markers ────────────────────────────────────────────

    def _publish_structure_markers(self):
        """Publish MarkerArray with placed-block cubes and target wireframe."""
        if self.pipeline.structure is None:
            return

        structure = self.pipeline.structure
        stamp = self.clock.now().to_msg()
        marker_array = MarkerArray()

        # First marker: DELETEALL to clear stale markers
        delete_all = Marker()
        delete_all.action = Marker.DELETEALL
        marker_array.markers.append(delete_all)

        # Placed blocks as solid colored cubes
        marker_id = 0
        for idx in np.ndindex(structure.placed_grid.shape):
            val = int(structure.placed_grid[idx])
            if val == 0:
                continue
            row, col, layer = idx
            pos = structure.get_world_position(row, col, layer)

            m = Marker()
            m.header.frame_id = 'world'
            m.header.stamp = stamp
            m.ns = 'placed_blocks'
            m.id = marker_id
            m.type = Marker.CUBE
            m.action = Marker.ADD
            m.pose.position.x = float(pos[0])
            m.pose.position.y = float(pos[1])
            m.pose.position.z = float(pos[2])
            m.pose.orientation.w = 1.0
            m.scale.x = BLOCK_SIZE
            m.scale.y = BLOCK_SIZE
            m.scale.z = BLOCK_SIZE
            m.color = _OBJTYPE_RGBA.get(val, _DEFAULT_RGBA)
            m.lifetime = Duration(sec=2)
            marker_array.markers.append(m)
            marker_id += 1

        # Current target cell as cyan wireframe
        if len(structure.blocks_to_place) > 0:
            row, col, layer = structure.blocks_to_place[0]
            center = structure.get_world_position(row, col, layer)

            m = Marker()
            m.header.frame_id = 'world'
            m.header.stamp = stamp
            m.ns = 'target_cell'
            m.id = 0
            m.type = Marker.LINE_LIST
            m.action = Marker.ADD
            m.scale.x = 0.002  # line width 2mm
            m.color = ColorRGBA(r=0.0, g=1.0, b=1.0, a=1.0)
            m.lifetime = Duration(sec=2)
            m.points = _wireframe_cube_points(center, BLOCK_SIZE)
            marker_array.markers.append(m)

        self.pub_structure_markers.publish(marker_array)

    # ── Manual point command (testing) ────────────────────────────────

    def recvpoint(self, pointmsg):
        """Execute a manual point-to-point trajectory for testing.

        Moves the arm above the specified XYZ, descends, closes
        the gripper, lifts, then returns to Q_READY.

        Arguments
        ---------
        pointmsg : PointCommandMsg
            Target position and orientation (degrees).
        """
        x = pointmsg.x
        y = pointmsg.y
        z = pointmsg.z
        tilt = pointmsg.t * np.pi / 180
        roll = pointmsg.r * np.pi / 180

        self.logger.info(
            f"Received point ({x}, {y}, {z})"
            f" tilt={pointmsg.t} roll={pointmsg.r}"
        )

        above_block_state = ManipulatorState(
            p=np.array([x, y, 0.1]),
            o=np.array([tilt, roll]),
        )
        q, qd = self.block_manipulator.ikin(above_block_state)
        if q is None:
            self.logger.error(
                "IK failed for above-block position"
            )
            return
        above_block_state.q = q
        above_block_state.qd = qd

        surround_block_state = ManipulatorState(
            p=np.array([x, y, 0.02]),
            o=np.array([tilt, roll]),
        )
        q, qd = self.block_manipulator.ikin(surround_block_state)
        surround_block_state.q = q
        surround_block_state.qd = qd

        q_ready_with_close_grip = Q_READY.copy()
        q_ready_with_close_grip[GRIPPER_JOINT_INDEX] = (
            GRIPPER_MOTOR_CLOSED_RAD
        )
        up_state = ManipulatorState(
            q=q_ready_with_close_grip,
            qd=np.zeros(len(JOINT_NAMES)),
        )

        above_block_state.close_gripper()

        self._publish_trajectory(
            [
                TrajectoryState(
                    final_state=above_block_state,
                    min_duration=4.0,
                ),
                TrajectoryState(
                    final_state=surround_block_state,
                    min_duration=2.0, delay_before=1.0,
                ),
                TrajectoryState(
                    final_state=up_state,
                    min_duration=4.0, delay_before=1.0,
                ),
                TrajectoryState(
                    final_state=ManipulatorState(
                        q=Q_READY.copy(),
                        qd=np.zeros(len(JOINT_NAMES)),
                    ),
                    min_duration=4.0, delay_before=5.0,
                ),
            ],
            TrajectoryCommandMsg.COMMAND_REPLACE,
        )
        self.pipeline.executing_trajectory = True
        
    def _recv_structure_json_path(self, msg: StringMsg):
        # Load new structure
        json_path = msg.data
        new_structure = BlockStructure.from_json(json_path, self.logger)
        
        # Update old structure fields
        self.structure.block_grid = new_structure.block_grid
        self.structure.blocks_to_place = deque()
        self.structure.visible_blocks = set()
        self.structure.invisible_blocks = set()
        self.structure.placed_grid = np.zeros_like(self.structure.block_grid, dtype=int)
        self.structure.detected_grid_objects = []
        
        # Logging
        self.logger.info(f"Successfully loaded new structure from {json_path}")
        self.logger.info(f"   block_grid={self.structure.block_grid}")
        self.logger.info(f"   blocks_to_place={self.structure.blocks_to_place}")
        self.logger.info(f"   visible_blocks={self.structure.visible_blocks}")
        self.logger.info(f"   invisible_blocks={self.structure.invisible_blocks}")
        self.logger.info(f"   placed_grid={self.structure.placed_grid}")
        self.logger.info(f"   detected_grid_objects={self.structure.detected_grid_objects}")

def main(args=None):
    """Spin the brain node until interrupted."""
    rclpy.init(args=args)
    node = BrainNode('brain')
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

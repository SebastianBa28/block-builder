"""Detector ROS node for camera-based block detection and 3D mapping.

Subscribes to the overhead camera image topic, runs the detection
pipeline (HSV filtering, contour extraction, shape classification),
and publishes detected contour info to the brain node.  Also manages
3D world mapping via depth camera point clouds and visual grip quality
checks via the external depth estimation server.
"""

import rclpy
import cv_bridge
import numpy as np

from collections import deque
from rclpy.node import Node
from std_msgs.msg import Bool
from sensor_msgs.msg import Image, JointState, PointCloud2
from visualization_msgs.msg import MarkerArray

from legobuilder.vision.detection import Detector, DetectorConfig
from legobuilder.vision.world_map import WorldMap
from legobuilder.vision import ros_bridge as det_ros_bridge
from legobuilder.vision import grip_check_handler
from legobuilder.vision import connection_check_handler
from legobuilder.kinematics.kinematic_chain import KinematicChain
from legobuilder.kinematics.transform_utils import T_from_Rp
from legobuilder.config import (
    HEARTBEAT_RATE, HEARTBEAT_TIMEOUT, WorldMapConfig,
    VISUAL_GRIP_CHECK, DEPTH_ANYTHING_MODEL_ID,
    DEPTH_SERVER_URL, TEST_GRIPPER,
    ENABLE_POINTCLOUD_CACHING,
    ENABLE_CAMERA_FUSION,
    BLOCK_SIZE,
    MAX_GRID_SIZE,
    CAMERA_JOINT_NAMES,
    JOINT_STATE_BUFFER_DURATION,
    PLACEMENT_GRID_COORDS,
    OVERHEAD_PURPLE_HSV,
    EE_PURPLE_HSV,
    GRID_CENTER_XY,
    CALIBRATE_Z,
    CALIBRATE_XY,
    XY_CAL_ROI_RADIUS_PX,
    OVERHEAD_CAM_POS,
    ERODE_DILATE_ITERATIONS,
    PUBLISH_DETECTOR_IMAGES,
    PUBLISH_DETECTOR_POINTCLOUDS,
    LOW_COMPUTE_MODE,
    GRIP_CONNECTION_DETECTION,
    CORNER_DETECTION_THRESHOLD,
    GRID_PLACEMENT,
)
from legobuilder.schemas import Color
from legobuilder_interfaces.msg import (
    ContourInfoArray, ScanRequestMsg,
    GripCheckRequestMsg, GripCheckResponseMsg,
    ConnectionCheckRequestMsg, ConnectionCheckResponseMsg,
    TableHeightMsg, TableHeightRequestMsg,
    CalibrationXYMsg, GridCoordsMsg,
)


class DetectorNode(Node):
    """ROS 2 node wrapping the detection pipeline and 3D world map.

    Processes overhead camera images at frame rate, extracts block
    contours via the Detector pipeline, and publishes
    ContourInfoArray messages for the brain node.  Also integrates
    end-effector depth camera point clouds into a probabilistic 3D
    world map and handles visual grip quality checks.

    Attributes
    ----------
    detector : Detector
        Core detection pipeline instance.
    world_map : WorldMap
        Probabilistic 3D voxel map built from depth scans.
    bridge : cv_bridge.CvBridge
        ROS image message converter.
    logger
        ROS logger instance.
    clock
        ROS clock for timestamping.
    start_time
        Node start time for elapsed-time computation.
    is_brain_online : bool
        Whether the brain node heartbeat is active.
    is_manipulator_online : bool
        Whether the manipulator node heartbeat is active.
    pub_contours
        Publisher for detected contour info.
    pub_world_map
        Publisher for 3D point cloud map.
    """

    # ── Lifecycle ──────────────────────────────────────────────────

    def __init__(self, name):
        """Initialise the detector node with all pub/sub and subsystems.

        Arguments
        ---------
        name : str
            ROS node name (used as namespace prefix for topics).
        """
        super().__init__(name)

        self.logger = self.get_logger()
        self.clock = self.get_clock()
        self.start_time = self.clock.now()
        self.detector = Detector(
            detector_config=DetectorConfig(),
            logger=self.logger,
            clock=self.clock,
            start_time=self.start_time,
        )

        self.pubraw = self.create_publisher(Image, name + '/raw', 3)
        self.pubann = self.create_publisher(
            Image, name + '/annotated', 3)
        self.pubseg = self.create_publisher(
            Image, name + '/segmentations', 3)

        self.pub_contours = self.create_publisher(
            ContourInfoArray, name + '/contour_info', 10)

        # ── Heartbeat ─────────────────────────────────────────────
        self.is_brain_online = False
        self.is_manipulator_online = False
        self.brain_last_seen = None
        self.manipulator_last_seen = None

        self.pub_heartbeat = self.create_publisher(
            Bool, name + '/heartbeat', 10)
        self.create_subscription(
            Bool, 'brain/heartbeat',
            self.recv_brain_heartbeat, 10)
        self.create_subscription(
            Bool, 'manipulator/heartbeat',
            self.recv_manipulator_heartbeat, 10)
        self.create_timer(1.0 / HEARTBEAT_RATE, self.heartbeat_tick)

        self.bridge = cv_bridge.CvBridge()

        # Queue size 1: only process the most recent frame
        self.sub = self.create_subscription(
            Image, '/image_raw', self.process_image_contours, 1
        )

        # Init early: KinematicChain.__init__ calls spin_once() which can
        # fire process_image_contours before __init__ finishes.
        self._pending_connection_check_id = None

        # ── 3D World Mapping ──────────────────────────────────────
        self.world_map = WorldMap(
            WorldMapConfig(), logger=self.logger)

        # Camera FK chain (4 DOFs: base, shoulder, elbow, wrist_tilt)
        self._camera_chain = KinematicChain(
            self, "world", "camera_optical_frame", CAMERA_JOINT_NAMES)

        # Blue mark FK chain (for XY calibration ROI centering)
        self._blue_mark_chain = KinematicChain(
            self, "world", "blue_mark", CAMERA_JOINT_NAMES)

        # Scan sync buffers
        self._joint_history = deque()       # (stamp, q, qd)
        self._depth_history = deque()       # (stamp, xyzrgb)
        self._scan_count = 0

        self._ee_pointcloud = None
        self.create_subscription(
            PointCloud2, '/ee_cam/depth/points',
            self._recv_ee_pointcloud, 1)
        self.create_subscription(
            JointState, '/joint_states',
            self._recv_joint_states, 1)
        self.create_subscription(
            ScanRequestMsg, 'brain/worldmap_scan_request',
            self._recv_scan_request, 10)

        self.pub_world_map = self.create_publisher(
            PointCloud2, name + '/world_map', 1)

        # PointCloud caching (brain-triggered rebuild)
        self._cached_pointcloud_msg = None
        self._pointcloud_dirty = True
        if ENABLE_POINTCLOUD_CACHING:
            from std_msgs.msg import Bool as BoolMsg
            self.create_subscription(
                BoolMsg, 'brain/pointcloud_rebuild',
                self._recv_pointcloud_rebuild, 10)

        from visualization_msgs.msg import Marker
        self.pub_world_bounds = self.create_publisher(
            Marker, name + '/world_bounds', 1)

        # ── Worldmap Contour Detection ───────────────────────────
        self.pub_worldmap_markers = self.create_publisher(
            MarkerArray, name + '/worldmap_markers', 1)
        self.pub_worldmap_contours = self.create_publisher(
            ContourInfoArray, name + '/worldmap_contours', 10)
        self.pub_grid_corners = self.create_publisher(
            MarkerArray, name + '/grid_corner_markers', 1)

        # Latest-scan-only world map for single-scan detection mode.
        # Use confidence_threshold=-1 so single-observation voxels
        # (log_odds = log_odds_prior = 0) pass the query filter.
        latest_cfg = WorldMapConfig(confidence_threshold=-2.0)
        self._latest_scan_map = WorldMap(latest_cfg, logger=self.logger)

        # ── EE Camera ─────────────────────────────────────────────
        self._ee_color_frame = None
        if VISUAL_GRIP_CHECK or TEST_GRIPPER or ENABLE_CAMERA_FUSION:
            self.create_subscription(
                Image, '/ee_cam/color/image_raw',
                self._recv_ee_color, 1)

        # ── Visual Grip Check ─────────────────────────────────────
        if VISUAL_GRIP_CHECK or TEST_GRIPPER:
            self.create_subscription(
                GripCheckRequestMsg, 'brain/grip_check_request',
                self._recv_grip_check_request, 10)
            self.pub_grip_check_response = self.create_publisher(
                GripCheckResponseMsg,
                name + '/grip_check_response', 10)
            self.detector.init_depth_anything_model(
                DEPTH_ANYTHING_MODEL_ID, DEPTH_SERVER_URL)

        # ── Connection Check ────────────────────────────────
        if GRIP_CONNECTION_DETECTION:
            self.create_subscription(
                ConnectionCheckRequestMsg,
                'brain/connection_check_request',
                self._recv_connection_check_request, 10)
            self.pub_connection_check_response = self.create_publisher(
                ConnectionCheckResponseMsg,
                name + '/connection_check_response', 10)

        self.logger.info("Ball detector running...")

        # ── Calibration ─────────────────────────────────────
        if CALIBRATE_Z:
            self.pub_table_height = self.create_publisher(
                TableHeightMsg, '/table_height', 1)

            self.create_subscription(
                TableHeightRequestMsg, 'brain/table_height_request',
                self._recv_table_height_request, 10)

        if CALIBRATE_XY:
            self.pub_calibration_xy = self.create_publisher(
                CalibrationXYMsg, '/calibration_xy', 1)

        # Grid coord detection from overhead camera
        self.pub_grid_coords = self.create_publisher(
            GridCoordsMsg, '/grid_coords', 1)

        # Latest overhead camera frame (stored for XY calibration scans)
        self._overhead_frame_hsv = None
        self._overhead_frame_rgb = None

    def get_t(self):
        """Return elapsed seconds since node start."""
        now = self.get_clock().now()
        return (now - self.start_time).nanoseconds * 1e-9

    def shutdown(self):
        """Destroy the ROS node."""
        self.destroy_node()

    # ── Heartbeat Callbacks ───────────────────────────────────────

    def recv_brain_heartbeat(self, msg: Bool):
        """Handle brain heartbeat -- track online status."""
        was_online = self.is_brain_online
        self.is_brain_online = True
        self.brain_last_seen = self.get_t()
        self.logger.debug(
            f"[RECV brain/heartbeat] t={self.brain_last_seen:.4f}"
        )
        if not was_online:
            self.logger.info("Brain is now online")

    def recv_manipulator_heartbeat(self, msg: Bool):
        """Handle manipulator heartbeat -- track online status."""
        was_online = self.is_manipulator_online
        self.is_manipulator_online = True
        self.manipulator_last_seen = self.get_t()
        self.logger.debug(
            f"[RECV manipulator/heartbeat]"
            f" t={self.manipulator_last_seen:.4f}"
        )
        if not was_online:
            self.logger.info("Manipulator is now online")

    def heartbeat_tick(self):
        """Publish heartbeat and check for timed-out nodes."""
        t = self.get_t()

        self.pub_heartbeat.publish(Bool(data=True))
        self.logger.debug(f"[PUB detector/heartbeat] t={t:.4f}")

        if self.is_brain_online and self.brain_last_seen is not None:
            if t - self.brain_last_seen > HEARTBEAT_TIMEOUT:
                self.logger.warn("Brain went offline")
                self.is_brain_online = False

        if (self.is_manipulator_online
                and self.manipulator_last_seen is not None):
            if t - self.manipulator_last_seen > HEARTBEAT_TIMEOUT:
                self.logger.warn("Manipulator went offline")
                self.is_manipulator_online = False

    # ── Image Processing Callback ─────────────────────────────────

    def process_image_contours(self, msg: Image):
        """Handle incoming camera frame: detect, annotate, and publish.

        Arguments
        ---------
        msg : Image
            ROS Image message from the overhead camera.
        """
        self.logger.debug("[RECV /image_raw] frame processing started")
        raw_frame, raw_frame_hsv = self.detector.process_img_msg(msg)
        self._overhead_frame_hsv = raw_frame_hsv
        self._overhead_frame_rgb = raw_frame.copy()

        # Process pending connection check on this fresh frame
        if self._pending_connection_check_id is not None:
            request_id = self._pending_connection_check_id
            self._pending_connection_check_id = None
            response = connection_check_handler.handle_connection_check_request(
                request_id, self.detector,
                raw_frame.copy(), raw_frame_hsv.copy(),
                self.clock.now().to_msg(), self.logger)
            self.pub_connection_check_response.publish(response)

        contour_infos, segmentations = (
            self.detector.perceive_from_image(
                raw_frame.copy(), raw_frame_hsv.copy(),
            )
        )

        if ENABLE_CAMERA_FUSION:
            self._run_camera_fusion(contour_infos)


        contour_msg = det_ros_bridge.build_contour_info_array(contour_infos, self.clock.now().to_msg())
        self.pub_contours.publish(contour_msg)
        self.logger.debug(
            f"[PUB detector/contour_info]"
            f" n_contours={len(contour_infos)}"
        )

        ann_frame = raw_frame.copy()
        for ci in contour_infos:
            self.detector.draw_annotated_shape_contour(ann_frame, ci)

        if PUBLISH_DETECTOR_IMAGES and not LOW_COMPUTE_MODE:
            self.pubraw.publish(self.bridge.cv2_to_imgmsg(raw_frame, "rgb8"))
            self.pubann.publish(self.bridge.cv2_to_imgmsg(ann_frame, "rgb8"))
            self.pubseg.publish(self.bridge.cv2_to_imgmsg(segmentations, "rgb8"))

    # ── 3D World Mapping Callbacks ────────────────────────────────

    @staticmethod
    def _point_in_quad(p, quad):
        """Check if point p is inside a convex quadrilateral (CCW vertices)."""
        for i in range(4):
            edge = quad[(i + 1) % 4] - quad[i]
            to_point = np.array(p) - quad[i]
            if edge[0] * to_point[1] - edge[1] * to_point[0] < 0:
                return False
        return True

    def process_worldmap_contours(self):
        """Run worldmap contour detection and publish results."""
        if not self.world_map.has_data:
            self.logger.info(
                "[worldmap_contours] No world map data, skipping")
            return

        detected_corners = self.detect_base_grid(self.world_map, min_corner_detection=2)
        # detected_corners = self.detect_base_grid_from_corner(self.world_map)
        level_results = self.detector.perceive_from_worldmap(self.world_map)
        self._apply_grid_correction(detected_corners, level_results)

        # Filter base grid false positives from level 0
        if detected_corners is not None:
            grid_quad = detected_corners[[0, 1, 3, 2]]  # bl,br,tr,tl (CCW)
            filtered_results = []
            for level, contour_infos in level_results:
                if level == 0:
                    contour_infos = [
                        c for c in contour_infos
                        if not self._point_in_quad(c.center_xy, grid_quad)
                    ]
                filtered_results.append((level, contour_infos))
            level_results = filtered_results

        all_contour_infos = []
        for _level, contour_infos in level_results:
            all_contour_infos.extend(contour_infos)

        self.logger.info(
            f"[worldmap_contours] Detected {len(all_contour_infos)}"
            f" blocks across {len(level_results)} levels")

        stamp = self.clock.now().to_msg()

        marker_array = det_ros_bridge.build_worldmap_marker_array(
            level_results, BLOCK_SIZE, stamp)
        self.pub_worldmap_markers.publish(marker_array)

        contour_msg = det_ros_bridge.build_contour_info_array(
            all_contour_infos, stamp
        )
        self.pub_worldmap_contours.publish(contour_msg)

    def _integrate_into_fresh_map(self) -> bool:
        """Find best scan pair, compute camera FK, integrate into _latest_scan_map.

        Returns True if _latest_scan_map has data after integration.
        """
        pair = self._find_best_scan_pair()
        if pair is None:
            return False

        xyzrgb, q, best_dt = pair
        q_cam = q[:4]
        self._camera_chain.fkin(q_cam)
        T_world_camera = T_from_Rp(self._camera_chain.tip_rot, self._camera_chain.tip_pos)

        self._latest_scan_map.reset()
        self._latest_scan_map.integrate(xyzrgb[:, :3], xyzrgb[:, 3:6], T_world_camera)
        return self._latest_scan_map.has_data

    def _process_latest_scan_contours(self):
        """Run contour detection on only the latest EE depth scan."""
        if not self._integrate_into_fresh_map():
            self.logger.info(
                "[worldmap_contours/latest] No valid scan pair or empty map")
            return

        self.logger.info(f"Detecting base grid...")
        detected_corners = self.detect_base_grid(self._latest_scan_map, min_corner_detection=2)
        # detected_corners = self.detect_base_grid_from_corner(self._latest_scan_map)
        level_results = self.detector.perceive_from_worldmap(self._latest_scan_map)
        for _level, contour_infos in level_results:
            self.logger.info(f"after perceive. level={_level}")
            for ci in contour_infos:
                self.logger.info(f"    center_xy={ci.center_xy}, center_z={ci.center_z}")
        self._apply_grid_correction(detected_corners, level_results)
        for _level, contour_infos in level_results:
            self.logger.info(f"after grid correction. level={_level}")
            for ci in contour_infos:
                self.logger.info(f"    center_xy={ci.center_xy}, center_z={ci.center_z}")

        if detected_corners is not None:
            grid_quad = detected_corners[[0, 1, 3, 2]]  # bl,br,tr,tl (CCW)
            filtered_results = []
            for level, contour_infos in level_results:
                if level == 0:
                    contour_infos = [
                        c for c in contour_infos
                        if not self._point_in_quad(c.center_xy, grid_quad)
                    ]
                filtered_results.append((level, contour_infos))
            level_results = filtered_results
        for _level, contour_infos in level_results:
            self.logger.info(f"after filter. level={_level}")
            for ci in contour_infos:
                self.logger.info(f"    center_xy={ci.center_xy}, center_z={ci.center_z}")

        all_contour_infos = []
        for _level, contour_infos in level_results:
            all_contour_infos.extend(contour_infos)

        self.logger.info(
            f"[worldmap_contours/latest] Detected {len(all_contour_infos)}"
            f" blocks across {len(level_results)} levels")

        stamp = self.clock.now().to_msg()

        marker_array = det_ros_bridge.build_worldmap_marker_array(
            level_results, BLOCK_SIZE, stamp)
        self.pub_worldmap_markers.publish(marker_array)

        contour_msg = det_ros_bridge.build_contour_info_array(
            all_contour_infos, stamp)
        self.pub_worldmap_contours.publish(contour_msg)

    def _process_z_calibration_scan(self):
        """Integrate fresh scan and publish table height for z-calibration."""
        if not self._integrate_into_fresh_map():
            self.logger.info("[Z-CALIBRATE] Fresh scan empty — no table height")
            return

        table_height = self._latest_scan_map.get_table_height()
        self.logger.info(f"[Z-CALIBRATE] Table height: {table_height:.4f}")
        self.pub_table_height.publish(TableHeightMsg(table_height=table_height))

    _xy_cal_save_idx = 0

    def _save_xy_cal_image(self, ann, fname):
        """Save an annotated XY calibration image to disk."""
        import os
        save_dir = '/home/robot/robotws/src/legobuilder/tmp/xy_calibration'
        os.makedirs(save_dir, exist_ok=True)
        import cv2
        cv2.imwrite(os.path.join(save_dir, fname), cv2.cvtColor(ann, cv2.COLOR_RGB2BGR))

    def _publish_xy_failed(self):
        """Publish a failed CalibrationXYMsg so the calibrator can advance."""
        self.pub_calibration_xy.publish(
            CalibrationXYMsg(x=0.0, y=0.0, success=False))

    def _process_xy_calibration_scan(self):
        """Detect blue circle in overhead camera and publish its world XY.

        Uses the existing camera FK chain and joint state buffer to
        compute where the EE camera mount is, then masks the overhead
        frame to a circular ROI around that pixel location so that
        other blue objects in the scene are ignored.
        """
        import cv2

        if self._overhead_frame_hsv is None:
            self.logger.warning("[XY-CALIBRATE] No overhead frame available")
            self._publish_xy_failed()
            return
        if not self.detector.mapper.is_calibrated:
            self.logger.warning("[XY-CALIBRATE] Mapper not calibrated yet")
            self._publish_xy_failed()
            return
        if not self._joint_history:
            self.logger.warning("[XY-CALIBRATE] No joint states available")
            self._publish_xy_failed()
            return

        mask = self.detector.filter_color(self._overhead_frame_hsv, Color.BLUE, erode_and_dilate=False)
        if mask is None:
            self.logger.warning("[XY-CALIBRATE] No blue mask produced")
            self._publish_xy_failed()
            return

        # Compute blue mark world position from FK
        _, q, _ = self._joint_history[-1]
        q_cam = q[:len(CAMERA_JOINT_NAMES)]
        self._blue_mark_chain.fkin(q_cam)
        mark_pos = self._blue_mark_chain.tip_pos  # blue mark 3D world position

        # Project blue mark onto z=0 table plane along overhead camera ray (parallax correction)
        h_frame, w_frame = self._overhead_frame_hsv.shape[:2]
        cx, cy = self.detector.mapper.pixel2world(w_frame // 2, h_frame // 2, no_offset=True)
        cz = OVERHEAD_CAM_POS[2]
        if abs(cz - mark_pos[2]) > 1e-6:
            t = cz / (cz - mark_pos[2])
            x_table = cx + t * (mark_pos[0] - cx)
            y_table = cy + t * (mark_pos[1] - cy)
        else:
            x_table, y_table = mark_pos[0], mark_pos[1]

        eu, ev = self.detector.mapper.world2pixel(x_table, y_table)

        # Apply circular ROI mask
        h, w = mask.shape[:2]
        roi_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(roi_mask, (eu, ev), XY_CAL_ROI_RADIUS_PX, 255, -1)
        mask = cv2.bitwise_and(mask, roi_mask)
        mask = cv2.dilate(mask, None, iterations=4)
        self.logger.info(
            f"[XY-CALIBRATE] Blue mark FK=({mark_pos[0]:.4f}, {mark_pos[1]:.4f}, {mark_pos[2]:.4f}), "
            f"ROI pixel ({eu},{ev}), radius={XY_CAL_ROI_RADIUS_PX}px")

        # Annotation: draw green ROI circle + red mask overlay
        ann = self._overhead_frame_rgb.copy()
        cv2.circle(ann, (eu, ev), XY_CAL_ROI_RADIUS_PX, (0, 255, 0), 2)
        ann[mask > 0] = (255, 0, 0)

        if not np.any(mask):
            self.logger.warning("[XY-CALIBRATE] No blue detected in ROI")
            if PUBLISH_DETECTOR_IMAGES and not LOW_COMPUTE_MODE:
                self.pubann.publish(self.bridge.cv2_to_imgmsg(ann, "rgb8"))
            self._save_xy_cal_image(ann, f"FAIL_{self._xy_cal_save_idx:03d}_no_blue.png")
            self._xy_cal_save_idx += 1
            self._publish_xy_failed()
            return

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            self.logger.warning("[XY-CALIBRATE] No contours found in blue mask")
            if PUBLISH_DETECTOR_IMAGES and not LOW_COMPUTE_MODE:
                self.pubann.publish(self.bridge.cv2_to_imgmsg(ann, "rgb8"))
            self._save_xy_cal_image(ann, f"FAIL_{self._xy_cal_save_idx:03d}_no_contours.png")
            self._xy_cal_save_idx += 1
            self._publish_xy_failed()
            return

        # Find the most circular contour of reasonable size
        best = None
        best_circularity = 0.0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 50:  # too small
                continue
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter ** 2)
            if circularity >= best_circularity:
                best_circularity = circularity
                best = cnt

        # if best is None or best_circularity < 0.3:
        #     self.logger.warning(
        #         f"[XY-CALIBRATE] No circular blue contour found "
        #         f"(best circularity={best_circularity:.2f})")
        #     self.pubann.publish(self.bridge.cv2_to_imgmsg(ann, "rgb8"))
        #     self._save_xy_cal_image(ann, f"FAIL_{self._xy_cal_save_idx:03d}_low_circularity.png")
        #     self._xy_cal_save_idx += 1
        #     self._publish_xy_failed()
        #     return

        M = cv2.moments(best)
        if M["m00"] == 0:
            if PUBLISH_DETECTOR_IMAGES and not LOW_COMPUTE_MODE:
                self.pubann.publish(self.bridge.cv2_to_imgmsg(ann, "rgb8"))
            self._save_xy_cal_image(ann, f"FAIL_{self._xy_cal_save_idx:03d}_zero_moment.png")
            self._xy_cal_save_idx += 1
            self._publish_xy_failed()
            return
        u = int(M["m10"] / M["m00"])
        v = int(M["m01"] / M["m00"])

        # Annotation: draw green contour + centroid (over the red mask)
        cv2.drawContours(ann, [best], -1, (0, 255, 0), 2)
        cv2.circle(ann, (u, v), 5, (0, 255, 0), -1)
        if PUBLISH_DETECTOR_IMAGES and not LOW_COMPUTE_MODE:
            self.pubann.publish(self.bridge.cv2_to_imgmsg(ann, "rgb8"))

        world_xy = self.detector.mapper.pixel2world(u, v)
        x, y = world_xy[0], world_xy[1]
        self.logger.info(
            f"[XY-CALIBRATE] Blue circle at pixel ({u},{v}) → "
            f"world ({x:.4f}, {y:.4f})")

        # Reject if offset is unreasonably large (>5cm in either axis)
        if abs(x - x_table) > 0.05 or abs(y - y_table) > 0.05:
            self.logger.warning(
                f"[XY-CALIBRATE] Offset too large: "
                f"dx={x - x_table:.4f}, dy={y - y_table:.4f} — skipping")
            self._save_xy_cal_image(ann, f"FAIL_{self._xy_cal_save_idx:03d}_large_offset.png")
            self._xy_cal_save_idx += 1
            self._publish_xy_failed()
            return

        # Save annotated image to disk
        self._save_xy_cal_image(ann, f"xy=({x:.2f},{y:.2f})-uv_cnt=({u},{v}).png")

        self.pub_calibration_xy.publish(
            CalibrationXYMsg(x=x, y=y, true_x=x_table, true_y=y_table, success=True))

    def _detect_grid_coords(self):
        """Detect purple corner blocks from overhead camera and publish grid corners."""
        import cv2

        if self._overhead_frame_hsv is None:
            self.logger.warning("[GRID-COORDS] No overhead frame available")
            self.pub_grid_coords.publish(GridCoordsMsg(corners=[0.0]*8, success=False))
            return

        # HSV mask for purple
        min_limits = OVERHEAD_PURPLE_HSV[:, 0]
        max_limits = OVERHEAD_PURPLE_HSV[:, 1]
        mask = cv2.inRange(self._overhead_frame_hsv, min_limits, max_limits)

        # Morphological cleanup: erode → dilate(2x) → erode
        iters = ERODE_DILATE_ITERATIONS
        mask = cv2.erode(mask, None, iterations=iters)
        mask = cv2.dilate(mask, None, iterations=2 * iters)
        mask = cv2.erode(mask, None, iterations=iters)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            self.logger.warning("[GRID-COORDS] No purple contours found")
            self.pub_grid_coords.publish(GridCoordsMsg(corners=[0.0]*8, success=False))
            return

        # Concatenate all contour points and find bounding rotated rect
        all_points = np.concatenate(contours)
        rect = cv2.minAreaRect(all_points)
        box_pts = cv2.boxPoints(rect)  # 4 pixel corners

        # Save annotated debug image
        import os
        ann = self._overhead_frame_rgb.copy()
        ann[mask > 0] = (255, 0, 0)  # red overlay on purple mask
        cv2.drawContours(ann, contours, -1, (0, 255, 0), 2)  # green contours
        box_int = np.intp(box_pts)
        cv2.drawContours(ann, [box_int], 0, (255, 255, 0), 3)  # yellow minAreaRect
        for i, pt in enumerate(box_pts):
            cv2.putText(ann, f"{i}:({int(pt[0])},{int(pt[1])})",
                        (int(pt[0]) + 5, int(pt[1]) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        save_dir = '/home/robot/robotws/src/legobuilder/tmp/grid_calibration'
        os.makedirs(save_dir, exist_ok=True)
        cv2.imwrite(os.path.join(save_dir, 'grid_cal_raw.png'),
                    cv2.cvtColor(ann, cv2.COLOR_RGB2BGR))
        self.logger.info(f"[GRID-COORDS] Debug image saved to {save_dir}/grid_cal_raw.png")

        # Convert pixel corners to world coordinates
        world_corners = []
        for pt in box_pts:
            wx, wy = self.detector.mapper.pixel2world(int(pt[0]), int(pt[1]))
            world_corners.append([wx, wy])
        world_corners = np.array(world_corners)

        # Sort into [bl, br, tl, tr]:
        #   bl = min_x + min_y, br = max_x + min_y
        #   tl = min_x + max_y, tr = max_x + max_y
        sorted_by_y = world_corners[np.argsort(world_corners[:, 1])]
        bottom = sorted_by_y[:2]  # two lowest y
        top = sorted_by_y[2:]     # two highest y
        bl = bottom[np.argmin(bottom[:, 0])]
        br = bottom[np.argmax(bottom[:, 0])]
        tl = top[np.argmin(top[:, 0])]
        tr = top[np.argmax(top[:, 0])]

        self.logger.info(
            f"[GRID-COORDS] Detected corners: "
            f"BL=({bl[0]:.4f},{bl[1]:.4f}) BR=({br[0]:.4f},{br[1]:.4f}) "
            f"TL=({tl[0]:.4f},{tl[1]:.4f}) TR=({tr[0]:.4f},{tr[1]:.4f})"
        )

        corners_flat = [bl[0], bl[1], br[0], br[1], tl[0], tl[1], tr[0], tr[1]]
        self.pub_grid_coords.publish(
            GridCoordsMsg(corners=corners_flat, success=True)
        )

    def _apply_grid_correction(self, detected_corners: np.ndarray, level_results):
        """Apply affine transform from detected grid corners to expected grid positions."""
        import cv2

        M, inliers = cv2.estimateAffinePartial2D(
            detected_corners.astype(np.float32).reshape(-1, 1, 2),
            PLACEMENT_GRID_COORDS.astype(np.float32).reshape(-1, 1, 2),
        )
        if M is None:
            self.logger.warn("[grid_correction] Could not estimate affine transform")
            return

        self.logger.info(
            f"[grid_correction] Affine transform: "
            f"tx={M[0,2]:.4f}, ty={M[1,2]:.4f}, "
            f"scale={np.sqrt(M[0,0]**2 + M[1,0]**2):.4f}, "
            f"angle={np.degrees(np.arctan2(M[1,0], M[0,0])):.2f} deg"
        )

        for _level, contour_infos in level_results:
            for ci in contour_infos:
                # Transform center_xy
                pt = np.array([[ci.center_xy]], dtype=np.float32)
                transformed = cv2.transform(pt, M)
                ci.center_xy = (float(transformed[0, 0, 0]), float(transformed[0, 0, 1]))

                # Transform corner_xys
                if hasattr(ci, 'corner_xys') and ci.corner_xys:
                    corners = np.array([ci.corner_xys], dtype=np.float32)
                    t_corners = cv2.transform(corners, M)
                    ci.corner_xys = [(float(c[0]), float(c[1])) for c in t_corners[0]]

                # Transform face_conns_xyzs (XY only, keep Z)
                if hasattr(ci, 'face_conns_xyzs') and ci.face_conns_xyzs:
                    for i, (fx, fy, fz) in enumerate(ci.face_conns_xyzs):
                        fp = np.array([[[fx, fy]]], dtype=np.float32)
                        tp = cv2.transform(fp, M)
                        ci.face_conns_xyzs[i] = (float(tp[0, 0, 0]), float(tp[0, 0, 1]), fz)

    def detect_base_grid(self, world_map, min_corner_detection: int = 2) -> np.ndarray:
        """Detect the base grid location from a WorldMap using corner detection.

        Queries all voxels above BLOCK_SIZE/2 via query_region, builds a
        binary occupancy mask, applies Sobel edge detection and
        goodFeaturesToTrack corner detection, then extrapolates detected
        corners to the full 4-corner grid using MAX_GRID_SIZE dimensions.

        Parameters
        ----------
        world_map : WorldMap
            Probabilistic 3D voxel map to sample heights from.
        min_corner_detection : int
            Minimum number of corners required before falling back to
            PLACEMENT_GRID_COORDS.  Default 2.

        Returns
        -------
        np.ndarray
            Shape (4, 2) array with [bl, br, tl, tr] world XY coordinates.
        """
        import cv2
        import math

        # ── Grid physical dimensions ─────────────────────────────────
        grid_w = MAX_GRID_SIZE["length"] * BLOCK_SIZE  # x-extent
        grid_h = MAX_GRID_SIZE["width"] * BLOCK_SIZE   # y-extent
        grid_diagonal = math.sqrt(grid_w**2 + grid_h**2)
        search_radius = grid_diagonal / 2 + 2 * BLOCK_SIZE

        cx, cy = GRID_CENTER_XY
        voxel_res = world_map.config.voxel_resolution

        # ── (a) Query voxels and build binary occupancy mask ─────────
        result = world_map.query_region(
            xy_center=np.array([cx, cy]),
            radius=search_radius,
            z_min=BLOCK_SIZE / 2,
            z_max=world_map.config.workspace_z_max,
        )
        keys = result['keys']

        if len(keys) == 0:
            self.logger.warn(
                "[detect_base_grid] No voxels above threshold, using fallback"
            )
            return PLACEMENT_GRID_COORDS.copy()

        # Compute local pixel grid from search region
        cfg = world_map.config
        bx = cfg.workspace_base_xy
        r_ws = cfg.workspace_outer_radius
        bounds_min_x = bx[0] - r_ws
        bounds_min_y = bx[1] - r_ws

        x_min_world = cx - search_radius
        y_min_world = cy - search_radius
        ix_offset = int(np.floor((x_min_world - bounds_min_x) / voxel_res))
        iy_offset = int(np.floor((y_min_world - bounds_min_y) / voxel_res))
        nx = int(np.ceil(2 * search_radius / voxel_res))
        ny = int(np.ceil(2 * search_radius / voxel_res))

        occupancy = np.zeros((ny, nx), dtype=np.uint8)
        local_ix = keys[:, 0] - ix_offset
        local_iy = keys[:, 1] - iy_offset
        valid = (local_ix >= 0) & (local_ix < nx) & (local_iy >= 0) & (local_iy < ny)
        occupancy[local_iy[valid], local_ix[valid]] = 255

        # ── (b) Morphological cleanup ─────────────────────────────────
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        cleaned = cv2.morphologyEx(occupancy, cv2.MORPH_CLOSE, kernel, iterations=3)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=2)

        # ── (c) Find contours → largest → minAreaRect ────────────────
        contours, _ = cv2.findContours(
            cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )

        if not contours:
            self.logger.warn(
                "[detect_base_grid] No contours found, using fallback"
            )
            return PLACEMENT_GRID_COORDS.copy()

        largest = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest)
        (cpx, cpy), (rect_w_px, rect_h_px), angle_deg = rect

        # Convert pixel center to world coordinates
        center_world = np.array([
            x_min_world + cpx * voxel_res,
            y_min_world + cpy * voxel_res,
        ])

        # ── (d) Build 4 corners from center + angle + known dims ─────
        theta = math.radians(angle_deg)
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        R = np.array([[cos_t, -sin_t],
                       [sin_t,  cos_t]])

        half_offsets = np.array([
            [-grid_w / 2, -grid_h / 2],
            [+grid_w / 2, -grid_h / 2],
            [-grid_w / 2, +grid_h / 2],
            [+grid_w / 2, +grid_h / 2],
        ])
        raw_corners = np.array([center_world + R @ off for off in half_offsets])

        # ── (e) Sort into bl, br, tl, tr by world position ───────────
        # Sort by y to split bottom/top pairs, then by x within each
        sorted_by_y = raw_corners[np.argsort(raw_corners[:, 1])]
        bottom = sorted_by_y[:2]
        top = sorted_by_y[2:]
        bottom = bottom[np.argsort(bottom[:, 0])]
        top = top[np.argsort(top[:, 0])]
        grid_corners = np.array([bottom[0], bottom[1], top[0], top[1]])

        # ── Debug: plot occupancy + cleaned + minAreaRect ─────────────
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        x_extent = [x_min_world, x_min_world + nx * voxel_res]
        y_extent = [y_min_world, y_min_world + ny * voxel_res]
        ref = PLACEMENT_GRID_COORDS
        labels = ['bl', 'br', 'tl', 'tr']

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Panel 1: raw occupancy
        axes[0].imshow(occupancy, cmap='gray', origin='lower',
                        extent=[*x_extent, *y_extent])
        axes[0].set_title('Raw Occupancy Mask')
        axes[0].set_xlabel('X (m)')
        axes[0].set_ylabel('Y (m)')
        axes[0].plot(ref[:, 0], ref[:, 1], 'r+', markersize=12,
                      markeredgewidth=2, label='Ref corners')
        axes[0].legend(fontsize=8)

        # Panel 2: cleaned occupancy
        axes[1].imshow(cleaned, cmap='gray', origin='lower',
                        extent=[*x_extent, *y_extent])
        axes[1].set_title('Cleaned (close+open)')
        axes[1].set_xlabel('X (m)')
        axes[1].plot(ref[:, 0], ref[:, 1], 'r+', markersize=12,
                      markeredgewidth=2, label='Ref corners')
        # Draw minAreaRect box
        box_pts = cv2.boxPoints(rect)
        box_world = np.array([
            [x_min_world + p[0] * voxel_res,
             y_min_world + p[1] * voxel_res] for p in box_pts
        ])
        box_closed = np.vstack([box_world, box_world[0]])
        axes[1].plot(box_closed[:, 0], box_closed[:, 1], 'y-',
                      linewidth=2, label='minAreaRect')
        axes[1].legend(fontsize=8)

        # Panel 3: final detected corners vs reference
        axes[2].imshow(occupancy, cmap='gray', origin='lower',
                        extent=[*x_extent, *y_extent], alpha=0.5)
        axes[2].set_title('Detected vs Reference Corners')
        axes[2].set_xlabel('X (m)')
        for i, lbl in enumerate(labels):
            axes[2].plot(ref[i, 0], ref[i, 1], 'r+', markersize=14,
                          markeredgewidth=2)
            axes[2].plot(grid_corners[i, 0], grid_corners[i, 1], 'go',
                          markersize=10)
            axes[2].annotate(lbl, (grid_corners[i, 0], grid_corners[i, 1]),
                              textcoords='offset points', xytext=(5, 5),
                              fontsize=9, color='lime')
        axes[2].plot([], [], 'r+', markersize=10, markeredgewidth=2,
                      label='Reference')
        axes[2].plot([], [], 'go', markersize=8, label='Detected')
        axes[2].legend(fontsize=8)

        plt.tight_layout()
        plt.savefig('/home/robot/robotws/src/legobuilder/tmp/detect_base_grid_debug.png', dpi=150)
        plt.close(fig)
        self.logger.debug(
            "[detect_base_grid] Debug plot saved to "
            "/tmp/detect_base_grid_debug.png"
        )

        # ── Logging ──────────────────────────────────────────────────
        self.logger.info(
            f"[detect_base_grid] minAreaRect center=({center_world[0]:.4f}, "
            f"{center_world[1]:.4f}), angle={angle_deg:.1f} deg"
        )
        self.logger.info(
            f"[detect_base_grid] Grid corners [bl,br,tl,tr]: "
            f"bl=({grid_corners[0,0]:.4f}, {grid_corners[0,1]:.4f}), "
            f"br=({grid_corners[1,0]:.4f}, {grid_corners[1,1]:.4f}), "
            f"tl=({grid_corners[2,0]:.4f}, {grid_corners[2,1]:.4f}), "
            f"tr=({grid_corners[3,0]:.4f}, {grid_corners[3,1]:.4f})"
        )

        return grid_corners

    def detect_base_grid_from_corner(self, world_map) -> np.ndarray:
        """Detect base grid location by finding purple corner blocks.

        Queries base-grid-level voxels, filters for purple hue, clusters
        into individual blocks, then classifies each as a corner or center
        purple by distance to expected corner positions. Corners that are
        close enough to an expected position are used to fit a similarity
        transform that maps the reference grid to the detected positions.

        Parameters
        ----------
        world_map : WorldMap
            Probabilistic 3D voxel map.

        Returns
        -------
        np.ndarray
            Shape (4, 2) array with [bl, br, tl, tr] world XY coordinates.
        """
        import cv2
        import math

        cfg = world_map.config
        voxel_res = cfg.voxel_resolution

        # ── (a) Query base-grid-level voxels ──────────────────────────
        z_level = (BLOCK_SIZE if GRID_PLACEMENT else 0)
        z_min = z_level - BLOCK_SIZE / 2
        z_max = z_level + BLOCK_SIZE / 2

        grid_w = MAX_GRID_SIZE["length"] * BLOCK_SIZE
        grid_h = MAX_GRID_SIZE["width"] * BLOCK_SIZE
        search_radius = math.sqrt(grid_w**2 + grid_h**2) / 2 + 2 * BLOCK_SIZE
        cx, cy = GRID_CENTER_XY

        result = world_map.query_region(
            xy_center=np.array([cx, cy]),
            radius=search_radius,
            z_min=z_min,
            z_max=z_max,
        )
        centers = result['centers']
        colors = result['colors']

        if len(centers) == 0:
            self.logger.warn(
                "[detect_base_grid_from_corner] No voxels at base level, "
                "using fallback"
            )
            return PLACEMENT_GRID_COORDS.copy()

        # ── (b) Filter purple voxels by HSV hue ──────────────────────
        rgb_uint8 = (np.clip(colors, 0, 1) * 255).astype(np.uint8)
        hsv = cv2.cvtColor(rgb_uint8.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV)
        hues = hsv[:, 0, 0].astype(np.float32)
        sats = hsv[:, 0, 1].astype(np.float32)
        vals = hsv[:, 0, 2].astype(np.float32)

        h_min, h_max = EE_PURPLE_HSV[0]
        s_min, s_max = EE_PURPLE_HSV[1]
        v_min, v_max = EE_PURPLE_HSV[2]
        purple_mask = (
            (hues >= h_min) & (hues <= h_max)
            & (sats >= 0) & (sats <= s_max)
            & (vals >= 0) & (vals <= v_max)
        )

        # ── Debug: hue histogram ──────────────────────────────────────
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.hist(hues, bins=180, range=(0, 180), color='gray', alpha=0.7,
                    label=f'All voxels (n={len(hues)})')
            ax.axvline(h_min, color='purple', linestyle='--', linewidth=2,
                       label=f'h_min={h_min}')
            ax.axvline(h_max, color='magenta', linestyle='--', linewidth=2,
                       label=f'h_max={h_max}')
            ax.set_xlabel('Hue (0-179)')
            ax.set_ylabel('Count')
            ax.set_title('Voxel Hue Distribution at Base Grid Level')
            ax.legend()
            plt.tight_layout()
            plt.savefig(
                '/home/robot/robotws/src/legobuilder/tmp/purple_hue_histogram.png',
                dpi=150)
            plt.close(fig)
            self.logger.info(
                "[detect_base_grid_from_corner] Hue histogram saved to "
                "tmp/purple_hue_histogram.png"
            )
        except Exception as e:
            self.logger.warn(
                f"[detect_base_grid_from_corner] Hue histogram failed: {e}"
            )

        purple_centers = centers[purple_mask]
        self.logger.info(
            f"[detect_base_grid_from_corner] Purple filter: "
            f"H=[{h_min},{h_max}], {np.sum(purple_mask)} of "
            f"{len(hues)} voxels passed"
        )
        if len(purple_centers) == 0:
            self.logger.warn(
                "[detect_base_grid_from_corner] No purple voxels detected, "
                "using fallback"
            )
            return PLACEMENT_GRID_COORDS.copy()

        self.logger.info(
            f"[detect_base_grid_from_corner] {len(purple_centers)} purple "
            f"voxels out of {len(centers)} total"
        )

        # ── (c) Build 2D occupancy mask and find contours ─────────────
        bx = cfg.workspace_base_xy
        r_ws = cfg.workspace_outer_radius
        bounds_min_x = bx[0] - r_ws
        bounds_min_y = bx[1] - r_ws

        x_min_world = cx - search_radius
        y_min_world = cy - search_radius
        nx = int(np.ceil(2 * search_radius / voxel_res))
        ny = int(np.ceil(2 * search_radius / voxel_res))

        occupancy = np.zeros((ny, nx), dtype=np.uint8)
        for pc in purple_centers:
            ix = int(np.floor((pc[0] - x_min_world) / voxel_res))
            iy = int(np.floor((pc[1] - y_min_world) / voxel_res))
            if 0 <= ix < nx and 0 <= iy < ny:
                occupancy[iy, ix] = 255

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        cleaned = cv2.morphologyEx(occupancy, cv2.MORPH_CLOSE, kernel, iterations=3)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(
            cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )
        if not contours:
            self.logger.warn(
                "[detect_base_grid_from_corner] No contours from purple "
                "voxels, using fallback"
            )
            return PLACEMENT_GRID_COORDS.copy()

        # ── (d) Get centroids of each purple cluster ──────────────────
        centroids = []
        for cnt in contours:
            M = cv2.moments(cnt)
            if M['m00'] < 1:
                continue
            px = M['m10'] / M['m00']
            py = M['m01'] / M['m00']
            wx = x_min_world + px * voxel_res
            wy = y_min_world + py * voxel_res
            centroids.append(np.array([wx, wy]))

        if not centroids:
            self.logger.warn(
                "[detect_base_grid_from_corner] No valid purple block "
                "centroids, using fallback"
            )
            return PLACEMENT_GRID_COORDS.copy()

        self.logger.info(
            f"[detect_base_grid_from_corner] {len(centroids)} purple block "
            f"centroids detected"
        )

        # ── (e) Classify corners vs center purples ────────────────────
        # For each centroid, find nearest expected corner. If distance
        # is below threshold, it's a corner block.
        expected_corners = PLACEMENT_GRID_COORDS  # shape (4, 2): bl, br, tl, tr
        matched = {}  # corner_idx -> detected_xy

        for centroid in centroids:
            dists = np.linalg.norm(expected_corners - centroid, axis=1)
            min_idx = int(np.argmin(dists))
            min_dist = dists[min_idx]

            if min_dist < CORNER_DETECTION_THRESHOLD:
                # If this corner already matched, keep the closer one
                if min_idx not in matched or min_dist < np.linalg.norm(
                    expected_corners[min_idx] - matched[min_idx]
                ):
                    matched[min_idx] = centroid

        corner_labels = ['bl', 'br', 'tl', 'tr']
        for idx, pos in matched.items():
            self.logger.info(
                f"[detect_base_grid_from_corner] Matched {corner_labels[idx]}"
                f" at ({pos[0]:.4f}, {pos[1]:.4f}), "
                f"expected ({expected_corners[idx, 0]:.4f}, "
                f"{expected_corners[idx, 1]:.4f})"
            )

        # ── (f) Extrapolate full grid from matched corners ────────────
        n_matched = len(matched)
        if n_matched == 0:
            self.logger.warn(
                "[detect_base_grid_from_corner] No corners matched, "
                "using fallback"
            )
            return PLACEMENT_GRID_COORDS.copy()

        if n_matched == 1:
            # Single corner: translate all expected corners by offset
            idx = next(iter(matched))
            offset = matched[idx] - expected_corners[idx]
            grid_corners = expected_corners + offset
            self.logger.info(
                f"[detect_base_grid_from_corner] 1 corner matched, "
                f"translating by ({offset[0]:.4f}, {offset[1]:.4f})"
            )
        else:
            # ≥2 corners: fit similarity transform
            src = np.array([expected_corners[i] for i in matched])
            dst = np.array([matched[i] for i in matched])
            M, _ = cv2.estimateAffinePartial2D(
                src.astype(np.float32).reshape(-1, 1, 2),
                dst.astype(np.float32).reshape(-1, 1, 2),
            )
            if M is not None:
                pts = expected_corners.astype(np.float32).reshape(-1, 1, 2)
                transformed = cv2.transform(pts, M)
                grid_corners = transformed.reshape(-1, 2)
                self.logger.info(
                    f"[detect_base_grid_from_corner] {n_matched} corners "
                    f"matched, fitted similarity transform"
                )
            else:
                # Transform fitting failed, fall back to mean offset
                offsets = np.array([
                    matched[i] - expected_corners[i] for i in matched
                ])
                mean_offset = offsets.mean(axis=0)
                grid_corners = expected_corners + mean_offset
                self.logger.warn(
                    "[detect_base_grid_from_corner] Similarity transform "
                    "failed, using mean offset"
                )

        # ── (g) Debug plot ────────────────────────────────────────────
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            x_extent = [x_min_world, x_min_world + nx * voxel_res]
            y_extent = [y_min_world, y_min_world + ny * voxel_res]

            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            # Panel 1: purple occupancy + centroids
            axes[0].imshow(cleaned, cmap='gray', origin='lower',
                           extent=[*x_extent, *y_extent])
            axes[0].set_title('Purple Voxels (cleaned)')
            axes[0].set_xlabel('X (m)')
            axes[0].set_ylabel('Y (m)')
            for c in centroids:
                axes[0].plot(c[0], c[1], 'rx', markersize=10, markeredgewidth=2)
            for idx, pos in matched.items():
                axes[0].plot(pos[0], pos[1], 'go', markersize=12)
                axes[0].annotate(corner_labels[idx], (pos[0], pos[1]),
                                 textcoords='offset points', xytext=(5, 5),
                                 fontsize=9, color='lime')
            axes[0].legend(['All centroids', 'Matched corners'], fontsize=8)

            # Panel 2: final grid vs reference
            axes[1].imshow(occupancy, cmap='gray', origin='lower',
                           extent=[*x_extent, *y_extent], alpha=0.3)
            axes[1].set_title('Detected vs Reference Grid Corners')
            axes[1].set_xlabel('X (m)')
            ref = PLACEMENT_GRID_COORDS
            for i, lbl in enumerate(corner_labels):
                axes[1].plot(ref[i, 0], ref[i, 1], 'r+', markersize=14,
                             markeredgewidth=2)
                axes[1].plot(grid_corners[i, 0], grid_corners[i, 1], 'go',
                             markersize=10)
                axes[1].annotate(lbl, (grid_corners[i, 0], grid_corners[i, 1]),
                                 textcoords='offset points', xytext=(5, 5),
                                 fontsize=9, color='lime')
            axes[1].plot([], [], 'r+', markersize=10, markeredgewidth=2,
                         label='Reference')
            axes[1].plot([], [], 'go', markersize=8, label='Detected')
            axes[1].legend(fontsize=8)

            plt.tight_layout()
            # plt.savefig('/tmp/detect_base_grid_corner_debug.png', dpi=150)
            plt.savefig('/home/robot/robotws/src/legobuilder/tmp/detect_base_grid_corner_debug.png', dpi=150)
            plt.close(fig)
            self.logger.info(
                "[detect_base_grid_from_corner] Debug plot saved to "
                "/tmp/detect_base_grid_corner_debug.png"
            )
        except Exception as e:
            self.logger.warn(
                f"[detect_base_grid_from_corner] Debug plot failed: {e}"
            )

        # ── Logging ──────────────────────────────────────────────────
        self.logger.info(
            f"[detect_base_grid_from_corner] Grid corners [bl,br,tl,tr]: "
            f"bl=({grid_corners[0,0]:.4f}, {grid_corners[0,1]:.4f}), "
            f"br=({grid_corners[1,0]:.4f}, {grid_corners[1,1]:.4f}), "
            f"tl=({grid_corners[2,0]:.4f}, {grid_corners[2,1]:.4f}), "
            f"tr=({grid_corners[3,0]:.4f}, {grid_corners[3,1]:.4f})"
        )

        return grid_corners

    def _recv_joint_states(self, msg: JointState):
        """Buffer joint state for scan sync matching."""
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        q = np.array(msg.position)
        qd = np.array(msg.velocity)
        self._joint_history.append((stamp, q, qd))
        cutoff = stamp - JOINT_STATE_BUFFER_DURATION
        while self._joint_history and self._joint_history[0][0] < cutoff:
            self._joint_history.popleft()

    def _recv_ee_pointcloud(self, msg: PointCloud2):
        """Buffer end-effector point cloud for scan sync matching."""
        result = det_ros_bridge.process_ee_pointcloud(
            msg, self.logger)
        if result is None:
            return
        xyzrgb, stamp_msg = result
        self._ee_pointcloud = xyzrgb  # keep latest for grip check
        stamp = stamp_msg.stamp.sec + stamp_msg.stamp.nanosec * 1e-9
        self._depth_history.append((stamp, xyzrgb))
        cutoff = stamp - JOINT_STATE_BUFFER_DURATION
        while self._depth_history and self._depth_history[0][0] < cutoff:
            self._depth_history.popleft()

    def _recv_scan_request(self, msg: ScanRequestMsg):
        """Dispatch scan requests by request_type."""
        if msg.request_type == ScanRequestMsg.SIMPLY_ACCUMULATE:
            self._integrate_latest_scan(msg.request_id)
        elif msg.request_type == ScanRequestMsg.DETECT_FROM_ACCUMULATED:
            self._integrate_latest_scan(msg.request_id)
            self.process_worldmap_contours()
        elif msg.request_type == ScanRequestMsg.DETECT_FROM_LATEST:
            self._process_latest_scan_contours()
        elif msg.request_type == ScanRequestMsg.DETECT_GRID_COORDS:
            self._detect_grid_coords()
        elif msg.request_type == ScanRequestMsg.IN_Z_CALIBRATION:
            self._process_z_calibration_scan()
        elif msg.request_type == ScanRequestMsg.IN_XY_CALIBRATION:
            self._process_xy_calibration_scan()

    def _integrate_latest_scan(self, request_id: str):
        """Integrate the best-matching depth/joint pair into the world map.

        The brain's scan request acts as a trigger.  The detector searches
        its own depth and joint-state buffers for the freshest pair where
        the arm was stationary, computes the camera FK, and integrates.
        """
        pair = self._find_best_scan_pair()
        if pair is None:
            self.logger.debug(
                f"[scan {request_id}] no valid depth/joint pair")
            return

        xyzrgb, q, best_dt = pair

        # Camera FK at matched joint state
        q_cam = q[:4]
        self._camera_chain.fkin(q_cam)
        T_world_camera = T_from_Rp(
            self._camera_chain.tip_rot, self._camera_chain.tip_pos
        )

        # Integrate
        self.world_map.integrate(
            xyzrgb[:, :3], xyzrgb[:, 3:6], T_world_camera)
        self._scan_count += 1
        self._pointcloud_dirty = True

        self.logger.debug(
            f"[scan {request_id}] integrated (dt={best_dt:.3f}s)")

        if self.world_map.should_publish():
            self._publish_world_map()

    def _find_best_scan_pair(self):
        """Find the best (depth, joint) pair from buffered data.

        Iterates depth frames newest-first.  For each, finds the closest
        joint state and applies velocity + cooldown gates.  Returns the
        first valid match (freshest data that passed all gates), or None.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, float] or None
            (xyzrgb, q, dt) on success, None if no valid pair exists.
        """
        if not self._depth_history or not self._joint_history:
            return None

        dt_threshold = WorldMapConfig.scan_sync_dt_threshold
        vel_threshold = WorldMapConfig.scan_sync_velocity_threshold
        cooldown = WorldMapConfig.scan_sync_cooldown_samples

        for d_stamp, xyzrgb in reversed(self._depth_history):
            # Find closest joint state to this depth frame
            best_idx = None
            best_dt = float('inf')
            for i, (j_stamp, q, qd) in enumerate(self._joint_history):
                dt = abs(j_stamp - d_stamp)
                if dt < best_dt:
                    best_dt = dt
                    best_idx = i

            if best_idx is None or best_dt > dt_threshold:
                continue

            _, q, qd = self._joint_history[best_idx]

            # Velocity gate
            if qd is not None and np.linalg.norm(qd) > vel_threshold:
                continue

            # Cooldown gate
            start = max(0, best_idx - cooldown)
            cooldown_ok = True
            for i in range(start, best_idx):
                _, _, qd_prev = self._joint_history[i]
                if qd_prev is not None and np.linalg.norm(qd_prev) > vel_threshold:
                    cooldown_ok = False
                    break
            if not cooldown_ok:
                continue

            return (xyzrgb, q, best_dt)

        return None

    def _recv_pointcloud_rebuild(self, msg):
        """Mark PointCloud cache as dirty on brain rebuild signal."""
        self._pointcloud_dirty = True
        self.logger.debug("[RECV brain/pointcloud_rebuild]")

    # ── Publishing Helpers ────────────────────────────────────────

    def _publish_world_map(self):
        """Publish the accumulated 3D map as a PointCloud2 message."""
        result = det_ros_bridge.build_world_map_msg(
            self.world_map, self.clock.now().to_msg(),
            ENABLE_POINTCLOUD_CACHING,
            self._pointcloud_dirty,
            self._cached_pointcloud_msg, self.logger)
        if result is not None:
            msg, self._cached_pointcloud_msg, self._pointcloud_dirty = result
            if PUBLISH_DETECTOR_POINTCLOUDS and not LOW_COMPUTE_MODE:
                self.pub_world_map.publish(msg)
        self._publish_world_bounds()
        self._publish_grid_corners()

    def _publish_grid_corners(self):
        """Publish purple cylinder markers at grid corner positions."""
        grid_msg = det_ros_bridge.build_grid_corner_markers(
            PLACEMENT_GRID_COORDS, BLOCK_SIZE, self.clock.now().to_msg()
        )
        self.pub_grid_corners.publish(grid_msg)

    def _publish_world_bounds(self):
        """Publish workspace bounds as a wireframe Marker."""
        marker = det_ros_bridge.build_world_bounds_marker(
            self.world_map.config, self.clock.now().to_msg())
        self.pub_world_bounds.publish(marker)

    # ── Camera Fusion ─────────────────────────────────────────────

    def _run_camera_fusion(self, contour_infos):
        """Fill detected block rectangles into the voxel store."""
        from legobuilder.vision.profiling import TimingContext
        from legobuilder.vision.fusion import contours_to_pointcloud

        with TimingContext("camera_fusion", self.logger):
            pts, colors = contours_to_pointcloud(
                contour_infos,
                z_height=BLOCK_SIZE,
                voxel_resolution=self.world_map.config.voxel_resolution,
            )
            if len(pts) > 0:
                self.world_map.integrate_world_points(pts, colors)

        self.logger.debug(
            f"[FUSION] {len(pts)} overhead pts integrated")

    # ── Visual Grip Check Callbacks ───────────────────────────────

    def _recv_ee_color(self, msg: Image):
        """Cache the latest end-effector color camera frame."""
        self._ee_color_frame = self.bridge.imgmsg_to_cv2(
            msg, "rgb8")

    def _recv_grip_check_request(self, msg: GripCheckRequestMsg):
        """Run grip analysis and publish result."""
        response = grip_check_handler.handle_grip_check_request(
            msg.request_id, self._ee_color_frame,
            self.detector, self.clock.now().to_msg(),
            self.logger)
        self.pub_grip_check_response.publish(response)

    def _recv_connection_check_request(
        self, msg: ConnectionCheckRequestMsg,
    ):
        """Store request; handler runs on next fresh camera frame."""
        self.logger.info(
            f"[RECV brain/connection_check_request] id={msg.request_id}"
            " — waiting for next frame")
        self._pending_connection_check_id = msg.request_id

    def _recv_table_height_request(self, msg):
        table_height = self.world_map.get_table_height()
        response_msg = TableHeightMsg(table_height=table_height)
        self.pub_table_height.publish(response_msg)


def main(args=None):
    """Entry point for the detector node."""
    rclpy.init(args=args)
    node = DetectorNode('detector')
    rclpy.spin(node)
    node.shutdown()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

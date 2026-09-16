"""Core detection pipeline for block recognition.

Processes overhead camera images through a multi-stage pipeline:
HSV color filtering, morphological cleanup, contour extraction and
shape classification (circle, rectangle, square), pixel-to-world
coordinate mapping, and per-color segmentation overlay.
"""

import numpy as np
from numpy import ndarray as arr
import cv2
import cv_bridge

from pydantic import BaseModel
from collections import defaultdict

from legobuilder.schemas import (
    ContourInfo,
    CircleContourInfo,
    RectangleContourInfo,
    SquareContourInfo,
    ObjectType, Object, Color
)
from legobuilder.vision.mapping import Mapper
from sklearn.cluster import DBSCAN


from legobuilder.config import (
    USE_MULTIPLE_PERSPECTIVES,
    ARUCO_SETTINGS,
    PIXEL_TO_WORLD_OFFSET,
    HSV_LIMITS,
    ERODE_DILATE_ITERATIONS,
    CIRCLE_CONTOUR_TOLERANCE,
    RECTANGLE_CONTOUR_TOLERANCE,
    SQUARE_CONTOUR_TOLERANCE,
    SQUARE_LENGTH_RANGE,
    BLOCK_SIZE,
    ASPECT_RATIO_THRESHOLD,
    BLOCK_FILL_RATIO,
    CLUSTERING_SETTINGS,
    PRESENCE_DURATION_THRESHOLD,
    ABSENCE_DURATION_TOLERANCE,
    STABILITY_THRESHOLDS,
    SMOOTHING_WINDOW_DURATION,
    CALIBRATION_INTERVAL,
    BLOCK_SIZE,
    WORLDMAP_CONTOUR_MAX_LEVELS,
    WORLDMAP_CONTOUR_HUE_THRESHOLD,
    WORLDMAP_CONTOUR_ERODE_DILATE_ITERS,
    WORLDMAP_CONTOUR_MIN_VOXELS,
    WORLDMAP_MASK_SCALE,
    WORLDMAP_COLOR_HUES,
    SHOW_LEVEL_SEGMENTATIONS_VISUALIZATION,
    EE_CAM_HSV_LIMITS,
    CONNECTION_DETECTION_CELL_SIZE,
    GRID_PLACEMENT
)


# ── Configuration ─────────────────────────────────────────────────────


class DetectorConfig(BaseModel):
    """Pydantic configuration model for the detection pipeline.

    Aggregates all tuneable parameters from config.py into a single
    validated object passed to the Detector constructor.

    Attributes
    ----------
    use_multiple_perspectives : bool
        Whether to require all ArUco quadrants for calibration.
    aruco_settings : dict
        Per-quadrant ArUco marker configuration.
    pixel_to_world_offset : tuple[float, float]
        Additive (dx, dy) correction for coordinate mapping.
    calibration_interval : float
        Seconds between automatic recalibrations.
    hsv_limits : dict
        Per-color HSV range arrays keyed by Color.
    erode_dilate_iterations : int
        Morphological open/close iteration count.
    circle_contour_tolerance : float
        Area-ratio tolerance for circle classification.
    rectangle_contour_tolerance : float
        Area-ratio tolerance for rectangle classification.
    square_contour_tolerance : float
        Aspect-ratio tolerance for square classification.
    square_length_range : tuple[float, float]
        Minimum and maximum side length (world metres) for a valid square.
    aspect_ratio_threshold: float
        Maximum aspect ratio (length/width) for square classification.
    clustering_settings : dict
        DBSCAN parameters (eps, min_samples) per color.
    presence_duration_threshold : float
        Seconds a detection must persist to be considered stable.
    absence_duration_tolerance : float
        Seconds a detection may vanish before being dropped.
    stability_thresholds : dict
        Per-attribute stability thresholds for temporal filtering.
    smoothing_window_duration : float
        EWMA smoothing window in seconds.
    """

    use_multiple_perspectives: bool = USE_MULTIPLE_PERSPECTIVES
    aruco_settings: dict = ARUCO_SETTINGS
    pixel_to_world_offset: tuple[float, float] = PIXEL_TO_WORLD_OFFSET
    calibration_interval: float = CALIBRATION_INTERVAL
    hsv_limits: dict = HSV_LIMITS # Top cam
    ee_cam_hsv_limits: dict = EE_CAM_HSV_LIMITS # End-effector cam
    erode_dilate_iterations: int = ERODE_DILATE_ITERATIONS
    circle_contour_tolerance: float = CIRCLE_CONTOUR_TOLERANCE
    rectangle_contour_tolerance: float = RECTANGLE_CONTOUR_TOLERANCE
    square_contour_tolerance: float = SQUARE_CONTOUR_TOLERANCE
    square_length_range: tuple[float, float] = SQUARE_LENGTH_RANGE
    block_fill_ratio: float = BLOCK_FILL_RATIO
    aspect_ratio_threshold: float = ASPECT_RATIO_THRESHOLD
    clustering_settings: dict = CLUSTERING_SETTINGS
    presence_duration_threshold: float = PRESENCE_DURATION_THRESHOLD
    absence_duration_tolerance: float = ABSENCE_DURATION_TOLERANCE
    stability_thresholds: dict = STABILITY_THRESHOLDS
    smoothing_window_duration: float = SMOOTHING_WINDOW_DURATION


# ── Utilities ─────────────────────────────────────────────────────────


def get_ewma_weights(T: int, half_life: int = 10) -> list[float]:
    """Compute exponentially weighted moving average weights.

    Arguments
    ---------
    T : int
        Number of samples in the window.
    half_life : int
        Decay half-life in samples.

    Returns
    -------
    list[float]
        Normalised weights, oldest-first.
    """
    h = 1 / half_life
    p = lambda t: 2**(-h * (t-1)) / sum([2**(-h*s) for s in range(0, T)])  # noqa: E501, E731
    return [p(t) for t in range(1, T+1)][::-1]


# ── Voxel Grid Mapper ─────────────────────────────────────────────────


class VoxelGridMapper:
    """Lightweight pixel-to-world mapper for voxel grid images.

    Provides the same pixel2world(u, v) interface as Mapper, using
    the linear voxel-to-world transformation. Used when running the
    contour pipeline on 2D images projected from the voxel store.
    """

    def __init__(self, resolution: float, bounds_min: np.ndarray):
        self.resolution = resolution
        self.bounds_min = bounds_min
        self.is_calibrated = True

    def pixel2world(self, u, v) -> tuple[float, float]:
        """Convert voxel-grid pixel (u=ix, v=iy) to world XY."""
        x = (u + 0.5) * self.resolution + self.bounds_min[0]
        y = (v + 0.5) * self.resolution + self.bounds_min[1]
        return (x, y)


# ── Detector ──────────────────────────────────────────────────────────


class Detector:
    """Multi-color block detection pipeline.

    Processes camera frames through HSV color filtering, morphological
    cleanup, contour extraction, shape classification (circle, rectangle,
    square), and pixel-to-world coordinate conversion.  Currently only
    square detection is active; circle and rectangle paths are disabled
    but retained for future use.

    Attributes
    ----------
    hsv_limits : dict
        Per-color HSV range arrays.
    erode_dilate_iterations : int
        Morphological iteration count.
    circle_contour_tolerance : float
        Circle area-ratio tolerance.
    rectangle_contour_tolerance : float
        Rectangle area-ratio tolerance.
    square_contour_tolerance : float
        Square aspect-ratio tolerance.
    square_length_range : tuple[float, float]
        Minimum and maximum side length (world metres) for a valid square.
    block_fill_ratio: float
        Minimum ratio of block-colored pixels within square contour for block classification during rectangle separation.
    aspect_ratio_threshold: float
        Maximum aspect ratio (length/width) for square classification.
    mapper : Mapper
        ArUco-based pixel-to-world coordinate mapper.
    bridge : cv_bridge.CvBridge
        ROS image message converter.
    curr_frame_idx : int
        Monotonically increasing frame counter.
    grip_analyzer : GripAnalyzer or None
        Depth-based grip quality analyser (initialised on demand).
    """

    red = (255, 0, 0)
    green = (0, 255, 0)
    blue = (0, 0, 255)
    yellow = (255, 255, 0)
    white = (255, 255, 255)
    orange = (255, 165, 0)

    # ── Lifecycle ─────────────────────────────────────────────────

    def __init__(
        self,
        detector_config: DetectorConfig,
        logger,
        clock,
        start_time: float
    ):
        """Initialise the detector with configuration and ROS handles.

        Arguments
        ---------
        detector_config : DetectorConfig
            Validated detection parameters.
        logger
            ROS-compatible logger.
        clock
            ROS clock for timestamping.
        start_time : float
            Node start time for elapsed-time computation.
        """
        self.hsv_limits = detector_config.hsv_limits
        self.ee_cam_hsv_limits = detector_config.ee_cam_hsv_limits
        self.erode_dilate_iterations = detector_config.erode_dilate_iterations
        self.circle_contour_tolerance = detector_config.circle_contour_tolerance
        self.rectangle_contour_tolerance = detector_config.rectangle_contour_tolerance
        self.square_contour_tolerance = detector_config.square_contour_tolerance
        self.square_length_range = detector_config.square_length_range
        self.block_fill_ratio = detector_config.block_fill_ratio
        self.aspect_ratio_threshold = detector_config.aspect_ratio_threshold
        self.clustering_settings = detector_config.clustering_settings
        self.presence_duration_threshold = detector_config.presence_duration_threshold
        self.absence_duration_tolerance = detector_config.absence_duration_tolerance
        self.stability_thresholds = detector_config.stability_thresholds
        self.smoothing_window_duration = detector_config.smoothing_window_duration

        self.logger = logger
        self.clock = clock
        self.start_time = start_time
        self.curr_frame_idx = 0

        self.estimated_rate = 15.0  # Hz

        self.mapper = Mapper(
            use_multiple_perspectives=detector_config.use_multiple_perspectives,
            aruco_settings=detector_config.aruco_settings,
            pixel_to_world_offset=detector_config.pixel_to_world_offset,
            calibration_interval=detector_config.calibration_interval,
            logger=logger
        )

        self.bridge = cv_bridge.CvBridge()

        self._depth_server_url = None
        self.grip_analyzer = None

    def get_t(self):
        """Return elapsed seconds since node start."""
        now = self.clock.now()
        return (now - self.start_time).nanoseconds * 1e-9

    # ── Public API ────────────────────────────────────────────────

    def perceive_from_image(self, frame: arr, frame_hsv: arr, connection_detection: bool = False) -> tuple[arr, list[ContourInfo], arr]:
        """Run the full detection pipeline on a single camera frame.

        Processes the frame through color filtering, contour extraction,
        shape classification, and builds a per-color segmentation overlay.

        Arguments
        ---------
        frame : arr
            RGB image from the camera.
        frame_hsv : arr
            HSV image from the camera.

        Returns
        -------
        tuple[list[ContourInfo], arr]
            (contour_infos, segmentation_overlay).
        """
        segmentations = np.zeros(frame.shape, dtype=np.uint8)

        calibrated = self.mapper.ensure_valid_mapping(frame)
        if not calibrated:
            self.logger.warning(
                "Mapper not calibrated yet. Cannot perceive objects.")
            return [], segmentations

        contour_infos = []
        overlap_count = np.zeros(frame.shape[:2], dtype=np.uint8)

        # 1. First build a combined mask and segmentation overlay, 
        # while counting overlaps for the white overlay later
        combined_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        for color in self.hsv_limits.keys():
            mask = self.filter_color(frame_hsv, color)
            combined_mask[mask > 0] = color.value
            overlap_count += (mask > 0).astype(np.uint8)
            rgb_color = Color.to_rgb(color)
            if rgb_color is None:
                self.logger.debug(
                    f"Unknown color '{color}' for annotation."
                )
                rgb_color = (0, 0, 0)
            segmentations[(combined_mask == color.value) & (overlap_count >= 1)] = rgb_color

        # 2. Erode and dilate the combined mask to combine attached blocks of different colors
        binary_mask = (combined_mask > 0).astype(np.uint8)
        kernel = np.ones((10,10), np.uint8)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

        # 3. Run contour detection on the combined mask, determining connections and colors of 
        # all blocks in one pass
        contours = self._get_contours(binary_mask)
        if connection_detection:
                cell_size = CONNECTION_DETECTION_CELL_SIZE
                # Remove all contours with side lengths less than cell_size, as these are not the block raised to the camera
                contours = [contour for contour in contours if cv2.contourArea(contour) >= cell_size ** 2]
        else:
            cell_size = 43  # Default value tuned for block detection
        for contour in contours:
            new_contour_infos = self._process_contour(contour, binary_mask, cell_size)
            for contour_info in new_contour_infos:
                if contour_info is not None:
                    # Average the value around the center to determine the color
                    u, v = contour_info.center_uv
                    neighborhood = combined_mask[max(0, v-5):v+5, max(0, u-5):u+5]
                    valid_pixels = neighborhood[neighborhood > 0]
                    if len(valid_pixels) > 0:
                        avg_color = round(np.mean(valid_pixels))
                    else:
                        avg_color = 0
                    if avg_color not in Color:
                        self.logger.debug(
                            f"Unknown color value '{avg_color}' for annotation.")
                        continue
                    contour_info.color = Color(avg_color)
                    contour_infos.append(contour_info)

        # if connection_detection:
        # self.logger.info(f"Detected {len(contour_infos)} connections in the current frame with cell_size {cell_size}.")
        segmentations[overlap_count > 1] = self.white
        h, w = segmentations.shape[:2]
        cv2.putText(
            segmentations,
            f'DETECTOR/SEGMENTATIONS ({w}x{h})', (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4,
            cv2.LINE_AA
        )

        self.curr_frame_idx += 1

        return contour_infos, segmentations

    def perceive_from_worldmap(self, world_map) -> list[tuple[int, list[ContourInfo]]]:
        """Detect blocks from the 3D voxel store layer by layer.

        Uses a combined-mask approach (like perceive_from_image): builds a
        single mask with all colors, runs contour detection once on the
        binary union, then determines each block's color by sampling the
        combined mask at its center. This correctly detects connections
        between adjacent blocks of different colors.

        Arguments
        ---------
        world_map
            WorldMap instance to query voxels from.

        Returns
        -------
        list[tuple[int, list[ContourInfo]]]
            (level, contour_infos) pairs for each occupied level.
        """
        cfg = world_map.config
        resolution = cfg.voxel_resolution

        # Workspace AABB from config
        bx = cfg.workspace_base_xy
        r = cfg.workspace_outer_radius
        bounds_min = np.array([bx[0] - r, bx[1] - r, cfg.workspace_z_min])
        bounds_max = np.array([bx[0] + r, bx[1] + r, cfg.workspace_z_max])

        # Grid dimensions in pixels (XY only)
        grid_w = int(np.ceil((bounds_max[0] - bounds_min[0]) / resolution))
        grid_h = int(np.ceil((bounds_max[1] - bounds_min[1]) / resolution))

        scaled_res = resolution / WORLDMAP_MASK_SCALE
        voxel_mapper = VoxelGridMapper(scaled_res, bounds_min)

        # Reference hue per color (midpoint of HSV H range)
        ref_colors = list(WORLDMAP_COLOR_HUES.keys())
        ref_hue_vals = np.array([WORLDMAP_COLOR_HUES[c] for c in ref_colors], dtype=np.float32)

        COLOR_RGB = {
            Color.RED: (255, 0, 0), Color.GREEN: (0, 255, 0),
            Color.BLUE: (0, 0, 255), Color.YELLOW: (255, 255, 0),
            Color.ORANGE: (255, 165, 0),
        }

        original_mapper = self.mapper
        all_level_results = []
        level_segmentations = []

        try:
            self.mapper = voxel_mapper

            for level in range(0, WORLDMAP_CONTOUR_MAX_LEVELS + 1):
                z_level = level * BLOCK_SIZE + (+BLOCK_SIZE if GRID_PLACEMENT else 0) # need to apply offset since voxel_store.integrate() calibrates for table height (which in this case is the foundation grid height)
                z_min = z_level - BLOCK_SIZE / 2 if level > 1 else z_level - BLOCK_SIZE / 4
                z_max = z_level + BLOCK_SIZE / 2

                result = world_map.query_region(
                    xy_center=cfg.workspace_base_xy,
                    radius=cfg.workspace_outer_radius,
                    z_min=z_min,
                    z_max=z_max,
                )
                
                # self.logger.info(f"Layer {level}: Queried {len(result['keys'])} voxels between z={z_min:.3f}m and z={z_max:.3f}m")

                keys = result['keys']
                colors = result['colors']

                if len(keys) < WORLDMAP_CONTOUR_MIN_VOXELS:
                    continue

                # Convert voxel RGB to HSV hue (OpenCV 0-179 scale)
                rgb_uint8 = (np.clip(colors, 0, 1) * 255).astype(np.uint8)
                hsv_img = cv2.cvtColor(rgb_uint8.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV)
                voxel_hues = hsv_img[:, 0, 0].astype(np.float32)


                # Step 2: Build combined mask with Color enum values
                combined_mask = np.zeros((grid_h, grid_w), dtype=np.uint8)
                level_seg = np.zeros_like(combined_mask)
                for ci_idx, color in enumerate(self.ee_cam_hsv_limits):
                    limits = self.ee_cam_hsv_limits[color]

                    color_mask_1d = (
                        (hsv_img[:, 0, 0] >= limits[0][0]) & (hsv_img[:, 0, 0] <= limits[0][1]) &
                        (hsv_img[:, 0, 1] >= limits[1][0]) & (hsv_img[:, 0, 1] <= limits[1][1]) &
                        (hsv_img[:, 0, 2] >= limits[2][0]) & (hsv_img[:, 0, 2] <= limits[2][1])
                    )
                    if not np.any(color_mask_1d):
                        continue
                    matched_keys = keys[color_mask_1d]
                    ix = np.clip(matched_keys[:, 0], 0, grid_w - 1)
                    iy = np.clip(matched_keys[:, 1], 0, grid_h - 1)
                    combined_mask[iy, ix] = color.value
                    level_seg[combined_mask > 0] = color.value

                # Step 3: Morphological close on binary mask at original resolution
                binary_mask = (combined_mask > 0).astype(np.uint8) * 255
                iters = WORLDMAP_CONTOUR_ERODE_DILATE_ITERS
                binary_mask = cv2.erode(binary_mask, None, iterations=iters)
                binary_mask = cv2.dilate(binary_mask, None, iterations=2 * iters)
                binary_mask = cv2.erode(binary_mask, None, iterations=iters)

                # Step 4: Upscale for higher-fidelity contour detection
                scaled_w = grid_w * WORLDMAP_MASK_SCALE
                scaled_h = grid_h * WORLDMAP_MASK_SCALE
                binary_mask = cv2.resize(
                    binary_mask, (scaled_w, scaled_h),
                    interpolation=cv2.INTER_NEAREST,
                )
                combined_mask_scaled = cv2.resize(
                    combined_mask, (scaled_w, scaled_h),
                    interpolation=cv2.INTER_NEAREST,
                )

                # Segmentation overlay
                level_seg = np.zeros((scaled_h, scaled_w, 3), dtype=np.uint8)
                for color in ref_colors:
                    rgb = COLOR_RGB.get(color, (128, 128, 128))
                    level_seg[combined_mask_scaled == color.value] = rgb

                # Step 5: Contour detection on combined binary mask
                contours = self._get_contours(binary_mask, apply_erode_dilate=False)
                cell_px = int(round(BLOCK_SIZE / scaled_res))

                level_contour_infos = []
                for contour in contours:
                    new_infos = self._process_contour(contour, binary_mask, cell_px, layer=level)
                    for ci in new_infos:
                        if ci is None:
                            continue
                        # Step 6: Determine color by sampling combined mask
                        u, v = ci.center_uv
                        r_nb = 5 * WORLDMAP_MASK_SCALE
                        neighborhood = combined_mask_scaled[
                            max(0, v - r_nb):v + r_nb,
                            max(0, u - r_nb):u + r_nb,
                        ]
                        valid_pixels = neighborhood[neighborhood > 0]
                        if len(valid_pixels) == 0:
                            continue
                        avg_color = round(np.mean(valid_pixels))
                        if avg_color not in Color:
                            self.logger.debug(f"Unknown color value '{avg_color}' at center ({u},{v}).")
                            continue
                        ci.color = Color(avg_color)
                        ci.z_level = level
                        ci.center_z = level*BLOCK_SIZE + BLOCK_SIZE / 2 + (-BLOCK_SIZE if not GRID_PLACEMENT else 0)
                        level_contour_infos.append(ci)

                if level_contour_infos:
                    all_level_results.append((level, level_contour_infos))
                    level_segmentations.append((level, level_seg))

        finally:
            self.mapper = original_mapper

        # Debug visualization
        if level_segmentations and SHOW_LEVEL_SEGMENTATIONS_VISUALIZATION:
            import matplotlib.pyplot as plt
            n = len(level_segmentations)
            fig, axes = plt.subplots(1, n, figsize=(6 * n, 6), squeeze=False)
            for i, (level, seg) in enumerate(level_segmentations):
                axes[0, i].imshow(seg, origin='lower')
                axes[0, i].set_title(f'Level {level} (z={level * BLOCK_SIZE:.3f}m)')
                axes[0, i].axis('off')
            fig.suptitle('Worldmap Contour Detection (combined)', fontsize=14)
            fig.tight_layout()
            plt.show()
        
        self.logger.info(f"Detected contours in {len(all_level_results)} levels of the world map.")

        return all_level_results

    def init_depth_anything_model(
        self,
        model_id: str = 'depth-anything/Depth-Anything-V2-Base-hf',
        server_url: str = 'http://localhost:8000'
    ):
        """Configure the external depth estimation server connection.

        Arguments
        ---------
        model_id : str
            HuggingFace model identifier (informational only).
        server_url : str
            Base URL of the depth estimation FastAPI server.
        """
        from legobuilder.vision.grip_analyzer import GripAnalyzer

        self._depth_server_url = server_url
        self.grip_analyzer = GripAnalyzer()
        self.logger.info(f"Depth server URL: {server_url}")

    def analyze_grip(
        self, color_frame: np.ndarray, request_id: str = ''
    ):
        """Run depth estimation via external server and grip analysis.

        Encodes the frame as JPEG, sends it to the depth server, and
        runs the grip analyser on the returned depth map.

        Arguments
        ---------
        color_frame : np.ndarray
            RGB image from the end-effector camera.
        request_id : str
            Optional identifier for correlating request/response.

        Returns
        -------
        tuple[GripAnalysisResult, np.ndarray or None]
            (result, depth_map) where *depth_map* is uint8 or
            *None* on failure.
        """
        import urllib.request
        import urllib.parse
        import cv2
        from legobuilder.vision.grip_analyzer import (
            GripAnalysisResult, GripQuality)

        if self._depth_server_url is None:
            return (
                GripAnalysisResult(quality=GripQuality.NO_FRAME),
                None)

        if not self._check_depth_server():
            self.logger.warn(
                "Depth server not reachable, try again shortly")
            return (
                GripAnalysisResult(quality=GripQuality.NO_FRAME),
                None)

        _, jpeg_bytes = cv2.imencode(
            '.jpg',
            cv2.cvtColor(color_frame, cv2.COLOR_RGB2BGR))

        boundary = b'----DepthBoundary'
        body = (
            b'--' + boundary + b'\r\n'
            b'Content-Disposition: form-data;'
            b' name="file"; filename="frame.jpg"\r\n'
            b'Content-Type: image/jpeg\r\n\r\n'
            + jpeg_bytes.tobytes()
            + b'\r\n--' + boundary + b'--\r\n'
        )
        qs = urllib.parse.urlencode({'request_id': request_id})
        req = urllib.request.Request(
            f"{self._depth_server_url}/depth?{qs}",
            data=body,
            headers={
                'Content-Type':
                    f'multipart/form-data;'
                    f' boundary={boundary.decode()}'},
            method='POST',
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                dh = int(resp.headers['X-Depth-Height'])
                dw = int(resp.headers['X-Depth-Width'])
                depth_map = np.frombuffer(
                    resp.read(), dtype=np.uint8).reshape(dh, dw)
        except Exception as e:
            self.logger.error(
                f"Depth server request failed: {e}")
            return (
                GripAnalysisResult(quality=GripQuality.NO_FRAME),
                None)

        return self.grip_analyzer.analyze(depth_map), depth_map

    def draw_annotated_shape_contour(
        self, frame: arr, contour_info: ContourInfo
    ):
        """Draw shape outline and world-coordinate label on the frame.

        Arguments
        ---------
        frame : arr
            RGB image to annotate (modified in-place).
        contour_info : ContourInfo
            Detected shape info with pixel and world coordinates.
        """
        if contour_info.is_circle:
            (ur, vr) = contour_info.center_uv
            radius = contour_info.radius
            center_xy = contour_info.center_xy

            # cv2.circle(
            #     frame, (ur, vr), int(radius), self.yellow, 2)
            cv2.circle(frame, (ur, vr), 3, self.blue, -1)

            # cv2.putText(
            #     frame,
            #     f"({center_xy[0]:.2f}, {center_xy[1]:.2f})",
            #     (ur - 60, vr - 20),
            #     cv2.FONT_HERSHEY_SIMPLEX, 1.2,
            #     (255, 0, 0), 3, cv2.LINE_AA)

        elif contour_info.is_rectangle:
            (ur, vr) = contour_info.center_uv
            corner_uvs = contour_info.corner_uvs
            center_xy = contour_info.center_xy

            for i in range(len(corner_uvs)):
                u1, v1 = corner_uvs[i]
                u2, v2 = corner_uvs[(i+1) % len(corner_uvs)]
                cv2.line(
                    frame, (u1, v1), (u2, v2), self.yellow, 2)
            cv2.circle(frame, (ur, vr), 3, self.red, -1)

            angle_deg = np.degrees(contour_info.angle)
            cv2.putText(
                frame,
                f"({center_xy[0]:.2f}, {center_xy[1]:.2f}),"
                f" {angle_deg:.1f} deg",
                (ur - 60, vr - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                (255, 0, 0), 3, cv2.LINE_AA)

        elif contour_info.is_square:
            (ur, vr) = contour_info.center_uv
            corner_uvs = contour_info.corner_uvs
            center_xy = contour_info.center_xy
            corner_conns_px = contour_info.corner_conns_px
            
            # Edges
            for i, (u1, v1) in enumerate(corner_uvs):
                u2, v2 = corner_uvs[(i+1) % len(corner_uvs)]
                cv2.line(frame, (u1, v1), (u2, v2), self.yellow, 2)
            # Connections
            for (u, v) in corner_conns_px:
                cv2.circle(frame, (u, v), 3, self.red, -1)
            # Center
            cv2.circle(frame, (ur, vr), 3, self.yellow, -1)

            angle_deg = np.degrees(contour_info.angle)
            cv2.putText(
                frame,
                f"({center_xy[0]:.3f}, {center_xy[1]:.3f}),"
                f" {angle_deg:.1f} deg",
                (ur - 60, vr - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                (255, 0, 0), 3, cv2.LINE_AA)
            # cv2.putText(
            #     frame,
            #     f"{angle_deg:.1f} deg",
            #     (ur - 20, vr - 20),
            #     cv2.FONT_HERSHEY_SIMPLEX, 1.2,
            #     (255, 0, 0), 3, cv2.LINE_AA)

        else:
            self.logger.warning("Contour is neither circle nor rectangle?")

    # ── Color Filtering ───────────────────────────────────────────

    def process_img_msg(self, msg) -> tuple[arr, arr]:
        """Convert a ROS Image message to RGB and HSV arrays.

        Arguments
        ---------
        msg
            ROS Image message (must use rgb8 encoding).

        Returns
        -------
        tuple[arr, arr]
            (rgb_frame, hsv_frame).
        """
        assert(msg.encoding == "rgb8")
        frame = self.bridge.imgmsg_to_cv2(msg, "passthrough")
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        return frame, frame_hsv

    def filter_color(self, frame_hsv: arr, color: Color, erode_and_dilate: bool = True) -> arr | None:
        """Apply HSV thresholding and morphological cleanup for one color.

        Arguments
        ---------
        frame_hsv : arr
            HSV image.
        color : Color
            Target color to filter.

        Returns
        -------
        arr or None
            Binary mask, or *None* if no HSV limits for *color*.
        """
        if color not in self.hsv_limits:
            self.logger.warning(
                f"HSV limits for color '{color}' not found.")
            return None
        min_limits = self.hsv_limits[color][:, 0]
        max_limits = self.hsv_limits[color][:, 1]
        binary = cv2.inRange(frame_hsv, min_limits, max_limits)

        if erode_and_dilate:
            # Open-close-open to remove noise then fill gaps
            iters = self.erode_dilate_iterations
            binary = cv2.erode(binary, None, iterations=iters)
            binary = cv2.dilate(binary, None, iterations=2*iters)
            binary = cv2.erode(binary, None, iterations=iters)

        return binary

    # ── Contour Extraction ────────────────────────────────────────

    def _get_contours(self, binary: arr, apply_erode_dilate: bool = True) -> list[arr]:
        """Extract external contours from a binary mask.

        Arguments
        ---------
        binary : arr
            Binary uint8 mask from color filtering.

        Returns
        -------
        list[arr]
            List of OpenCV contour arrays.
        """
        if apply_erode_dilate:
            iters = self.erode_dilate_iterations
            binary = cv2.erode( binary, None, iterations=iters)
            binary = cv2.dilate(binary, None, iterations=2*iters)
            binary = cv2.erode( binary, None, iterations=iters)

        (contours, hierarchy) = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        return contours

    # ── Shape Classification ──────────────────────────────────────

    def _get_contour_angle(self, corners: list[tuple]) -> float:
        """Compute the orientation angle of a contour given its corner coordinates.

        Arguments
        ---------
        corners : list[tuple]
            World/pixel coordinates of the contour corners.

        Returns
        -------
        float
            Orientation angle in radians, where 0 means aligned with the x-axis.
        """
        bottom_corner_index = np.argmin([c[1] for c in corners])
        bottom_corner = corners[bottom_corner_index]

        remaining_corners = [
            c for i, c in enumerate(corners)
            if i != bottom_corner_index]
        right_corner_index = np.argmax(
            [c[0] for c in remaining_corners])
        right_corner = remaining_corners[right_corner_index]

        delta_x = right_corner[0] - bottom_corner[0]
        delta_y = right_corner[1] - bottom_corner[1]
        angle = np.arctan2(delta_y, delta_x)

        if angle > np.pi / 2:
            angle -= np.pi / 2
        elif angle < 0:
            angle += np.pi / 2
        return angle

    def _angle_to_quaternion(
        self, angle: float
    ) -> tuple[float, float, float, float]:
        """Convert a 2D orientation angle to a Z-axis quaternion (x, y, z, w)."""
        half = angle / 2.0
        return (0.0, 0.0, np.sin(half), np.cos(half))

    def _get_dominant_edge_angle(self, contour: arr) -> float:
        """Compute the dominant edge direction of a contour.

        Extracts edge vectors between consecutive contour points,
        maps their angles to [0, pi/2) (exploiting 90-degree symmetry
        of square blocks), and returns the length-weighted circular
        mean angle.

        Arguments
        ---------
        contour : arr
            OpenCV contour array (Nx1x2).

        Returns
        -------
        float
            Dominant edge angle in radians, in [0, pi/2).
        """
        # Simplify staircase contour into polygon with actual edge directions
        epsilon = max(2.0, 0.02 * cv2.arcLength(contour, True))
        contour = cv2.approxPolyDP(contour, epsilon, True)

        pts = contour.reshape(-1, 2).astype(np.float64)
        edges = np.diff(pts, axis=0, append=pts[:1]) # wrap around
        lengths = np.linalg.norm(edges, axis=1)

        # Filter out degenerate edges
        valid = lengths > 1e-6
        edges = edges[valid]
        lengths = lengths[valid]

        if len(edges) == 0:
            return 0.0

        # Edge angles mapped to [0, pi/2) via 4x frequency trick
        # (0, pi/2, pi, 3pi/2 all map to the same value)
        angles = np.arctan2(edges[:, 1], edges[:, 0])
        mapped = angles * 4.0
        wx = np.sum(lengths * np.cos(mapped))
        wy = np.sum(lengths * np.sin(mapped))
        mean_mapped = np.arctan2(wy, wx)
        dominant = (mean_mapped / 4.0) % (np.pi / 2.0)

        return dominant

    def _edge_aligned_min_area_rect(self, contour: arr):
        """Compute a bounding rectangle aligned to the contour's dominant edge direction.

        Unlike cv2.minAreaRect (which minimizes area regardless of edge
        directions), this aligns the rectangle to the dominant edge angle
        of the contour.

        Arguments
        ---------
        contour : arr
            OpenCV contour array (Nx1x2).

        Returns
        -------
        tuple
            ((center_x, center_y), (width, height), angle_degrees)
        """
        theta = self._get_dominant_edge_angle(contour)

        # Rotate contour points by -theta so block edges become axis-aligned
        pts = contour.reshape(-1, 2).astype(np.float64)
        cos_t = np.cos(-theta)
        sin_t = np.sin(-theta)
        rot_pts = np.column_stack([
            pts[:, 0] * cos_t - pts[:, 1] * sin_t,
            pts[:, 0] * sin_t + pts[:, 1] * cos_t,
        ])

        # Axis-aligned bounding box in rotated space
        x_min, y_min = rot_pts.min(axis=0)
        x_max, y_max = rot_pts.max(axis=0)
        w = x_max - x_min
        h = y_max - y_min

        # Center in rotated space, then rotate back
        cx_rot = (x_min + x_max) / 2.0
        cy_rot = (y_min + y_max) / 2.0
        cos_t_fwd = np.cos(theta)
        sin_t_fwd = np.sin(theta)
        cx = cx_rot * cos_t_fwd - cy_rot * sin_t_fwd
        cy = cx_rot * sin_t_fwd + cy_rot * cos_t_fwd

        return ((cx, cy), (w, h), np.degrees(theta))

    def _check_rectangularity(
        self, contour: arr, tolerance: float
    ) -> dict | None:
        """Check if a contour is rectangular and return its geometry.

        Compares the enclosing-circle area to the min-area-rectangle
        area.  Returns a geometry dict or None if not rectangular.
        """
        rect = self._edge_aligned_min_area_rect(contour)
        # rect = cv2.minAreaRect(contour)
        (ur, vr), (width_px, length_px), _ = rect
        corners = cv2.boxPoints(rect)

        (_, radius) = cv2.minEnclosingCircle(contour)
        circle_area = np.pi * (radius ** 2)
        rect_area = width_px * length_px
        if rect_area == 0:
            return None
        if abs(circle_area - rect_area) / rect_area <= (1 - tolerance):
            return None

        corner_uvs = [
            (int(c[0]), int(c[1])) for c in corners]
        corner_xys = [
            self.mapper.pixel2world(c[0], c[1]) for c in corners]
        center_xy = self.mapper.pixel2world(int(ur), int(vr))

        return dict(
            center_uv=(int(ur), int(vr)),
            width_px=width_px,
            length_px=length_px,
            corner_uvs=corner_uvs,
            corner_xys=corner_xys,
            center_xy=center_xy,
        )

    def _classify_rectangle(
        self, contour: arr
    ) -> tuple[bool, RectangleContourInfo | None]:
        """Classify contour as a rectangle and build its ContourInfo."""
        geom = self._check_rectangularity(contour, self.rectangle_contour_tolerance)
        if geom is None:
            return False, None

        corner_xys = geom['corner_xys']
        length_m = np.sqrt(
              (corner_xys[0][0] - corner_xys[1][0])**2
            + (corner_xys[0][1] - corner_xys[1][1])**2
        )
        width_m = np.sqrt(
              (corner_xys[1][0] - corner_xys[2][0])**2
            + (corner_xys[1][1] - corner_xys[2][1])**2
        )
        angle = self._get_contour_angle(corner_xys)
        angle_px = np.pi/2 - self._get_contour_angle(geom['corner_uvs'])

        return True, RectangleContourInfo(
            t=self.get_t(),
            frame_idx=self.curr_frame_idx,
            is_rectangle=True,
            center_uv=geom['center_uv'],
            center_xy=geom['center_xy'],
            length_px=geom['length_px'],
            width_px=geom['width_px'],
            length_m=length_m,
            width_m=width_m,
            corner_uvs=geom['corner_uvs'],
            corner_xys=corner_xys,
            angle=angle,
            angle_px=angle_px,
            quaternion=self._angle_to_quaternion(angle),
        )

    def _classify_block(
        self, contour: arr
    ) -> tuple[bool, SquareContourInfo | None]:
        """Classify contour as a single square block.

        Checks rectangularity, world-space side-length range, and
        pixel-space aspect ratio.
        """
        geom = self._check_rectangularity(
            contour, self.square_contour_tolerance
        )
        if geom is None:
            return False, None

        corner_xys = geom['corner_xys']
        side_lengths = [
            np.sqrt(
                (corner_xys[i][0] - corner_xys[(i + 1) % 4][0])**2
                + (corner_xys[i][1] - corner_xys[(i + 1) % 4][1])**2)
            for i in range(4)
        ]
        min_side = min(side_lengths)
        max_side = max(side_lengths)
        min_allowed, max_allowed = self.square_length_range

        if min_side < min_allowed or max_side > max_allowed:
            return False, None
        if max_side / min_side > self.aspect_ratio_threshold:
            return False, None

        width_px = geom['width_px']
        length_px = geom['length_px']
        aspect_ratio = max(width_px, length_px) / min(width_px, length_px)
        if abs(aspect_ratio - 1) >= self.square_contour_tolerance:
            return False, None

        angle = self._get_contour_angle(corner_xys)

        return True, SquareContourInfo(
            t=self.get_t(),
            frame_idx=self.curr_frame_idx,
            is_square=True,
            center_uv=geom['center_uv'],
            center_xy=geom['center_xy'],
            size_px=length_px,
            corner_uvs=geom['corner_uvs'],
            corner_xys=corner_xys,
            center_z=BLOCK_SIZE/2,
            angle=angle,
            quaternion=self._angle_to_quaternion(angle),
        )

    def _split_rectangle_into_blocks(
        self, rect_info: RectangleContourInfo, mask: arr,
        cell_size_approx: int = 43, layer: int = 1,
    ) -> list[SquareContourInfo]:
        """Split a larger rectangle into individual block cells.

        Overlays a grid aligned to the rectangle's orientation and
        checks each cell's mask fill ratio to find blocks.
        """
        corners_by_v = sorted(rect_info.corner_uvs, key=lambda c: c[1])
        top_corners = corners_by_v[2:]
        bottom_corners = corners_by_v[:2]
        top_left = np.array(min(top_corners, key=lambda c: c[0]))
        top_right = np.array(max(top_corners, key=lambda c: c[0]))
        bottom_left = np.array(min(bottom_corners, key=lambda c: c[0]))
        bottom_right = np.array(max(bottom_corners, key=lambda c: c[0]))

        left_side_len = np.linalg.norm(bottom_left - top_left)
        top_side_len = np.linalg.norm(top_right - top_left)

        if left_side_len < top_side_len:
            long_start, long_end = top_left, top_right
            short_start, short_end = top_left, bottom_left
            length, width = top_side_len, left_side_len
        else:
            long_start, long_end = top_left, bottom_left
            short_start, short_end = bottom_left, bottom_right
            length, width = left_side_len, top_side_len

        n_long = round(length / cell_size_approx)
        n_short = round(width / cell_size_approx)
        if n_long == 0 or n_short == 0:
            return []
        cell_size = (length / n_long + width / n_short) / 2

        long_step = long_end - long_start
        long_step = long_step / np.linalg.norm(long_step) * cell_size
        short_step = short_end - short_start
        short_step = (short_step / np.linalg.norm(short_step) * cell_size)

        # Phase 1: Build occupancy grid
        grid = np.zeros((n_short, n_long), dtype=np.uint8)

        for i in range(n_short):
            for j in range(n_long):
                origin = top_left + long_step * j + short_step * i
                center_uv = origin + long_step / 2 + short_step / 2
                theor_square = cv2.boxPoints((
                    (int(center_uv[0]), int(center_uv[1])),
                    (cell_size, cell_size),
                    np.degrees(rect_info.angle_px))
                )

                temp_mask = np.zeros(mask.shape, dtype=np.uint8)
                cv2.fillPoly(
                    temp_mask,
                    [theor_square.astype(np.int32)], 1)
                actual_area = np.count_nonzero((mask != 0) & (temp_mask == 1))
                theor_area = cell_size * cell_size

                self.logger.debug(
                    f"Grid cell ({i}, {j}) at pixel "
                    f"({int(origin[0])}, {int(origin[1])}): "
                    f"actual_area={actual_area:.1f}, "
                    f"theor_area={theor_area:.1f}, "
                    f"fill_ratio={actual_area / theor_area:.2f}"
                )

                if actual_area > self.block_fill_ratio * theor_area:
                    grid[i, j] = 1

        if grid.shape[0] > 1 or grid.shape[1] > 1:
            self.logger.debug(f"Grid of detected blocks within rectangle:\n{grid}")

        # Phase 2: Create SquareContourInfos with corner connections
        # Grid-based corners have deterministic ordering:
        #     c0 -------- c1
        #     |   (i,j)   |
        #     c3 -------- c2
        #
        # Adjacency -> shared corners:
        #   (i-1, j): c0, c1   (i+1, j): c2, c3
        #   (i, j-1): c0, c3   (i, j+1): c1, c2
        block_infos = []
        for i in range(n_short):
            for j in range(n_long):
                if not grid[i, j]:
                    continue

                c0 = top_left + long_step * j + short_step * i
                c1 = top_left + long_step * (j + 1) + short_step * i
                c2 = (top_left + long_step * (j + 1)
                      + short_step * (i + 1))
                c3 = top_left + long_step * j + short_step * (i + 1)
                corners_px = [c0, c1, c2, c3]

                corner_uvs = [(int(c[0]), int(c[1])) for c in corners_px]
                corner_xys = [self.mapper.pixel2world(int(c[0]), int(c[1])) for c in corners_px]
                center_uv = c0 + long_step / 2 + short_step / 2
                center_uv = (int(center_uv[0]), int(center_uv[1]))
                center_xy = self.mapper.pixel2world(center_uv[0], center_uv[1])

                conn_idx = []
                unique_corner_conn_idxs = set()
                if i > 0 and grid[i - 1, j]:
                    conn_idx.append((0, 1))
                    unique_corner_conn_idxs.update([0, 1])
                if i < n_short - 1 and grid[i + 1, j]:
                    conn_idx.append((2, 3))
                    unique_corner_conn_idxs.update([2, 3])
                if j > 0 and grid[i, j - 1]:
                    conn_idx.append((0, 3))
                    unique_corner_conn_idxs.update([0, 3])
                if j < n_long - 1 and grid[i, j + 1]:
                    conn_idx.append((1, 2))
                    unique_corner_conn_idxs.update([1, 2])
                
                corner_conns_px = [corner_uvs[k] for k in unique_corner_conn_idxs]
                    
                # NOTE: the vertical faces (meaning for z != this block's z) will never be detected as a connection
                # TODO: implement this if really needed. Prob not needed since even for the world map we won't be able
                # to see blocks under/above
                face_conns_xys = [
                    list(np.mean([corner_xys[k1], corner_xys[k2]], axis=0)) for (k1, k2) in conn_idx
                ]
                face_z = layer * BLOCK_SIZE - BLOCK_SIZE / 2
                face_conns_xyzs = [
                    (xys[0], xys[1], face_z) for xys in face_conns_xys
                ]

                block_infos.append(SquareContourInfo(
                    t=self.get_t(),
                    frame_idx=self.curr_frame_idx,
                    is_square=True,
                    center_uv=center_uv,
                    center_xy=center_xy,
                    size_px=cell_size,
                    corner_uvs=corner_uvs,
                    corner_xys=corner_xys,
                    center_z=BLOCK_SIZE/2,
                    angle=rect_info.angle,
                    quaternion=rect_info.quaternion,
                    corner_conns_px=corner_conns_px,
                    face_conns_xyzs=face_conns_xyzs
                ))

        return block_infos
    
    def _process_contour(
        self, contour: arr, mask: arr, cell_size_approx: int = 43,
        layer: int = 1,
    ) -> list[ContourInfo]:
        """Classify a contour as a single block or split a rectangle into blocks.

        Tries square classification first; if that fails, tries rectangle
        classification and splits the rectangle into individual block cells
        using the binary mask for fill-ratio verification.

        Arguments
        ---------
        contour : arr
            OpenCV contour array from cv2.findContours.
        mask : arr
            Binary uint8 mask for the current color (used by
            _split_rectangle_into_blocks for per-cell fill checks).
        cell_size_approx : int
            Expected block side length in pixels for grid splitting
            (default 43 for camera images, ~7 for voxel grid).

        Returns
        -------
        list[ContourInfo]
            Detected SquareContourInfo objects, or empty list if the
            contour does not match any known shape.
        """
        is_square, square_info = self._classify_block(contour)
        if is_square and square_info is not None:
            return [square_info]

        is_rect, rect_info = self._classify_rectangle(contour)
        if is_rect and rect_info is not None:
            block_infos = self._split_rectangle_into_blocks(rect_info, mask, cell_size_approx, layer)
            # block_infos = self._split_rectangle_into_blocks_experiment(rect_info, mask, cell_size_approx, contour, layer)
            return block_infos

        return []

    # ── Depth / Grip Helpers ──────────────────────────────────────

    def _check_depth_server(self) -> bool:
        """Check if the external depth server is reachable and ready."""
        import urllib.request
        try:
            req = urllib.request.Request(
                f"{self._depth_server_url}/health", method='GET')
            with urllib.request.urlopen(req, timeout=1) as resp:
                return b'"ready":true' in resp.read()
        except Exception:
            return False

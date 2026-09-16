"""ArUco-based perspective transform calibration.

Maps pixel coordinates from the overhead camera to world XY coordinates
(metres) using ArUco marker detection.  Supports single- and
multi-perspective modes with inverse-distance-weighted interpolation
across multiple marker sets.
"""

import time

import numpy as np
import cv2

from legobuilder.config import ARUCO_DX, ARUCO_DY

# ── Image Pre-processing ─────────────────────────────────────────────


def increase_contrast_clahe(frame, clip_limit=2.0, grid_size=(8, 8)):
    """Enhance image contrast via CLAHE on the L channel.

    Arguments
    ---------
    frame : np.ndarray
        RGB image.
    clip_limit : float
        CLAHE clip limit.
    grid_size : tuple[int, int]
        CLAHE tile grid size.

    Returns
    -------
    np.ndarray
        Contrast-enhanced RGB image.
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
    l_ch, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    cl = clahe.apply(l_ch)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

# ── Mapper ────────────────────────────────────────────────────────────


class Mapper:
    """Pixel-to-world coordinate mapper using ArUco perspective transforms.

    Detects four ArUco markers (IDs 1-4, DICT_4X4_50) in quadrants of the
    overhead camera frame and computes per-quadrant perspective transforms.
    In multi-perspective mode all four transforms are required; in
    single-perspective mode the available transforms are combined via
    inverse-distance weighting for smoother coordinate conversion.

    Attributes
    ----------
    use_multiple_perspectives : bool
        When *True*, use only the first available perspective transform.
        When *False*, blend all available transforms via inverse-distance
        weighting.
    aruco_settings : dict
        Per-quadrant ArUco configuration keyed by 'tl', 'tr',
        'bl', 'br' with cnt_x/cnt_y world offsets.
    pixel_to_world_offset : tuple[float, float]
        Additive (dx, dy) correction applied after transform.
    calibration_interval : float
        Seconds between automatic recalibrations (0 disables).
    perspective_transforms : dict[str, np.ndarray]
        Computed 3x3 perspective matrices keyed by quadrant name.
    marker_pixel_centers : dict[str, np.ndarray]
        Mean pixel position of each quadrant's 4 markers.
    is_calibrated : bool
        Whether at least one successful calibration has occurred.
    """

    # ── Lifecycle ─────────────────────────────────────────────────────

    def __init__(
        self,
        use_multiple_perspectives: bool,
        aruco_settings: dict,
        pixel_to_world_offset: tuple[float, float],
        calibration_interval: float,
        logger,
    ):
        """Initialise the mapper.

        Arguments
        ---------
        use_multiple_perspectives : bool
            Perspective blending mode.
        aruco_settings : dict
            Per-quadrant ArUco configuration.
        pixel_to_world_offset : tuple[float, float]
            Additive correction for pixel-to-world conversion.
        calibration_interval : float
            Seconds between automatic recalibrations.
        logger
            ROS-compatible logger instance.
        """
        self.use_multiple_perspectives = use_multiple_perspectives
        self.aruco_settings = aruco_settings
        self.pixel_to_world_offset = pixel_to_world_offset
        self.calibration_interval = calibration_interval
        self.logger = logger
        self.perspective_transforms = {}
        self.marker_pixel_centers = {}
        self.is_calibrated = False
        self._last_calibration_time = 0.0
        self._needs_recalibration = False

    # ── Public API ────────────────────────────────────────────────────

    def ensure_valid_mapping(self, frame: np.ndarray):
        """Ensure a valid perspective transform exists, recalibrating if needed.

        On first call, performs initial calibration.  Subsequent calls
        trigger recalibration when calibration_interval has elapsed.
        Falls back to the previous transform on recalibration failure.

        Arguments
        ---------
        frame : np.ndarray
            BGR image from the overhead camera.

        Returns
        -------
        bool
            *True* when a valid mapping is available.
        """
        if not self.is_calibrated:
            self.set_mapping(frame)
            if self.perspective_transforms == {}:
                self.logger.warning(
                    "Perspective transform not set. Either markers"
                    " not present or obstructed by objects.")
                return False
            self.logger.info("Perspective transform set")
            self.is_calibrated = True
            self._last_calibration_time = time.time()
            return True

        if self._needs_recalibration:
            # Safe: set_mapping assigns new dicts, so old refs stay valid
            prev_transforms = self.perspective_transforms
            prev_centers = self.marker_pixel_centers
            self.set_mapping(frame)
            if self.perspective_transforms == {}:
                self.perspective_transforms = prev_transforms
                self.marker_pixel_centers = prev_centers
                self.logger.warning("Recalibration failed, keeping previous calibration.")
                return True
            self.logger.info("Perspective transform recalibrated")
            self._needs_recalibration = False
            self._last_calibration_time = time.time()
            return True

        if (self.calibration_interval > 0
                and (time.time() - self._last_calibration_time) >= self.calibration_interval):
            self._needs_recalibration = True

        return True

    # ── Private Helpers ───────────────────────────────────────────────

    def set_mapping(self, frame: np.ndarray):
        """Detect ArUco markers and compute perspective transforms.

        Searches each quadrant (tl, tr, bl, br) for four ArUco markers,
        computes the perspective transform from pixel to world coordinates
        for each detected quadrant, and stores the results.

        Arguments
        ---------
        frame : np.ndarray
            RGB image from the overhead camera.
        """
        if frame is None:
            self.logger.warning(
                "set_perspective_transform: Frame is None")
            return

        frame = frame.copy()

        perspective_transforms = {}
        marker_pixel_centers = {}
        missing_markers = []

        for name, _ in self.aruco_settings.items():
            # Build a quadrant mask to isolate each marker set
            if name == 'tl':
                mask_incl = np.zeros(frame.shape[:2], dtype="uint8")
                mask_incl[0:frame.shape[0]//2, 0:frame.shape[1]//2] = 1
                frame = increase_contrast_clahe(
                    frame, clip_limit=3.0, grid_size=(8, 8))
            elif name == 'tr':
                mask_incl = np.ones(frame.shape[:2], dtype="uint8")
                # mask_incl[0:frame.shape[0]//2, frame.shape[1]//2:frame.shape[1]] = 1
            elif name == 'bl':
                mask_incl = np.zeros(frame.shape[:2], dtype="uint8")
                mask_incl[frame.shape[0]//2:frame.shape[0],
                          0:frame.shape[1]//2] = 1
            elif name == 'br':
                mask_incl = np.zeros(frame.shape[:2], dtype="uint8")
                mask_incl[frame.shape[0]//2:frame.shape[0],
                          frame.shape[1]//2:frame.shape[1]] = 1
            else:
                self.logger.warning(
                    f"set_perspective_transform: Unknown Aruco"
                    f" setting name {name}")
                continue
            new_frame = cv2.bitwise_and(frame, frame, mask=mask_incl)

            markerCorners, markerIds, _ = cv2.aruco.detectMarkers(
                new_frame,
                cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))

            if (markerIds is None or len(markerIds) != 4
                    or set(markerIds.flatten()) != set([1, 2, 3, 4])):
                self.logger.warning(
                    f"set_perspective_transform: marker {name}"
                    " not detected")
                missing_markers.append(name)
                continue

            uvMarkers = np.zeros((4, 2), dtype='float32')
            for i in range(4):
                uvMarkers[markerIds[i]-1, :] = np.mean(
                    markerCorners[i], axis=1)

            # Physical marker spacing (metres) from workspace centre
            DX = ARUCO_DX / 2
            DY = ARUCO_DY / 2
            x0 = 0
            y0 = 0
            xyMarkers = np.float32(
                [[x0+dx, y0+dy] for (dx, dy) in
                 [(-DX, DY), (DX, DY), (-DX, -DY), (DX, -DY)]])

            M = cv2.getPerspectiveTransform(uvMarkers, xyMarkers)
            perspective_transforms[name] = M
            marker_pixel_centers[name] = np.mean(uvMarkers, axis=0)

        if len(missing_markers) > 0 and self.use_multiple_perspectives:
            self.logger.warning(
                f"set_perspective_transform: Missing markers"
                f" ({missing_markers}) for multiple perspectives,"
                " not setting transforms")
            return
        else:
            self.perspective_transforms = perspective_transforms
            self.marker_pixel_centers = marker_pixel_centers

        self.logger.debug(
            f"Perspective transforms: {self.perspective_transforms}")

    def pixel2world(self, u: int, v: int, no_offset: bool = False) -> tuple[float, float]:
        """Convert a pixel coordinate to world XY using stored transforms.

        In single-perspective mode uses only the first available
        transform.  Otherwise blends all transforms via inverse-distance
        weighting relative to each quadrant's marker centre.

        Arguments
        ---------
        u : int
            Pixel column (horizontal).
        v : int
            Pixel row (vertical).

        Returns
        -------
        tuple[float, float]
            World (x, y) in metres.
        """
        def _pixel_to_world(uv, M, aruco_cnt_xy):
            """Apply a single perspective transform with offset."""
            u_px, v_px = uv
            cnt_x, cnt_y = aruco_cnt_xy
            uvObj = np.float32([u_px, v_px])
            xyObj = cv2.perspectiveTransform(
                uvObj.reshape(1, 1, 2), M).reshape(2)
            return (cnt_x + xyObj[0], cnt_y + xyObj[1])

        if not self.use_multiple_perspectives:
            world_coords = []
            distances = []
            for name, M in self.perspective_transforms.items():
                aruco_settings = self.aruco_settings[name]
                world_coord = _pixel_to_world(
                    (u, v), M,
                    (aruco_settings['cnt_x'], aruco_settings['cnt_y']))
                world_coords.append(world_coord)

                marker_uv = self.marker_pixel_centers[name]
                dist = np.sqrt(
                    (u - marker_uv[0])**2 + (v - marker_uv[1])**2)
                distances.append(dist)

            world_coords = np.array(world_coords)
            distances = np.array(distances)

            weights = 1.0 / (distances + 1e-6)
            weights = weights / np.sum(weights)

            weighted_coord = np.sum(
                world_coords * weights[:, np.newaxis], axis=0)
            ox = 0.0 if no_offset else self.pixel_to_world_offset[0]
            oy = 0.0 if no_offset else self.pixel_to_world_offset[1]
            return (
                weighted_coord[0] + ox,
                weighted_coord[1] + oy,
            )
        else:
            first_name = list(self.perspective_transforms.keys())[0]
            aruco_settings = self.aruco_settings[first_name]
            M = self.perspective_transforms[first_name]
            aruco_cnt_xy = (
                aruco_settings['cnt_x'], aruco_settings['cnt_y'])
            world_coord = _pixel_to_world((u, v), M, aruco_cnt_xy)
            ox = 0.0 if no_offset else self.pixel_to_world_offset[0]
            oy = 0.0 if no_offset else self.pixel_to_world_offset[1]
            return (
                world_coord[0] + ox,
                world_coord[1] + oy,
            )

    def world2pixel(self, x: float, y: float) -> tuple[int, int]:
        """Convert a world XY coordinate to pixel UV (inverse of pixel2world).

        Uses the first available perspective transform. Inverts the
        forward mapping: subtracts offsets, then applies the inverse
        perspective matrix.

        Arguments
        ---------
        x : float
            World x in metres.
        y : float
            World y in metres.

        Returns
        -------
        tuple[int, int]
            Pixel (u, v) coordinates.
        """
        if not self.perspective_transforms:
            return (0, 0)

        first_name = list(self.perspective_transforms.keys())[0]
        aruco_settings = self.aruco_settings[first_name]
        M = self.perspective_transforms[first_name]
        cnt_x, cnt_y = aruco_settings['cnt_x'], aruco_settings['cnt_y']

        # Undo the offset and ArUco center shift
        wx = x - self.pixel_to_world_offset[0] - cnt_x
        wy = y - self.pixel_to_world_offset[1] - cnt_y

        M_inv = np.linalg.inv(M)
        world_pt = np.float32([wx, wy])
        pixel_pt = cv2.perspectiveTransform(
            world_pt.reshape(1, 1, 2), M_inv).reshape(2)
        return (int(round(pixel_pt[0])), int(round(pixel_pt[1])))

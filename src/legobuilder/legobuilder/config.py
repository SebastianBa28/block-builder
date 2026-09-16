"""Unified configuration for the legobuilder package.

All tunable parameters for brain, vision, and kinematics modules are
consolidated here: node rates, joint limits, gripper settings, trajectory
planning, collision detection, IK solver, object dimensions, HSV color
thresholds, detection/tracking, structure build demo, gravity compensation,
and 3D world mapping.
"""

import numpy as np
from math import pi, atan2
from dataclasses import dataclass, field
from legobuilder.schemas import ObjectType
from legobuilder.vision.hsv_presets import HSV_PRESETS, Color
from rclpy.impl.rcutils_logger import RcutilsLogger

# ── Logging Control ──────────────────────────────────────────────────
INFO_MODE = False  # Set False to suppress all self.logger.info() calls

_original_rcutils_info = RcutilsLogger.info

def _guarded_info(self, message, **kwargs):
    if INFO_MODE:
        return _original_rcutils_info(self, message, **kwargs)

RcutilsLogger.info = _guarded_info

def wrap_roll(angle: float) -> float:
    """Wrap an angle to [-pi, pi] to match IK solver roll range."""
    if angle > pi:
        angle -= 2 * pi
    elif angle < -pi:
        angle += 2 * pi
    return angle

# ── Node Rates ──────────────────────────────────────────────────────
MANIPULATOR_RATE = 100.0  # Hz
BRAIN_RATE = 5  # Hz

# ── Heartbeat Settings ──────────────────────────────────────────────
HEARTBEAT_RATE = 1.0  # Hz
HEARTBEAT_TIMEOUT = 3.0  # seconds before considering a node offline

# ── Robot Configuration ─────────────────────────────────────────────
JOINT_NAMES_3DOF = ["base", "shoulder", "elbow"]
Q_READY_3DOF = np.array([pi/2, 0, pi/2])

MOTOR_NAMES = ['4.5', '8.6', '4.4', '4.3', '4.1', '4.2']
JOINT_NAMES = [
    "base", "shoulder", "elbow",
    "wrist_tilt", "wrist_roll", "tip_slide",
]
CAMERA_JOINT_NAMES = ["base", "shoulder", "elbow", "wrist_tilt"]
TILT_JOINT_INDICES = [
    JOINT_NAMES.index("shoulder"),
    JOINT_NAMES.index("elbow"),
    JOINT_NAMES.index("wrist_tilt"),
]
ROLL_JOINT_INDICES = [
    JOINT_NAMES.index("base"),
    JOINT_NAMES.index("wrist_roll"),
]
TILT_LIMITS = (-5/4 * pi, pi/4)  # radians
NUM_DOFS = 6
LAMBDA = 20.0  # damped-least-squares damping factor for IK

# Workspace bounds (relative to base motor)
OUTER_RADIUS = 0.6  # meters (conservative, gripper facing down)
INNER_RADIUS = 0.25  # meters
# Gap: BASE_MOTOR_POS is approximate — measure actual base motor XYZ
# for accurate reachability checks
BASE_MOTOR_POS = np.array([0.7635, 0.1016, 0.0254])

# ── Gripper ─────────────────────────────────────────────────────────
# Motor radians <-> URDF prismatic meters conversion.
# In the URDF, higher prismatic value = fingers closer (closed).
GRIPPER_MOTOR_CLOSED_RAD   = -0.3
GRIPPER_MOTOR_OPEN_RAD     =  0.0
GRIPPER_PRISMATIC_CLOSED_M =  0.015
GRIPPER_PRISMATIC_OPEN_M   =  0.0
GRIPPER_JOINT_INDEX = JOINT_NAMES.index("tip_slide")  # 5
GRIP_FAILURE_EFFORT_THRESHOLD = -0.7   # Nm — effort > this means no block held
GRIP_FAILURE_CONSECUTIVE_FRAMES = 10   # ticks at 100Hz = 100ms debounce

# ── Idle chomp animation ────────────────────────────────────────────
IDLE_CHOMP               = True  # enable/disable idle gripper animation
IDLE_CHOMP_BURST_COUNT   = 4     # open/close cycles per burst
IDLE_CHOMP_DURATION      = 0.3   # seconds for each open or close motion
IDLE_CHOMP_PAUSE         = 5.0   # seconds between bursts
IDLE_CHOMP_INITIAL_DELAY = 1.0   # seconds idle before first chomp starts
IDLE_CHOMP_CLOSED_RAD    = -0.15 # half-closed to reduce motor strain

# ── Idle structure scanning ────────────────────────────────────────
IDLE_SCAN_INTERVAL       = 4.0   # seconds between idle structure scans

def grip_check_roll(p: np.ndarray) -> float:
    """Task-space roll that aligns gripper perpendicular to the arm.

    Computes wrist_roll such that the gripper fingers are
    perpendicular to the radial direction from the base motor.
    The relationship is roll = -atan2(dy, dx) + pi/2 where
    (dx, dy) is the vector from the base motor to the target.

    Arguments
    ---------
    p : np.ndarray
        3D target position [x, y, z].

    Returns
    -------
    float
        Task-space roll angle (radians).
    """
    return wrap_roll(
        -atan2(p[1] - BASE_MOTOR_POS[1], p[0] - BASE_MOTOR_POS[0])
        + pi / 2
    )

# Diagonal grip detection
DIAGONAL_GRIP_MAX_RETRIES = 2

# Visual grip check (Depth Anything V2)
VISUAL_GRIP_CHECK = True
DEPTH_ANYTHING_MODEL_ID = 'depth-anything/Depth-Anything-V2-Base-hf'
DEPTH_SERVER_URL = 'http://localhost:8000'
DASHBOARD_SERVER_URL = 'http://localhost:8001'

# Image crop: remove left portion (background) before depth estimation
GRIP_IMAGE_CROP_X = 0.65  # fraction of image width to remove from the left

# Column-std roll detection (10 columns extending left of block trapezoid)
GRIP_COLUMN_STD_THRESHOLD = 8.5  # per-column std above this = "high" (CALIBRATE)
GRIP_DIAGONAL_MIN_COLS = 8  # >= this many high-std columns = DIAGONAL
GRIP_PARTIAL_MIN_COLS = 4  # >= this many high-std columns = PARTIAL

# High detection (block_mean - finger_mean, computed at runtime)
GRIP_HIGH_THRESHOLD = 15.0  # block protrudes above this = HIGH (CALIBRATE)

# Brain timeout waiting for grip check response
GRIP_CHECK_TIMEOUT = 3.0  # seconds

GRIP_CONNECTION_DETECTION = True
CONNECTION_DETECTION_TIMEOUT = 2.0   # seconds to wait for contour data
CONNECTION_DETECTION_RADIUS = 0.4   # meters — FK radius to count blocks near gripper
CONNECTION_DETECTION_POS = np.array([0.75, 0.28, 0.0]) # Detection position
CONNECTION_DROP_HEIGHT = 0.10        # meters above table for drop
CONNECTION_DETECTION_CELL_SIZE = 150  # pixels — block cell size when arm is at detection pose (closer to camera)


def gripper_rad_to_meters(rad_value: float) -> float:
    """Convert gripper motor position from radians to prismatic meters.

    Linearly interpolates between closed and open limits, then clamps
    the result to the valid prismatic range.

    Arguments
    ---------
    rad_value : float
        Motor position in radians.

    Returns
    -------
    float
        Equivalent prismatic distance in meters.
    """
    t = (rad_value - GRIPPER_MOTOR_CLOSED_RAD) / (
        GRIPPER_MOTOR_OPEN_RAD - GRIPPER_MOTOR_CLOSED_RAD
    )
    meters = (
        GRIPPER_PRISMATIC_CLOSED_M
        + t * (GRIPPER_PRISMATIC_OPEN_M - GRIPPER_PRISMATIC_CLOSED_M)
    )
    lo = min(GRIPPER_PRISMATIC_CLOSED_M, GRIPPER_PRISMATIC_OPEN_M)
    hi = max(GRIPPER_PRISMATIC_CLOSED_M, GRIPPER_PRISMATIC_OPEN_M)
    return max(lo, min(hi, meters))


# ── Trajectory Planning ─────────────────────────────────────────────
DURATION = 6.0  # seconds — default trajectory duration
ARC_HEIGHT = 0.2  # meters
ARC_SPEED = 0.05  # m/s
Q_READY = np.array([
    0, 0, pi/2, -0.785398, 0, GRIPPER_MOTOR_OPEN_RAD,
])

Q_SCAN = np.array([
    0.70, -0.05, 1.24, -1.09, 0, GRIPPER_MOTOR_OPEN_RAD,
])

Q_CHOMP = np.array([
    -0.70, 0, pi/2, -0.785398, 0, GRIPPER_MOTOR_OPEN_RAD,
])

Q_CONNECTION_CHECK = np.array([
    -0.2, 0.2, pi/2, pi/2, 0.2, GRIPPER_MOTOR_CLOSED_RAD,
])

# ── Block Separation ────────────────────────────────────────────────
SEPARATION_TILT_HEIGHT = 0.1  # m above block to bring before tilting
SEPARATION_TILT_ANGLE = pi / 3  # radians to tilt by before separating
SEPARATION_INNER_THRESHOLD = 0.30  # m, radial distance below which we nudge outward
SEPARATION_RADIAL_NUDGE = 0.05     # m, how far to move outward

# ── Collision Detection & Recovery ──────────────────────────────────
COLLISION_DETECTION = True
POSITION_ERROR_THRESHOLD = 1.0  # radians
VELOCITY_ERROR_THRESHOLD = 6.0  # rad/s
EFFORT_ERROR_THRESHOLD = 3.0  # Nm
COLLISION_WAIT_DURATION = 2.0  # seconds to wait after collision

# ── Jerk Prevention Harness ────────────────────────────────────────
JERK_PREVENTION = True  # toggleable like COLLISION_DETECTION
JERK_THRESHOLD_RAD = 0.5  # joint-space norm threshold (rad)
JERK_IGNORE_COUNT = 50  # consecutive bad ticks before smooth recovery (~0.5s at 100Hz)
JERK_RECOVERY_DURATION = 2.0  # seconds for smooth trajectory to target

LOOP_GAP_RECOVERY = False

# ── Inverse Kinematics ──────────────────────────────────────────────
IK_MAX_ITERATIONS = 10_000
IK_TOLERANCE = 1e-3
IK_SIM_DT = 0.05

# ── Object Dimensions ──────────────────────────────────────────────
BLOCK_SIZE = 0.034036   # meters
DISK_DIMS = {
    'radius': 0.02,      # meters
    'thickness': 0.0075,  # meters
}
STRIP_DIMS = {
    'length': 0.112,  # meters
    'width': 0.025,   # meters
}

# ── Vision / Detection Settings ─────────────────────────────────────
HSV_PRESET = 'day'
HSV_LIMITS = HSV_PRESETS[HSV_PRESET]
EE_CAM_HSV_LIMITS = HSV_PRESETS['depth_camera']

# ArUco marker positions for perspective transform
ARUCO_DX = 1.334 # Distance between aruco markers in meters in x direction
ARUCO_DY = 0.679 # Distance between aruco markers in meters in y direction
ARUCO_SETTINGS = {
    'tr': {
        'cnt_x': 0.17 + ARUCO_DX / 2,
        'cnt_y': 0.022 + ARUCO_DY / 2,
    },
}
USE_MULTIPLE_PERSPECTIVES = False

USE_ERROR_MAP = True

# Pixel-to-world coordinate offset (x, y) in meters
PIXEL_TO_WORLD_OFFSET = (-0.01, 0.01)
# PIXEL_TO_WORLD_OFFSET = (-0.06, -0.02)
# PIXEL_TO_WORLD_OFFSET = (0.0, 0.0)

# Calibration interval in seconds (0 = once, >0 = periodic)
CALIBRATION_INTERVAL = 100.0

ERODE_DILATE_ITERATIONS = 2  # morphological open/close iterations

# Contour detection tolerances (lower = stricter)
CIRCLE_CONTOUR_TOLERANCE = 0.2
RECTANGLE_CONTOUR_TOLERANCE = 1.0
SQUARE_CONTOUR_TOLERANCE = 1.5 # side-length ratio threshold for squareness
SQUARE_LENGTH_RANGE = (0.02, 0.055) # meters — minimum and maximum side length for square classification
BLOCK_FILL_RATIO = 0.6  # minimum ratio of block-colored pixels within square contour for block classification during rectangle separation
ASPECT_RATIO_THRESHOLD = 1.5  # maximum aspect ratio for square classification (length/width)

# ── Worldmap Contour Detection ─────────────────────────────────────
WORLDMAP_CONTOUR_MAX_LEVELS = 10           # max height levels to scan
WORLDMAP_CONTOUR_HUE_THRESHOLD = 50        # max circular hue distance (0-179 OpenCV scale)
WORLDMAP_CONTOUR_ERODE_DILATE_ITERS = 0    # morphological iterations for voxel grid
WORLDMAP_CONTOUR_MIN_VOXELS = 3            # min voxels per level to consider occupied
WORLDMAP_MASK_SCALE = 1                    # upscale factor for voxel grid masks before morphological processing
WORLDMAP_COLOR_HUES = {
    color: (limits[0, 0] + limits[0, 1]) / 2.0
    for color, limits in EE_CAM_HSV_LIMITS.items()
}
WORLDMAP_COLOR_HUES = {
    Color.GREEN: 80,
    Color.YELLOW: 30,
    Color.BLUE: 100
}


# ── Object Detection / Tracking ─────────────────────────────────────
BUFFER_DURATION = 2.0  # seconds

CLUSTERING_SETTINGS = {
    ObjectType.DISK: {
        'eps': 0.04,
        'min_samples': 2,
        'length_threshold': 2,
        'error_threshold': 0.02,
        'max_items': 2,
    },
    ObjectType.STRIP: {
        'eps': 0.06,
        'min_samples': 2,
        'length_threshold': 2,
        'error_threshold': 0.02,
        'max_items': 2,
    },
    ObjectType.YELLOW_BLOCK: {
        'eps': 0.02,
        'min_samples': 2,
        'length_threshold': 2,
        'error_threshold': 0.02,
        'max_items': 7,
    },
    ObjectType.BLUE_BLOCK: {
        'eps': 0.02,
        'min_samples': 2,
        'length_threshold': 2,
        'error_threshold': float('inf'),
        'max_items': 7,
    },
    ObjectType.GREEN_BLOCK: {
        'eps': 0.02,
        'min_samples': 2,
        'length_threshold': 2,
        'error_threshold': 0.02,
        'max_items': 7,
    },
    ObjectType.RED_BLOCK: {
        'eps': 0.02,
        'min_samples': 2,
        'length_threshold': 2,
        'error_threshold': 0.02,
        'max_items': 7,
    },
}

STABILITY_THRESHOLDS = {
    ObjectType.DISK: {
        'position': 0.1,  # meters
        'quaternion': float('inf'),
    },
    ObjectType.STRIP: {
        'position': float('inf'),
        'quaternion': float('inf'),
    },
    ObjectType.YELLOW_BLOCK: {
        'position': 0.2,
        'quaternion': float('inf'),
        'angle': float('inf'),  # radians
    },
    ObjectType.BLUE_BLOCK: {
        'position': 0.2,
        'quaternion': float('inf'),
        'angle': float('inf'),
    },
    ObjectType.GREEN_BLOCK: {
        'position': 0.2,
        'quaternion': float('inf'),
        'angle': float('inf'),
    },
    ObjectType.RED_BLOCK: {
        'position': 0.2,
        'quaternion': float('inf'),
        'angle': float('inf'),
    },
}

# Presence/absence duration thresholds
PRESENCE_DURATION_THRESHOLD = 0.5  # seconds
ABSENCE_DURATION_TOLERANCE = 1.0  # seconds

# Moving average smoothing window
SMOOTHING_WINDOW_DURATION = 1.5  # seconds

# ── Block Inventory ────────────────────────────────────────────────
AVAILABLE_BLOCKS: dict[ObjectType, int] = {
    ObjectType.GREEN_BLOCK: 14,
    ObjectType.YELLOW_BLOCK: 14,
    ObjectType.BLUE_BLOCK: 14,
}

# ── Grid Limits ───────────────────────────────────────────────────
MAX_GRID_SIZE = {"length": 6, "width": 6, "height": 6}

# ── Structure Build Demo ────────────────────────────────────────────
STRUCTURE_BUILD_DEMO = True
PLACEMENT_GRID_COORDS = np.array([
    [0.362, 0.324],   # bl
    [0.563, 0.324],   # br
    [0.362, 0.525],   # tl
    [0.563, 0.525],   # tr
])
PLACEMENT_GRID_COORD_OFFSET = np.array([BLOCK_SIZE/2,  -BLOCK_SIZE/2])
PLACEMENT_GRID_COORDS +=PLACEMENT_GRID_COORD_OFFSET
# PLACEMENT_GRID_COORDS = np.array([
#     [0.354 - BLOCK_SIZE / 2, 0.352 - BLOCK_SIZE / 2],   # bl
#     [0.532 + BLOCK_SIZE / 2, 0.352 - BLOCK_SIZE / 2],   # br
#     [0.354 - BLOCK_SIZE / 2, 0.527 + BLOCK_SIZE / 2],   # tl
#     [0.532 + BLOCK_SIZE / 2, 0.527 + BLOCK_SIZE / 2],   # tr
# ])

GRID_CENTER_XY = (
    sum(c[0] for c in PLACEMENT_GRID_COORDS) / len(PLACEMENT_GRID_COORDS),
    sum(c[1] for c in PLACEMENT_GRID_COORDS) / len(PLACEMENT_GRID_COORDS),
)

GRID_PLACEMENT = True   # If true, specify origin via grid index, else by (x,y)


# Purple HSV range for overhead camera
OVERHEAD_PURPLE_HSV = np.array([[120, 145], [146, 255], [120, 255]])
EE_PURPLE_HSV = np.array([[110, 133], [92, 255], [113, 255]])
CORNER_DETECTION_THRESHOLD = 1.5 * BLOCK_SIZE
PLACEMENT_PUSH = False
FLOATING_PUSH_DISTANCE = 0.000  # meters — horizontal push for floating blocks

# ── Placement Verification ─────────────────────────────────────
SHOW_LEVEL_SEGMENTATIONS_VISUALIZATION = False
PLACEMENT_VERIFICATION = True
VERIFY_SCAN_HEIGHT = 0.50              # m above target block for EE camera
VERIFY_SCAN_DWELL_TIME = 3.0           # seconds stationary for scan integration
VERIFY_SCAN_INTERVAL = 0.3              # seconds between scan requests during dwell
VERIFY_DETECT_TIMEOUT = 3.0             # seconds to wait for detect response
VERIFY_POSITION_TOLERANCE = BLOCK_SIZE / 2  # ~17mm for position match
VERIFY_MIN_ORIGIN_VOTES = 2             # min matched blocks to update origin
MAX_PLACEMENT_VERIFY_RETRIES = 2
MAX_VERIFY_GRIP_RETRIES = 2             # max recovery grasp retries on grip failure
VERIFY_MOVE_DURATION = 2.0              # seconds for verification movements
ORBIT_SCAN_POSITIONS = 3
ORBIT_SCAN_RADIUS = 0.00             # m lateral offset from target
ORBIT_SCAN_HEIGHT = 0.50              # m above target for orbit viewpoints
ORBIT_SCAN_DWELL_TIME = 1.0            # seconds at each orbit position
HISTORICAL_PLACED_GRID_RESCAN_COUNT = 1  # number of orbit rescans to trigger when next_target_block falls in the previous placed_grid
VERIFY_SCAN_TILT = -3 * pi / 4         # tilt that makes EE camera face straight down
VERIFY_SCAN_SOURCE = 1 # 0 = accumulated worldmap, 1 = latest scan worldmap
VERIFY_TARGET_CENTERED = False  # True = scan/orbit around target block, False = grid center
ENABLE_IK_PRECOMPUTE = True  # Speculative IK for next queued object
EE_DEPTH_SCAN = True  # end-effector depth camera world mapping
ENABLE_POINTCLOUD_CACHING = False  # off preserves current rebuild-every-scan behavior
ENABLE_CAMERA_FUSION = False  # cross-validate overhead + EE camera detections
RECORD_BAG = False
PUBLISH_DETECTOR_IMAGES = True
PUBLISH_DETECTOR_POINTCLOUDS = True
LOW_COMPUTE_MODE = False

PILE_X_MIN = 0.65  # blocks with center x > this are in the pickup pile
PILE_CENTER = np.array([1.03, 0.46, 0.0])  # reference point for pile drop and connection detection
MAX_PENDING_OBJECTS = 1
QUEUE_MOVEMENT_THRESHOLD = 0.06  # meters — matches DBSCAN eps
GRIP_FAILURE_RECOVERY = True

# Approximate XY occlusion radius per link (motor half-width + margin)
ARM_OCCLUSION_RADII = {
    'base':       0.035,  # X8 motor (~22.5 mm)
    'shoulder':   0.035,  # X8 motor
    'elbow':      0.025,  # X5 motor (~15.6 mm)
    'wrist_tilt': 0.025,  # X5 motor
    'wrist_roll': 0.025,  # X5 motor
    'tip_slide':  0.050,  # gripper base (100 mm wide / 2)
}

BLOCK_TOP_OFFSET = 0.01
PICK_Z = BLOCK_SIZE / 4 + BLOCK_TOP_OFFSET # Small offset to pick from the topbottom of block not middle
PICK_TILT = -pi          # gripper pointing straight down
# ── Gravity Compensation ────────────────────────────────────────────
STARTUP_GRAV_DURATION = 5.0  # seconds
TEST_GRAVITY = False
TEST_GRIPPER = False
TEST_GRIPPER_EFFORT = -1.0  # Nm, constant closing effort (CALIBRATE)


# ── Calibration ────────────────────────────────────────────
CALIBRATE_Z = False # Whether to produce a z error map before building structure
CALIBRATE_XY = False # Whether to produce an xy error map before building structure
USE_EXISTING_XY_CALIBRATIONS = True  # If CALIBRATE_XY is False, load xy_calibrations.csv instead of offsets_default.csv
USE_EXISTING_Z_CALIBRATIONS = False   # If CALIBRATE_Z is False, load z_calibration.csv on startup
CALIBRATE_GRID_COORDS = False        # detect grid corners from overhead camera at startup

# XY-calibration grid: sweep X and Y across the reachable workspace at a
# fixed safe height with the wrist pointing straight down.
CALIBRATE_XY_POSES = []
_xy_height = 0.08                       # fixed z for all xy-cal poses
_xy_tilt = -3 / 4 * pi                  # EE camera facing straight down
_xy_steps = [
    (0.362, 0.324),   # bl
    (0.563, 0.324),   # br
    (0.362, 0.525),   # tl
    (0.563, 0.525),   # tr
    (0.4625, 0.4245),  # middle
    (0.3, 0.3),
    (0.3, 0.5),
    (0.6, 0.3),
    (0.6, 0.6),
    (0.75, 0.45),
    (0.75, 0.65),
    (0.9, 0.3),
    (0.9, 0.6),
    (1.2, 0.3),
    (1.2, 0.6)
]
for (_x, _y)in _xy_steps:
    _r = np.sqrt((_x - BASE_MOTOR_POS[0])**2 + (_y - BASE_MOTOR_POS[1])**2)
    if INNER_RADIUS <= _r <= OUTER_RADIUS:
        p = np.array([_x, _y, _xy_height])
        CALIBRATE_XY_POSES.append(
            (p, np.array([_xy_tilt, grip_check_roll(p)]))
        )
        
CALIBRATE_MOVE_DURATION = 3.0      # seconds for arm movements
CALIBRATE_WAIT_DURATION = 2.0      # seconds to dwell at scan position
CALIBRATE_SAFE_HEIGHT = 0.25       # meters — intermediate Z to lift to between moves

# Define sequence of (position, orientation) tuples for z-calibration at startup.
# These are converted to ManipulatorState objects in calibration.py to avoid
# circular imports (config → block_manipulator → kinematic_chain → config).
CALIBRATE_Z_POSES = []
_layers = 2     # number of z-height layers to sample
_samples = 4
# number of radial (y) positions per layer
_p0 = np.array([0.764, 0.45, 0.05]) # starting pos: arm is right in front of base motor
_dp = (0.6 - 0.37) / (_samples - 1) # step size in y to sweep from near (0.37m) to far reach (0.66m)
_dh = 0.1 # each layer is _dh m higher than previous
for _i in range(_layers):
    for _j in range(_samples):
        _p = _p0 + np.array([0, _j * _dp, _i * _dh + .08])
        CALIBRATE_Z_POSES.append((_p.copy(), np.array([-3/4 * pi, 0]))) # have camera point ~straight down

XY_CAL_ROI_RADIUS_PX = 200  # pixel radius of circular ROI mask for blue circle detection
OVERHEAD_CAM_POS = (0.829, 0.33, 1.05)  # overhead camera world position (x, y, z) in metres
XY_CAL_SCAN_TIMEOUT = 3.0   # seconds to wait for blue circle detection before skipping pose

# ── 3D World Mapping ────────────────────────────────────────────────


@dataclass
class WorldMapConfig:
    """Configuration for 3D world mapping and voxel grid.

    Includes Open3D WorldMap settings (range filtering, outlier removal,
    downsampling) and sparse VoxelStore settings (resolution, workspace
    bounds, log-odds priors, observation limits, decay, frustum FOV,
    and pruning).

    Frustum-based miss model: when processing a depth scan, any voxel that 
    falls within the camera's field of view but is not observed (i.e. no 
    point hits it) is considered a "miss" and has its log-odds decremented 
    accordingly — but only if its current confidence (log-odds) is above 
    the confidence_threshold

    Attributes
    ----------
    range_min : float
        Minimum depth range in meters for point cloud filtering.
    range_max : float
        Maximum depth range in meters for point cloud filtering.
    voxel_resolution : float
        Voxel side length in meters.
    workspace_base_xy : np.ndarray
        XY position of the base motor in world frame.
    workspace_inner_radius : float
        Inner cylindrical workspace bound in meters (with leeway).
    workspace_outer_radius : float
        Outer cylindrical workspace bound in meters (with leeway).
    workspace_z_min : float
        Minimum Z workspace bound in meters.
    workspace_z_max : float
        Maximum Z workspace bound in meters.
    log_odds_prior : float
        Initial log-odds for new voxels (negative = skeptical).
    log_odds_hit : float
        Log-odds increment on observation.
    log_odds_miss : float
        Log-odds decrement on miss.
    log_odds_max : float
        Upper clamp for log-odds accumulation.
    log_odds_min : float
        Lower clamp for log-odds accumulation.
    obs_count_max : int
        Maximum observation count per voxel (caps decay recovery).
    decay_after_scans : int
        Number of scans before decay activates.
    decay_factor : float
        Multiplicative decay for stale positive log-odds (0 < f < 1).
    prune_interval : int
        Scans between dead-voxel garbage collection sweeps.
    confidence_threshold : float
        Log-odds floor for queries and frustum miss.
        When querying (from brain usually), only voxels above this threshold
        are considered occupied. When applying a frustum-based miss model,
        voxels with a low confidence (below this threshold) are ignored.
    publish_every_n_scans : int
        RViz PointCloud2 publishing throttle (1 = every scan).
    camera_hfov : float
        Horizontal FOV in radians for frustum-based miss model.
    camera_vfov : float
        Vertical FOV in radians for frustum-based miss model.
    kalman_enabled : bool
        Enable Kalman filter refinement of voxel positions.
    z_remainder_threshold : float
        Maximum distance (in voxel units) from the nearest block-height
        boundary for a voxel to survive filtering.  float('inf')
        disables the filter entirely (default). 
        voxel units = meters / voxel_resolution
    block_height : float
        Physical block thickness in metres, used to convert
        z_remainder_threshold from voxel units to metric space.
    max_voxels : int
        Upper bound on active voxel count.  Eviction triggers when
        the store exceeds this limit.
    kalman_process_noise : float
        Kalman Q: process noise (low for static scene).
    kalman_measurement_noise : float
        Kalman R: measurement noise (~3 mm std at 0.3-0.6 m).
    """

    range_min: float = 0.1
    range_max: float = 1.2

    voxel_resolution: float = 0.002  # m/voxel

    workspace_base_xy: np.ndarray = field(
        default_factory=lambda: BASE_MOTOR_POS[:2].copy()
    )
    workspace_inner_radius: float = INNER_RADIUS - 0.1
    workspace_outer_radius: float = OUTER_RADIUS + 0.1
    workspace_z_min: float = 0.0
    workspace_z_max: float = 0.4

    log_odds_prior: float = 0
    # log_odds_hit: float = 0.8
    # log_odds_miss: float = -0.8
    # log_odds_max: float = 4.0
    # log_odds_min: float = -4.0
    log_odds_hit: float = 4.0
    log_odds_miss: float = -0.2
    log_odds_max: float = float('inf')
    log_odds_min: float = -2

    obs_count_max: int = 20

    decay_after_scans: int = 6
    decay_factor: float = 0.5

    prune_interval: int = 10

    confidence_threshold: float = 1.5 # log-odds threshold for queries and frustum miss (requires ~3 consistent hits)

    publish_every_n_scans: int = 1

    # https://www.bhphotovideo.com/c/product/1495418-REG/intel_82635d435idk5p_realsense_depth_camera_d435i.html
    camera_hfov: float = 1.21  # 69.4 deg
    camera_vfov: float = 0.74  # 42.5 deg

    z_remainder_threshold: float = float('inf')  # in voxel units (m / voxel_resolution)

    max_voxels: int = 100_000
    
    # Loose
    scan_sync_dt_threshold: float = 0.2         # seconds - max time difference between depth frame and joint state for sync
    scan_sync_velocity_threshold: float = 2.0   # rad/s - disregard depth frames when joint speed exceeds this
    scan_sync_cooldown_samples: int = 0         # joint states to skip after a fast movement

    # Kalman refinement; TODO: tune, disable, or replace with more sophisticated method if it doesn't help
    kalman_enabled: bool = False
    kalman_process_noise: float = 1e-8       # Q: very low for static scene (blocks don't move)
    kalman_measurement_noise: float = 9e-6   # R: ~3mm std per axis (RealSense at 0.3-0.6m)

PERPETUAL_SCAN_INTERVAL = None  # seconds between map scans; None don't do a perpetual worldmap scan
JOINT_STATE_BUFFER_DURATION = 1.0  # seconds of joint history for scan sync

# Trapezoid ROI corners (normalized) for grip analysis
GRIP_TOP_FINGER_TRAP = [
    (0.75, 0.348), (0.825, 0.188), (0.998, 0.138), (0.994, 0.31),
]
GRIP_BLOCK_TRAP = [
    (0.797, 0.36), (0.997, 0.331), (0.997, 0.688), (0.806, 0.656),
]
GRIP_BOTTOM_FINGER_TRAP = [
    (0.827, 0.825), (0.742, 0.665), (0.997, 0.7), (0.997, 0.871),
]

"""Post-placement verification and recovery.

Verifies block placement by moving the EE camera above the target,
scanning the worldmap, detecting blocks, re-estimating the structure
origin from detected neighbors, and checking the target block position.
If verification fails, performs an orbit scan to localize the misplaced
block and re-attempts the grasp-approach-place sequence.

Follows the callback-injection pattern (like RecoveryHandler): receives
injected functions for ROS operations — no ROS imports.
"""

from collections import defaultdict
from math import pi, cos, sin
from typing import Callable, Optional

import numpy as np

from legobuilder.kinematics.trajectory import TrajectoryState
from legobuilder.kinematics.block_manipulator import ManipulatorState

from legobuilder.brain.block_structure import BlockStructure, is_point_reachable
from legobuilder.schemas import Object

from legobuilder.config import (
    BLOCK_SIZE,
    GRID_CENTER_XY,
    MAX_VERIFY_GRIP_RETRIES,
    ORBIT_SCAN_DWELL_TIME,
    ORBIT_SCAN_HEIGHT,
    ORBIT_SCAN_POSITIONS,
    ORBIT_SCAN_RADIUS,
    VERIFY_DETECT_TIMEOUT,
    VERIFY_MOVE_DURATION,
    VERIFY_SCAN_DWELL_TIME,
    VERIFY_SCAN_HEIGHT,
    VERIFY_SCAN_INTERVAL,
    VERIFY_SCAN_SOURCE,
    VERIFY_SCAN_TILT,
    VERIFY_TARGET_CENTERED,
    BLOCK_TOP_OFFSET,
    PICK_Z,
    grip_check_roll,
)

from legobuilder_interfaces.msg import (
    IKRequestMsg,
    TrajectoryCommandMsg,
)


XY_TOL = BLOCK_SIZE / 4   # ~8.5mm
Z_TOL = BLOCK_SIZE / 2    # ~17mm
MAX_RECOVERY_ATTEMPTS = 2


def match_detections_to_grid(
    detections: list[Object],
    structure: BlockStructure,
    target_cell: tuple[int, int, int],
    logger=None,
) -> dict:
    """Match worldmap detections to grid cells using 3D distance.

    Builds a candidate set from visible placed cells (anchors) plus the
    target cell, then greedily assigns detections to cells prioritising
    colour match first and closest distance second.

    Arguments
    ---------
    detections : list[Object]
        Detected objects from the worldmap.
    structure : BlockStructure
        The target block structure (block_grid holds expected types,
        placed_grid holds already-placed blocks).
    target_cell : tuple[int, int, int]
        (row, col, layer) of the block being verified.

    Returns
    -------
    dict
        result : str
            'verified', 'retry', or 'failed'.
        cell_matches : dict
            Mapping of (r, c, l) -> Object for each assigned cell.
        target_match : Object or None
            Detection assigned to the target cell, if any.
        unassigned : list[Object]
            Detections that could not be assigned to any cell.
    """
    if not detections:
        return {
            'result': 'retry',
            'cell_matches': {},
            'target_match': None,
            'unassigned': [],
        }

    row_t, col_t, layer_t = target_cell
    expected_type_val = int(structure.block_grid[row_t, col_t, layer_t])

    # Build candidate cells: placed anchors + target
    candidates = {}  # (r, c, l) -> expected_type_value (int)
    sim_grid = structure.get_simulated_detection_grid()
    for r in range(sim_grid.shape[0]):
        for c in range(sim_grid.shape[1]):
            for lay in range(sim_grid.shape[2]):
                val = int(sim_grid[r, c, lay])
                if val != 0:
                    candidates[(r, c, lay)] = val
    # Always include target cell
    candidates[target_cell] = expected_type_val

    # Pre-compute cell world positions
    cell_positions = {
        cell: structure.get_world_position(*cell)
        for cell in candidates
    }

    if logger:
        target_pos = cell_positions[target_cell]
        logger.info(
            f"match_detections: {len(candidates)} candidates, "
            f"target={target_cell} expected_pos="
            f"[{target_pos[0]:.4f}, {target_pos[1]:.4f}, "
            f"{target_pos[2]:.4f}], "
            f"expected_type={expected_type_val}, "
            f"XY_TOL={XY_TOL:.4f}m, Z_TOL={Z_TOL:.4f}m"
        )
        for det_idx, det in enumerate(detections):
            ddx, ddy, ddz = det.center_xyz
            tp = cell_positions[target_cell]
            delta_x = abs(ddx - float(tp[0]))
            delta_y = abs(ddy - float(tp[1]))
            delta_z = abs(ddz - float(tp[2]))
            logger.info(
                f"  det[{det_idx}] type={det.obj_type.name} "
                f"pos=[{ddx:.4f}, {ddy:.4f}, {ddz:.4f}] "
                f"vs target: "
                f"dx={delta_x:.4f}"
                f"({'ok' if delta_x < XY_TOL else 'FAIL'}), "
                f"dy={delta_y:.4f}"
                f"({'ok' if delta_y < XY_TOL else 'FAIL'}), "
                f"dz={delta_z:.4f}"
                f"({'ok' if delta_z < Z_TOL else 'FAIL'})"
            )

    # Score each (detection, cell) pair within tolerance
    pairs = []
    for det_idx, det in enumerate(detections):
        dx, dy, dz = det.center_xyz
        for cell, pos in cell_positions.items():
            ex, ey, ez = float(pos[0]), float(pos[1]), float(pos[2])
            if (abs(dx - ex) < XY_TOL
                    and abs(dy - ey) < XY_TOL
                    and abs(dz - ez) < Z_TOL):
                color_match = (det.obj_type.value == candidates[cell])
                dist = (
                    (dx - ex) ** 2 + (dy - ey) ** 2 + (dz - ez) ** 2
                ) ** 0.5
                pairs.append((cell, det_idx, color_match, dist))

    # Sort: colour match first (True before False), then closest
    pairs.sort(key=lambda p: (not p[2], p[3]))

    # Greedy assignment
    assigned_cells: dict[tuple, Object] = {}
    assigned_dets: set[int] = set()
    for cell, det_idx, _color_match, _dist in pairs:
        if cell not in assigned_cells and det_idx not in assigned_dets:
            assigned_cells[cell] = detections[det_idx]
            assigned_dets.add(det_idx)

    unassigned = [d for i, d in enumerate(detections)
                  if i not in assigned_dets]
    target_match = assigned_cells.get(target_cell)

    if logger:
        for cell, det in assigned_cells.items():
            tag = "TARGET" if cell == target_cell else "anchor"
            logger.info(
                f"  assigned: {tag} {cell} <- "
                f"{det.obj_type.name} "
                f"[{det.center_xyz[0]:.4f}, "
                f"{det.center_xyz[1]:.4f}, "
                f"{det.center_xyz[2]:.4f}]"
            )
        for det in unassigned:
            logger.info(
                f"  unassigned: {det.obj_type.name} "
                f"[{det.center_xyz[0]:.4f}, "
                f"{det.center_xyz[1]:.4f}, "
                f"{det.center_xyz[2]:.4f}]"
            )

    # Determine result
    if target_match is None:
        result = 'retry'
    elif target_match.obj_type.value != expected_type_val:
        result = 'failed'
    else:
        result = 'verified'

    return {
        'result': result,
        'cell_matches': assigned_cells,
        'target_match': target_match,
        'unassigned': unassigned,
    }


def identify_misplaced_block(
    unassigned: list[Object],
    expected_type_val: int,
    target_world_pos: np.ndarray,
) -> Optional[Object]:
    """Find the closest color-matched unassigned detection to a target cell.

    Filters ``unassigned`` detections by color (``obj_type.value ==
    expected_type_val``), then returns the one closest to
    ``target_world_pos`` by 3D Euclidean distance.  Returns ``None``
    when no color-matched detection exists.  There is no maximum
    distance threshold -- any color match is accepted.

    Arguments
    ---------
    unassigned : list[Object]
        Detections not assigned to any grid cell.
    expected_type_val : int
        Integer value of the expected ``ObjectType`` for the target cell.
    target_world_pos : np.ndarray
        (3,) world-frame position of the target cell.

    Returns
    -------
    Object or None
        Closest color-matched detection, or ``None`` if none match.
    """
    color_matched = [
        det for det in unassigned
        if det.obj_type.value == expected_type_val
    ]
    if not color_matched:
        return None
    return min(
        color_matched,
        key=lambda det: np.linalg.norm(
            np.array(det.center_xyz) - target_world_pos
        ),
    )

class PlacementVerifier:
    """Post-placement verification and recovery state machine.

    After the build pipeline completes a place phase, the verifier
    moves the EE camera above the target cell, scans, detects blocks,
    re-estimates the structure origin, and checks placement accuracy.
    On failure it performs an orbit scan, identifies the misplaced
    block, and re-attempts the full grasp-approach-place sequence.

    Attributes
    ----------
    structure : BlockStructure
        Reference to the target block structure.
    logger
        Logger instance for status messages.
    """

    # Verify states
    IDLE = 'idle'
    VERIFY_MOVE_TO_SCAN = 'verify_move_to_scan'
    VERIFY_DWELL_SCAN = 'verify_dwell_scan'
    VERIFY_DETECT = 'verify_detect'
    RECOVERY_ORBIT = 'recovery_orbit'
    RECOVERY_ORBIT_DWELL = 'recovery_orbit_dwell'
    RECOVERY_ORBIT_DETECT = 'recovery_orbit_detect'
    RECOVERY_GRASP = 'recovery_grasp'
    RECOVERY_GRIP_RETRY_LIFT = 'recovery_grip_retry_lift'
    RECOVERY_APPROACH = 'recovery_approach'
    RECOVERY_PLACE = 'recovery_place'

    PICK_TILT = -pi
    PICK_DURATION = 3.0
    PLACE_DURATION = 3.0
    GRIPPER_DELAY = 0.5
    APPROACH_HEIGHT = 0.1

    # ── Lifecycle ─────────────────────────────────────────────────────

    def __init__(
        self,
        send_ik_request_fn: Callable,
        publish_trajectory_fn: Callable,
        request_scan_fn: Callable,
        create_timer_fn: Callable,
        destroy_timer_fn: Callable,
        get_t_fn: Callable,
        structure: Optional[BlockStructure],
        logger,
    ):
        self._send_ik_request = send_ik_request_fn
        self._publish_trajectory = publish_trajectory_fn
        self._request_scan = request_scan_fn
        self._create_timer = create_timer_fn
        self._destroy_timer = destroy_timer_fn
        self._get_t = get_t_fn

        self.structure = structure
        self.logger = logger

        self._state: str = self.IDLE
        self._target_cell: Optional[tuple] = None
        self._pickup_context: Optional[dict] = None
        self._grip_retry_count: int = 0
        self._last_verification_result: Optional[dict] = None
        self._is_orbit_fallback: bool = False
        self._recovery_attempt_count: int = 0
        self._is_grip_retry_rescan: bool = False

        # IK tracking
        self._ik_id: Optional[str] = None
        self._ik_data: Optional[dict] = None
        self._waiting_for_ik: bool = False

        # Timers
        self._scan_timer = None
        self._detect_timeout_timer = None
        self._dwell_end_timer = None

        # Orbit scan state
        self._orbit_waypoints: list = []
        self._orbit_index: int = 0
        self._accumulated_detections: list = []

        # Misplaced block for recovery
        self._misplaced_obj: Optional[Object] = None

        # Result callback set by pipeline
        self._on_result: Optional[Callable] = None

    # ── Entry point ───────────────────────────────────────────────────

    def start_verification(
        self,
        target_cell: tuple,
        pickup_context: dict,
        on_result: Callable,
    ) -> None:
        """Begin verification after a placement.

        Arguments
        ---------
        target_cell : tuple
            (row, col, layer) of the placed block.
        pickup_context : dict
            Pipeline pickup context (contains pickup_roll, etc.).
        on_result : callable
            Callback: on_result(result_str) where result is
            'verified' or 'failed'.
        """
        if self.is_active:
            self.logger.warning(
                'start_verification called while active — resetting first'
            )
            self.reset()

        self._target_cell = target_cell
        self._pickup_context = dict(pickup_context)
        self._on_result = on_result
        self._grip_retry_count = 0
        self._accumulated_detections = []
        self._last_verification_result = None
        self._is_orbit_fallback = False
        self._recovery_attempt_count = 0
        self._is_grip_retry_rescan = False

        self.logger.info(
            f"Starting placement verification for "
            f"cell {target_cell}"
        )
        self._transition_to_move_to_scan()

    # ── State transitions ─────────────────────────────────────────────

    def _transition_to_move_to_scan(self) -> None:
        """Plan and send trajectory to position EE camera above target."""
        self._is_orbit_fallback = False
        self._state = self.VERIFY_MOVE_TO_SCAN

        scan_states = self._plan_verify_scan_trajectory()
        if scan_states is None:
            self.logger.warning(
                "Cannot plan scan trajectory, failing verification"
            )
            self._finish('failed')
            return

        request_id = self._send_ik_request(
            scan_states, IKRequestMsg.REQUEST_PRIMARY,
        )
        self._ik_id = request_id
        self._waiting_for_ik = True
        self._ik_data = {
            'trajectory_states': scan_states,
            'phase': 'verify_scan',
        }

    def _transition_to_dwell_scan(self) -> None:
        """Arm is above target; start periodic scanning."""
        self._state = self.VERIFY_DWELL_SCAN
        self.logger.info(
            f"Verify dwell: scanning for "
            f"{VERIFY_SCAN_DWELL_TIME}s"
        )
        self._cancel_timer('_scan_timer')
        self._cancel_timer('_dwell_end_timer')
        self._scan_timer = self._create_timer(
            VERIFY_SCAN_INTERVAL, self._on_scan_tick,
        )
        self._dwell_end_timer = self._create_timer(
            VERIFY_SCAN_DWELL_TIME, self._on_dwell_complete,
        )

    def _transition_to_detect(self) -> None:
        """Request worldmap contour detection and start timeout."""
        self._state = self.VERIFY_DETECT
        self.logger.info("Verify detect: requesting worldmap detection")
        # Map VERIFY_SCAN_SOURCE (0=accumulated, 1=latest) to request types
        from legobuilder_interfaces.msg import ScanRequestMsg
        if VERIFY_SCAN_SOURCE == 1:
            self._request_scan(ScanRequestMsg.DETECT_FROM_LATEST)
        else:
            self._request_scan(ScanRequestMsg.DETECT_FROM_ACCUMULATED)
        self._cancel_timer('_detect_timeout_timer')
        self._detect_timeout_timer = self._create_timer(
            VERIFY_DETECT_TIMEOUT, self._on_detect_timeout,
        )

    def _transition_to_orbit(self) -> None:
        """Generate orbit waypoints and start orbit scan sequence."""
        self._state = self.RECOVERY_ORBIT
        self._is_orbit_fallback = True
        self._orbit_waypoints = self._generate_orbit_waypoints()
        self._orbit_index = 0
        self._accumulated_detections = []

        if not self._orbit_waypoints:
            self.logger.warning(
                "No reachable orbit waypoints, failing"
            )
            self._finish('failed')
            return

        self.logger.info(
            f"Starting orbit scan with "
            f"{len(self._orbit_waypoints)} waypoints"
        )
        self._move_to_orbit_waypoint()

    def _move_to_orbit_waypoint(self) -> None:
        """Move to the current orbit waypoint."""
        if self._orbit_index >= len(self._orbit_waypoints):
            if VERIFY_SCAN_SOURCE == 1:
                # Source=1: accumulated Object lists across waypoints
                clustered = self._cluster_detections(
                    self._accumulated_detections,
                )
                self.logger.info(
                    f"Orbit clustered {len(self._accumulated_detections)} "
                    f"detections into {len(clustered)}"
                )
                result = self._check_placement(clustered, 'orbit')
                if result == 'verified':
                    self._finish('verified')
                else:
                    self._finish('failed')
            else:
                # Source=0: request single DETECT_FROM_ACCUMULATED
                self._transition_to_detect()
            return

        self._state = self.RECOVERY_ORBIT

        pos, orient = self._orbit_waypoints[self._orbit_index]
        states = [
            TrajectoryState(
                final_state=ManipulatorState(
                    p=pos,
                    o=orient,
                    gripper_open=True,
                ),
                min_duration=VERIFY_MOVE_DURATION,
            ),
        ]

        request_id = self._send_ik_request(
            states, IKRequestMsg.REQUEST_PRIMARY,
        )
        self._ik_id = request_id
        self._waiting_for_ik = True
        self._ik_data = {
            'trajectory_states': states,
            'phase': 'orbit_move',
        }

    def _transition_to_orbit_dwell(self) -> None:
        """Dwell and scan at the current orbit position."""
        self._state = self.RECOVERY_ORBIT_DWELL
        self._cancel_timer('_scan_timer')
        self._cancel_timer('_dwell_end_timer')
        self._scan_timer = self._create_timer(
            VERIFY_SCAN_INTERVAL, self._on_scan_tick,
        )
        self._dwell_end_timer = self._create_timer(
            ORBIT_SCAN_DWELL_TIME, self._on_orbit_dwell_complete,
        )

    def _transition_to_orbit_detect(self) -> None:
        """Request detection at current orbit position."""
        self._state = self.RECOVERY_ORBIT_DETECT
        from legobuilder_interfaces.msg import ScanRequestMsg
        if VERIFY_SCAN_SOURCE == 1:
            self._request_scan(ScanRequestMsg.DETECT_FROM_LATEST)
        else:
            self._request_scan(ScanRequestMsg.DETECT_FROM_ACCUMULATED)
        self._cancel_timer('_detect_timeout_timer')
        self._detect_timeout_timer = self._create_timer(
            VERIFY_DETECT_TIMEOUT, self._on_orbit_detect_timeout,
        )

    # ── Callback dispatchers ──────────────────────────────────────────

    def on_trajectory_complete(self) -> None:
        """Dispatch trajectory completion by current verify state."""
        if self._waiting_for_ik:
            self.logger.warning(
                'Ignoring trajectory_complete while waiting for IK'
            )
            return
        if self._state == self.VERIFY_MOVE_TO_SCAN:
            if VERIFY_SCAN_SOURCE == 1:
                self._transition_to_detect()
            else:
                self._transition_to_dwell_scan()
        elif self._state == self.RECOVERY_ORBIT:
            from legobuilder_interfaces.msg import ScanRequestMsg
            if VERIFY_SCAN_SOURCE == 0:
                self._request_scan(ScanRequestMsg.SIMPLY_ACCUMULATE)
                self._orbit_index += 1
                self._move_to_orbit_waypoint()
            else:
                self._transition_to_orbit_detect()
        elif self._state == self.RECOVERY_GRASP:
            self._on_recovery_grasp_complete()
        elif self._state == self.RECOVERY_GRIP_RETRY_LIFT:
            self._on_grip_retry_lift_complete()
        elif self._state == self.RECOVERY_APPROACH:
            self._on_recovery_approach_complete()
        elif self._state == self.RECOVERY_PLACE:
            self._on_recovery_place_complete()

    def on_grip_failure(self) -> None:
        """Handle grip failure during recovery grasp.

        If within the retry limit, plans a lift trajectory (open gripper,
        move to safe height above misplaced block), then retries the
        recovery grasp.  If retries exhausted, finishes with 'failed'.
        """
        if self._state != self.RECOVERY_GRASP:
            self.logger.warning(
                f"on_grip_failure called in unexpected state "
                f"{self._state}, ignoring"
            )
            return

        self._grip_retry_count += 1
        if self._grip_retry_count > MAX_VERIFY_GRIP_RETRIES:
            self.logger.warning(
                f"Recovery grip retry limit reached "
                f"({MAX_VERIFY_GRIP_RETRIES}), failing verification"
            )
            self._finish('failed')
            return

        self.logger.info(
            f"Grip failure during recovery, "
            f"retry {self._grip_retry_count}/{MAX_VERIFY_GRIP_RETRIES}"
        )
        self._plan_grip_retry_lift()

    def _plan_grip_retry_lift(self) -> None:
        """Plan a lift trajectory with open gripper above the misplaced block."""
        obj = self._misplaced_obj
        if obj is None:
            self.logger.error("No misplaced object for grip retry lift")
            self._finish('failed')
            return

        x, y, _ = obj.center_xyz
        p_above = np.array([
            x, y, obj.center_xyz[2] + self.APPROACH_HEIGHT,
        ])

        if not is_point_reachable(p_above):
            self.logger.warning(
                f"Grip retry lift position {p_above} not reachable"
            )
            self._finish('failed')
            return

        pick_roll = -obj.angle if obj.angle is not None else 0.0

        

        lift_o = np.array([self.PICK_TILT, pick_roll])

        states = [
            TrajectoryState(
                final_state=ManipulatorState(
                    p=np.array([x, y, obj.center_xyz[2]]),
                    o=lift_o,
                    gripper_open=True,
                ),
                min_duration=self.GRIPPER_DELAY,
            ),
            TrajectoryState(
                final_state=ManipulatorState(
                    p=p_above,
                    o=lift_o,
                    gripper_open=True,
                ),
                min_duration=self.PICK_DURATION / 2,
            ),
        ]

        self._state = self.RECOVERY_GRIP_RETRY_LIFT

        request_id = self._send_ik_request(
            states, IKRequestMsg.REQUEST_PRIMARY,
        )
        self._ik_id = request_id
        self._waiting_for_ik = True
        self._ik_data = {
            'trajectory_states': states,
            'phase': 'recovery_grip_retry_lift',
        }

    def _on_grip_retry_lift_complete(self) -> None:
        """Lift done after grip failure, rescan for misplaced block."""
        self.logger.info(
            "Grip retry lift complete, rescanning for misplaced block"
        )
        self._is_grip_retry_rescan = True
        self._transition_to_move_to_scan()

    def on_ik_response(
        self,
        request_id: str,
        success: bool,
        q_solutions: list[float],
    ) -> None:
        """Handle IK response during verification/recovery phases."""
        if request_id != self._ik_id:
            return
        self._ik_id = None
        self._waiting_for_ik = False

        if not success:
            self.logger.error(
                f"Verification IK failed in state {self._state}"
            )
            self._finish('failed')
            return

        data = self._ik_data
        self._ik_data = None

        from legobuilder.brain.ros_bridge import apply_ik_solutions
        trajectory_states = data['trajectory_states']
        apply_ik_solutions(trajectory_states, q_solutions)

        phase = data.get('phase')

        if phase == 'verify_scan':
            self._publish_trajectory(
                trajectory_states,
                TrajectoryCommandMsg.COMMAND_REPLACE,
            )
        elif phase == 'orbit_move':
            self._publish_trajectory(
                trajectory_states,
                TrajectoryCommandMsg.COMMAND_REPLACE,
            )
        elif phase == 'recovery_grasp':
            self._state = self.RECOVERY_GRASP
            self._publish_trajectory(
                trajectory_states,
                TrajectoryCommandMsg.COMMAND_REPLACE,
            )
        elif phase == 'recovery_grip_retry_lift':
            self._state = self.RECOVERY_GRIP_RETRY_LIFT
            self._publish_trajectory(
                trajectory_states,
                TrajectoryCommandMsg.COMMAND_REPLACE,
            )
        elif phase == 'recovery_approach':
            self._state = self.RECOVERY_APPROACH
            self._publish_trajectory(
                trajectory_states,
                TrajectoryCommandMsg.COMMAND_REPLACE,
            )
        elif phase == 'recovery_place':
            self._state = self.RECOVERY_PLACE
            self._publish_trajectory(
                trajectory_states,
                TrajectoryCommandMsg.COMMAND_REPLACE,
            )

    def on_worldmap_detections(
        self, detected_objects: list[Object],
    ) -> None:
        """Receive worldmap contour detection results."""
        if self._state == self.VERIFY_DETECT:
            self._cancel_timer('_detect_timeout_timer')
            result = self._check_placement(detected_objects, 'primary')
            if result == 'verified':
                self._finish('verified')
            elif result == 'failed':
                self._finish('failed')
            elif result == 'retry':
                if self._is_orbit_fallback:
                    self._attempt_recovery()
                elif self._has_visible_misplaced_block():
                    self.logger.info(
                        "Misplaced block visible in primary scan, "
                        "skipping orbit"
                    )
                    self._is_orbit_fallback = True
                    self._attempt_recovery()
                else:
                    self._transition_to_orbit()

        elif self._state == self.RECOVERY_ORBIT_DETECT:
            self._cancel_timer('_detect_timeout_timer')
            self._accumulated_detections.extend(detected_objects)
            self._orbit_index += 1
            self._move_to_orbit_waypoint()

    # ── Timer callbacks ───────────────────────────────────────────────

    def _on_scan_tick(self) -> None:
        """Periodic scan during dwell."""
        from legobuilder_interfaces.msg import ScanRequestMsg
        self._request_scan(ScanRequestMsg.SIMPLY_ACCUMULATE)

    def _on_dwell_complete(self) -> None:
        """Dwell period finished; stop scanning, request detection."""
        self._cancel_timer('_dwell_end_timer')
        self._cancel_timer('_scan_timer')
        self._transition_to_detect()

    def _on_detect_timeout(self) -> None:
        """Detection timed out during verify phase."""
        self._cancel_timer('_detect_timeout_timer')
        self.logger.warning("Verify detect timed out")
        if self._is_orbit_fallback:
            self._finish('failed')
        else:
            self._transition_to_orbit()

    def _on_orbit_dwell_complete(self) -> None:
        """Orbit dwell finished; request detection."""
        self._cancel_timer('_dwell_end_timer')
        self._cancel_timer('_scan_timer')
        self._transition_to_orbit_detect()

    def _on_orbit_detect_timeout(self) -> None:
        """Detection timed out during orbit scan."""
        self._cancel_timer('_detect_timeout_timer')
        self.logger.warning(
            f"Orbit detect timed out at waypoint "
            f"{self._orbit_index}"
        )
        # Move to next waypoint without detections
        self._orbit_index += 1
        self._move_to_orbit_waypoint()

    # ── Verification logic ────────────────────────────────────────────

    def _check_placement(
        self, detected_objects: list[Object], scan_type: str,
    ) -> str:
        """Check if the target block is correctly placed using grid matching.

        Delegates to match_detections_to_grid() for 3D distance-based
        detection-to-cell assignment and stores the full result on
        _last_verification_result for debugging and telemetry.

        Arguments
        ---------
        detected_objects : list[Object]
            Detected objects from the worldmap.
        scan_type : str
            'primary' or 'orbit' indicating which scan produced these.

        Returns
        -------
        str
            'verified', 'retry', or 'failed'.
        """
        match_result = match_detections_to_grid(
            detected_objects, self.structure, self._target_cell,
            logger=self.logger,
        )

        self._last_verification_result = {
            'result': match_result['result'],
            'target_cell': self._target_cell,
            'target_match': match_result['target_match'],
            'matched_cells': match_result['cell_matches'],
            'unassigned_detections': match_result['unassigned'],
            'num_detections': len(detected_objects),
            'scan_type': scan_type,
        }

        self.logger.info(
            f"Verification check ({scan_type}): "
            f"{match_result['result']} — "
            f"{len(match_result['cell_matches'])} matched, "
            f"{len(match_result['unassigned'])} unassigned, "
            f"{len(detected_objects)} detections"
        )

        return match_result['result']

    # ── Recovery grasp/approach/place ─────────────────────────────────

    @staticmethod
    def _cluster_detections(detections: list[Object]) -> list[Object]:
        """Cluster duplicate detections from multiple viewpoints.

        Groups detections by object type, runs DBSCAN with
        eps=BLOCK_SIZE/2 to merge nearby duplicates, and returns
        one representative Object per cluster with median position.
        """
        from sklearn.cluster import DBSCAN

        by_type: dict = defaultdict(list)
        for det in detections:
            by_type[det.obj_type].append(det)

        clustered = []
        for obj_type, dets in by_type.items():
            centers = np.array([d.center_xyz for d in dets])
            labels = DBSCAN(
                eps=BLOCK_SIZE / 2, min_samples=1,
            ).fit(centers).labels_

            for label in set(labels):
                if label == -1:
                    continue
                members = [d for d, l in zip(dets, labels) if l == label]
                median_xyz = tuple(
                    np.median([m.center_xyz for m in members], axis=0)
                )
                rep = min(
                    members,
                    key=lambda m: np.linalg.norm(
                        np.array(m.center_xyz) - np.array(median_xyz)
                    ),
                )
                clustered.append(Object(
                    t=rep.t,
                    frame_idx=rep.frame_idx,
                    obj_type=obj_type,
                    center_uv=rep.center_uv,
                    center_xyz=median_xyz,
                    angle=rep.angle,
                    corner_xys=rep.corner_xys,
                    corner_uvs=rep.corner_uvs,
                    face_conns_xyzs=rep.face_conns_xyzs,
                ))

        return clustered

    def _has_visible_misplaced_block(self) -> bool:
        """Check if unassigned detections contain a color match for target."""
        lvr = self._last_verification_result
        if lvr is None:
            return False
        unassigned = lvr.get('unassigned_detections', [])
        if not unassigned:
            return False
        row_t, col_t, layer_t = self._target_cell
        expected_type_val = int(
            self.structure.block_grid[row_t, col_t, layer_t]
        )
        return any(
            det.obj_type.value == expected_type_val
            for det in unassigned
        )

    def _get_all_detections(self) -> list[Object]:
        """Get all detected objects from the last verification result."""
        lvr = self._last_verification_result
        if lvr is None:
            return []
        detections = list(lvr.get('matched_cells', {}).values())
        detections.extend(lvr.get('unassigned_detections', []))
        return detections

    def _attempt_recovery(self) -> None:
        """Attempt to recover a misplaced block.

        Called when 'retry' result occurs after orbit scan.
        Identifies the misplaced block from the last verification
        result's unassigned detections, then triggers recovery grasp.
        """
        if self._is_grip_retry_rescan:
            self._is_grip_retry_rescan = False
        else:
            self._recovery_attempt_count += 1
        if self._recovery_attempt_count > MAX_RECOVERY_ATTEMPTS:
            self.logger.warning(
                f"Recovery attempt limit reached "
                f"({MAX_RECOVERY_ATTEMPTS}), failing"
            )
            self._finish('failed')
            return

        lvr = self._last_verification_result
        if lvr is None:
            self.logger.error("No verification result for recovery")
            self._finish('failed')
            return

        unassigned = lvr.get('unassigned_detections', [])
        row_t, col_t, layer_t = self._target_cell
        expected_type_val = int(
            self.structure.block_grid[row_t, col_t, layer_t]
        )
        target_world_pos = self.structure.get_world_position(
            row_t, col_t, layer_t,
        )

        misplaced = identify_misplaced_block(
            unassigned, expected_type_val, target_world_pos,
        )
        if misplaced is None:
            self.logger.warning(
                "No color-matched unassigned detection for recovery"
            )
            self._finish('failed')
            return

        self.logger.info(
            f"Identified misplaced block at "
            f"({misplaced.center_xyz[0]:.3f}, "
            f"{misplaced.center_xyz[1]:.3f}, "
            f"{misplaced.center_xyz[2]:.3f})"
        )
        self._misplaced_obj = misplaced
        self._grip_retry_count = 0
        self._plan_recovery_grasp()

    def _plan_recovery_grasp(self) -> None:
        """Plan a straight-down grasp of the misplaced block."""
        obj = self._misplaced_obj
        x, y, z = obj.center_xyz
        p_pick = np.array([x, y, z + BLOCK_TOP_OFFSET])

        if not is_point_reachable(p_pick):
            self.logger.warning(
                f"Recovery pick position {p_pick} not reachable"
            )
            self._finish('failed')
            return

        all_detections = self._get_all_detections()
        pick_roll = self.structure.compute_pickup_roll(obj, all_detections)
        pick_o = np.array([self.PICK_TILT, pick_roll])

        states = [
            # Approach above misplaced block
            TrajectoryState(
                final_state=ManipulatorState(
                    p=p_pick + np.array([0, 0, self.APPROACH_HEIGHT]),
                    o=pick_o,
                    gripper_open=True,
                ),
                min_duration=self.PICK_DURATION,
            ),
            # Descend
            TrajectoryState(
                final_state=ManipulatorState(
                    p=p_pick.copy(),
                    o=pick_o,
                    gripper_open=True,
                ),
                min_duration=self.PICK_DURATION / 2,
            ),
            # Close gripper
            TrajectoryState(
                final_state=ManipulatorState(
                    p=p_pick.copy(),
                    o=pick_o,
                    gripper_open=False,
                ),
                min_duration=self.GRIPPER_DELAY,
                delay_before=self.GRIPPER_DELAY,
            ),
            # Lift
            TrajectoryState(
                final_state=ManipulatorState(
                    p=p_pick + np.array([0, 0, self.APPROACH_HEIGHT]),
                    o=pick_o,
                    gripper_open=False,
                ),
                min_duration=self.PICK_DURATION,
            ),
        ]

        # Store recovery context
        self._pickup_context['p_pick'] = p_pick
        self._pickup_context['pickup_roll'] = pick_roll

        request_id = self._send_ik_request(
            states, IKRequestMsg.REQUEST_PRIMARY,
        )
        self._ik_id = request_id
        self._waiting_for_ik = True
        self._ik_data = {
            'trajectory_states': states,
            'phase': 'recovery_grasp',
        }

    def _on_recovery_grasp_complete(self) -> None:
        """Grasp done, plan approach to structure."""
        row, col, layer = self._target_cell
        pickup_roll = self._pickup_context.get('pickup_roll', 0.0)

        mating_faces = self.structure.get_horizontal_mating_faces(
            row, col, layer, use_placed_grid=True,
        )
        placement_roll = self.structure.compute_placement_roll(
            mating_faces, pickup_roll,
        )
        self._pickup_context['placement_roll'] = placement_roll

        approach_states = self.structure.plan_approach(
            self._target_cell, placement_roll,
        )

        request_id = self._send_ik_request(
            approach_states, IKRequestMsg.REQUEST_PRIMARY,
        )
        self._ik_id = request_id
        self._waiting_for_ik = True
        self._ik_data = {
            'trajectory_states': approach_states,
            'phase': 'recovery_approach',
        }

    def _on_recovery_approach_complete(self) -> None:
        """Approach done, plan placement."""
        row, col, layer = self._target_cell
        placement_roll = self._pickup_context.get(
            'placement_roll', 0.0,
        )

        place_states = self.structure.plan_placement(
            row, col, layer, placement_roll,
            return_to_ready=False,
        )

        request_id = self._send_ik_request(
            place_states, IKRequestMsg.REQUEST_PRIMARY,
        )
        self._ik_id = request_id
        self._waiting_for_ik = True
        self._ik_data = {
            'trajectory_states': place_states,
            'phase': 'recovery_place',
        }

    def _on_recovery_place_complete(self) -> None:
        """Recovery placement done, re-verify."""
        self.logger.info(
            "Recovery placement complete, re-verifying"
        )
        self._transition_to_move_to_scan()

    # ── Trajectory planning ───────────────────────────────────────────

    def _plan_verify_scan_trajectory(
        self,
    ) -> Optional[list[TrajectoryState]]:
        """Plan trajectory to position EE camera above target block.

        Returns
        -------
        list[TrajectoryState] or None
            Single-state trajectory, or None if unreachable.
        """
        if VERIFY_TARGET_CENTERED:
            center = self.structure.get_world_position(*self._target_cell)
        else:
            center = np.array([GRID_CENTER_XY[0], GRID_CENTER_XY[1], 0.0])
        scan_pos = np.array([center[0], center[1], VERIFY_SCAN_HEIGHT])

        if not is_point_reachable(scan_pos):
            return None

        scan_roll = grip_check_roll(scan_pos)

        return [
            TrajectoryState(
                final_state=ManipulatorState(
                    p=scan_pos,
                    o=np.array([VERIFY_SCAN_TILT, scan_roll]),
                    gripper_open=True,
                ),
                min_duration=VERIFY_MOVE_DURATION,
            ),
        ]

    def _generate_orbit_waypoints(
        self,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Generate orbit scan waypoints around target block.

        Returns
        -------
        list of (position, orientation)
            Reachable waypoints at ORBIT_SCAN_HEIGHT above target block,
            offset laterally by ORBIT_SCAN_RADIUS.
        """
        if VERIFY_TARGET_CENTERED:
            target_pos = self.structure.get_world_position(*self._target_cell)
            center_x, center_y = target_pos[0], target_pos[1]
        else:
            center_x, center_y = GRID_CENTER_XY

        waypoints = []
        for i in range(ORBIT_SCAN_POSITIONS):
            angle = 2 * pi * i / ORBIT_SCAN_POSITIONS
            offset_x = ORBIT_SCAN_RADIUS * cos(angle)
            offset_y = ORBIT_SCAN_RADIUS * sin(angle)
            pos = np.array([
                center_x + offset_x,
                center_y + offset_y,
                ORBIT_SCAN_HEIGHT,
            ])

            if not is_point_reachable(pos):
                continue

            scan_roll = grip_check_roll(pos)
            orient = np.array([VERIFY_SCAN_TILT, scan_roll])
            waypoints.append((pos, orient))

        return waypoints

    # ── Helpers ────────────────────────────────────────────────────────

    def _cancel_timer(self, attr_name: str) -> None:
        """Cancel and destroy a timer by attribute name."""
        timer = getattr(self, attr_name, None)
        if timer is not None:
            timer.cancel()
            self._destroy_timer(timer)
            setattr(self, attr_name, None)

    def _finish(self, result: str) -> None:
        """Clean up and report result to pipeline."""
        self._cancel_timer('_scan_timer')
        self._cancel_timer('_detect_timeout_timer')
        self._cancel_timer('_dwell_end_timer')
        self._state = self.IDLE
        self._ik_id = None
        self._ik_data = None
        self._waiting_for_ik = False
        self._misplaced_obj = None
        self._orbit_waypoints = []
        self._accumulated_detections = []
        self._last_verification_result = None
        self._is_orbit_fallback = False

        self.logger.info(
            f"Verification result: {result} for "
            f"cell {self._target_cell}"
        )

        if self._on_result is not None:
            self._on_result(result)

    def reset(self) -> None:
        """Reset all verifier state (called on pipeline clear)."""
        self._cancel_timer('_scan_timer')
        self._cancel_timer('_detect_timeout_timer')
        self._cancel_timer('_dwell_end_timer')
        self._state = self.IDLE
        self._target_cell = None
        self._pickup_context = None
        self._grip_retry_count = 0
        self._recovery_attempt_count = 0
        self._ik_id = None
        self._ik_data = None
        self._waiting_for_ik = False
        self._misplaced_obj = None
        self._orbit_waypoints = []
        self._orbit_index = 0
        self._accumulated_detections = []
        self._on_result = None
        self._last_verification_result = None
        self._is_orbit_fallback = False

    @property
    def is_active(self) -> bool:
        """True if verification is in progress."""
        return self._state != self.IDLE

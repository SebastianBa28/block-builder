"""Structure scanning and state reconstruction.

Scans the build area (overhead + optional orbit), reconstructs the
placed_grid via rebuild_placed_grid, then selects the next target
block via next_target_block.  Returns a ScanResult to the pipeline.

Follows the callback-injection pattern: receives injected functions
for ROS operations — no ROS imports.
"""

from collections import defaultdict
from dataclasses import dataclass
from math import pi, cos, sin
from typing import Callable, Optional

import numpy as np

from legobuilder.kinematics.trajectory import TrajectoryState
from legobuilder.kinematics.block_manipulator import ManipulatorState

from legobuilder.brain.block_structure import BlockStructure, is_point_reachable
from legobuilder.brain.reachability import get_unreachable_cells
from legobuilder.brain.rebuild_state import rebuild_placed_grid
from legobuilder.brain.next_target_block import next_target_block
from legobuilder.schemas import Object, ObjectType

from legobuilder.config import (
    BLOCK_SIZE,
    GRID_CENTER_XY,
    HISTORICAL_PLACED_GRID_RESCAN_COUNT,
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
    grip_check_roll,
    PILE_X_MIN
)

from legobuilder_interfaces.msg import (
    IKRequestMsg,
    TrajectoryCommandMsg,
    ScanRequestMsg,
)


@dataclass
class ScanResult:
    """Result returned by StructureScanner after a complete scan cycle."""

    placed_grid: np.ndarray
    """Rebuilt placed_grid from detections."""

    target_cell: Optional[tuple]
    """Next cell to fill (row, col, layer), or None if build is complete."""

    block_type: Optional[ObjectType]
    """ObjectType needed at target_cell, or None if done."""

    unassigned: list
    """Detected objects not matched to any grid cell (misplaced blocks)."""

    is_mistake: bool = False
    """True if target_cell contains a wrong-color block that must be removed."""


class StructureScanner:
    """Structure scanning FSM: overhead scan, optional orbit, state rebuild.

    After scanning, calls rebuild_placed_grid to reconstruct the grid
    from detections, then next_target_block to choose the next block.
    Returns a ScanResult to the pipeline via callback.

    Attributes
    ----------
    structure : BlockStructure
        Reference to the target block structure.
    logger
        Logger instance for status messages.
    """

    # States
    IDLE = 'idle'
    MOVE_TO_SCAN = 'move_to_scan'
    DWELL_SCAN = 'dwell_scan'
    DETECT = 'detect'
    ORBIT = 'orbit'
    ORBIT_DWELL = 'orbit_dwell'
    ORBIT_DETECT = 'orbit_detect'

    def __init__(
        self,
        send_ik_request_fn: Callable,
        publish_trajectory_fn: Callable,
        request_scan_fn: Callable,
        create_timer_fn: Callable,
        destroy_timer_fn: Callable,
        get_t_fn: Callable,
        structure: BlockStructure,
        logger,
        post_event_fn=None,
        unreachable_tracker=None,
    ):
        self._send_ik_request = send_ik_request_fn
        self._publish_trajectory = publish_trajectory_fn
        self._request_scan = request_scan_fn
        self._create_timer = create_timer_fn
        self._destroy_timer = destroy_timer_fn
        self._get_t = get_t_fn

        self.structure = structure
        self.logger = logger
        self._post_event = post_event_fn or (lambda e: None)
        self._unreachable_tracker = unreachable_tracker

        self._state: str = self.IDLE
        self._skip_orbit: bool = False
        self._orbit_attempted: bool = False
        self._on_result: Optional[Callable] = None

        # IK tracking
        self._ik_id: Optional[str] = None
        self._ik_data: Optional[dict] = None
        self._waiting_for_ik: bool = False

        # Timers
        self._scan_timer = None
        self._detect_timeout_timer = None
        self._dwell_end_timer = None

        # Orbit state
        self._orbit_waypoints: list = []
        self._orbit_index: int = 0
        self._accumulated_detections: list = []

        # Historical and rescan state
        self._historical_placed_grid: Optional[np.ndarray] = None  # persists across start_scan calls
        self._rescan_for_target_count: int = 0
        self._primary_detections: list = []

    def _emit(self, event_type: str, data: dict) -> None:
        self._post_event({
            'type': event_type,
            'source': 'scanner',
            'data': data,
            'timestamp': self._get_t(),
        })

    # ── Entry point ───────────────────────────────────────────────────

    def start_scan(
        self,
        on_result: Callable,
        skip_orbit: bool = False,
    ) -> None:
        """Begin structure scan. Calls on_result(ScanResult) when done.

        Parameters
        ----------
        on_result : callable
            Callback receiving a ScanResult.
        skip_orbit : bool
            If True, never fall back to orbit (used for first scan).
        """
        if self.is_active:
            self.logger.warning(
                'start_scan called while active — resetting first'
            )
            self.reset()

        self._on_result = on_result
        self._skip_orbit = skip_orbit
        self._orbit_attempted = False
        self._accumulated_detections = []
        self._rescan_for_target_count = 0
        self._primary_detections = []
        # _historical_placed_grid intentionally NOT reset here

        self.logger.debug("Starting structure scan")
        self._transition_to_move_to_scan()

    # ── State transitions ─────────────────────────────────────────────

    def _transition_to_move_to_scan(self) -> None:
        """Plan and send trajectory to overhead scan position."""
        self._state = self.MOVE_TO_SCAN
        self._emit('state_change', {'state': self.MOVE_TO_SCAN})

        scan_states = self._plan_scan_trajectory()
        if scan_states is None:
            self.logger.warning(
                "Cannot plan scan trajectory, finishing with empty result"
            )
            self._finish_with_empty()
            return

        request_id = self._send_ik_request(
            scan_states, IKRequestMsg.REQUEST_PRIMARY,
        )
        self._ik_id = request_id
        self._waiting_for_ik = True
        self._ik_data = {
            'trajectory_states': scan_states,
            'phase': 'scan_move',
        }

    def _transition_to_dwell_scan(self) -> None:
        """Arm is at scan position; start periodic scanning."""
        self._state = self.DWELL_SCAN
        self._emit('state_change', {'state': self.DWELL_SCAN})
        self.logger.info(
            f"Scan dwell: scanning for {VERIFY_SCAN_DWELL_TIME}s"
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
        self._state = self.DETECT
        self._emit('state_change', {'state': self.DETECT})
        self.logger.debug("Scan detect: requesting worldmap detection")
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
        self._state = self.ORBIT
        self._orbit_attempted = True
        self._emit('state_change', {'state': self.ORBIT})
        self._orbit_waypoints = self._generate_orbit_waypoints()
        self._orbit_index = 0
        self._accumulated_detections = []

        if not self._orbit_waypoints:
            self.logger.warning(
                "No reachable orbit waypoints, finishing with empty"
            )
            self._finish_with_empty()
            return

        self.logger.debug(
            f"Starting orbit scan with "
            f"{len(self._orbit_waypoints)} waypoints"
        )
        self._move_to_orbit_waypoint()

    def _move_to_orbit_waypoint(self) -> None:
        """Move to the current orbit waypoint."""
        if self._orbit_index >= len(self._orbit_waypoints):
            if VERIFY_SCAN_SOURCE == 1:
                # Accumulated Object lists across waypoints
                clustered = self._cluster_detections(
                    self._accumulated_detections,
                )
                self.logger.info(
                    f"Orbit clustered {len(self._accumulated_detections)} "
                    f"detections into {len(clustered)}"
                )
                self._process_detections(clustered)
            else:
                # Request single DETECT_FROM_ACCUMULATED
                self._transition_to_detect()
            return

        self._state = self.ORBIT
        self._emit('state_change', {'state': self.ORBIT, 'waypoint': self._orbit_index})

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
        self._state = self.ORBIT_DWELL
        self._emit('state_change', {'state': self.ORBIT_DWELL})
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
        self._state = self.ORBIT_DETECT
        self._emit('state_change', {'state': self.ORBIT_DETECT})
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
        """Dispatch trajectory completion by current state."""
        if self._waiting_for_ik:
            self.logger.warning(
                'Ignoring trajectory_complete while waiting for IK'
            )
            return
        if self._state == self.MOVE_TO_SCAN:
            if VERIFY_SCAN_SOURCE == 1:
                self._transition_to_detect()
            else:
                self._transition_to_dwell_scan()
        elif self._state == self.ORBIT:
            if VERIFY_SCAN_SOURCE == 0:
                self._request_scan(ScanRequestMsg.SIMPLY_ACCUMULATE)
                self._orbit_index += 1
                self._move_to_orbit_waypoint()
            else:
                self._transition_to_orbit_detect()

    def on_ik_response(
        self,
        request_id: str,
        success: bool,
        q_solutions: list[float],
    ) -> None:
        """Handle IK response during scan phases."""
        if request_id != self._ik_id:
            return
        self._ik_id = None
        self._waiting_for_ik = False

        if not success:
            self.logger.error(
                f"Scanner IK failed in state {self._state}"
            )
            self._finish_with_empty()
            return

        data = self._ik_data
        self._ik_data = None

        from legobuilder.brain.ros_bridge import apply_ik_solutions
        trajectory_states = data['trajectory_states']
        apply_ik_solutions(trajectory_states, q_solutions)

        self._publish_trajectory(
            trajectory_states,
            TrajectoryCommandMsg.COMMAND_REPLACE,
        )

    def on_worldmap_detections(
        self, detected_objects: list[Object],
    ) -> None:
        """Receive worldmap contour detection results."""
        if self._state == self.DETECT:
            self._cancel_timer('_detect_timeout_timer')
            self._primary_detections = list(detected_objects)
            self._process_detections(detected_objects)

        elif self._state == self.ORBIT_DETECT:
            self._cancel_timer('_detect_timeout_timer')
            self._accumulated_detections.extend(detected_objects)
            self._orbit_index += 1
            self._move_to_orbit_waypoint()
    
        self.structure.detected_grid_objects = [
            obj for obj in detected_objects
            if obj.center_xyz[0] <= PILE_X_MIN and obj.obj_type.is_block()
        ]
        

    # ── Timer callbacks ───────────────────────────────────────────────

    def _on_scan_tick(self) -> None:
        """Periodic scan during dwell."""
        self._request_scan(ScanRequestMsg.SIMPLY_ACCUMULATE)

    def _on_dwell_complete(self) -> None:
        """Dwell period finished; stop scanning, request detection."""
        self._cancel_timer('_dwell_end_timer')
        self._cancel_timer('_scan_timer')
        self._transition_to_detect()

    def _on_detect_timeout(self) -> None:
        """Detection timed out during primary scan."""
        self._cancel_timer('_detect_timeout_timer')
        self.logger.warning("Scan detect timed out")
        if self._skip_orbit:
            self._finish_with_empty()
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
            f"Orbit detect timed out at waypoint {self._orbit_index}"
        )
        self._orbit_index += 1
        self._move_to_orbit_waypoint()

    # ── Detection processing ─────────────────────────────────────────

    def _process_detections(self, detections: list[Object]) -> None:
        """Rebuild placed_grid from detections and decide next action.

        After overhead scan, if few cells matched compared to expected
        visible cells, triggers orbit fallback for better coverage
        (unless skip_orbit is set).
        """
        placed_grid, unassigned = rebuild_placed_grid(
            detections, self.structure, logger=self.logger,
        )

        _save_placed_grid_figure(placed_grid)
        _save_unassigned_figure(unassigned)

        # Check if orbit fallback is needed
        if not self._skip_orbit and not self._orbit_attempted and self._state == self.DETECT:
            expected_visible = int(np.count_nonzero(
                self.structure.block_grid
            ))
            matched_count = int(np.count_nonzero(placed_grid))
            # Only trigger orbit if we expected some blocks and got < 50%
            if expected_visible > 0 and matched_count < expected_visible * 0.5:
                self.logger.debug(
                    f"Low match rate ({matched_count}/{expected_visible}), "
                    f"triggering orbit fallback"
                )
                self._transition_to_orbit()
                return

        self._finish_with_result(placed_grid, unassigned)

    def _finish_with_result(
        self,
        placed_grid: np.ndarray,
        unassigned: list[Object],
    ) -> None:
        """Build ScanResult and invoke callback."""
        # Pass full list of unassigned types for priority matching
        unassigned_types = (
            [det.obj_type for det in unassigned] if unassigned else None
        )

        # Reachability analysis: detect unreachable cells and exclude them
        current_unreachable = get_unreachable_cells(
            placed_grid, unassigned, self.structure,
        )
        if self._unreachable_tracker is not None:
            self._unreachable_tracker.update(
                current_unreachable,
                block_grid=self.structure.block_grid,
                placed_grid=placed_grid,
            )
            self.logger.info(f"unreachable_blocks: {self._unreachable_tracker.get_all_statuses()}")
            skip_cells = (
                set(current_unreachable.keys())
                | self._unreachable_tracker.ignored_cells
            )
        else:
            skip_cells = (
                set(current_unreachable.keys())
                if current_unreachable else None
            )

        # Filter unreachable blocks from unassigned so the pipeline
        # doesn't attempt to grasp them.
        if skip_cells and unassigned:
            unassigned = [
                obj for obj in unassigned
                if not (
                    obj.obj_type.is_block()
                    and self.structure.world_to_grid(*obj.center_xyz) in skip_cells
                )
            ]
            unassigned_types = (
                [det.obj_type for det in unassigned] if unassigned else None
            )

        target_cell, block_type, is_mistake = next_target_block(
            self.structure.block_grid,
            placed_grid,
            unassigned_types=unassigned_types,
            skip_cells=skip_cells,
        )

        result = ScanResult(
            placed_grid=placed_grid,
            target_cell=target_cell,
            block_type=block_type,
            unassigned=unassigned,
            is_mistake=is_mistake,
        )

        # If target cell was occupied in the previous scan's placed_grid and we
        # haven't already done an orbit rescan this cycle, trigger one orbit scan
        # seeded with the current overhead detections to confirm.
        if (
            self._rescan_for_target_count < HISTORICAL_PLACED_GRID_RESCAN_COUNT
            and target_cell is not None
            and self._historical_placed_grid is not None
            and int(self._historical_placed_grid[target_cell[0], target_cell[1], target_cell[2]]) != 0
        ):
            self.logger.warning(
                f"Target cell {target_cell} was occupied in historical placed_grid "
                f"(val={int(self._historical_placed_grid[target_cell[0], target_cell[1], target_cell[2]])})"
                f" — triggering orbit rescan {self._rescan_for_target_count + 1}"
                f"/{HISTORICAL_PLACED_GRID_RESCAN_COUNT} to verify"
            )
            self._rescan_for_target_count += 1
            self._cleanup_timers()
            self._ik_id = None
            self._ik_data = None
            self._waiting_for_ik = False
            self._transition_to_orbit()
            # Seed orbit accumulation with primary overhead detections so that
            # the final cluster combines both overhead and orbit viewpoints.
            if self._orbit_waypoints:
                self._accumulated_detections = list(self._primary_detections)
            return

        target_diag = ""
        if target_cell is not None:
            r, c, l = target_cell
            bg_val = int(self.structure.block_grid[r, c, l])
            pg_val = int(placed_grid[r, c, l])
            target_diag = f", bg={bg_val}, pg={pg_val}"
        self.logger.info(
            f"Scan complete: "
            f"{int(np.count_nonzero(placed_grid))} cells matched, "
            f"{len(unassigned)} unassigned, "
            f"target_cell={target_cell}, "
            f"block_type={block_type}, "
            f"is_mistake={is_mistake}"
            f"{target_diag}"
        )

        self._cleanup_timers()
        self._state = self.IDLE
        self._emit('state_change', {'state': self.IDLE})
        self._emit('scan_result', {
            'placed_count': int(np.count_nonzero(placed_grid)),
            'unassigned_count': len(unassigned),
            'target_cell': str(target_cell) if target_cell else None,
        })
        self._ik_id = None
        self._ik_data = None
        self._waiting_for_ik = False
        self._orbit_waypoints = []
        self._accumulated_detections = []

        # Publish unreachable block state to dashboard on every scan
        if self._unreachable_tracker is not None:
            self._publish_unreachable_event()

        self._historical_placed_grid = placed_grid.copy()

        if self._on_result is not None:
            self._on_result(result)

    def _publish_unreachable_event(self):
        """Publish unreachable block state to dashboard."""
        import time as _time
        statuses = self._unreachable_tracker.get_all_statuses()
        blocks = []
        for cell, status in statuses.items():
            row, col, layer = cell
            world_xyz = self.structure.get_world_position(row, col, layer)
            blocks.append({
                'grid_cell': {'row': int(row), 'col': int(col), 'layer': int(layer)},
                'world_xyz': {
                    'x': float(world_xyz[0]),
                    'y': float(world_xyz[1]),
                    'z': float(world_xyz[2]),
                },
                'block_type': status.block_type,
                'reason': status.reason,
                'nod_attempts': status.nod_attempts,
                'permanently_ignored': status.permanently_ignored,
            })
        self._post_event({
            'type': 'unreachable_blocks',
            'source': 'scanner',
            'timestamp': _time.time(),
            'data': {'blocks': blocks},
        })

    def _finish_with_empty(self) -> None:
        """Finish with an empty/zero placed_grid."""
        empty_grid = np.zeros_like(
            self.structure.block_grid, dtype=int,
        )
        self._finish_with_result(empty_grid, [])

    # ── DBSCAN clustering ─────────────────────────────────────────────

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

    # ── Trajectory planning ───────────────────────────────────────────

    def _plan_scan_trajectory(self) -> Optional[list[TrajectoryState]]:
        """Plan trajectory to overhead scan position above grid center.

        Returns
        -------
        list[TrajectoryState] or None
            Single-state trajectory, or None if unreachable.
        """
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
        """Generate orbit scan waypoints around grid center.

        Returns
        -------
        list of (position, orientation)
            Reachable waypoints at ORBIT_SCAN_HEIGHT above grid center,
            offset laterally by ORBIT_SCAN_RADIUS.
        """
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

    def _cleanup_timers(self) -> None:
        """Cancel all active timers."""
        self._cancel_timer('_scan_timer')
        self._cancel_timer('_detect_timeout_timer')
        self._cancel_timer('_dwell_end_timer')

    def reset(self) -> None:
        """Reset all scanner state."""
        self._cleanup_timers()
        self._state = self.IDLE
        self._emit('state_change', {'state': self.IDLE})
        self._skip_orbit = False
        self._orbit_attempted = False
        self._ik_id = None
        self._ik_data = None
        self._waiting_for_ik = False
        self._orbit_waypoints = []
        self._orbit_index = 0
        self._accumulated_detections = []
        self._rescan_for_target_count = 0
        self._primary_detections = []
        # _historical_placed_grid intentionally NOT reset here
        self._on_result = None

    @property
    def is_active(self) -> bool:
        """True if a scan is in progress."""
        return self._state != self.IDLE


def _save_placed_grid_figure(placed_grid: np.ndarray) -> None:
    """Save a 3D voxel visualization of the rebuilt placed_grid to disk."""
    import os
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    grid = placed_grid
    filled = grid != 0

    colors = np.zeros(grid.shape + (4,))
    for val, rgb in BlockStructure.BLOCK_COLORS.items():
        mask = grid == val
        colors[mask] = (*rgb, 0.9)

    ax.voxels(filled, facecolors=colors, edgecolors='gray', linewidth=0.5)

    ax.set_xlabel('Col')
    ax.set_ylabel('Row')
    ax.set_zlabel('Layer')
    ax.set_xlim(0, grid.shape[1])
    ax.set_ylim(0, grid.shape[0])
    ax.set_zlim(0, grid.shape[2])
    ax.set_xticks(range(grid.shape[1] + 1))
    ax.set_yticks(range(grid.shape[0] + 1))
    ax.set_zticks(range(grid.shape[2] + 1))
    ax.set_aspect('equal')
    ax.set_title(f"Rebuilt placed_grid ({int(np.count_nonzero(grid))} blocks)")

    plt.tight_layout()
    save_dir = '/home/robot/robotws/src/legobuilder/tmp/structure_scanner'
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'placed_grid.png'), dpi=150)
    plt.close(fig)


def _save_unassigned_figure(unassigned: list[Object]) -> None:
    """Save a 3D visualization of unassigned (off-grid) detections."""
    if not unassigned:
        return

    import os
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    half = BLOCK_SIZE / 2

    for obj in unassigned:
        cx, cy, cz = obj.center_xyz
        angle = obj.angle if obj.angle is not None else 0.0
        ca, sa = np.cos(angle), np.sin(angle)

        # Local corners of a square block face (XY), centered at origin
        local = np.array([
            [-half, -half],
            [ half, -half],
            [ half,  half],
            [-half,  half],
        ])
        # Rotate and translate
        rotated = np.column_stack([
            local[:, 0] * ca - local[:, 1] * sa + cx,
            local[:, 0] * sa + local[:, 1] * ca + cy,
        ])

        z_bot = cz
        z_top = cz + BLOCK_SIZE

        # 6 faces of the prism
        bot = [(rotated[i, 0], rotated[i, 1], z_bot) for i in range(4)]
        top = [(rotated[i, 0], rotated[i, 1], z_top) for i in range(4)]
        faces = [bot, top]
        for i in range(4):
            j = (i + 1) % 4
            faces.append([bot[i], bot[j], top[j], top[i]])

        color_val = obj.obj_type.value if obj.obj_type else 1
        rgb = BlockStructure.BLOCK_COLORS.get(color_val, (0.5, 0.5, 0.5))

        poly = Poly3DCollection(faces, alpha=0.9, linewidth=0.5,
                                edgecolor='gray')
        poly.set_facecolor((*rgb, 0.9))
        ax.add_collection3d(poly)

    # Compute axis limits from all block centers
    xs = [o.center_xyz[0] for o in unassigned]
    ys = [o.center_xyz[1] for o in unassigned]
    zs = [o.center_xyz[2] for o in unassigned]
    margin = BLOCK_SIZE * 2
    ax.set_xlim(min(xs) - margin, max(xs) + margin)
    ax.set_ylim(min(ys) - margin, max(ys) + margin)
    ax.set_zlim(min(0, min(zs) - margin), max(zs) + BLOCK_SIZE + margin)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_aspect('equal')
    ax.set_title(f"Unassigned blocks ({len(unassigned)})")

    plt.tight_layout()
    save_dir = '/home/robot/robotws/src/legobuilder/tmp/structure_scanner'
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'unassigned.png'), dpi=150)
    plt.close(fig)

"""Calibration routine for grid coordinates, Z offsets, and XY offsets.

Detects purple corner blocks from the overhead camera to compute grid
coordinates.  Also calibrates Z offsets by scanning the table surface
and XY offsets by detecting a blue circle on the EE camera mount.
Follows the callback-injection pattern (no ROS imports).
"""

from typing import Callable, Optional
from collections import deque

import csv
import numpy as np

from legobuilder.kinematics.trajectory import TrajectoryState
from legobuilder.kinematics.block_manipulator import ManipulatorState

from legobuilder.brain.block_structure import BlockStructure
from legobuilder.config import (
    BLOCK_SIZE,
    CALIBRATE_MOVE_DURATION,
    CALIBRATE_WAIT_DURATION,
    CALIBRATE_Z,
    CALIBRATE_Z_POSES,
    CALIBRATE_XY,
    CALIBRATE_XY_POSES,
    XY_CAL_SCAN_TIMEOUT,
    CALIBRATE_SAFE_HEIGHT,
    CALIBRATE_GRID_COORDS,
    JOINT_NAMES,
    Q_READY,
)

from legobuilder_interfaces.msg import (
    IKRequestMsg,
    TrajectoryCommandMsg,
    ScanRequestMsg,
)


class Calibrator:
    """State machine that calibrates world positions.

    Detects grid corners from the overhead camera (purple blocks).
    Calibrates Z error map by scanning the table surface.
    Calibrates XY offsets by detecting a blue circle on the EE camera mount.

    States: IDLE → DETECT_GRID_COORDS → MOVE_TO_POS <--> SCAN
                                       → MOVE_TO_XY_POS <--> SCAN → DONE
    """

    IDLE = 'idle'
    DETECT_GRID_COORDS = 'detect_grid_coords'
    MOVE_TO_POS = 'move_to_pos'
    MOVE_TO_XY_POS = 'move_to_xy_pos'
    SCAN = 'scan'
    MOVE_TO_READY = 'move_to_ready'
    DONE = 'done'

    def __init__(
        self,
        send_ik_request_fn: Callable,
        publish_trajectory_fn: Callable,
        request_scan_fn: Callable,
        request_height_fn: Callable,
        create_timer_fn: Callable,
        destroy_timer_fn: Callable,
        get_t_fn: Callable,
        logger,
        fkin_fn: Optional[Callable] = None,
        get_current_q_fn: Optional[Callable] = None,
        post_event_fn=None,
    ):
        self._send_ik_request = send_ik_request_fn
        self._publish_trajectory = publish_trajectory_fn
        self._request_scan = request_scan_fn
        self._request_height = request_height_fn
        self._create_timer = create_timer_fn
        self._destroy_timer = destroy_timer_fn
        self._get_t = get_t_fn
        self.logger = logger
        self._fkin_fn = fkin_fn
        self._get_current_q_fn = get_current_q_fn
        self._post_event = post_event_fn or (lambda e: None)

        self._state: str = self.IDLE
        self._on_complete: Optional[Callable] = None

        # IK tracking
        self._ik_id: Optional[str] = None
        self._ik_data: Optional[dict] = None
        self._waiting_for_ik: bool = False

        # Calibration of z
        self.table_heights = {} # (r) position of tip: table_height_detected
        self.calibrate_z_states = deque(
            ManipulatorState(p=p, o=o) for p, o in CALIBRATE_Z_POSES
        )

        # Calibration of xy
        self.xy_offsets = []  # list of (true_x, true_y, true_z, placed_x, placed_y, placed_z)
        self.calibrate_xy_states = deque(
            ManipulatorState(p=p, o=o) for p, o in CALIBRATE_XY_POSES
        )
        self._xy_cal_timer = None


    def _emit_state(self) -> None:
        self._post_event({
            'type': 'state_change',
            'source': 'calibrator',
            'data': {'state': self._state},
            'timestamp': self._get_t(),
        })

    # ── Public API ──────────────────────────────────────────────────

    def start_calibration(self, on_complete: Callable) -> None:
        """Begin the calibration sequence."""
        self._on_complete = on_complete
        self.logger.info(
            "[CALIBRATE] Starting calibration sequence"
        )
        if CALIBRATE_GRID_COORDS:
            self._move_to_ready()
        elif CALIBRATE_Z and len(self.calibrate_z_states) > 0:
            self._move_to_pos()
        elif CALIBRATE_XY and len(self.calibrate_xy_states) > 0:
            self._move_to_xy_pos()
        else:
            self._done()

    @property
    def is_active(self) -> bool:
        """True if calibration is in progress."""
        return self._state not in (self.IDLE, self.DONE)

    @property
    def is_done(self) -> bool:
        """True if calibration has completed."""
        return self._state == self.DONE

    # ── IK / trajectory callbacks ───────────────────────────────────

    def on_ik_response(
        self,
        request_id: str,
        success: bool,
        q_solutions: list[float],
    ) -> None:
        """Handle IK response during calibration."""
        if request_id != self._ik_id:
            return
        self._ik_id = None
        self._waiting_for_ik = False

        if not success:
            self.logger.error("[CALIBRATE] IK failed for grid center")
            self._state = self.DONE
            self._emit_state()
            if self._on_complete is not None:
                self._on_complete()
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

    def on_trajectory_complete(self) -> None:
        """Handle trajectory completion during calibration."""
        if self._state == self.MOVE_TO_READY:
            self._detect_grid_coords()
        elif self._state == self.MOVE_TO_POS:
            self._state = self.SCAN
            self._emit_state()
            self.logger.info("[CALIBRATE] At pos for z calibration, requesting scan")
            self._request_scan(ScanRequestMsg.IN_Z_CALIBRATION)
        elif self._state == self.MOVE_TO_XY_POS:
            self._state = self.SCAN
            self._emit_state()
            self.logger.info("[CALIBRATE] At pos for xy calibration, requesting scan")
            self._request_scan(ScanRequestMsg.IN_XY_CALIBRATION)
            self._xy_cal_timer = self._create_timer(
                XY_CAL_SCAN_TIMEOUT, self._on_xy_scan_timeout)

    def on_height_recieved(self, height) -> None:
        manipulator_state = self.calibrate_z_states.popleft()
        x, y, z = manipulator_state.p[0], manipulator_state.p[1], manipulator_state.p[2]
        r = np.sqrt(x**2 + y**2)
        self.table_heights[(r, z)] = height

        if len(self.calibrate_z_states) > 0:
            self._move_to_pos()
        else:
            self._write_z_calibration()
            self._advance_after_z()

    def on_xy_position_received(self, placed_x, placed_y, true_x, true_y) -> None:
        """Handle detected blue circle position from overhead camera."""
        self._cancel_xy_timer()
        self.calibrate_xy_states.popleft()

        # NOTE: these 3 lines are the correct logic but for some reason grid calibration is off i think -firdavs
        # # CSV column order: (fk_x, fk_y, fk_z, cam_x, cam_y, cam_z)
        # # IK solver computes offset = col0 - col3 (fk - cam) per axis
        # self.xy_offsets.append((true_x, true_y, 0.0, placed_x, placed_y, 0.0))
        
        # NOTE: swapping order when CALIBRATE_GRID = False -> works fine
        # CSV column order: (cam_x, cam_y, cam_z, fk_x, fk_y, fk_z)                                                                                                               
        # IK solver computes offset = col0 - col3 (cam - fk) per axis                                                                                                             
        self.xy_offsets.append((placed_x, placed_y, 0.0, true_x, true_y, 0.0))  
        self.logger.info(
            f"[XY-CALIBRATE] FK=({true_x:.4f}, {true_y:.4f}) "
            f"cam=({placed_x:.4f}, {placed_y:.4f}) "
            f"err=({true_x - placed_x:.4f}, {true_y - placed_y:.4f})")

        if len(self.calibrate_xy_states) > 0:
            self._move_to_xy_pos()
        else:
            self._write_xy_calibration()
            self._advance_after_xy()


    def on_xy_detection_failed(self) -> None:
        """Handle failed blue circle detection — skip this pose and advance."""
        self._cancel_xy_timer()
        if not self.calibrate_xy_states:
            self.logger.warning("[XY-CALIBRATE] Detection failed but no poses left")
            self._advance_after_xy()
            return
        skipped = self.calibrate_xy_states.popleft()
        self.logger.warning(
            f"[XY-CALIBRATE] Detection failed at "
            f"({skipped.p[0]:.3f}, {skipped.p[1]:.3f}), skipping")
        if len(self.calibrate_xy_states) > 0:
            self._move_to_xy_pos()
        else:
            self._write_xy_calibration()
            self._advance_after_xy()

    def _on_xy_scan_timeout(self) -> None:
        """Timer callback: xy scan timed out without any response."""
        self._cancel_xy_timer()
        if not self.calibrate_xy_states:
            self.logger.warning("[XY-CALIBRATE] Scan timed out but no poses left")
            self._advance_after_xy()
            return
        skipped = self.calibrate_xy_states.popleft()
        self.logger.warning(
            f"[XY-CALIBRATE] Scan timed out at "
            f"({skipped.p[0]:.3f}, {skipped.p[1]:.3f}), skipping")
        if len(self.calibrate_xy_states) > 0:
            self._move_to_xy_pos()
        else:
            self._write_xy_calibration()
            self._advance_after_xy()

    def _cancel_xy_timer(self) -> None:
        """Cancel and destroy the xy calibration timeout timer."""
        if self._xy_cal_timer is not None:
            self._xy_cal_timer.cancel()
            self._destroy_timer(self._xy_cal_timer)
            self._xy_cal_timer = None

    # ── Internal ────────────────────────────────────────────────────

    Z_CALIBRATION_PATH = 'src/legobuilder/offsets/z_calibration.csv'
    XY_CALIBRATION_PATH = 'src/legobuilder/offsets/xy_calibrations.csv'

    def _write_z_calibration(self):
        """Write z-calibration table_heights to CSV file for IK solver."""
        self.logger.info(
            f"[CALIBRATE] Writing z calibration ({len(self.table_heights)} "
            f"entries) to {self.Z_CALIBRATION_PATH}"
        )
        with open(self.Z_CALIBRATION_PATH, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['r', 'z', 'table_height'])
            for (r, z), h in self.table_heights.items():
                writer.writerow([round(r, 4), round(z, 4), round(h, 4)])

    def _write_xy_calibration(self):
        """Write xy-calibration offsets to CSV file for IK solver."""
        self.logger.info(
            f"[CALIBRATE] Writing xy calibration ({len(self.xy_offsets)} "
            f"entries) to {self.XY_CALIBRATION_PATH}"
        )
        with open(self.XY_CALIBRATION_PATH, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['true_x', 'true_y', 'true_z', 'placed_x', 'placed_y', 'placed_z'])
            for row in self.xy_offsets:
                writer.writerow([round(v, 4) for v in row])

    def _advance_after_z(self):
        """Advance to next calibration phase after z-calibration completes."""
        if CALIBRATE_XY and len(self.calibrate_xy_states) > 0:
            self._move_to_xy_pos()
        else:
            self._done()

    def _advance_after_xy(self):
        """Advance to next calibration phase after xy-calibration completes."""
        self._done()

    def _done(self):
        """Sets the state of the calibrator to done."""
        self._state = self.DONE
        self._emit_state()
        self.logger.info("[CALIBRATE] Calibration scan complete")
        if self._on_complete is not None:
            self._on_complete()


    def _move_to_pos(self) -> None:
        """Move arm to next position in calibration of z."""
        self._state = self.MOVE_TO_POS
        self._emit_state()
        target = self.calibrate_z_states[0]
        traj_states = [
            TrajectoryState(
                final_state=target,
                min_duration=CALIBRATE_MOVE_DURATION,
                delay_after=CALIBRATE_WAIT_DURATION,
            ),
        ]
        request_id = self._send_ik_request(
            traj_states, IKRequestMsg.REQUEST_PRIMARY,
            in_calibration=True,
        )
        self._ik_id = request_id
        self._waiting_for_ik = True
        self._ik_data = {
            'trajectory_states': traj_states,
        }

    def _move_to_xy_pos(self) -> None:
        """Move arm to next position in calibration of xy."""
        self._state = self.MOVE_TO_XY_POS
        self._emit_state()
        target = self.calibrate_xy_states[0]
        lift_state = ManipulatorState(
            p=np.array([target.p[0], target.p[1], CALIBRATE_SAFE_HEIGHT]),
            o=target.o.copy(),
        )
        traj_states = [
            TrajectoryState(
                final_state=lift_state,
                min_duration=CALIBRATE_MOVE_DURATION,
                delay_after=CALIBRATE_WAIT_DURATION/2,
            ),
            TrajectoryState(
                final_state=target,
                min_duration=CALIBRATE_MOVE_DURATION,
                delay_after=CALIBRATE_WAIT_DURATION,
            ),
        ]
        request_id = self._send_ik_request(
            traj_states, IKRequestMsg.REQUEST_PRIMARY,
            in_calibration=True,
        )
        self._ik_id = request_id
        self._waiting_for_ik = True
        self._ik_data = {
            'trajectory_states': traj_states,
        }


    def _move_to_ready(self) -> None:
        """Move arm to Q_READY so it doesn't occlude the grid."""
        self._state = self.MOVE_TO_READY
        self._emit_state()
        self.logger.info("[CALIBRATE] Moving arm to Q_READY before grid detection")
        self._publish_trajectory(
            [TrajectoryState(
                final_state=ManipulatorState(
                    q=Q_READY.copy(),
                    qd=np.zeros(len(JOINT_NAMES)),
                ),
                min_duration=3.0,
                delay_after=1.0,
            )],
            TrajectoryCommandMsg.COMMAND_REPLACE,
        )

    def _detect_grid_coords(self) -> None:
        """Request overhead camera grid corner detection."""
        self._state = self.DETECT_GRID_COORDS
        self._emit_state()
        self.logger.info("[CALIBRATE] Requesting grid coord detection from overhead camera")
        self._request_scan(ScanRequestMsg.DETECT_GRID_COORDS)

    def on_grid_coords_received(self, corners: np.ndarray) -> None:
        """Handle detected grid corners and advance to next phase.

        Arguments
        ---------
        corners : np.ndarray
            Shape (4, 2) array of [bl, br, tl, tr] world XY coordinates.
        """
        self.logger.info(
            f"[CALIBRATE] Grid corners received: "
            f"BL=({corners[0][0]:.4f}, {corners[0][1]:.4f}) "
            f"BR=({corners[1][0]:.4f}, {corners[1][1]:.4f}) "
            f"TL=({corners[2][0]:.4f}, {corners[2][1]:.4f}) "
            f"TR=({corners[3][0]:.4f}, {corners[3][1]:.4f})"
        )
        if CALIBRATE_Z and len(self.calibrate_z_states) > 0:
            self._move_to_pos()
        elif CALIBRATE_XY and len(self.calibrate_xy_states) > 0:
            self._move_to_xy_pos()
        else:
            self._done()

"""Block separation trajectory planning.

Generates trajectory states to separate a gripped block from magnetically
attached neighbors. Currently handles single-face connections; multi-face
strategies will be added later.
"""

from math import pi
from typing import Optional

import numpy as np

from legobuilder.config import (
    BASE_MOTOR_POS,
    BLOCK_SIZE,
    BLOCK_TOP_OFFSET,
    PILE_CENTER,
    SEPARATION_TILT_HEIGHT,
    SEPARATION_TILT_ANGLE,
    SEPARATION_INNER_THRESHOLD,
    SEPARATION_RADIAL_NUDGE,
    grip_check_roll as get_grip_check_roll,
    wrap_roll,
    PICK_Z
)
from legobuilder.schemas import Direction, Object
from legobuilder.kinematics.trajectory import TrajectoryState
from legobuilder.kinematics.block_manipulator import ManipulatorState


class BlockSeparation:
    """Plans separation trajectories for magnetically connected blocks."""

    def __init__(self, pick_tilt: float, pick_duration: float, logger):
        self.pick_tilt = pick_tilt
        self.pick_duration = pick_duration
        self.logger = logger

    def find_clearing_target(
        self,
        target_obj: Object,
        detected_objects: list[Object],
    ) -> Optional[Object]:
        """Find a neighbor of target_obj that has ≤1 face connection.

        Matches target's face_conns_xyzs positions to other detected
        objects' center_xyz. Returns the first match with ≤1 connection,
        or None if no separable neighbor found.
        """
        for face_pos in target_obj.face_conns_xyzs:
            face_xy = np.array(face_pos[:2])
            for det_obj in detected_objects:
                if det_obj is target_obj:
                    continue
                obj_xy = np.array(det_obj.center_xyz[:2])
                dist = np.linalg.norm(face_xy - obj_xy)
                if dist < BLOCK_SIZE and len(det_obj.face_conns_xyzs) <= 1:
                    return det_obj
        return None

    def adjust_pickup_roll(
        self,
        pickup_roll: float,
        connected_directions: set[Direction],
    ) -> float:
        """Rotate pickup roll to avoid collision with the attached neighbor."""
        if Direction.LEFT in connected_directions:
            return wrap_roll(pickup_roll + pi / 2)
        elif Direction.RIGHT in connected_directions:
            return wrap_roll(pickup_roll - pi / 2)
        elif Direction.ABOVE in connected_directions:
            return wrap_roll(pickup_roll + pi)
        return pickup_roll

    def plan_single_face(
        self,
        p_pick: np.ndarray,
        connected_directions: set[Direction],
    ) -> list[TrajectoryState]:
        """Generate separation trajectory for a block with one face connection.

        4 states: radial align → lift → tilt → lower-with-tilt.
        """
        grip_roll = get_grip_check_roll(p_pick)
        self.logger.info(f"connected_directions: {connected_directions}")

        states = []

        # Radial nudge: if block is too close to base motor, move outward
        # before tilting to avoid wrist-to-lower-arm collision
        p_rel = p_pick[:2] - BASE_MOTOR_POS[:2]
        r_xy = np.linalg.norm(p_rel)
        if r_xy < SEPARATION_INNER_THRESHOLD:
            d_hat = p_rel / r_xy
            p_nudged = p_pick.copy()
            p_nudged[:2] += d_hat * SEPARATION_RADIAL_NUDGE
            states.append(TrajectoryState(
                final_state=ManipulatorState(
                    p=p_nudged,
                    o=np.array([self.pick_tilt, grip_roll]),
                    gripper_open=False,
                ),
                min_duration=self.pick_duration,
            ))
            p_pick = p_nudged
            grip_roll = get_grip_check_roll(p_pick)
            self.logger.info(
                f"Radial nudge: r_xy={r_xy:.3f}m < {SEPARATION_INNER_THRESHOLD}m, "
                f"nudged {SEPARATION_RADIAL_NUDGE}m outward"
            )

        states.extend([
            # Align gripped block radially
            TrajectoryState(
                final_state=ManipulatorState(
                    p=p_pick,
                    o=np.array([self.pick_tilt, grip_roll]),
                    gripper_open=False,
                ),
                min_duration=self.pick_duration,
            ),
            # Lift
            TrajectoryState(
                final_state=ManipulatorState(
                    p=p_pick + np.array([0, 0, SEPARATION_TILT_HEIGHT]),
                    o=np.array([self.pick_tilt, grip_roll]),
                    gripper_open=False,
                ),
                min_duration=self.pick_duration / 1.5,
            ),
            # Tilt backward
            TrajectoryState(
                final_state=ManipulatorState(
                    p=p_pick + np.array([0, 0, SEPARATION_TILT_HEIGHT]),
                    o=np.array([
                        self.pick_tilt + SEPARATION_TILT_ANGLE,
                        grip_roll,
                    ]),
                    gripper_open=False,
                ),
                min_duration=self.pick_duration / 1.5,
            ),
            # Lower with tilt (breaks magnetic contact)
            TrajectoryState(
                final_state=ManipulatorState(
                    p=p_pick,
                    o=np.array([
                        self.pick_tilt + SEPARATION_TILT_ANGLE,
                        grip_roll,
                    ]),
                    gripper_open=False,
                ),
                min_duration=self.pick_duration / 1.5,
            ),
        ])

        return states

    def plan_relocate_and_separate(
        self,
        p_pick: np.ndarray,
        connected_directions: set[Direction],
        center_z: float,
    ) -> list[TrajectoryState]:
        """Move connected blocks above pile area, then separate.

        Used when a connected block is picked up from the 6x6 grid area.
        Moves the group laterally over the pile before doing the tilt
        maneuver so detached blocks fall into the pile, not onto the
        structure.

        Parameters
        ----------
        p_pick : np.ndarray
            Original pick position on the grid.
        connected_directions : set[Direction]
            Directions with magnetic connections.
        center_z : float
            The detected center_z of the block being picked.
        """
        # Pile surface is BLOCK_SIZE lower than the base grid surface,
        # so when relocating from the grid, lower by one BLOCK_SIZE.
        pick_z = center_z + PICK_Z
        if center_z >= BLOCK_SIZE:
            pile_z = pick_z - BLOCK_SIZE
        else:
            pile_z = pick_z

        p_pile = np.array([PILE_CENTER[0], PILE_CENTER[1], pile_z])
        grip_roll = get_grip_check_roll(p_pick)

        states = [
            # Lift above pick position to clear structure blocks
            TrajectoryState(
                final_state=ManipulatorState(
                    p=p_pick + np.array([0, 0, SEPARATION_TILT_HEIGHT]),
                    o=np.array([self.pick_tilt, grip_roll]),
                    gripper_open=False,
                ),
                min_duration=self.pick_duration / 1.5,
            ),
            # Move laterally above pile area
            TrajectoryState(
                final_state=ManipulatorState(
                    p=p_pile + np.array([0, 0, SEPARATION_TILT_HEIGHT]),
                    o=np.array([self.pick_tilt, get_grip_check_roll(p_pile)]),
                    gripper_open=False,
                ),
                min_duration=self.pick_duration,
            ),
        ]

        # Separation tilt maneuver at the pile position
        states.extend(self.plan_single_face(p_pile, connected_directions))

        return states

    def plan(
        self,
        num_connected_faces: int,
        p_pick: np.ndarray,
        connected_directions: set[Direction],
    ) -> Optional[list[TrajectoryState]]:
        """Dispatch to the appropriate separation strategy.

        Returns None if separation is not needed (0 faces) or not
        yet supported (2+ faces).
        """
        if num_connected_faces == 0:
            return None
        elif num_connected_faces == 1:
            return self.plan_single_face(p_pick, connected_directions)
        else:
            return None

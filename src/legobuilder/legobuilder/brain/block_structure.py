"""3D grid model for block structure assembly and trajectory planning.

Represents a discrete (row, col, layer) grid of magnetic cubic blocks and
provides methods for planning grasp, approach, and placement trajectories.
Used by BuildPipeline to coordinate the 3-phase build state machine.
"""

import numpy as np
import json
import os
from collections import deque
from math import atan2, cos, pi, sin
from copy import deepcopy
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt

from legobuilder.schemas import ObjectType, Object, Color, Direction
from legobuilder.kinematics.trajectory import TrajectoryState
from legobuilder.kinematics.block_manipulator import ManipulatorState, BlockManipulator

from math import sqrt

from legobuilder.brain.separation import BlockSeparation
from legobuilder.config import (
    Q_READY,
    Q_SCAN,
    JOINT_NAMES,
    BLOCK_SIZE,
    OUTER_RADIUS,
    INNER_RADIUS,
    BASE_MOTOR_POS,
    PLACEMENT_GRID_COORDS,
    GRID_PLACEMENT,
    PLACEMENT_PUSH,
    FLOATING_PUSH_DISTANCE,
    BLOCK_TOP_OFFSET,
    PICK_Z,
    PICK_TILT,
    grip_check_roll as get_grip_check_roll,
    wrap_roll,
    PILE_X_MIN,
    MAX_GRID_SIZE,
)


def is_point_reachable(p: np.ndarray) -> bool:
    """Check if a point is within the robot's reachable workspace.

    Tests radial distance from the base motor against inner/outer
    radius limits and rejects points in the negative-x, negative-y,
    or below-ground half-spaces.

    Arguments
    ---------
    p : np.ndarray
        3D world position [x, y, z].

    Returns
    -------
    bool
        True if the point is reachable.
    """
    p_rel = p - BASE_MOTOR_POS
    r_xy = sqrt(p_rel[0]**2 + p_rel[1]**2)
    if (r_xy > OUTER_RADIUS or r_xy < INNER_RADIUS
            or p[0] <= 0 or p[1] <= 0 or p[2] < 0):
        return False
    return True


_DEBUG_DIR = os.path.expanduser('/home/robot/robotws/src/legobuilder/tmp')


def debug_compute_pickup_roll(obj_face_centers, other_face_centers, obj, augmented):
    """Save debug plots for compute_pickup_roll face/corner geometry."""
    os.makedirs(_DEBUG_DIR, exist_ok=True)

    # Plot 1: Face Centers
    fig, ax = plt.subplots(figsize=(10, 10))
    # Target face centers
    for i, fc in enumerate(obj_face_centers):
        ax.scatter(fc[0], fc[1], c='red', s=100, zorder=5)
        ax.annotate(f'target face {i}', (fc[0], fc[1]),
                    textcoords='offset points', xytext=(5, 5), fontsize=8, color='red')
    # Other face centers
    for ofc in other_face_centers:
        ax.scatter(ofc[0], ofc[1], c='blue', s=40, zorder=4, alpha=0.6)
    # Target center
    ax.scatter(obj.center_xyz[0], obj.center_xyz[1], c='red', s=200, marker='x', zorder=6)
    ax.annotate('target center', (obj.center_xyz[0], obj.center_xyz[1]),
                textcoords='offset points', xytext=(5, -10), fontsize=8, color='red')
    ax.set_aspect('equal')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title('Face Centers (red=target, blue=other)')
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(_DEBUG_DIR, 'debug_face_centers.png'), dpi=150, bbox_inches='tight')
    plt.show()
    plt.close(fig)

    # Plot 2: Object Corners
    fig, ax = plt.subplots(figsize=(10, 10))
    # Target obj corners as closed polygon
    if obj.corner_xys is not None and len(obj.corner_xys) == 4:
        xs = [c[0] for c in obj.corner_xys] + [obj.corner_xys[0][0]]
        ys = [c[1] for c in obj.corner_xys] + [obj.corner_xys[0][1]]
        ax.plot(xs, ys, 'r-', linewidth=2, zorder=5, label='target')
        ax.scatter([c[0] for c in obj.corner_xys], [c[1] for c in obj.corner_xys],
                   c='red', s=60, zorder=6)
    ax.scatter(obj.center_xyz[0], obj.center_xyz[1], c='red', s=200, marker='x', zorder=7)
    # Augmented objects
    for det in augmented:
        if det is obj:
            continue
        if det.corner_xys is not None and len(det.corner_xys) == 4:
            xs = [c[0] for c in det.corner_xys] + [det.corner_xys[0][0]]
            ys = [c[1] for c in det.corner_xys] + [det.corner_xys[0][1]]
            ax.plot(xs, ys, 'b-', linewidth=1.5, alpha=0.7, zorder=3)
            ax.scatter([c[0] for c in det.corner_xys], [c[1] for c in det.corner_xys],
                       c='blue', s=30, zorder=4, alpha=0.7)
        center = det.center_xyz if hasattr(det, 'center_xyz') else None
        if center is not None:
            ax.scatter(center[0], center[1], c='blue', s=100, marker='+', zorder=5, alpha=0.7)
    ax.set_aspect('equal')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title('Object Corners (red=target, blue=augmented)')
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(_DEBUG_DIR, 'debug_corners.png'), dpi=150, bbox_inches='tight')
    plt.show()
    plt.close(fig)


def _make_virtual_object(
    obj_type_val: int,
    center_xyz: tuple[float, float, float],
    angle: float = 0.0,
) -> Object:
    """Create a virtual Object from type value and position.

    Reconstructs corner_xys from center and angle so the object can
    participate in face-distance calculations.
    """
    cx, cy = center_xyz[0], center_xyz[1]
    half = BLOCK_SIZE / 2
    ca, sa = cos(angle), sin(angle)
    offsets = [(-half, half), (half, half), (half, -half), (-half, -half)]
    corner_xys = [
        (cx + dx * ca - dy * sa, cy + dx * sa + dy * ca)
        for dx, dy in offsets
    ]
    return Object(
        t=0,
        frame_idx=0,
        obj_type=obj_type_val,
        center_uv=(0, 0),
        center_xyz=center_xyz,
        angle=angle,
        corner_xys=corner_xys,
        corner_uvs=None,
        face_conns_xyzs=None,
    )


class BlockStructure:
    """3D grid of magnetic cubic blocks for structure assembly.

    Tracks block positions in a discrete (row, col, layer) grid and
    provides methods for planning grasp, approach, and placement
    trajectories.  The grid stores ObjectType.value ints
    (0 = empty) and supports 6-connected adjacency.

    Attributes
    ----------
    block_grid : np.ndarray
        3D int array (length x width x height).  Non-zero values
        are ObjectType enum values indicating block color.
    placed_grid : np.ndarray
        Same shape as block_grid.  Tracks which blocks the robot
        has physically placed (vs. pre-loaded from JSON).
    origin : np.ndarray
        World XYZ position of the grid's (0, 0, 0) corner.
    roll : float
        Structure rotation in radians about the Z axis.
    blocks_to_place : deque
        Queue of (row, col, layer) tuples for the robot to place.
    logger
        ROS logger instance.
    """

    # Approach/retreat clearance above the target cell (meters)
    APPROACH_HEIGHT = 0.1
    # Magnitude of the lateral approach offset (meters)
    APPROACH_VECTOR_MAGNITUDE = 0.1
    # Grasp height: 3/4 of block thickness above the table (~0.017 m)
    PICK_DURATION = 3.0      # seconds for pick motions
    PLACE_DURATION = 3.0     # seconds for place motions
    GRIPPER_DELAY = 0.5      # seconds delay for gripper open/close

    # ── Lifecycle ─────────────────────────────────────────────────────

    def __init__(
        self,
        logger,
        max_length: int = 8,
        max_width: int = 8,
        max_height: int = 8,
        roll: float = 0.0,
        origin: Optional[np.ndarray] = None,
    ):
        """Initialize an empty block structure grid.

        Arguments
        ---------
        logger
            ROS logger instance.
        max_length : int
            Grid extent along the row axis.
        max_width : int
            Grid extent along the column axis.
        max_height : int
            Grid extent along the layer axis.
        roll : float
            Structure rotation about Z (radians).
        origin : np.ndarray | None
            World XYZ of the grid origin.
            If None, determine based on PLACEMENT_GRID_COORDS
        """
        self.block_grid = np.zeros(
            (max_length, max_width, max_height), dtype=int,
        )
        self.blocks_to_place = deque()
        self.visible_blocks = set() # Set of (row, col, layer) tuples for placed blocks currently visible to the camera (i.e. not occluded by other blocks above them)
        self.invisible_blocks = set() # Set of (row, col, layer) tuples for placed blocks that are currently occluded by other blocks above them
        self.logger = logger
        self.placed_grid = np.zeros_like(self.block_grid, dtype=int)
    
        self.detected_grid_objects: list = []

        self.separator = BlockSeparation(
            pick_tilt=PICK_TILT,
            pick_duration=self.PICK_DURATION,
            logger=self.logger,
        )

        if origin is None:
            self.origin = self.get_grid_origin(PLACEMENT_GRID_COORDS)
            self.roll = 0
        else:
            self.origin = np.array(origin, dtype=float)
            self.roll = roll
            
    @classmethod
    def from_json(cls, filename: str, logger, origin: np.ndarray = None, roll: float = 0.0):
        """Parse a JSON assembly file and return a BlockStructure.

        The JSON blocks array defines the placement order the robot
        will follow.

        Arguments
        ---------
        filename : str
            Path to JSON file containing a list of block placements.
        logger
            ROS logger instance.
        origin : np.ndarray | None
            World XYZ of the grid origin.
            If None, determine based on PLACEMENT_GRID_COORDS
        roll : float
            Structure rotation about Z (radians).

        Returns
        -------
        BlockStructure
            Populated grid with blocks_to_place queue set.
        """
        with open(filename, 'r') as f:
            data = json.load(f)

        metadata = data.get("metadata", None)
        grid_size = metadata.get("gridSize", None) if metadata else None
        if not grid_size or len(grid_size) != 3:
            raise ValueError(
                "JSON metadata must contain 'gridSize' with 3 integers"
            )

        max_length = grid_size.get('length', None)
        max_width = grid_size.get('width', None)
        max_height = grid_size.get('height', None)

        if not all(
            isinstance(x, int) and x > 0
            for x in [max_length, max_width, max_height]
        ):
            raise ValueError("Grid dimensions must be positive integers")

        structure = cls(
            logger=logger,
            max_length=max_length,
            max_width=max_width,
            max_height=max_height,
            origin=origin,
            roll=roll,
        )

        blocks = data.get("blocks", None)
        for block in blocks:
            pos = block.get("position", None)
            color = block.get("color", None)
            if pos is None or color is None:
                raise ValueError(
                    "Each block must have 'position' and 'color'"
                )
            col = pos.get("x", None)
            row = pos.get("y", None)
            layer = pos.get("z", None)
            if col is None or row is None or layer is None:
                raise ValueError(
                    "Block position must have 'x', 'y', and 'z'"
                )
            block_type = ObjectType.block_from_color(Color.from_str(color))
            structure.place_block(row, col, layer, block_type)
            structure.blocks_to_place.append((row, col, layer))
        return structure

    # ── Grid operations ───────────────────────────────────────────────

    def get_height(self, row: int, col: int) -> int:
        """Return the highest occupied layer + 1 at (row, col), or 0."""
        column = self.block_grid[row, col, :]
        nonzero = np.nonzero(column)[0]
        if len(nonzero) == 0:
            return 0
        return int(nonzero[-1]) + 1

    def has_layer_neighbor(self, row: int, col: int, layer: int) -> bool:
        """
        Check if a cell has an adjacent occupied neighbor in its layer.
        """
        grid = self.block_grid
        for dr, dc in [
            (1, 0), (-1, 0), (0, 1), (0, -1)
        ]:
            nr, nc = row + dr, col + dc
            if (0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]):
                if grid[nr, nc, layer] != 0:
                    return True
        return False


    def has_neighbor(self, row: int, col: int, layer: int) -> bool:
        """Check if a cell has an adjacent occupied neighbor or is on ground."""
        if layer == 0:
            return True
        grid = self.block_grid
        for dr, dc, dl in [
            (1, 0, 0), (-1, 0, 0), (0, 1, 0),
            (0, -1, 0), (0, 0, 1), (0, 0, -1),
        ]:
            nr, nc, nl = row + dr, col + dc, layer + dl
            if nl == 0:
                return True  # ground surface counts as support
            if (0 <= nr < grid.shape[0]
                    and 0 <= nc < grid.shape[1]
                    and 0 <= nl < grid.shape[2]):
                if grid[nr, nc, nl] != 0:
                    return True
        return False

    def place_block(
        self, row: int, col: int, layer: int, block_type: ObjectType,
    ) -> None:
        """Place a block at (row, col, layer).

        Validates bounds, emptiness, block type, and 6-connected
        adjacency before writing to the grid.
        """
        grid = self.block_grid
        if not (0 <= row < grid.shape[0]
                and 0 <= col < grid.shape[1]
                and 0 <= layer < grid.shape[2]):
            raise ValueError(
                f"Position ({row}, {col}, {layer}) "
                f"out of bounds {grid.shape}"
            )
        if grid[row, col, layer] != 0:
            raise ValueError(
                f"Cell ({row}, {col}, {layer}) is already occupied"
            )
        if not block_type.is_block():
            raise ValueError(f"{block_type} is not a block type")
        if not self.has_neighbor(row, col, layer):
            raise ValueError(
                f"No adjacent neighbor at ({row}, {col}, {layer})"
            )
        grid[row, col, layer] = block_type.value

    def place_top_block(
        self, row: int, col: int, block_type: ObjectType,
    ) -> int:
        """Place a block on top of the column at (row, col).

        Returns
        -------
        int
            The layer index where the block was placed.
        """
        layer = self.get_height(row, col)
        if layer >= self.block_grid.shape[2]:
            raise ValueError(f"Column ({row}, {col}) is full")
        self.place_block(row, col, layer, block_type)
        return layer

    def get_block(self, row: int, col: int, layer: int):
        """Return ObjectType at (row, col, layer), or None if empty."""
        val = self.block_grid[row, col, layer]
        if val == 0:
            return None
        return ObjectType(val)

    def remove_block(self, row: int, col: int, layer: int) -> ObjectType:
        """Remove and return the block at (row, col, layer)."""
        val = self.block_grid[row, col, layer]
        if val == 0:
            raise ValueError(
                f"Cell ({row}, {col}, {layer}) is empty"
            )
        self.block_grid[row, col, layer] = 0
        return ObjectType(val)

    def get_world_position(
        self, row: int, col: int, layer: int,
    ) -> np.ndarray:
        """Convert grid (row, col, layer) to world XYZ (cell center).

        Applies the structure's origin offset and yaw rotation.
        """
        size = BLOCK_SIZE
        # col -> local X, row -> local Y
        dx = col * size + size / 2
        dy = row * size + size / 2
        c, s = np.cos(self.roll), np.sin(self.roll)
        return np.array([
            self.origin[0] + c * dx - s * dy,
            self.origin[1] + s * dx + c * dy,
            self.origin[2] + (layer - 1) * size + (size / 2),
        ])

    def world_to_grid(
        self, x: float, y: float, z: float,
    ) -> tuple[int, int, int]:
        """Convert world XYZ to nearest grid (row, col, layer).

        Inverse of :meth:`get_world_position`.
        """
        size = BLOCK_SIZE
        c, s = np.cos(self.roll), np.sin(self.roll)
        # Undo rotation: transpose of [c, -s; s, c]
        rx, ry = x - self.origin[0], y - self.origin[1]
        local_x = c * rx + s * ry
        local_y = -s * rx + c * ry
        col = round((local_x - size / 2) / size)
        row = round((local_y - size / 2) / size)
        layer = round((z - self.origin[2] - size / 2) / size) + 1
        return row, col, layer

    def confirm_block_placed(self):
        """Mark the next queued block as physically placed.

        Pops the front of blocks_to_place and copies the
        corresponding value into placed_grid.
        """
        block_placed = self.blocks_to_place.popleft()
        self.visible_blocks.add(block_placed)
        # If the block is placed at layer > 0, it may occlude blocks below it, so we need to check if any visible blocks become invisible
        for layer in range(block_placed[-1]):
            below_block = (block_placed[0], block_placed[1], layer)
            if below_block in self.visible_blocks:
                self.visible_blocks.discard(below_block)
                self.invisible_blocks.add(below_block)
            elif below_block in self.invisible_blocks:
                break # No need to check further down since they would already be occluded

        self.placed_grid[block_placed] = self.block_grid[block_placed]

    # ── Origin estimation & neighbor queries ─────────────────────────
    
    def get_grid_origin(self, placement_grid_coords: list[tuple[float, float]]) -> np.ndarray:
        return np.array([
            placement_grid_coords[0][0],
            placement_grid_coords[0][1],
            BLOCK_SIZE
        ])

    def get_grid_roll(self, placement_grid_coords: list[tuple[float, float]]) -> float:
        db = np.array(placement_grid_coords[1]) - np.array(placement_grid_coords[0]) # br - bl
        return np.arctan2(db[1], db[0])

    def update_grid_coords(self, corners: np.ndarray) -> None:
        """Update grid origin and roll from detected corner positions.

        Arguments
        ---------
        corners : np.ndarray
            Shape (4, 2) array of [bl, br, tl, tr] world XY coordinates.
        """
        self.origin = np.array([corners[0][0], corners[0][1], BLOCK_SIZE])
        dx = corners[1][0] - corners[0][0]
        dy = corners[1][1] - corners[0][1]
        self.roll = np.arctan2(dy, dx)
        self.logger.info(f"BlockStructure.update_grid_coords: roll={self.roll}")

    def get_simulated_detection_grid(self) -> np.ndarray:
        """
        Return a 3D grid of the same shape as block_grid where each cell contains the
        ObjectType value of the block in that cell, or 0 if empty. The grid is simulated
        as if it was observed by the end effector camera, so it does not contain any blocks 
        that are not visible from above (e.g. they have another block above them).

        Returns
        -------
        np.ndarray
            3D int array with ObjectType values for visible blocks and 0 for empty cells.
        """
        simulated_grid = np.zeros_like(self.block_grid, dtype=int)
        for row in range(self.block_grid.shape[0]):
            for col in range(self.block_grid.shape[1]):
                for layer in range(self.block_grid.shape[2]):
                    if self.placed_grid[row, col, layer] != 0:
                        # Check if the block is visible (no blocks above it)
                        if layer + 1 >= self.block_grid.shape[2] or self.placed_grid[row, col, layer + 1] == 0:
                            simulated_grid[row, col, layer] = self.placed_grid[row, col, layer]
        return simulated_grid

    def get_same_layer_neighbors(
        self, row: int, col: int, layer: int, radius: int = 1,
    ) -> list[tuple[int, int, int]]:
        """Return all valid grid cells within row±radius, col±radius.

        Includes the center cell itself.  Used by origin estimation
        and verification checking.

        Arguments
        ---------
        row : int
            Center row index.
        col : int
            Center column index.
        layer : int
            Layer index.
        radius : int
            Search radius in grid cells.

        Returns
        -------
        list[tuple[int, int, int]]
            Valid (row, col, layer) tuples within the neighborhood.
        """
        cells = []
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                nr, nc = row + dr, col + dc
                if (0 <= nr < self.block_grid.shape[0]
                        and 0 <= nc < self.block_grid.shape[1]
                        and 0 <= layer < self.block_grid.shape[2]):
                    cells.append((nr, nc, layer))
        return cells

    def estimate_origin_from_detections(
        self,
        detected_objects: list,
        target_cell: tuple[int, int, int],
    ) -> tuple:
        """Re-estimate structure origin from detected blocks near target.

        For each cell in the 3x3 neighborhood of target_cell that has
        an expected block (in block_grid) AND is either the target or
        already placed (in placed_grid), find the closest detected
        object matching the expected ObjectType and approximate z.
        Back-calculate the origin each match implies and take the
        median.

        Arguments
        ---------
        detected_objects : list[Object]
            Detected objects from worldmap contour detection.
        target_cell : tuple[int, int, int]
            (row, col, layer) of the block being verified.

        Returns
        -------
        tuple
            (median_origin_xy as np.ndarray or None,
             confidence as float).
            confidence = matched_count / expected_count.
            Returns (None, 0.0) if no blocks could be matched.
        """
        from legobuilder.schemas import ObjectType as OT

        row, col, layer = target_cell
        neighbors = self.get_same_layer_neighbors(row, col, layer)

        s = BLOCK_SIZE
        c_r, s_r = np.cos(self.roll), np.sin(self.roll)

        expected_cells = []
        for (nr, nc, nl) in neighbors:
            val = self.block_grid[nr, nc, nl]
            if val == 0:
                continue
            # Must be either the target cell or already placed
            is_target = (nr == row and nc == col and nl == layer)
            is_placed = self.placed_grid[nr, nc, nl] != 0
            if is_target or is_placed:
                expected_cells.append((nr, nc, nl, OT(val)))

        if not expected_cells:
            return (None, 0.0)

        origin_estimates = []
        expected_count = len(expected_cells)
        used_detection_indices = set()

        for (nr, nc, nl, expected_type) in expected_cells:
            expected_z = self.origin[2] + (nl - 1) * s + (s / 2 - 0.01)

            # Find closest detected object matching type and z
            best_obj = None
            best_dist = float('inf')
            best_idx = -1
            expected_world = self.get_world_position(nr, nc, nl)

            for idx, obj in enumerate(detected_objects):
                if idx in used_detection_indices:
                    continue
                if obj.obj_type != expected_type:
                    continue
                obj_z = obj.center_xyz[2]
                if abs(obj_z - expected_z) > s * 0.75:
                    continue
                dx = obj.center_xyz[0] - expected_world[0]
                dy = obj.center_xyz[1] - expected_world[1]
                dist = np.sqrt(dx**2 + dy**2)
                if dist < best_dist:
                    best_dist = dist
                    best_obj = obj
                    best_idx = idx

            if best_obj is None or best_dist > s * 3:
                continue

            used_detection_indices.add(best_idx)

            # Back-calculate origin from this detection
            dx_local = nc * s + s / 2
            dy_local = nr * s + s / 2
            est_ox = best_obj.center_xyz[0] - (
                c_r * dx_local - s_r * dy_local
            )
            est_oy = best_obj.center_xyz[1] - (
                s_r * dx_local + c_r * dy_local
            )
            origin_estimates.append(np.array([est_ox, est_oy]))

        matched = len(origin_estimates)
        if matched == 0:
            return (None, 0.0)

        median_origin = np.median(
            np.array(origin_estimates), axis=0,
        )
        confidence = matched / expected_count

        from legobuilder.config import VERIFY_MIN_ORIGIN_VOTES
        if matched >= VERIFY_MIN_ORIGIN_VOTES:
            drift = np.linalg.norm(median_origin - self.origin[:2])
            if drift > 0.001:  # Only update if >1mm drift
                self.logger.info(
                    f"Origin re-estimated: "
                    f"({median_origin[0]:.4f}, {median_origin[1]:.4f})"
                    f" drift={drift:.4f}m "
                    f"({matched}/{expected_count} votes)"
                )
                self.origin[:2] = median_origin

        return (median_origin, confidence)

    # ── Approach and placement geometry ───────────────────────────────

    def compute_approach_vector(
        self, row: int, col: int, layer: int,
        magnitude: float = APPROACH_VECTOR_MAGNITUDE,
    ) -> np.ndarray:
        """Compute approach/retreat offset for placing at (row, col, layer).

        Finds the direction pointing away from all physically placed
        neighbors so the arm approaches from the side with the lowest
        collision risk.  The horizontal component is rotated by the
        structure's yaw angle.

        Direction examples (before scaling):
        - Stacking (neighbor below only)     -> [0, 0, 1] (straight up)
        - Side attach (neighbor to one side) -> [-1, 0, 0] (open side)
        - Stairway (below + one side)        -> [-0.7, 0, 0.7] (diagonal)

        Arguments
        ---------
        row : int
            Grid row index.
        col : int
            Grid column index.
        layer : int
            Grid layer index.
        magnitude : float
            Length of the approach vector (meters).

        Returns
        -------
        np.ndarray
            3D approach offset in world coordinates.
        """
        grid = self.placed_grid

        # Grid offset (dr, dc, dl) -> world direction
        # row -> y,  col -> x,  layer -> z
        offsets = [
            ((1, 0, 0),  np.array([0.0,  1.0, 0.0])),   # row+  -> +y
            ((-1, 0, 0), np.array([0.0, -1.0, 0.0])),   # row-  -> -y
            ((0, 1, 0),  np.array([1.0,  0.0, 0.0])),   # col+  -> +x
            ((0, -1, 0), np.array([-1.0, 0.0, 0.0])),   # col-  -> -x
            ((0, 0, 1),  np.array([0.0,  0.0, 1.0])),   # layer+ -> +z
            ((0, 0, -1), np.array([0.0,  0.0, -1.0])),  # layer- -> -z
        ]

        toward = np.zeros(3)

        for (dr, dc, dl), world_dir in offsets:
            nr, nc, nl = row + dr, col + dc, layer + dl
            if nl < 0:
                toward += world_dir
                continue
            if (0 <= nr < grid.shape[0]
                    and 0 <= nc < grid.shape[1]
                    and 0 <= nl < grid.shape[2]):
                if grid[nr, nc, nl] != 0:
                    toward += world_dir

        away = -toward

        # max(away[2], 0.0) -> Never approach from below
        # max(away[2], 0.5) -> Approach with at least ~28deg downward
        # max(away[2], 1.0) -> Approach with at least ~45deg downward
        # param in max = tan(angle)
        away[2] = max(away[2], 1.0)

        if np.linalg.norm(away) < 1e-6:
            away = np.array([0.0, 0.0, 1.0])

        # Rotate horizontal component by structure yaw
        c, s = np.cos(self.roll), np.sin(self.roll)
        ax, ay = away[0], away[1]
        away[0] = c * ax - s * ay
        away[1] = s * ax + c * ay

        approach_vec = (away / np.linalg.norm(away)) * magnitude

        self.logger.debug(
            f"compute_approach_vector({row},{col},{layer}): "
            f"toward={toward}, vec={approach_vec}"
        )

        return approach_vec

    def get_horizontal_mating_faces(
        self, row, col, layer, use_placed_grid=False
    ) -> set:
        """Return horizontal directions with an occupied neighbor.

        Checks the four horizontal neighbors at the same layer and
        returns a set of direction strings for those that are occupied.

        Returns
        -------
        set of str
            Subset of 'row+', 'row-', 'col+', 'col-' indicating occupied neighbors.
        """
        if use_placed_grid:
            grid = self.placed_grid
        else:
            grid = self.block_grid
        faces = set()
        for dr, dc, label in [
            (1, 0, 'row+'), (-1, 0, 'row-'),
            (0, 1, 'col+'), (0, -1, 'col-'),
        ]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                if grid[nr, nc, layer] != 0:
                    faces.add(label)
        return faces

    def is_block_floating(self, row: int, col: int, layer: int) -> bool:
        """Return True if the block has no support directly below it."""
        if layer <= 1:
            return False
        return self.placed_grid[row, col, layer - 1] == 0

    def _compute_floating_push_direction(
        self, row: int, col: int, layer: int,
    ) -> Optional[np.ndarray]:
        """Compute horizontal push direction toward same-layer neighbors.

        Sums unit vectors toward each occupied horizontal neighbor,
        normalizes, and rotates by the structure's yaw.  Returns None
        if there are no horizontal mating faces or opposing faces cancel.
        """
        mating = self.get_horizontal_mating_faces(
            row, col, layer, use_placed_grid=True,
        )
        if not mating:
            return None

        # col -> x, row -> y (matching get_world_position convention)
        face_to_dir = {
            'col+': np.array([1.0, 0.0]),
            'col-': np.array([-1.0, 0.0]),
            'row+': np.array([0.0, 1.0]),
            'row-': np.array([0.0, -1.0]),
        }

        push_local = np.zeros(2)
        for face in mating:
            push_local += face_to_dir[face]

        norm = np.linalg.norm(push_local)
        if norm < 1e-6:
            return None  # opposing faces cancel out
        push_local /= norm

        # Rotate by structure yaw
        c, s = np.cos(self.roll), np.sin(self.roll)
        push_world = np.array([
            c * push_local[0] - s * push_local[1],
            s * push_local[0] + c * push_local[1],
            0.0,
        ])
        return push_world

    def compute_placement_roll(
        self, mating_faces: set, grip_roll: float,
    ) -> float:
        """Choose placement roll that avoids obstructing mating faces.

        Gripper fingers span one axis of the block.  When grip_roll
        aligns with self.roll (mod pi), fingers run along the col
        axis and obstruct col+/col- faces.  Rotated 90 degrees they
        run along the row axis and obstruct row+/row- faces.

        The grip_roll is first snapped to the nearest grid-aligned
        angle (self.roll + n * pi/2) so placement always aligns
        with the structure grid.

        Arguments
        ---------
        mating_faces : set
            Horizontal directions with occupied neighbors.
        grip_roll : float
            Current gripper roll (radians).

        Returns
        -------
        float
            Chosen placement roll (radians).
        """
        delta_raw = grip_roll - self.roll
        n = round(delta_raw / (pi / 2))
        grip_roll = self.roll + n * (pi / 2)

        result = grip_roll  # default: keep snapped roll

        if not mating_faces:
            self.logger.debug(f"not mating_faces roll = {grip_roll}")
        else:
            delta = (grip_roll - self.roll) % pi
            fingers_along_col = delta < pi / 4 or delta > 3 * pi / 4

            if fingers_along_col:
                obstructed = mating_faces & {'col+', 'col-'}
                alt_obstructed = mating_faces & {'row+', 'row-'}
            else:
                obstructed = mating_faces & {'row+', 'row-'}
                alt_obstructed = mating_faces & {'col+', 'col-'}

            if obstructed and not alt_obstructed:
                self.logger.debug(
                    f"compute_placement_roll: rotating 90 deg to avoid "
                    f"obstructing {obstructed}, "
                    f"grip_roll: {grip_roll + pi / 2}"
                )
                result = grip_roll + pi / 2
            elif obstructed and len(alt_obstructed) < len(obstructed):
                self.logger.warning(
                    f"compute_placement_roll: both axes have mating "
                    f"faces ({obstructed} vs {alt_obstructed}), "
                    f"choosing fewer conflicts "
                    f"returning {grip_roll + pi / 2}"
                )
                result = grip_roll + pi / 2
            elif obstructed:
                self.logger.warning(
                    f"compute_placement_roll: both axes have mating "
                    f"faces ({obstructed} vs {alt_obstructed}), "
                    f"keeping current roll {grip_roll}"
                )

        return wrap_roll(result)

    def compute_pickup_roll(
        self,
        obj: Object,
        all_detections: list[Object],
        alternate_grip_convention: bool = False
    ) -> float:
        """Compute pickup roll that avoids the most obstructed side.

        For each of the 4 faces of *obj*, computes the minimum distance
        to any face of any other detected block (including invisible
        blocks and nearby base-grid positions).  The face with the
        highest obstruction (smallest min distance) is identified, and
        the roll is chosen so the gripper jaws avoid that side.

        Arguments
        ---------
        obj : Object
            The block to be picked up.
        all_detections : list[Object]
            Other detected blocks in the scene.
        alternate_grip_convention : bool
            If true check worst_face_idx in (2,3) else (0,1)

        Returns
        -------
        float
            Chosen pickup roll (radians).
        """
        base_roll = -obj.angle if obj.angle is not None else 0.0

        # self.logger.info(f"compute_pickup_roll center_xyzs")
        # for d in all_detections:
        #     self.
        
        if obj.corner_xys is None or len(obj.corner_xys) != 4:
            return base_roll

        # Augment detections with invisible (occluded) placed blocks
        augmented = list(all_detections)
        for r, c, l in self.invisible_blocks:
            pos = self.get_world_position(r, c, l)
            augmented.append(_make_virtual_object(
                self.placed_grid[r, c, l], (pos[0], pos[1], pos[2]),
                angle=self.roll,
            ))

        # If block is near the table, include closest base-grid positions
        if obj.center_xyz[2] < BLOCK_SIZE:
            ox, oy = obj.center_xyz[0], obj.center_xyz[1]
            base_positions = []
            for row in range(MAX_GRID_SIZE['length']):
                for col in range(MAX_GRID_SIZE['width']):
                    pos = self.get_world_position(row, col, 0)
                    dist = np.linalg.norm(pos[:2] - np.array([ox, oy]))
                    base_positions.append((dist, pos))
            base_positions.sort(key=lambda t: t[0])
            for _, pos in base_positions[:3]:
                augmented.append(_make_virtual_object(
                    1, (pos[0], pos[1], pos[2]),
                    angle=self.roll,
                ))

        # Compute face centers of the target object
        obj_corners_3d = [
            np.array([cx, cy, obj.center_xyz[2]])
            for (cx, cy) in obj.corner_xys
        ]
        face_indices = [(0, 1), (2, 3), (0, 3), (1, 2)]
        obj_face_centers = [
            (obj_corners_3d[i] + obj_corners_3d[j]) / 2
            for (i, j) in face_indices
        ]

        # Collect face centers from all other blocks
        other_face_centers = []
        for det in augmented:
            if det is obj:
                continue
            if det.corner_xys is None or len(det.corner_xys) != 4:
                continue
            det_corners_3d = [
                np.array([cx, cy, det.center_xyz[2]])
                for (cx, cy) in det.corner_xys
            ]
            for (i, j) in face_indices:
                other_face_centers.append(
                    (det_corners_3d[i] + det_corners_3d[j]) / 2
                )

        if not other_face_centers:
            return base_roll

        debug_compute_pickup_roll(obj_face_centers, other_face_centers, obj, augmented)

        # For each face, find min distance to any other block's face
        face_min_dists = []
        for fc in obj_face_centers:
            min_dist = min(
                np.linalg.norm(fc - ofc) for ofc in other_face_centers
            )
            self.logger.info(f"fc={fc}, min_dist={min_dist}")
            face_min_dists.append(min_dist)

        worst_face_idx = int(np.argmin(face_min_dists))
        self.logger.info(f"worst_face_idx={worst_face_idx}")
        
        # Indices 0,1 correspond to the faces the gripper jaws grip
        # at base_roll.  When they are obstructed, rotate by pi/2.
        check_idxs = (0, 1) if not alternate_grip_convention else (2,3)
        # if worst_face_idx in (0, 1):
        # if worst_face_idx in (2, 3):
        if worst_face_idx in check_idxs:
            return -wrap_roll(base_roll + pi / 2)
        return base_roll

    # ── 3-Phase trajectory planning ───────────────────────────────────

    def plan_grasp(
        self,
        detected_objects: list[Object],
        target_cell: tuple | None = None,
        block_type: ObjectType | None = None,
    ):
        """Plan grasp trajectory: approach, descend, close, lift.

        Selects the first detected object whose type matches the
        target block, then generates a 4-state pick trajectory.

        Arguments
        ---------
        detected_objects : list[Object]
            Currently visible pile objects.
        target_cell : tuple or None
            (row, col, layer) to place into. If None, reads from
            blocks_to_place[0].
        block_type : ObjectType or None
            Type of block to pick. If None, reads from block_grid
            at target_cell.

        Returns
        -------
        tuple
            (obj, states, context) on success, or
            (None, None, None) if no suitable block is found.
            *context* is a dict with keys 'pickup_roll',
            'target_cell', 'p_pick'.
        """
        # TODO: If block is connected on one side: approach with appropriate roll to avoid
        # collision with other block, then before lifting up, rotate so that block is radially outward
        # then tilt to separate and pick up

        if target_cell is not None:
            row, col, layer = target_cell
        elif len(self.blocks_to_place) > 0:
            row, col, layer = self.blocks_to_place[0]
        else:
            return None, None, None

        if block_type is not None:
            target_block = block_type
        else:
            target_block = self.get_block(row, col, layer)
        blocks_attached = 0
        clearing_obj = None
        # Sort: fewer connections first, then closer to arm base
        sorted_objects = sorted(
            detected_objects,
            key=lambda obj: (
                len(obj.face_conns_xyzs),
                np.linalg.norm(obj.center_xyz[:2] - BASE_MOTOR_POS[:2]),
            ),
        )   
        for detected_obj in sorted_objects:
            if detected_obj.obj_type == target_block:
                num_connected_faces = len(detected_obj.face_conns_xyzs)
                x, y = detected_obj.center_xyz[0], detected_obj.center_xyz[1]
                center_z = detected_obj.center_xyz[2]
                p_pick = np.array([x, y, center_z + PICK_Z])

                if not is_point_reachable(p_pick):
                    continue

                if num_connected_faces >= 2:
                    # Try to find a separable neighbor to clear first
                    clearing_obj = self.separator.find_clearing_target(
                        detected_obj, sorted_objects
                    )
                    if clearing_obj is None:
                        continue
                    # Plan grasp for the clearing target instead
                    obj = clearing_obj
                    x, y = obj.center_xyz[0], obj.center_xyz[1]
                    center_z = obj.center_xyz[2]
                    p_pick = np.array([x, y, center_z + BLOCK_TOP_OFFSET])
                    if not is_point_reachable(p_pick):
                        clearing_obj = None
                        continue
                    num_connected_faces = len(obj.face_conns_xyzs)
                    blocks_attached = min(num_connected_faces, 1)
                    self.logger.info(
                        f"Clearing neighbor at ({x:.3f}, {y:.3f}) "
                        f"to free target block {target_block}"
                    )
                    break

                if num_connected_faces == 1:
                    blocks_attached = 1

                obj = detected_obj
                break
        else:
            self.logger.info(f"No available reachable blocks of type {target_block}")
            return None, None, None

        x, y = obj.center_xyz[0], obj.center_xyz[1]
        center_z = obj.center_xyz[2]
        if clearing_obj is not None:
            pick_z = center_z + BLOCK_TOP_OFFSET + BLOCK_SIZE/2
        else:
            pick_z = center_z + PICK_Z
        p_pick = np.array([x, y, pick_z])

        # pickup_roll = -obj.angle if obj.angle is not None else 0.0
        # if p_pick[0] <= PILE_X_MIN:
        #     pickup_roll = self.compute_pickup_roll(obj, self.detected_grid_objects)
        # else:
        #     pickup_roll = -obj.angle if obj.angle is not None else 0.0

        if p_pick[0] > PILE_X_MIN:
            pickup_roll = -obj.angle if obj.angle is not None else 0.0
            connected_directions = self.get_connected_directions(obj)
            if blocks_attached == 1:
                self.logger.debug(
                    f"Block at ({row}, {col}, {layer}) has one attached block, "
                    f"planning separation trajectory"
                )
                pickup_roll = self.separator.adjust_pickup_roll(
                    pickup_roll, connected_directions
                )
        else:
            pickup_roll = self.compute_pickup_roll(obj, self.detected_grid_objects, alternate_grip_convention=True)
            self.logger.info(f"plan_grasp pickup_roll: {pickup_roll}, len(det_grid_objs)={len(self.detected_grid_objects)}")


        # if blocks_attached == 1:
        #     self.logger.debug(
        #         f"Block at ({row}, {col}, {layer}) has one attached block, "
        #         f"planning separation trajectory"
        #     )
        #     pickup_roll = self.separator.adjust_pickup_roll(
        #         pickup_roll, connected_directions
        #     )

        pick_o = np.array([PICK_TILT, pickup_roll])

        self.logger.debug(
            f"plan_grasp: {target_block} at ({x:.3f}, {y:.3f}) "
            f"with roll {np.degrees(pick_o[1])}, target cell "
            f"(row={row}, col={col}, layer={layer})"
        )

        states = [
            # 1. Approach above block
            TrajectoryState(
                final_state=ManipulatorState(
                    p=p_pick + np.array([0, 0, self.APPROACH_HEIGHT]),
                    o=pick_o,
                    gripper_open=True,
                ),
                min_duration=self.PICK_DURATION,
            ),
            # 2. Descend to block
            TrajectoryState(
                final_state=ManipulatorState(
                    p=p_pick,
                    o=pick_o,
                    gripper_open=True,
                ),
                min_duration=self.PICK_DURATION / 2,
            ),
            # 3. Close gripper
            TrajectoryState(
                final_state=ManipulatorState(
                    p=p_pick,
                    o=pick_o,
                    gripper_open=False,
                ),
                min_duration=self.GRIPPER_DELAY,
                delay_before=self.GRIPPER_DELAY,
            )
        ]

        # Separation trajectory (if attached)
        if blocks_attached == 1:
            if p_pick[0] > PILE_X_MIN:
                sep_states = self.separator.plan_single_face(
                    p_pick, connected_directions
                )
            else:     # NOTE: ignoring as this should be solvable after failing connection check
                sep_states = []
            #     # Block is on the grid — relocate above pile before separating
            #     sep_states = self.separator.plan_relocate_and_separate(
            #         p_pick, connected_directions, center_z,
            #     )
            #     new_p_pick = sep_states[-1].final_state.p
            #     new_p_pick[2] = p_pick[2]
            #     p_pick = new_p_pick
                
            states.extend(sep_states)
            
        # 4. Lift (roll aligned for grip check)
        # Lift higher when picking from base grid area to avoid
        # collision with structure blocks during wrist tilt.
        grip_check_height = self.APPROACH_HEIGHT
        if p_pick[0] <= PILE_X_MIN:
            grip_check_height = self.APPROACH_HEIGHT * 2
        
        states.extend([
            TrajectoryState(
                final_state=ManipulatorState(
                    p=p_pick + np.array([0, 0, grip_check_height]),
                    o=np.array([
                        PICK_TILT + np.pi/4,
                        get_grip_check_roll(p_pick),
                    ]),
                    gripper_open=False,
                ),
                min_duration=self.PICK_DURATION,
            ),
        ])

        context = {
            'pickup_roll': pickup_roll if obj.angle is not None else 0.0,
            'target_cell': (row, col, layer),
            'p_pick': p_pick,
            'clearing': clearing_obj is not None,
            'detected_objects': detected_objects if clearing_obj is not None else None,
        }

        return obj, states, context

    def get_connected_directions_grid(self, obj: Object) -> Direction | None:
        connected_directions = set()

        for (x, y, z) in obj.face_conns_xyzs:
            # Get closest angles to side 
            side_xy = np.array([x, y])

            dx = side_xy[0] - obj.center_xyz[0]
            dy = side_xy[1] - obj.center_xyz[1]
            direction = None
            if abs(dy) > abs(dx):
                direction = Direction.ABOVE if dy > 0 else Direction.BELOW
            else:
                direction = Direction.RIGHT if dx > 0 else Direction.LEFT
            
            if direction:
                # self.logger.info(f'Determined {direction} is connected to the block')
                connected_directions.add(direction)

        return connected_directions
    
    def get_connected_directions(self, obj: Object) -> Direction | None:
        connected_directions = set()

        for (x, y, z) in obj.face_conns_xyzs:
            # Get closest angles to side 
            side_xy = np.array([x, y])

            direction = None
            # TODO: uncomment when needed
            # if abs(side_xy[0] - obj.center_xyz[0]) < 1e-3 and abs(side_xy[1] - obj.center_xyz[1]) < 1e-3:
            #     if z > obj.center_xyz[2]:
            #         direction = Direction.FRONT
            #     else:
            #         direction = Direction.BACK
            # self.logger.info(f"side_xy: {side_xy}, center_xy: {obj.center_xyz[:2]}")
            if side_xy[0] <= obj.center_xyz[0] and side_xy[1] >= obj.center_xyz[1]:
                direction = Direction.ABOVE
            elif side_xy[0] >= obj.center_xyz[0] and side_xy[1] >= obj.center_xyz[1]:
                direction = Direction.RIGHT
            elif side_xy[0] <= obj.center_xyz[0] and side_xy[1] <= obj.center_xyz[1]:
                direction = Direction.LEFT
            elif side_xy[0] >= obj.center_xyz[0] and side_xy[1] <= obj.center_xyz[1]:
                direction = Direction.BELOW
            
            if direction:
                # self.logger.info(f'Determined {direction} is connected to the block')
                connected_directions.add(direction)

        return connected_directions
        
    def plan_approach(
        self, target_cell, pickup_roll,
    ) -> list[TrajectoryState]:
        """Plan approach trajectory toward the structure.

        Moves from the lift position to an approach waypoint offset
        from the target cell.  The approach direction is computed to
        avoid occupied neighbors.

        Arguments
        ---------
        target_cell : tuple
            (row, col, layer) grid coordinates.
        pickup_roll : float
            Current gripper roll (radians).

        Returns
        -------
        list[TrajectoryState]
            Two waypoints: raised approach and final approach.
        """
        row, col, layer = target_cell
        p_place = self.get_world_position(row, col, layer)
        approach_o = np.array([PICK_TILT, pickup_roll])
        approach_vec = self.compute_approach_vector(row, col, layer)

        p = p_place + approach_vec
        if not is_point_reachable(p):
            self.logger.info(
                f'p_place + approach_vec: {p}, '
                f'reachable: {is_point_reachable(p)} '
                f'-> will place from height'
            )
            p = p_place + np.array([0, 0, self.APPROACH_HEIGHT])

        return [
            # 1. Raised approach
            TrajectoryState(
                final_state=ManipulatorState(
                    p=p + np.array([0, 0, self.APPROACH_HEIGHT]),
                    o=approach_o,
                    gripper_open=False,
                ),
                min_duration=self.PLACE_DURATION,
            ),
            # 2. Final approach
            TrajectoryState(
                final_state=ManipulatorState(
                    p=p,
                    o=approach_o,
                    gripper_open=False,
                ),
                min_duration=self.PLACE_DURATION/2,
            ),
        ]

    def plan_placement(
        self, row, col, layer, placement_roll,
        return_to_ready: bool = True,
    ) -> list[TrajectoryState]:
        """Plan placement trajectory: descend, release, retreat, ready.

        Retreat direction matches the approach vector (away from
        neighbors).

        Arguments
        ---------
        row : int
            Target grid row.
        col : int
            Target grid column.
        layer : int
            Target grid layer.
        placement_roll : float
            Gripper roll for placement (radians).
        return_to_ready : bool
            If True, append a final state returning to Q_SCAN.
            Set False when verification follows immediately.

        Returns
        -------
        list[TrajectoryState]
            Three or four waypoints for the place phase.
        """
        p_place = self.get_world_position(row, col, layer)
        # p_place = self.get_world_position(row, col, layer) + np.array([0, 0, PICK_Z])

        if len(self.get_horizontal_mating_faces(row, col, layer, use_placed_grid=True)) > 1:
            self.logger.info(
                f"Block at ({row}, {col}, {layer}) has multiple mating faces, "
                f"adding extra height to placement to reduce risk of collision"
            )
            p_place[2] += BLOCK_SIZE / 2  # Drop from half a block height

        place_o = np.array([PICK_TILT, placement_roll])
        approach_vec = self.compute_approach_vector(row, col, layer)
        p_retreat = p_place + approach_vec
        if not is_point_reachable(p_retreat):
            p_retreat = p_place + np.array([0, 0, self.APPROACH_HEIGHT])

        # Small upward nudge to help magnetic release
        offset_z = np.array([0, 0, 0.015])

        descend_state = TrajectoryState(
            final_state=ManipulatorState(
                p=p_place.copy(),
                o=place_o,
                gripper_open=False,
            ),
            min_duration=self.PLACE_DURATION / 2,
        )

        if PLACEMENT_PUSH and \
          (layer == 1 and (approach_vec[0] != 0 or approach_vec[1] != 0)): # Ensure not approaching from above
            connect_states = [
                # 1. Push forward a little bit to ensure good connection
                TrajectoryState(
                    final_state=ManipulatorState(
                        p=p_place.copy() - approach_vec * 0.5,
                        o=place_o,
                        gripper_open=False,
                    ),
                    min_duration=self.GRIPPER_DELAY,
                    delay_before=self.GRIPPER_DELAY,
                ),
                # 2. Retreat back to original place position
                deepcopy(descend_state)
            ]
        else:
            connect_states = []

        # Connecting block: push horizontally into mating faces for magnetic contact
        if len(self.get_horizontal_mating_faces(row, col, layer, use_placed_grid=True)) >= 1:
            push_dir = self._compute_floating_push_direction(row, col, layer)
            if push_dir is not None:
                self.logger.info(
                    f"Floating block at ({row},{col},{layer}): "
                    f"pushing {FLOATING_PUSH_DISTANCE}m toward mating faces"
                )
                descend_state.final_state.p += push_dir * FLOATING_PUSH_DISTANCE

        retreat_states = [
            TrajectoryState(
                final_state=ManipulatorState(
                    p=p_place.copy() + offset_z,
                    o=place_o,
                    gripper_open=True,
                ),
                min_duration=self.GRIPPER_DELAY,
                delay_before=self.GRIPPER_DELAY,
            ),
            # 3. Retreat along approach vector
            TrajectoryState(
                final_state=ManipulatorState(
                    p=p_retreat,
                    o=place_o,
                    gripper_open=True,
                ),
                min_duration=self.PLACE_DURATION / 2,
            ),
        ]

        states = [descend_state] + connect_states + retreat_states
        if return_to_ready:
            states.append(
                # 4. Return to ready
                TrajectoryState(
                    final_state=ManipulatorState(
                        q=Q_SCAN.copy(),
                        qd=np.zeros(len(JOINT_NAMES)),
                    ),
                    min_duration=self.PICK_DURATION,
                ),
            )
        return states

    def plan_regrip(
        self, p_pick, new_roll,
    ) -> list[TrajectoryState]:
        """Plan recovery trajectory for a diagonal or bad grip.

        Robot is at approach height above the pickup with gripper
        closed.  Puts the block back, re-grips with *new_roll*,
        and lifts again.

        Arguments
        ---------
        p_pick : np.ndarray
            3D pickup position [x, y, z].
        new_roll : float
            Corrected gripper roll (radians).

        Returns
        -------
        list[TrajectoryState]
            Six waypoints: descend, open, retreat, descend, close, lift.
        """
        old_o = np.array([PICK_TILT, new_roll - pi / 4])
        new_o = np.array([PICK_TILT, new_roll])

        return [
            # 1. Descend to block (gripper still closed)
            TrajectoryState(
                final_state=ManipulatorState(
                    p=p_pick.copy(),
                    o=old_o,
                    gripper_open=False,
                ),
                min_duration=self.PICK_DURATION / 2,
            ),
            # 2. Open gripper
            TrajectoryState(
                final_state=ManipulatorState(
                    p=p_pick.copy(),
                    o=old_o,
                    gripper_open=True,
                ),
                min_duration=self.GRIPPER_DELAY,
                delay_before=self.GRIPPER_DELAY,
            ),
            # 3. Retreat upward
            TrajectoryState(
                final_state=ManipulatorState(
                    p=p_pick + np.array([0, 0, self.APPROACH_HEIGHT]),
                    o=new_o,
                    gripper_open=True,
                ),
                min_duration=self.PICK_DURATION / 2,
            ),
            # 4. Descend with new roll
            TrajectoryState(
                final_state=ManipulatorState(
                    p=p_pick.copy(),
                    o=new_o,
                    gripper_open=True,
                ),
                min_duration=self.PICK_DURATION / 2,
            ),
            # 5. Close gripper
            TrajectoryState(
                final_state=ManipulatorState(
                    p=p_pick.copy(),
                    o=new_o,
                    gripper_open=False,
                ),
                min_duration=self.GRIPPER_DELAY,
                delay_before=self.GRIPPER_DELAY,
            ),
            # 6. Lift (roll aligned for grip check)
            TrajectoryState(
                final_state=ManipulatorState(
                    p=p_pick + np.array([0, 0, self.APPROACH_HEIGHT]),
                    o=np.array([
                        PICK_TILT,
                        get_grip_check_roll(p_pick),
                    ]),
                    gripper_open=False,
                ),
                min_duration=self.PICK_DURATION / 2,
            ),
        ]

    # ── Visualization ─────────────────────────────────────────────────

    BLOCK_COLORS = {
        1: (1.0, 1.0, 0.0),   # YELLOW_BLOCK
        2: (0.0, 0.0, 1.0),   # BLUE_BLOCK
        3: (0.0, 1.0, 0.0),   # GREEN_BLOCK
        4: (1.0, 0.0, 0.0),   # RED_BLOCK
    }

    def visualize(self):
        """Open an interactive 3D popup showing the block grid."""
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

        grid = self.block_grid
        filled = grid != 0

        colors = np.zeros(grid.shape + (4,))
        for val, rgb in self.BLOCK_COLORS.items():
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
        ax.set_title(repr(self))

        plt.tight_layout()
        plt.show(block=True)

    # ── Dunder methods ────────────────────────────────────────────────

    def __repr__(self) -> str:
        """Return a compact summary: dimensions and block count."""
        l, w, h = self.block_grid.shape
        n = int(np.count_nonzero(self.block_grid))
        return f"BlockStructure({l}x{w}x{h}, {n} blocks)"

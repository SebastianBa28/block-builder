"""Pydantic data models and enums for detected objects and contours.

Defines the shared vocabulary of colors, object types, contour shapes, and
detected objects used across the brain, kinematics, and vision subpackages.
"""

from pydantic import BaseModel
from enum import Enum
from typing import Optional
import numpy as np
from dataclasses import dataclass


# ── Color and Object Type Enums ─────────────────────────────────────

RED_RGB = (255, 0, 0)
GREEN_RGB = (0, 255, 0)
BLUE_RGB = (0, 0, 255)
YELLOW_RGB = (255, 255, 0)
WHITE_RGB = (255, 255, 255)
ORANGE_RGB = (255, 165, 0)

class Color(Enum):
    """Named colors used for block detection and HSV filtering.

    Attributes
    ----------
    ORANGE : int
        Orange (1).
    YELLOW : int
        Yellow (2).
    BLUE : int
        Blue (3).
    GREEN : int
        Green (4).
    RED : int
        Red (5).
    """
    ORANGE = 1
    YELLOW = 2
    BLUE = 3
    GREEN = 4
    RED = 5

    @classmethod
    def to_rgb(cls, color):
        """Return the RGB tuple for this color."""
        if color == cls.RED:
            return RED_RGB
        elif color == cls.GREEN:
            return GREEN_RGB
        elif color == cls.BLUE:
            return BLUE_RGB
        elif color == cls.YELLOW:
            return YELLOW_RGB
        elif color == cls.ORANGE:
            return ORANGE_RGB
        else:
            return None

    @classmethod
    def from_str(cls, color_str: str):
        """Return the Color member matching a case-insensitive string.

        Arguments
        ---------
        color_str : str
            Color name (e.g. 'yellow', 'BLUE').

        Returns
        -------
        Color
            The matching enum member.

        Raises
        ------
        ValueError
            If *color_str* does not match any known color.
        """
        color_str = color_str.lower()
        if color_str == 'orange':
            return Color.ORANGE
        elif color_str == 'yellow':
            return Color.YELLOW
        elif color_str == 'blue':
            return Color.BLUE
        elif color_str == 'green':
            return Color.GREEN
        elif color_str == 'red':
            return Color.RED
        else:
            raise ValueError(f"Unknown color string: {color_str}")


class Direction(Enum):
    """Direction of a connected neighbor relative to a block.

    """
    ABOVE = (0, 1, 0)   # corners 0, 1
    BELOW = (0, -1, 0)  # corners 2, 3
    LEFT  = (-1, 0, 0)  # corners 0, 3
    RIGHT = (1, 0, 0)   # corners 1, 2
    FRONT = (0, 0, 1)
    BACK  = (0, 0, -1)

class ObjectType(Enum):
    """Block and object types with integer values for grid storage.

    The .value ints are stored directly in BlockStructure.block_grid
    (0 = empty cell). Block types map one-to-one with Color members.

    Attributes
    ----------
    YELLOW_BLOCK : int
        Yellow block (1).
    BLUE_BLOCK : int
        Blue block (2).
    GREEN_BLOCK : int
        Green block (3).
    RED_BLOCK : int
        Red block (4).
    DISK : int
        Disk object (5).
    STRIP : int
        Strip object (6).
    """

    YELLOW_BLOCK = 1
    BLUE_BLOCK = 2
    GREEN_BLOCK = 3
    RED_BLOCK = 4

    DISK = 5
    STRIP = 6

    def is_block(self):
        """Return True if this type is one of the four colored blocks."""
        return self in {
            ObjectType.YELLOW_BLOCK,
            ObjectType.BLUE_BLOCK,
            ObjectType.GREEN_BLOCK,
            ObjectType.RED_BLOCK,
        }

    @classmethod
    def block_from_color(cls, color: Color):
        """Return the block ObjectType corresponding to a Color.

        Arguments
        ---------
        color : Color
            A Color enum member (YELLOW, BLUE, GREEN, or RED).

        Returns
        -------
        ObjectType
            The matching block type.

        Raises
        ------
        ValueError
            If *color* has no corresponding block type.
        """
        if color == Color.YELLOW:
            return ObjectType.YELLOW_BLOCK
        elif color == Color.BLUE:
            return ObjectType.BLUE_BLOCK
        elif color == Color.GREEN:
            return ObjectType.GREEN_BLOCK
        elif color == Color.RED:
            return ObjectType.RED_BLOCK
        else:
            raise ValueError(f"No block type for color {color}")


# ── Contour Info Models ─────────────────────────────────────────────

class ContourInfo(BaseModel):
    """Base contour detection result from a single video frame.

    Attributes
    ----------
    t : float
        Timestamp in seconds.
    frame_idx : int
        Source frame index.
    is_circle : bool
        True if the contour was classified as circular.
    is_rectangle : bool
        True if the contour was classified as rectangular.
    is_square : bool
        True if the contour was classified as square.
    color : Color
        Detected color of the contour.
    """

    t: float
    frame_idx: int
    is_circle: bool = False
    is_rectangle: bool = False
    is_square: bool = False
    color: Color = None


class CircleContourInfo(ContourInfo):
    """Contour detection result for a circular shape.

    Attributes
    ----------
    center_uv : tuple[int, int]
        Pixel coordinates (u, v) of the circle center.
    center_xy : tuple[float, float]
        World coordinates (x, y) of the circle center in meters.
    radius : int
        Circle radius in pixels.
    """

    center_uv: tuple[int, int]
    center_xy: tuple[float, float]
    radius: int


class RectangleContourInfo(ContourInfo):
    """Contour detection result for a rectangular shape.

    Attributes
    ----------
    center_uv : tuple[int, int]
        Pixel coordinates (u, v) of the rectangle center.
    center_xy : tuple[float, float]
        World coordinates (x, y) of the rectangle center in meters.
    length_px : int
        Rectangle length in pixels.
    width_px : int
        Rectangle width in pixels.
    length_m : float
        Rectangle length in meters.
    width_m : float
        Rectangle width in meters.
    corner_uvs : list[tuple[int, int]]
        Pixel coordinates of the four corners.
    corner_xys : list[tuple[float, float]]
        World coordinates of the four corners in meters.
    angle : float
        Orientation in radians north of the x-axis.
    quaternion : tuple[float, float, float, float]
        Orientation as [x, y, z, w] quaternion.
    """

    center_uv: tuple[int, int]
    center_xy: tuple[float, float]
    length_px: int
    width_px: int
    length_m: float
    width_m: float
    corner_uvs: list[tuple[int, int]] = None
    corner_xys: list[tuple[float, float]] = None
    angle: float = None
    angle_px: float = None
    quaternion: tuple[float, float, float, float] = None


class SquareContourInfo(ContourInfo):
    """Contour detection result for a square shape.

    Attributes
    ----------
    center_uv : tuple[int, int]
        Pixel coordinates (u, v) of the square center.
    center_xy : tuple[float, float]
        World coordinates (x, y) of the square center in meters.
    size_px : int
        Side length in pixels.
    corner_uvs : list[tuple[int, int]]
        Pixel coordinates of the four corners.
    corner_xys : list[tuple[float, float]]
        World coordinates of the four corners in meters.
    angle : float
        Orientation in radians north of the x-axis.
    quaternion : tuple[float, float, float, float]
        Orientation as [x, y, z, w] quaternion.
    corner_conns_px : list[tuple[int, int]]
        Pixel coordinates of the corners connected to other blocks.
    face_conns_xyzs 
    """

    center_uv: tuple[int, int]
    center_xy: tuple[float, float]
    size_px: int
    corner_uvs: list[tuple[int, int]]
    corner_xys: list[tuple[float, float]]
    angle: float
    corner_conns_px: list[tuple[int, int]] = []
    face_conns_xyzs: list[tuple[float, float, float]] = []
    quaternion: tuple[float, float, float, float] = None
    z_level: int = 0        # height layer index (1-based, 0 = unset/camera)
    center_z: float = 0.0   # world z coordinate


# ── Detected Object Model ───────────────────────────────────────────

class Object(BaseModel):
    """Unified detected object after clustering and temporal filtering.

    Represents a single real-world object with its estimated 3D position,
    orientation, and originating contour data.

    Attributes
    ----------
    t : float
        Timestamp in seconds.
    frame_idx : int
        Source frame index.
    obj_type : ObjectType
        Classified type of the object.
    center_xyz : tuple[float, float, float]
        World position (x, y, z) in meters. z is typically 0 for
        overhead-camera detections.
    angle : float
        Orientation in radians north of the x-axis.
    quaternion : tuple[float, float, float, float]
        Orientation as [x, y, z, w] quaternion.
    corner_xys : list[tuple[float, float]]
        World coordinates of the object's corners in meters.
    corner_conns : list[bool]
        Connection info for each corner (for squares); index matches that of corner_xys.
    obj_label : int
        Cluster label assigned by DBSCAN.
    contour_info : ContourInfo
        Original contour data from the detector.
    """

    t: float
    frame_idx: int
    obj_type: ObjectType
    center_uv: tuple[int, int]
    center_xyz: tuple[float, float, float]
    angle: float = None
    corner_xys: list[tuple[float, float]]
    corner_uvs: list[tuple[int, int]] = None
    # corner_conns: Optional[list[bool]] = None
    face_conns_xyzs: list[tuple[float, float, float]] = None
    quaternion: tuple[float, float, float, float] = None
    obj_label: Optional[int] = None
    contour_info: Optional[ContourInfo] = None

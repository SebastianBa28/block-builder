"""Grip quality analysis using depth estimation.

Analyses depth maps produced by Depth Anything V2 to determine whether
the gripper has successfully grasped a block.  Evaluates roll (diagonal
grip) via column-wise depth standard deviation and height (low/high
grip) by comparing block depth against finger depth.
"""

import numpy as np
from enum import IntEnum
from dataclasses import dataclass, field
from typing import List

from legobuilder.config import (
    GRIP_TOP_FINGER_TRAP, GRIP_BLOCK_TRAP, GRIP_BOTTOM_FINGER_TRAP,
    GRIP_IMAGE_CROP_X,
    GRIP_COLUMN_STD_THRESHOLD,
    GRIP_DIAGONAL_MIN_COLS, GRIP_PARTIAL_MIN_COLS,
)


# ── Data Types ────────────────────────────────────────────────────────


class GripQuality(IntEnum):
    """Discrete grip quality classifications.

    Attributes
    ----------
    GOOD : int
        Block is level and centred between fingers.
    DIAGONAL : int
        Block is tilted (roll error) -- many high-std columns.
    PARTIAL : int
        Block partially gripped -- some high-std columns.
    LOW : int
        Block sits below finger plane.
    HIGH : int
        Block sits above finger plane (detected via 180-degree check).
    NO_FRAME : int
        No depth frame available for analysis.
    """

    GOOD = 0
    DIAGONAL = 1
    PARTIAL = 2
    LOW = 3
    HIGH = 4
    NO_FRAME = 5


@dataclass
class GripAnalysisResult:
    """Result of a single grip quality analysis.

    Attributes
    ----------
    quality : GripQuality
        Overall classification.
    diagonal_score : float
        Mean column standard deviation (roll indicator).
    height_score : float
        block_mean - finger_mean depth difference.
    column_stds : list[float]
        Per-column depth standard deviations.
    n_high_columns : int
        Count of columns exceeding the std threshold.
    """

    quality: GripQuality
    diagonal_score: float = 0.0
    height_score: float = 0.0
    column_stds: List[float] = field(default_factory=list)
    n_high_columns: int = 0


# ── Analyzer ──────────────────────────────────────────────────────────


class GripAnalyzer:
    """Depth-based grip quality analyser.

    Uses three trapezoid regions (top finger, block, bottom finger)
    defined in normalised image coordinates.  The left portion of the
    image (background) is cropped before analysis.  Roll is evaluated
    via column-wise depth variance; height via block-vs-finger mean
    depth comparison.

    Attributes
    ----------
    _block_trap : list[tuple[float, float]]
        Crop-adjusted block trapezoid vertices (normalised).
    _top_finger_trap : list[tuple[float, float]]
        Crop-adjusted top finger trapezoid vertices.
    _bot_finger_trap : list[tuple[float, float]]
        Crop-adjusted bottom finger trapezoid vertices.
    """

    def __init__(self):
        """Initialise trapezoid regions adjusted for image crop."""
        self._block_trap = self._adjust_for_crop(GRIP_BLOCK_TRAP)
        self._top_finger_trap = self._adjust_for_crop(
            GRIP_TOP_FINGER_TRAP)
        self._bot_finger_trap = self._adjust_for_crop(
            GRIP_BOTTOM_FINGER_TRAP)

    @staticmethod
    def _adjust_for_crop(trap):
        """Remap normalised x-coords to cropped image space.

        Arguments
        ---------
        trap : list[tuple[float, float]]
            Trapezoid vertices in original normalised coordinates.

        Returns
        -------
        list[tuple[float, float]]
            Vertices remapped to the cropped coordinate space.
        """
        crop = GRIP_IMAGE_CROP_X
        scale = 1.0 - crop
        return [((x - crop) / scale, y) for x, y in trap]

    @staticmethod
    def _polygon_mask(h, w, vertices):
        """Create a boolean mask for a convex polygon.

        Uses the cross-product winding test (handles CW and CCW).

        Arguments
        ---------
        h : int
            Image height in pixels.
        w : int
            Image width in pixels.
        vertices : list[tuple[float, float]]
            Polygon vertices as normalised (x, y) tuples.

        Returns
        -------
        np.ndarray
            Boolean mask of shape (h, w).
        """
        pts = [(x * w, y * h) for x, y in vertices]
        Y, X = np.mgrid[:h, :w]
        n = len(pts)
        crosses = []
        for i in range(n):
            x1, y1 = pts[i]
            x2, y2 = pts[(i + 1) % n]
            crosses.append(
                (x2 - x1) * (Y - y1) - (y2 - y1) * (X - x1))
        all_pos = np.ones((h, w), dtype=bool)
        all_neg = np.ones((h, w), dtype=bool)
        for c in crosses:
            all_pos &= (c >= 0)
            all_neg &= (c <= 0)
        return all_pos | all_neg

    def _interpolate_trap_y(self, trap, x_frac):
        """Interpolate top/bottom Y bounds at a given X in a trapezoid.

        Trapezoid vertex order is (TL, TR, BR, BL).

        Arguments
        ---------
        trap : list[tuple[float, float]]
            Four vertices in (TL, TR, BR, BL) order.
        x_frac : float
            Normalised X position to sample.

        Returns
        -------
        tuple[float, float]
            (y_top, y_bot) normalised Y bounds.
        """
        tl, tr, br, bl = trap
        # Top edge: TL -> TR
        dx_top = tr[0] - tl[0]
        t_top = (x_frac - tl[0]) / dx_top if abs(dx_top) > 1e-9 else 0.5
        t_top = np.clip(t_top, 0, 1)
        y_top = tl[1] + t_top * (tr[1] - tl[1])
        # Bottom edge: BL -> BR
        dx_bot = br[0] - bl[0]
        t_bot = (x_frac - bl[0]) / dx_bot if abs(dx_bot) > 1e-9 else 0.5
        t_bot = np.clip(t_bot, 0, 1)
        y_bot = bl[1] + t_bot * (br[1] - bl[1])
        return float(y_top), float(y_bot)

    def analyze(self, depth_map: np.ndarray) -> GripAnalysisResult:
        """Analyse a depth map to determine grip quality.

        Runs two sequential checks:

        1. **Roll check** -- samples vertical columns within the block
           trapezoid and flags high depth variance as diagonal grip.
        2. **Height check** -- compares mean block depth against mean
           finger depth to detect low grips.

        Arguments
        ---------
        depth_map : np.ndarray
            2-D uint8 array (H, W) from Depth Anything.  Higher
            values are closer to camera.  Expected to be already
            cropped (left background portion removed).

        Returns
        -------
        GripAnalysisResult
            Classification and supporting metrics.
        """
        h, w = depth_map.shape[:2]

        # ── 1. Roll check via column depth variance ──────────
        block_xs = [x for x, y in self._block_trap]
        x_min, x_max = min(block_xs), max(block_xs)
        # Sample extends left of trapezoid for better coverage
        x_span = x_max - x_min
        col_positions = np.linspace(x_min - x_span * 0.5, x_max, 10)

        column_stds = []
        for cx in col_positions:
            y_top, y_bot = self._interpolate_trap_y(self._block_trap, cx)
            px = int(cx * w)
            py_top = max(0, int(y_top * h))
            py_bot = min(h, int(y_bot * h))
            px = max(0, min(w - 1, px))

            if py_bot > py_top + 1:
                col_depths = depth_map[py_top:py_bot, px].astype(np.float64)
                column_stds.append(float(np.std(col_depths)))
            else:
                column_stds.append(0.0)

        n_high = sum(1 for s in column_stds if s > GRIP_COLUMN_STD_THRESHOLD)
        diagonal_score = float(np.mean(column_stds))

        if n_high >= GRIP_DIAGONAL_MIN_COLS:
            return GripAnalysisResult(
                quality=GripQuality.DIAGONAL,
                diagonal_score=diagonal_score,
                column_stds=column_stds,
                n_high_columns=n_high,
            )

        if n_high >= GRIP_PARTIAL_MIN_COLS:
            return GripAnalysisResult(
                quality=GripQuality.PARTIAL,
                diagonal_score=diagonal_score,
                column_stds=column_stds,
                n_high_columns=n_high,
            )

        # ── 2. Height check: block vs finger depth ──────────
        block_mask = self._polygon_mask(h, w, self._block_trap)
        top_mask = self._polygon_mask(h, w, self._top_finger_trap)
        bot_mask = self._polygon_mask(h, w, self._bot_finger_trap)

        block_pixels = depth_map[block_mask].astype(np.float64)
        top_pixels = depth_map[top_mask].astype(np.float64)
        bot_pixels = depth_map[bot_mask].astype(np.float64)

        if len(block_pixels) == 0 or len(top_pixels) == 0 or len(bot_pixels) == 0:
            return GripAnalysisResult(quality=GripQuality.NO_FRAME)

        block_mean = float(np.mean(block_pixels))
        finger_mean = float(np.mean(top_pixels) + np.mean(bot_pixels)) / 2.0
        height_score = block_mean - finger_mean

        if height_score < 0:
            return GripAnalysisResult(
                quality=GripQuality.LOW,
                diagonal_score=diagonal_score,
                height_score=height_score,
                column_stds=column_stds,
                n_high_columns=n_high,
            )

        # HIGH is detected by checking LOW at 180 deg wrist roll, not here.

        # ── 3. All checks passed ──────────────────────────────
        return GripAnalysisResult(
            quality=GripQuality.GOOD,
            diagonal_score=diagonal_score,
            height_score=height_score,
            column_stds=column_stds,
            n_high_columns=n_high,
        )

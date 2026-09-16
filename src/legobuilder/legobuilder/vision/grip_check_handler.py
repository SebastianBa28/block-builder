"""Visual grip check handlers for the Detector node.

Grip check request handling and depth-map debug visualization.
Functions take explicit dependencies and return messages for
the caller to publish.
"""

import numpy as np

from legobuilder_interfaces.msg import GripCheckResponseMsg


def handle_grip_check_request(
    request_id, ee_color_frame, detector, stamp, logger,
):
    """Run depth estimation and grip analysis, return response message.

    Arguments
    ---------
    request_id : str
        Correlation ID from the grip check request.
    ee_color_frame : np.ndarray or None
        Cached end-effector color camera frame (RGB uint8).
    detector : Detector
        Core detection pipeline instance with analyze_grip method.
    stamp
        A ROS builtin_interfaces/Time message for the header.
    logger
        ROS logger for debug output.

    Returns
    -------
    GripCheckResponseMsg
        Populated response message ready for publishing.
    """
    logger.info(
        f"[RECV brain/grip_check_request] id={request_id}")

    response = GripCheckResponseMsg()
    response.header.stamp = stamp
    response.request_id = request_id

    if ee_color_frame is None:
        logger.warn(
            "No EE color frame available for grip check")
        response.grip_quality = (
            GripCheckResponseMsg.GRIP_NO_FRAME)
        return response

    result, depth_map = detector.analyze_grip(
        ee_color_frame, request_id=request_id)

    response.grip_quality = int(result.quality)
    response.diagonal_score = result.diagonal_score
    response.height_score = result.height_score

    logger.info(
        f"[PUB detector/grip_check_response]"
        f" id={request_id}"
        f" quality={result.quality.name}"
        f" diag={result.diagonal_score:.2f}"
        f" height={result.height_score:.2f}"
    )

    return response


def show_grip_debug(depth_map, result):
    """Show depth map with trapezoid overlays for grip debugging.

    Arguments
    ---------
    depth_map : np.ndarray
        2-D uint8 depth map from Depth Anything.
    result : GripAnalysisResult
        Analysis result with column_stds and quality metrics.
    """
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from legobuilder.config import (
        GRIP_TOP_FINGER_TRAP, GRIP_BLOCK_TRAP,
        GRIP_BOTTOM_FINGER_TRAP,
        GRIP_IMAGE_CROP_X, GRIP_COLUMN_STD_THRESHOLD,
        GRIP_HIGH_THRESHOLD,
    )

    h, w = depth_map.shape
    crop = GRIP_IMAGE_CROP_X
    scale = 1.0 - crop

    def to_cropped_pixels(trap):
        """Convert normalised trap coords to cropped pixels."""
        return [
            ((x - crop) / scale * w, y * h)
            for x, y in trap]

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.imshow(depth_map, cmap='gray')

    for trap, color in [
        (GRIP_TOP_FINGER_TRAP, 'blue'),
        (GRIP_BLOCK_TRAP, 'green'),
        (GRIP_BOTTOM_FINGER_TRAP, 'blue'),
    ]:
        pts = to_cropped_pixels(trap)
        poly = patches.Polygon(
            pts, closed=True, facecolor=color,
            alpha=0.25, edgecolor=color, linewidth=1.5)
        ax.add_patch(poly)

    block_adj = [
        ((x - crop) / scale, y)
        for x, y in GRIP_BLOCK_TRAP]
    block_xs = [x for x, y in block_adj]
    x_min, x_max = min(block_xs), max(block_xs)
    x_span = x_max - x_min
    col_positions = np.linspace(
        x_min - x_span * 0.5, x_max, 10)

    tl, tr, br, bl = block_adj
    for i, cx_pos in enumerate(col_positions):
        dx_top = tr[0] - tl[0]
        t_top = ((cx_pos - tl[0]) / dx_top
                 if abs(dx_top) > 1e-9 else 0.5)
        y_top = (tl[1]
                 + np.clip(t_top, 0, 1) * (tr[1] - tl[1]))
        dx_bot = br[0] - bl[0]
        t_bot = ((cx_pos - bl[0]) / dx_bot
                 if abs(dx_bot) > 1e-9 else 0.5)
        y_bot = (bl[1]
                 + np.clip(t_bot, 0, 1) * (br[1] - bl[1]))

        px = cx_pos * w
        py_top, py_bot = y_top * h, y_bot * h

        if i < len(result.column_stds):
            col_color = (
                'red'
                if result.column_stds[i]
                > GRIP_COLUMN_STD_THRESHOLD
                else 'lime')
            label = f'{result.column_stds[i]:.1f}'
        else:
            col_color, label = 'yellow', '?'
        ax.plot(
            [px, px], [py_top, py_bot],
            color=col_color, linewidth=2)
        ax.text(
            px, py_top - 5, label,
            ha='center', va='bottom',
            fontsize=7, color=col_color,
            fontweight='bold')

    stds_str = ', '.join(
        f'{s:.1f}' for s in result.column_stds)
    title = (
        f"Quality: {result.quality.name}  |  "
        f"High cols: {result.n_high_columns}/10\n"
        f"Col stds: [{stds_str}]"
        f"  (thresh={GRIP_COLUMN_STD_THRESHOLD})\n"
        f"Height: {result.height_score:.2f}"
        f"  (high_thresh={GRIP_HIGH_THRESHOLD})"
    )
    ax.set_title(
        title, fontsize=10, fontfamily='monospace')
    ax.set_axis_off()
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)

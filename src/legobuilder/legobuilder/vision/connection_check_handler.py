"""Connection check handler for the Detector node.

Performs a one-shot connection detection on the current overhead frame
and returns a response message with the count of blocks near the
connection detection position.
"""

import os

import cv2
import numpy as np

from legobuilder_interfaces.msg import ConnectionCheckResponseMsg

from legobuilder.config import (
    CONNECTION_DETECTION_POS,
    CONNECTION_DETECTION_RADIUS,
)

_SAVE_DIR = '/home/robot/robotws/src/legobuilder/tmp/connection_check'


def handle_connection_check_request(
    request_id, detector, overhead_frame_rgb, overhead_frame_hsv,
    stamp, logger,
):
    """Run connection detection on current overhead frame, return response.

    Arguments
    ---------
    request_id : str
        Correlation ID from the connection check request.
    detector : Detector
        Core detection pipeline instance.
    overhead_frame_rgb : np.ndarray or None
        Cached overhead camera frame (RGB uint8).
    overhead_frame_hsv : np.ndarray or None
        Cached overhead camera frame (HSV uint8).
    stamp
        A ROS builtin_interfaces/Time message for the header.
    logger
        ROS logger for debug output.

    Returns
    -------
    ConnectionCheckResponseMsg
        Populated response message ready for publishing.
    """
    logger.info(
        f"[RECV brain/connection_check_request] id={request_id}")

    response = ConnectionCheckResponseMsg()
    response.header.stamp = stamp
    response.request_id = request_id

    if overhead_frame_rgb is None or overhead_frame_hsv is None:
        logger.warn(
            "No overhead frame available for connection check")
        response.nearby_count = 0
        return response

    contour_infos, _seg = detector.perceive_from_image(
        overhead_frame_rgb.copy(),
        overhead_frame_hsv.copy(),
        connection_detection=True,
    )

    # Collect contours within radius of the detection position
    nearby = []
    for ci in contour_infos:
        if not hasattr(ci, 'center_xy'):
            continue
        dx = ci.center_xy[0] - CONNECTION_DETECTION_POS[0]
        dy = ci.center_xy[1] - CONNECTION_DETECTION_POS[1]
        if (dx * dx + dy * dy) <= CONNECTION_DETECTION_RADIUS ** 2:
            nearby.append(ci)

    response.nearby_count = len(nearby)

    # When exactly one block detected, report its color
    if len(nearby) == 1 and hasattr(nearby[0], 'color'):
        response.detected_color = nearby[0].color.name

    # Save annotated image for debugging
    _save_annotated_image(
        overhead_frame_rgb, contour_infos, detector,
        request_id, len(nearby), logger)

    logger.info(
        f"[PUB detector/connection_check_response]"
        f" id={request_id}"
        f" nearby_count={len(nearby)}"
        f" detected_color={response.detected_color!r}"
    )

    return response


def _save_annotated_image(
    frame_rgb, contour_infos, detector,
    request_id, nearby_count, logger,
):
    """Save an annotated connection-check image to disk."""
    ann = frame_rgb.copy()

    # Draw contour annotations (same as detector/annotated topic)
    for ci in contour_infos:
        detector.draw_annotated_shape_contour(ann, ci)

    # Draw the connection detection search region
    try:
        cu, cv = detector.mapper.world2pixel(
            float(CONNECTION_DETECTION_POS[0]),
            float(CONNECTION_DETECTION_POS[1]))
        # Approximate pixel radius: convert a point offset by the radius
        ru, rv = detector.mapper.world2pixel(
            float(CONNECTION_DETECTION_POS[0] + CONNECTION_DETECTION_RADIUS),
            float(CONNECTION_DETECTION_POS[1]))
        radius_px = int(np.hypot(ru - cu, rv - cv))
        cv2.circle(ann, (int(cu), int(cv)), radius_px, (0, 255, 255), 2)
        cv2.circle(ann, (int(cu), int(cv)), 4, (0, 255, 255), -1)
    except Exception:
        pass  # mapper may not be calibrated yet

    # Label with result
    cv2.putText(
        ann,
        f"id={request_id} nearby={nearby_count}",
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    os.makedirs(_SAVE_DIR, exist_ok=True)
    fname = f"conn_check_{request_id}_nearby={nearby_count}.png"
    cv2.imwrite(
        os.path.join(_SAVE_DIR, fname),
        cv2.cvtColor(ann, cv2.COLOR_RGB2BGR))

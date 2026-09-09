"""Named HSV threshold presets for different lighting conditions.

Each preset maps Color -> np.ndarray of shape (3, 2) where rows are
[H, S, V] and columns are [min, max].  Presets are selected at import
time via the HSV_PRESET key in config.py.

Available Presets
-----------------
night
    Regular overhead camera, dark room, original tuning.
night_2
    Regular overhead camera, dark room, re-tuned with wider S/V bands.
day
    Regular overhead camera, daytime ambient light.
depth_camera
    Intel RealSense D435 end-effector camera.
"""

import numpy as np

from legobuilder.schemas import Color

HSV_PRESETS: dict[str, dict[Color, np.ndarray]] = {
    'night': {
        Color.YELLOW: np.array([[21, 30], [204, 255], [160, 177]]),
        Color.BLUE:   np.array([[100, 179], [180, 255], [170, 209]]),
        Color.GREEN:  np.array([[48, 85], [0, 255], [70, 139]]),
        # Color.RED:    np.array([[120, 145], [146, 255], [120, 255]]) # PURPLE
    },
    'night_2': {
        Color.YELLOW: np.array([[21, 30], [150, 255], [150, 177]]),
        Color.BLUE:   np.array([[100, 179], [180, 255], [170, 209]]),
        Color.GREEN:  np.array([[48, 85], [0, 255], [40, 139]]),
        # Color.RED:    np.array([[120, 145], [146, 255], [120, 255]]) # PURPLE

    },
    'day': {
        Color.YELLOW: np.array([[21, 30], [135, 255], [157, 177]]),
        Color.BLUE:   np.array([[100, 179], [180, 255], [170, 255]]),
        Color.GREEN:  np.array([[48, 85], [0, 255], [27, 156]]),
        # Color.RED:    np.array([[120, 145], [146, 255], [120, 255]]) # PURPLE
    },
    'depth_camera': {
        Color.YELLOW: np.array([[22, 45], [70, 255], [82, 255]]),
        Color.BLUE:   np.array([[99, 120], [240, 255], [140, 255]]),
        Color.GREEN:  np.array([[54, 90], [92, 255], [59, 255]]),
    },
}

MIN_H = 10
MAX_H = 25
MIN_S = 76
MAX_S = 255
MIN_V = 210
MAX_V = 255

FOCUS = 0
AUTO_FOCUS = False
EXPOSURE = 90
AUTO_EXPOSURE = False
GAIN = 81
WHITE_BALANCE = 3488
AUTOWHITEBALANCE = False # 0=OFF, 1=ON
BRIGHTNESS = 28
CONTRAST = 182
SATURATION = 255
SHARPNESS = 255

ARUCO_SETTINGS = {
    'tl': {     # top left
        'cnt_x': 0.14,
        'cnt_y': 0.7975,
    },
    'br': {     # bottom right
        'cnt_x': 0.987,
        'cnt_y': 0.108 
    }
}
USE_MULTIPLE_PERSPECTIVES = True
WORLD_U = 24
WORLD_V = 12

DISK_DIMS = {
    'radius': 0.02,
    'thickness': 0.0075
}
STRIP_DIMS = {
    'length': 0.112,
    'width': 0.025
}

# Stability and buffer settings
CIRCLE_STABILITY_THRESHOLD = 0.2
RECTANGLE_STABILITY_THRESHOLD = 6.0
CIRCLE_BUFFER_DURATION = 3.0
RECTANGLE_BUFFER_DURATION = 6.0
T_CHECK_CIRCLE = 1.0
T_CHECK_RECTANGLE = 3.0
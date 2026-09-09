import numpy as np
from math import pi

RATE = 100.0            # Hertz
Q_READY = np.array([pi/2, 0, pi/2])
JOINT_NAMES = ["base", "shoulder", "elbow"]
DURATION = 5.0 # seconds
OUTER_RADIUS = 0.73 # About base motor
INNER_RADIUS = 0.25
BASE_MOTOR_POS = np.array([0.5565, 0.035, 0.0378])
NUM_DOFS = 3

COLLISION_DETECTION = False
POSITION_ERROR_THRESHOLD = 0.5  # radians
VELOCITY_ERROR_THRESHOLD = 0.7  # radians/sec
EFFORT_ERROR_THRESHOLD = 1.5    # Newton-meters
COLLISION_WAIT_DURATION = 1.0   # seconds

TEST_GRAVITY = False

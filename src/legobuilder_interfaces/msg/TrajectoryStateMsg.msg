# Pre-computed trajectory state from brain to manipulator

# Mode constants
int32 MODE_P=0
int32 MODE_Q=1
int32 mode

# Joint space (q values - computed by brain via IK)
float64[6] q   # final joint angles
float64[6] qd  # final joint velocities

# Cartesian space (for reference)
geometry_msgs/Point p   # final position
geometry_msgs/Vector3 pd  # final linear velocity

# Timing parameters
float64 min_duration
float64 delay_before
float64 delay_after

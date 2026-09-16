std_msgs/Header header
string request_id
int32 request_type

int32 REQUEST_PRIMARY=0
int32 REQUEST_PRECOMPUTE=1

# Per-waypoint data (parallel arrays, N waypoints)
# Cartesian targets (for waypoints needing IK)
float64[] positions        # flattened N*3: [x1,y1,z1, x2,y2,z2, ...]
float64[] tilts            # N tilts in radians
float64[] rolls            # N rolls in radians
bool[] gripper_open        # N gripper states

# Pre-specified joint solutions (for waypoints that already have q)
float64[] q_preset         # flattened N*6 joint angles
float64[] qd_preset        # flattened N*6 joint velocities
bool[] needs_ik            # N flags: true = solve IK, false = use q_preset

# Timing (passed through for response reconstruction)
float64[] min_durations    # N durations
float64[] delay_befores    # N delays
float64[] delay_afters     # N delays

# Calibration flag — skip offset application when true
bool in_calibration false

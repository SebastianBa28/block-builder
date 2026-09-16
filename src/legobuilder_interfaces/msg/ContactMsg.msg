# Sent from manipulator to brain when collision is detected
std_msgs/Header header

# Contact type constants
int32 CONTACT_COLLISION=0

int32 contact_type

# Diagnostic info
float64[3] position_error
float64[3] velocity_error
float64[3] effort_error

# Current joint state at collision
float64[3] q_actual
float64[3] qd_actual

# Number of states remaining in queue when collision occurred
int32 states_remaining

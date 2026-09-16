# Sent from manipulator to brain when grip failure is detected
std_msgs/Header header

# Actual gripper motor position (radians) when failure detected
float64 gripper_position

# Number of trajectory states remaining in queue
int32 states_remaining

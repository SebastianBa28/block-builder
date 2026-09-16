# Command from brain to manipulator for trajectory queue operations
std_msgs/Header header

# Command type constants
int32 COMMAND_APPEND_FRONT=0   # Add states to front of queue
int32 COMMAND_APPEND_BACK=1    # Add states to end of queue
int32 COMMAND_REPLACE=2        # Clear queue and add new states

int32 command

# Trajectory states to add
legobuilder_interfaces/TrajectoryStateMsg[] states

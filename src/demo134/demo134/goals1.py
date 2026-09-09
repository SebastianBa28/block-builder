'''demo134.py

   Demonstration node to interact with the HEBIs.

   Similar to 133a, this publishes joint commands every 10ms.  But the
   topic is now /joint_commands.

   To start the trajectory, the initialization also (temporarily)
   subscribes to /joint_states and remembers the initial joint
   positions.


   Node:        /demo
   Subscribe:   /joint_states (once)    sensor_msgs/JointState
   Publish:     /joint_commands         sensor_msgs/JointState
'''

import rclpy
import numpy as np
import tf2_ros
import sys

from math               import pi, sin, cos, acos, atan2, sqrt, fmod, exp
import numpy as np
from dataclasses import dataclass

from asyncio            import Future
from rclpy.node         import Node
from geometry_msgs.msg  import PoseStamped, TwistStamped
from geometry_msgs.msg  import TransformStamped
from sensor_msgs.msg    import JointState
from std_msgs.msg       import Header
from std_msgs.msg       import Float32MultiArray
from demo134.TrajectoryUtils    import goto5


#
#   Definitions
#
RATE = 100.0            # Hertz
NUM_DOFS = 3
BASE_INIT = 0.17
EPS = 1e-2

@dataclass
class TrajectoryState:
    name: str
    duration: float
    q: np.ndarray
    t_at_start: float = None

class Trajectory:
    def __init__(self, q0: np.ndarray):
        self.trajectory_states = []
        self.q0 = q0
        
    def _ensure_valid_state(self, state: TrajectoryState):
        # TODO: implement this correctly. IMPORTANT: can cause robot breaking
        # ensure that t_at_start from different states are not conflicting
        # if conflicting: will cause jerks
        if state.t_at_start is None:
            total_duration = sum(state.duration for state in self.trajectory_states)
            state.t_at_start = total_duration
        return state
    
    def add_state(self, state: TrajectoryState):
        state = self._ensure_valid_state(state)
        self.trajectory_states.append(state)
        
    def add_states(self, states: list[TrajectoryState]):
        for state in states:
            self.add_state(state)
    
    # def get_state(self, t: float) -> TrajectoryState:
    #     for state in self.trajectory_states:
    #         if t < state.t_at_start + state.duration:
    #             return state
    #     return self.trajectory_states[-1]
    
    def get_update(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(self.trajectory_states) == 0:
            return [], [], []
        q0 = self.q0.copy()
        for state in self.trajectory_states:
            if state.t_at_start <= t < state.t_at_start + state.duration:
                break
            else:
                q0 = state.q.copy()
        else:
            state = self.trajectory_states[-1]
            return state.q, np.zeros_like(state.q), np.zeros_like(state.q)
        t_state = t - state.t_at_start
        q, qd, qdd = goto5(t_state, state.duration, q0, state.q)
        return q, qd, qdd

#
#   DEMO Node Class
#
#   This inherits all the standard ROS node stuff, but adds an
#   update() method to be called regularly by an internal timer and a
#   shutdown method to stop the timer.
#
#   Arguments are the node name and a future object (to force a shutdown).
#
class DemoNode(Node):
    # Initialization.
    def __init__(self, name, future):
        # Initialize the node and store the future object (to end).
        super().__init__(name)
        self.future = future

        # Create a temporary subscriber to grab the initial position.
        self.q0 = self.grabfbk()

        ##############################################################
        # INITIALIZE YOUR TRAJECTORY DATA!

        self.q = self.q0
        self.get_logger().info(f"self.q0: {self.q0}")

        self.trajectory = Trajectory(self.q0)
        self.trajectory.add_states([
            TrajectoryState(name='1a', duration=3, q=np.array([self.q0[0], 0, self.q0[2]])),
            TrajectoryState(name='1b', duration=3, q=np.array([BASE_INIT, 0, 0])),
            # TrajectoryState(name='2',  duration=3, q=np.array([BASE_INIT + pi/2, 0, pi/2])),
            # TrajectoryState(name='3',  duration=3, q=np.array([BASE_INIT - pi/2, 0, pi/2])),
            # TrajectoryState(name='4',  duration=3, q=np.array([BASE_INIT - pi/2, -pi/2, 0])),
        ])

        ##############################################################
        # Setup the logistics of the node:
        # Add a publisher to send the joint commands.
        self.pubcmd = self.create_publisher(JointState, '/joint_commands', 10)

        # Do you need any other subscribers?  Place them here.
        # self.create_subscription(Float32MultiArray, '/goals', self.go_to_goal, 1)

        # Wait for a connection to happen.  This isn't necessary, but
        # means we don't start until the rest of the system is ready.
        self.get_logger().info("Waiting for a /joint_commands subscriber...")
        while(not self.count_subscribers('/joint_commands')):
            pass

        # Create a timer to keep calculating/sending commands.
        rate           = RATE
        self.starttime = self.get_clock().now()
        self.timer     = self.create_timer(1/rate, self.update)
        self.get_logger().info("Sending commands with dt of %f seconds (%fHz)" %
                               (self.timer.timer_period_ns * 1e-9, rate))

    # Shutdown
    def shutdown(self):
        # Destroy the timer, then shut down the node.
        self.timer.destroy()
        self.destroy_node()


    # Grab a single feedback - DO NOT CALL THIS REPEATEDLY!
    def grabfbk(self):
        # Create a temporary handler to grab the position.
        def cb(fbkmsg):
            self.grabpos   = list(fbkmsg.position)
            self.grabready = True

        # Temporarily subscribe to get just one message.
        sub = self.create_subscription(JointState, '/joint_states', cb, 1)
        self.grabready = False
        while not self.grabready:
            rclpy.spin_once(self)
        self.destroy_subscription(sub)

        # Return the values.
        return self.grabpos

    # Send a command.
    def sendcmd(self, pos, vel, eff = []):
        # Build up the message and publish.
        cmdmsg = JointState()
        cmdmsg.header.stamp    = self.get_clock().now().to_msg()
        cmdmsg.header.frame_id = 'world'
        cmdmsg.name            = ['base', 'shoulder', 'elbow']
        cmdmsg.position        = pos
        cmdmsg.velocity        = vel
        cmdmsg.effort          = eff
        self.pubcmd.publish(cmdmsg)


    ######################################################################
    # Handlers
    
    def get_t(self):
        now = self.get_clock().now()
        return (now - self.starttime).nanoseconds * 1e-9

    # Also place other callbacks here...
    # def callback(self, msg):

    # Timer (100Hz) update.
    def update(self):
        # Grab the current time.
        t = self.get_t()
        
        q, qd, qdd = self.trajectory.get_update(t)
        self.get_logger().info(f"t={t:.2f}, q={q}, qd={qd}, qdd={qdd}")
        
        # Send.
        self.sendcmd(q, qd, qdd)

#
#   Main Code
#
def main(args=None):
    # Initialize ROS.
    rclpy.init(args=args)

    # Create a future object to signal when the trajectory ends.
    future = Future()

    # Instantiate the DEMO node.
    node = DemoNode('demo', future)

    # Spin, meaning keep running (taking care of the timer callbacks
    # and message passing), until interrupted or the node is complete
    # (as signaled by the future object).
    rclpy.spin_until_future_complete(node, future)

    # Report the reason for shutting down.
    if future.done():
        node.get_logger().info("Stopping: " + future.result())
    else:
        node.get_logger().info("Stopping: Interrupted")

    # Shutdown the node and ROS.
    node.shutdown()
    rclpy.shutdown()

if __name__ == "__main__":
    main()

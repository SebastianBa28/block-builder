
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

from demo134.trajectory import Trajectory, TrajectoryState


#
#   Definitions
#
RATE = 100.0            # Hertz
NUM_DOFS = 3
BASE_INIT = 0.95
EPS = 1e-2


class DemoNode(Node):
    def __init__(self, name, future):
        super().__init__(name)
        self.future = future

        self.q0 = self.grabfbk()

        ##############################################################
        # INITIALIZE YOUR TRAJECTORY DATA!

        self.q = self.q0
        self.get_logger().info(f"self.q0: {self.q0}")

        # self.trajectory = Trajectory(self.q0)
        # self.trajectory.add_states([
        #     TrajectoryState(name='1a', duration=3, q=np.array([self.q0[0], 0, self.q0[2]])),
        #     TrajectoryState(name='1b', duration=3, q=np.array([BASE_INIT, 0, 0])),
        #     TrajectoryState(name='2',  duration=3, q=np.array([BASE_INIT + pi/2, 0, pi/2])),
        #     TrajectoryState(name='3',  duration=3, q=np.array([BASE_INIT - pi/2, 0, pi/2])),
        #     TrajectoryState(name='4',  duration=3, q=np.array([BASE_INIT - pi/2, -pi/2, 0])),
        # ])

        ##############################################################
        self.pubcmd = self.create_publisher(JointState, '/joint_commands', 10)

        # Do you need any other subscribers?  Place them here.
        # self.create_subscription(Float32MultiArray, '/goals', self.go_to_goal, 1)

        self.get_logger().info("Waiting for a /joint_commands subscriber...")
        while(not self.count_subscribers('/joint_commands')):
            pass

        rate           = RATE
        self.starttime = self.get_clock().now()
        self.timer     = self.create_timer(1/rate, self.update)
        self.get_logger().info("Sending commands with dt of %f seconds (%fHz)" %
                               (self.timer.timer_period_ns * 1e-9, rate))

    def shutdown(self):
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
        t = self.get_t()
        
        # q, qd, qdd = self.trajectory.get_update(t)
        
        # self.sendcmd(q, qd, qdd)

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

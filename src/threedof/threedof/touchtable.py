'''touchtable.py

   Skeleton node to interact with the HEBIs.

   Similar to 133a, this publishes joint commands every 10ms.  But the
   topic is now /joint_commands.  Relative to last week, it also
   listens to /joint_states, so it can compute the gravity commands
   and detect collisions.

   To start the trajectory, the initialization also (temporarily)
   subscribes to /joint_states and remembers the initial joint
   positions.


   Node:        /demo
   Subscribe:   /joint_states           sensor_msgs/JointState
   Publish:     /joint_commands         sensor_msgs/JointState

'''

import rclpy
import numpy as np
import tf2_ros
import sys

from math               import pi, sin, cos, acos, atan2, sqrt, fmod, exp

from asyncio            import Future
from rclpy.node         import Node
from geometry_msgs.msg  import PoseStamped, TwistStamped
from geometry_msgs.msg  import TransformStamped, Point
from sensor_msgs.msg    import JointState
from std_msgs.msg       import Header
from std_msgs.msg       import Float32MultiArray

from threedof.TrajectoryUtils import goto5
from threedof.trajectory import Trajectory, TrajectoryState, P_OR_Q
from threedof.KinematicChain import KinematicChain
from threedof.config import RATE, Q_READY, JOINT_NAMES, DURATION, OUTER_RADIUS, \
                              INNER_RADIUS, BASE_MOTOR_POS, COLLISION_DETECTION, \
                              POSITION_ERROR_THRESHOLD, COLLISION_WAIT_DURATION, \
                              VELOCITY_ERROR_THRESHOLD, EFFORT_ERROR_THRESHOLD, TEST_GRAVITY


def is_point_reachable(p: np.ndarray) -> bool:
    p_rel = p - BASE_MOTOR_POS

    #### Current implementation -- only care about table XY reachability
    r_xy = sqrt(p_rel[0]**2 + p_rel[1]**2)
    if r_xy > OUTER_RADIUS or r_xy < INNER_RADIUS or p[0] <= 0 or p[1] <= 0 or p[2] < 0:
        return False
    return True

def compute_gravity_constants(
    mass: float = 0.5,    # kg
    length: float = 0.35,  # m
    g: float = 9.81 # m/s^2
):
    A = length * mass * g
    B = 0  # assuming start in stand-up position
    return A, B

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
        self.get_logger().info("Initial positions: %r" % self.q0)

        ##############################################################
        # INITIALIZE YOUR DATA!

        # Actual states
        self.q = self.q0
        self.qd = np.zeros_like(self.q0)
        self.tau = np.zeros_like(self.q0)

        # Command states
        self.qc = None
        self.qdc = None
        self.tauc = None

        self.get_logger().info(f"self.q0: {self.q0}")

        self.chain = KinematicChain(self, "world", "tip", JOINT_NAMES)
        self.trajectory = Trajectory(self, self.q0, dt=1/RATE, chain=self.chain)
        self.trajectory.add_states([
            TrajectoryState(label="ready_1", min_duration=3, q=np.array([self.q0[0], 0, self.q0[2]]), mode=P_OR_Q.Q),
            TrajectoryState(label="ready_2", min_duration=2, q=Q_READY.copy(), mode=P_OR_Q.Q),
            
            TrajectoryState(label='point_1', min_duration=DURATION, p=np.array([0.331, 0.604, 0]), mode=P_OR_Q.P),
            TrajectoryState(label="ready", min_duration=2, q=Q_READY.copy(), mode=P_OR_Q.Q),
            TrajectoryState(label='point_2', min_duration=DURATION, p=np.array([0.142, 0.23, 0]), mode=P_OR_Q.P),
            
            TrajectoryState(label="ready_3", min_duration=DURATION, q=Q_READY, mode=P_OR_Q.Q),
            TrajectoryState(label="ready_4", min_duration=2, q=Q_READY, mode=P_OR_Q.Q),
            TrajectoryState(label='point_3', min_duration=5, p=np.array([0.818, 0.4175, 0.0]), mode=P_OR_Q.P),
        ])  
        
        self.grav_constants_shoulder = compute_gravity_constants(
            mass=0.52,
            length=0.35,
        )
        self.grav_constants_elbow = compute_gravity_constants(
            mass=0.03,
            length=0.15
        )
        
        ##############################################################
        self.pubcmd = self.create_publisher(JointState, '/joint_commands', 10)

        # Create a subscriber to continually receive joint state messages.
        self.q = self.q0.copy()
        self.create_subscription(JointState, '/joint_states', self.recvact, 10)
        self.create_subscription(Point, '/point', self.recvpoint, 10)
        self.get_logger().info("Waiting for a /joint_commands subscriber...")
        while(not self.count_subscribers('/joint_commands')):
            pass

        # Create a timer to keep calculating/sending commands.
        self.dt           = 1 / RATE
        self.starttime = self.get_clock().now()
        self.timer     = self.create_timer(self.dt, self.update)
        self.get_logger().info("Sending commands with dt of %f seconds (%fHz)" %
                               (self.timer.timer_period_ns * 1e-9, RATE))

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

    def get_t(self):
        now = self.get_clock().now()
        return (now - self.starttime).nanoseconds * 1e-9

    ######################################################################
    # Handlers

    # Receive actual state - called repeatedly by incoming messages.
    def recvact(self, msg):
        # Save the actual states.
        self.q = np.array(msg.position)
        self.qd = np.array(msg.velocity)
        self.tau = np.array(msg.effort)
    
    def recvpoint(self, pointmsg):
        # Extract the data.
        x = pointmsg.x
        y = pointmsg.y
        z = pointmsg.z
        prioritize = False  # TODO: input from msg
        p = np.array([x, y, z])
        
        # Report.
        self.get_logger().info("Received point %r, %r, %r" % (x,y,z))
        
        # Ensure point is reachable
        is_reachable = is_point_reachable(p)
        if not is_reachable:
            self.get_logger().info("Point is not reachable, ignoring.")
            return

        # Add to trajectory
        t = self.get_t()
        ready = TrajectoryState(
            label=f"ready-t={t}",
            q=Q_READY.copy(),
            min_duration=DURATION,
            mode=P_OR_Q.Q,
        )
        point = TrajectoryState(
            label=f"recvpoint-t={t}",
            p=p,
            min_duration=DURATION,
            mode=P_OR_Q.P,
        )
        self.trajectory.add_states([ready, point], prioritize=prioritize, t=t)

    def gravity(self, q: np.ndarray):
        q_shoulder = q[1]
        q_elbow = q[2]
        A_shoulder, B_shoulder = self.grav_constants_shoulder
        A_elbow, B_elbow = self.grav_constants_elbow
        tau_shoulder = A_shoulder * np.sin(q_shoulder) + B_shoulder * np.cos(q_shoulder)
        tau_elbow = A_elbow * np.sin(q_shoulder + q_elbow) + B_elbow * np.cos(q_shoulder + q_elbow)
        return -np.array([0, tau_shoulder, tau_elbow])

    def collision_detection(self, t: float):
        # Collision detection
        if COLLISION_DETECTION \
          and self.qc is not None and self.qdc is not None and self.tauc is not None:
            pos_error = np.linalg.norm(self.qc - self.q) > POSITION_ERROR_THRESHOLD
            vel_error = np.linalg.norm(self.qdc - self.qd) > VELOCITY_ERROR_THRESHOLD
            eff_error = np.linalg.norm(self.tauc - self.tau) > EFFORT_ERROR_THRESHOLD
            if pos_error or vel_error or eff_error:
                if pos_error:
                    self.get_logger().info(f"Position error: {np.linalg.norm(self.qc - self.q)}")
                elif vel_error:
                    self.get_logger().info(f"Velocity error: {np.linalg.norm(self.qdc - self.qd)}")
                else:
                    self.get_logger().info(f"Effort error: {np.linalg.norm(self.tauc - self.tau)}")
                self.get_logger().info("Collision detected! Clearing trajectory.")
                self.trajectory.clear_states(self.q)
                self.trajectory.add_state(
                    TrajectoryState(
                        label=f"ready_after_collision-t={t}",
                        q=Q_READY.copy(),
                        delay_before=COLLISION_WAIT_DURATION,
                        min_duration=DURATION,
                        mode=P_OR_Q.Q,
                    ),
                    t=t
                )

    # Timer (100Hz) update.
    def update(self):
        t = self.get_t()
        
        self.collision_detection(t)
        
        # q, qd = self.trajectory.get_update(t, self.q)  # when using velocity inverse IK
        q, qd = self.trajectory.get_update(t)
        tau = self.gravity(self.q)
        self.qc = q if len(q) > 0 else None
        self.qdc = qd if len(qd) > 0 else None
        self.tauc = tau if len(tau) > 0 else None
        
        # Update commands for when testing gravity
        if TEST_GRAVITY:
            q = np.array([])
            qd = np.array([])
        
        # Send commands
        self.sendcmd(q, qd, tau)

#
#   Main Code
#
def main(args=None):
    # Initialize ROS.
    rclpy.init(args=args)

    # Create a future object to signal when the trajectory ends.
    future = Future()

    # Instantiate the DEMO node.
    node = DemoNode('touchtable', future)

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

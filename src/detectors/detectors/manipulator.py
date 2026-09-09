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
from geometry_msgs.msg  import TransformStamped, Point, Pose
from sensor_msgs.msg    import JointState
from std_msgs.msg       import Header
from std_msgs.msg       import Float32MultiArray


from .kinematics.TrajectoryUtils import goto5
from .kinematics.trajectory import Trajectory, TrajectoryState, P_OR_Q
from .kinematics.KinematicChain import KinematicChain
from .kinematics.config import RATE, Q_READY, JOINT_NAMES, DURATION, OUTER_RADIUS, \
    INNER_RADIUS, BASE_MOTOR_POS, COLLISION_DETECTION, \
    POSITION_ERROR_THRESHOLD, COLLISION_WAIT_DURATION, \
    VELOCITY_ERROR_THRESHOLD, EFFORT_ERROR_THRESHOLD, TEST_GRAVITY

from detectors.config import DISK_DIMS, STRIP_DIMS

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

import numpy as np

def apply_quaternion(
    pos: tuple[float, float, float],                # x, y, z
    quaternion: tuple[float, float, float, float]   # x, y, z, w
) -> tuple[float, float, float]: 
    # Convert inputs to arrays (given in prompt)
    p = np.array([pos[0], pos[1], pos[2], 0.0])
    q = np.array([quaternion[0], quaternion[1], quaternion[2], quaternion[3]])
    q_conj = np.array([-quaternion[0], -quaternion[1], -quaternion[2], quaternion[3]])
    
    # Define Hamilton product for quaternion multiplication (q1 * q2)
    def quat_mult(q1, q2):
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        return np.array([
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,  # x
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,  # y
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,  # z
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2   # w
        ])

    # Apply rotation: p' = q * p * q_conj
    qp = quat_mult(q, p)
    p_applied = quat_mult(qp, q_conj)

    return (p_applied[0], p_applied[1], p_applied[2])

class ManipulatorNode(Node):
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

        self.start_time = self.get_clock().now()
        
        self.get_logger().info(f"type(start_time): {type(self.start_time)}")
        self.get_logger().info(f"type(clock): {type(self.get_clock())}")

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
        self.trajectory = Trajectory(self, self.q0, dt=1/RATE, chain=self.chain, clock=self.get_clock(), start_time=self.start_time)
        self.trajectory.add_states([
            TrajectoryState(min_duration=3, q=np.array([self.q0[0], 0, self.q0[2]]), mode=P_OR_Q.Q),
            TrajectoryState(min_duration=2, q=Q_READY.copy(), mode=P_OR_Q.Q),
        ])  
        
        self.grav_constants_shoulder = compute_gravity_constants(
            mass=0.52,
            length=0.35,
        )
        self.grav_constants_elbow = compute_gravity_constants(
            mass=0.03,
            length=0.15
        )
        
        self.touching_disk = False
        self.touching_strip = False
        
        ##############################################################
        self.pubcmd = self.create_publisher(JointState, '/joint_commands', 10)

        # Create a subscriber to continually receive joint state messages.
        self.q = self.q0.copy()
        self.create_subscription(JointState, '/joint_states', self.recvact, 10)
        self.create_subscription(Point, 'detector/disk_info', self.recv_disk, 10)
        self.create_subscription(Pose, 'detector/strip_info', self.recv_strip, 10)
        self.get_logger().info("Waiting for a /joint_commands subscriber...")
        while(not self.count_subscribers('/joint_commands')):
            pass

        # Create a timer to keep calculating/sending commands.
        self.dt           = 1 / RATE
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
        return (now - self.start_time).nanoseconds * 1e-9

    ######################################################################
    # Handlers

    # Receive actual state - called repeatedly by incoming messages.
    def recvact(self, msg):
        # Save the actual states.
        self.q = np.array(msg.position)
        self.qd = np.array(msg.velocity)
        self.tau = np.array(msg.effort)
        
    def recv_disk(self, pointmsg):
        # TODO: Add linear scaling offset in x offset = -0.04/0.66 * x
        
        if self.touching_disk or self.touching_strip:
            return
        
        x, y, z = pointmsg.x, pointmsg.y, DISK_DIMS['thickness']
        self.get_logger().info(f"Received disk at {x}, {y}, {z}")
        
        if not is_point_reachable(np.array([x, y, z])):
            self.get_logger().info("Disk point is not reachable, ignoring.")
            return
        
        self.touching_disk = True
        
        disk_touch = TrajectoryState(
            p=np.array([x, y, z]),
            min_duration=DURATION,
            delay_after=1.0,
            mode=P_OR_Q.P,
        )
        ready = TrajectoryState(
            q=Q_READY.copy(),
            min_duration=DURATION,
            mode=P_OR_Q.Q,
        )
        self.trajectory.add_states([disk_touch, ready])
        
    def recv_strip(self, posemsg):
        if self.touching_strip or self.touching_disk:
            return
        
        cnt_x = posemsg.position.x
        cnt_y = posemsg.position.y
        cnt_z = 0
        cnt = np.array([cnt_x, cnt_y, cnt_z])
        quaternion = posemsg.orientation
        quaternion = (quaternion.x, quaternion.y, quaternion.z, quaternion.w)
        self.get_logger().info(f"Received strip at {cnt_x}, {cnt_y}, {cnt_z} with quaternion {quaternion}")
        
        above_strip_v = (0, STRIP_DIMS['width']/2 + 0.02, 0)
        below_strip_v = (0, -(STRIP_DIMS['width']/2 + 0.02), 0)
        above_strip_p = apply_quaternion(above_strip_v, quaternion)
        below_strip_p = apply_quaternion(below_strip_v, quaternion)
        
        if not is_point_reachable(cnt + np.array(above_strip_p)) \
          or not is_point_reachable(cnt + np.array(below_strip_p)):
            self.get_logger().info("Strip points are not reachable, ignoring.")
            return
        
        self.touching_strip = True
        
        strip_touch_above = TrajectoryState(
            p=cnt + np.array(above_strip_p),
            min_duration=DURATION,
            # delay_after=1.0,
            mode=P_OR_Q.P,
        )
        strip_touch_center = TrajectoryState(
            p=cnt + np.array([0, 0, 0.1]),
            min_duration=1.0,
            mode=P_OR_Q.P,
        )
        strip_touch_below = TrajectoryState(
            p=cnt + np.array(below_strip_p),
            min_duration=3.0,
            # delay_after=2.0,
            mode=P_OR_Q.P,
        )
        ready = TrajectoryState(
            q=Q_READY.copy(),
            min_duration=DURATION,
            mode=P_OR_Q.Q,
        )
        self.trajectory.add_states(
            [strip_touch_above, strip_touch_center, strip_touch_below, ready] 
        )
        

    def gravity(self, q: np.ndarray):
        q_shoulder = q[1]
        q_elbow = q[2]
        A_shoulder, B_shoulder = self.grav_constants_shoulder
        A_elbow, B_elbow = self.grav_constants_elbow
        tau_shoulder = A_shoulder * np.sin(q_shoulder) + B_shoulder * np.cos(q_shoulder)
        tau_elbow = A_elbow * np.sin(q_shoulder + q_elbow) + B_elbow * np.cos(q_shoulder + q_elbow)
        return -np.array([0, tau_shoulder, tau_elbow])

    def collision_detection(self, t: float):
        def _get_error_thresholds():
            # return POSITION_ERROR_THRESHOLD, VELOCITY_ERROR_THRESHOLD, EFFORT_ERROR_THRESHOLD
            
            # make effort error threshold dynamic based on joint speeds
            vel_norm = np.linalg.norm(self.qd)
            dynamic_position_threshold = POSITION_ERROR_THRESHOLD + (0.2 * vel_norm)
            dynamic_velocity_threshold = VELOCITY_ERROR_THRESHOLD + (0.3 * vel_norm)
            dynamic_effort_threshold = EFFORT_ERROR_THRESHOLD + (0.5 * vel_norm)
            return dynamic_position_threshold, dynamic_velocity_threshold, dynamic_effort_threshold
        
        
        # Collision detection
        if COLLISION_DETECTION \
          and self.qc is not None and self.qdc is not None and self.tauc is not None:
            p_th, v_th, e_th = _get_error_thresholds()
            pos_error = np.linalg.norm(self.qc - self.q) > p_th
            vel_error = np.linalg.norm(self.qdc - self.qd) > v_th
            eff_error = np.linalg.norm(self.tauc - self.tau) > e_th
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
                        q=Q_READY.copy(),
                        delay_before=COLLISION_WAIT_DURATION,
                        min_duration=DURATION,
                        mode=P_OR_Q.Q,
                    ),
                )

    # Timer (100Hz) update.
    def update(self):
        t = self.get_t()
        
        # Check if ready to touch disk/strip
        if len(self.trajectory.trajectory_states) == 0:
            self.touching_disk = False
            self.touching_strip = False
        
        # self.get_logger().info(f"Num states: {len(self.trajectory.trajectory_states)}")
        
        self.collision_detection(t)
        
        q, qd = self.trajectory.get_update(t, )
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
    node = ManipulatorNode('manipulator', future)

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

"""Generic kinematic chain parsed from URDF with FK, Jacobian, and IK.

Reads a URDF robot description from the /robot_description ROS topic,
extracts the chain of joints between a specified base frame and tip frame,
and provides forward-kinematics (FK), linear/angular Jacobians, and two
inverse-kinematics (IK) solvers: Newton-Raphson iterative and real-time
velocity-based.
"""

import enum
import rclpy
import numpy as np

from dataclasses import dataclass, field
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from std_msgs.msg import String
from urdf_parser_py.urdf import Robot

from legobuilder.kinematics import transform_utils as tu
from legobuilder.kinematics.trajectory_utils import goto5
from legobuilder.config import Q_READY, IK_MAX_ITERATIONS, IK_TOLERANCE, IK_SIM_DT


# ── Module-Level Helpers ─────────────────────────────────────────────

def _info(node, string):
    """Log an info message prefixed with 'KinematicChain'."""
    node.get_logger().info("KinematicChain: " + string)


def _error(node, string):
    """Log an error and raise an exception."""
    node.get_logger().error("KinematicChain: " + string)
    raise Exception(string)


def _read_urdf(node):
    """Read the URDF XML string from the /robot_description topic.

    Blocks (spinning the node) until the message arrives.

    Arguments
    ---------
    node : Node
        ROS node used for subscription and spinning.

    Returns
    -------
    str
        Raw URDF XML string.
    """
    _info(node, "Waiting for the URDF to be published...")

    html = None

    def callback(msg):
        nonlocal html
        html = msg.data

    topic = '/robot_description'
    quality = QoSProfile(durability=DurabilityPolicy.TRANSIENT_LOCAL, depth=1)
    sub = node.create_subscription(String, topic, callback, quality)

    while html is None:
        rclpy.spin_once(node)

    node.destroy_subscription(sub)
    return html


def _parse_urdf(node, html, baseframe, tipframe):
    """Parse URDF XML and extract the joint chain from base to tip.

    Walks backwards from *tipframe* to *baseframe* through parent-child
    joint relationships and returns an ordered list of URDFStep
    objects representing each joint.

    Arguments
    ---------
    node : Node
        ROS node for logging.
    html : str
        Raw URDF XML string.
    baseframe : str
        Name of the base link.
    tipframe : str
        Name of the tip link.

    Returns
    -------
    list[URDFStep]
        Ordered joint steps from base to tip.
    """
    robot = Robot.from_xml_string(html)
    _info(node, "Processing URDF for robot '%s'" % robot.name)
    _info(node, "Building chain from '%s' to '%s'" % (baseframe, tipframe))

    chain = []

    frame = tipframe
    while frame != baseframe:
        joints = [j for j in robot.joints if j.child == frame]
        if len(joints) == 0:
            _error(node, "Unable find joint connecting to '%s'" % frame)
        elif len(joints) != 1:
            _error(node, "Unable find unique joint connecting to '%s'" % frame)
        joint = joints[0]

        if joint.parent == frame:
            _error(node, "Joint '%s' connects '%s' to itself" % (joint.name, frame))
        frame = joint.parent

        if joint.type == 'revolute' or joint.type == 'continuous':
            type = JointType.REVOLUTE
        elif joint.type == 'prismatic':
            type = JointType.LINEAR
        elif joint.type == 'fixed':
            type = JointType.FIXED
        else:
            _error(node, "Joint '%s' has unknown type '%s'" % (joint.name, joint.type))

        if type is JointType.FIXED:
            nlocal = None
        else:
            nlocal = tu.n_from_URDF_axis(joint.axis)
            mag = np.sqrt(np.inner(nlocal, nlocal))
            if abs(mag - 1) > 1e-6:
                _info(node, "WARNING Joint '%s' axis needed normalization" % joint.name)
            nlocal = nlocal / mag

        if joint.origin is None:
            _info(node, "WARNING Joint '%s' has no <origin>" % joint.name)
            Tshift = tu.Teye()
        else:
            Tshift = tu.T_from_URDF_origin(joint.origin)

        chain.insert(0, URDFStep(name=joint.name, type=type, Tshift=Tshift, nlocal=nlocal))

    return chain


def _damped_pinv(J: np.ndarray, gamma: float = 0.001) -> np.ndarray:
    """Damped pseudo-inverse of Jacobian J via SVD.

    Singular values below gamma are regularised to avoid
    numerical instability near kinematic singularities.

    Arguments
    ---------
    J : np.ndarray
        Jacobian matrix (m x n).
    gamma : float
        Damping threshold.  Values below gamma are scaled by
        s / gamma**2 instead of inverted.

    Returns
    -------
    np.ndarray
        Damped pseudo-inverse (n x m).
    """
    U, S, Vt = np.linalg.svd(J)
    S_plus = np.zeros((len(Vt), len(U.T)))
    for i in range(len(S)):
        if gamma <= 0.0:
            S_plus[i][i] = 1 / S[i] if S[i] != 0.0 else 0.0
        else:
            if S[i] < gamma:
                S_plus[i][i] = S[i] / (gamma**2)
            else:
                S_plus[i][i] = 1 / S[i]
    return Vt.T @ S_plus @ U.T


# ── Data Classes ─────────────────────────────────────────────────────

class JointType(enum.Enum):
    """Enum for URDF joint types supported by the kinematic chain."""

    FIXED = 0
    REVOLUTE = 1
    LINEAR = 2


class IKinConfig:
    """Base class for inverse-kinematics configuration."""

    pass


@dataclass
class NRIKinConfig(IKinConfig):
    """Newton-Raphson iterative IK configuration.

    Attributes
    ----------
    x_desired : np.ndarray
        Target task-space vector for the IK solver.
    q_guess : np.ndarray
        Initial joint-angle seed.  Defaults to Q_READY.
    max_iter : int
        Maximum Newton-Raphson iterations before giving up.
    tolerance : float
        Convergence threshold on task-space error norm.
    sim_dt : float
        Step size for the NR joint-angle update.
    gamma : float
        Damping factor for the pseudo-inverse.
    fkin_packager : callable
        Function (q) -> (x, J) that returns the task-space
        vector and corresponding Jacobian for the IK error loop.
    """

    x_desired: np.ndarray
    q_guess: np.ndarray = field(default_factory=lambda: Q_READY.copy())
    max_iter: int = IK_MAX_ITERATIONS
    tolerance: float = IK_TOLERANCE
    sim_dt: float = IK_SIM_DT
    gamma: float = 0.001
    fkin_packager: callable = None


@dataclass
class VInvIKinConfig(IKinConfig):
    """Velocity-based IK configuration for real-time trajectory tracking.

    Used per control-loop timestep to compute joint velocities that
    track a position interpolation path.

    Attributes
    ----------
    p_init : np.ndarray
        Interpolation start position (3D).
    p_final : np.ndarray
        Interpolation end position (3D).
    t : float
        Time elapsed in the active movement phase.
    duration : float
        Total movement duration.
    q_curr : np.ndarray
        Current joint angles q (6D).
    dt : float
        Control loop timestep.
    lambda_ : float
        Position-error feedback gain.
    pos_error : np.ndarray
        Accumulated position error (modified in-place each call).
    gamma : float
        Damping factor for the pseudo-inverse.
    """

    p_init: np.ndarray
    p_final: np.ndarray
    t: float
    duration: float
    q_curr: np.ndarray
    dt: float
    lambda_: float
    pos_error: np.ndarray
    gamma: float = 0.001


class URDFStep:
    """Single step (joint) in a parsed kinematic chain.

    Attributes
    ----------
    name : str
        URDF joint name.
    type : JointType
        Joint type (FIXED, REVOLUTE, or LINEAR).
    Tshift : np.ndarray
        4x4 transform T from the previous frame to this joint's frame.
    nlocal : np.ndarray or None
        Joint axis in the local frame.  None for FIXED joints.
    dof : int or None
        Index into the active-DOF vector.  None for FIXED joints.
    """

    def __init__(self, name, type, Tshift, nlocal):
        """Initialize a URDFStep."""
        self.name = name
        self.type = type
        self.Tshift = Tshift
        self.nlocal = nlocal
        self.dof = None


# ── KinematicChain ───────────────────────────────────────────────────

class KinematicChain:
    """Kinematic chain parsed from URDF for FK, Jacobian, and IK.

    Reads the URDF from /robot_description, extracts the joint chain
    between a base and tip frame, and provides forward-kinematics and
    inverse-kinematics methods.  Caches intermediate joint positions
    and transforms after each FK call for reuse.

    Attributes
    ----------
    node : Node
        ROS node used for URDF subscription and logging.
    chain : list[URDFStep]
        Ordered list of joint steps from base to tip.
    steps : int
        Total number of steps (including fixed joints).
    dofs : int
        Number of active (non-fixed) degrees of freedom.
    joint_positions : dict[str, np.ndarray]
        Cached world positions of each active joint (set by fkin).
    joint_transforms : dict[str, np.ndarray]
        Cached 4x4 world transforms of each active joint.
    tip_pos : np.ndarray
        Cached 3D tip position from the last fkin call.
    tip_rot : np.ndarray
        Cached 3x3 tip rotation from the last fkin call.
    """

    # ── Lifecycle ────────────────────────────────────────────────────

    def __init__(self, node, baseframe, tipframe, expectedjointnames):
        """Initialize the chain from a URDF on /robot_description.

        Arguments
        ---------
        node : Node
            ROS node for topic subscription and logging.
        baseframe : str
            URDF link name for the chain base.
        tipframe : str
            URDF link name for the chain tip.
        expectedjointnames : list[str]
            Expected active joint names.  Raises if the parsed chain
            does not match.
        """
        self.node = node

        html = _read_urdf(node)
        self.chain = _parse_urdf(node, html, baseframe, tipframe)
        self.steps = len(self.chain)

        dof = 0
        for step in self.chain:
            if step.type == JointType.FIXED:
                step.dof = None
            else:
                step.dof = dof
                dof += 1
        self.dofs = dof

        _info(node, "URDF has %d steps, %d active DOFs:" % (self.steps, self.dofs))
        for (i, step) in enumerate(self.chain):
            dof_str = "      " if step.dof is None else "DOF #%d" % step.dof
            _info(node, "Step #%d %-8s %s '%s'" % (i, step.type.name, dof_str, step.name))

        jointnames = [s.name for s in self.chain if s.type != JointType.FIXED]
        if jointnames != list(expectedjointnames):
            _error(node, "Chain does not match expected names: " + str(expectedjointnames))

    # ── Public API ───────────────────────────────────────────────────

    def fkin(self, q):
        """Compute forward kinematics for joint configuration q.

        Walks up the chain, accumulating the 4x4 transform T for each
        joint, and builds the 3xN linear (Jv) and angular (Jw) Jacobians.
        Caches joint_positions, joint_transforms, tip_pos,
        and tip_rot for later retrieval.

        Arguments
        ---------
        q : np.ndarray
            Joint angles/positions array (length == self.dofs).

        Returns
        -------
        ptip : np.ndarray
            3D tip position in world frame.
        Rtip : np.ndarray
            3x3 tip rotation matrix R in world frame.
        Jv : np.ndarray
            3xN linear velocity Jacobian.
        Jw : np.ndarray
            3xN angular velocity Jacobian.
        """
        if len(q) != self.dofs:
            _error(self.node, "Given %d joint angles, expected %d" % (len(q), self.dofs))

        type_list = []
        p_list = []
        n_list = []

        joint_positions = {}
        joint_transforms = {}

        T = tu.Teye()

        for step in self.chain:
            if step.type is JointType.REVOLUTE:
                T = T @ step.Tshift @ tu.T_from_Rp(tu.Rotn(step.nlocal, q[step.dof]), tu.pzero())
            elif step.type is JointType.LINEAR:
                T = T @ step.Tshift @ tu.T_from_Rp(tu.Reye(), step.nlocal * q[step.dof])
            else:
                T = T @ step.Tshift

            if step.type != JointType.FIXED:
                type_list.append(step.type)
                p_list.append(tu.p_from_T(T))
                n_list.append(tu.R_from_T(T) @ step.nlocal)
                joint_positions[step.name] = tu.p_from_T(T)
                joint_transforms[step.name] = T.copy()

        ptip = tu.p_from_T(T)
        Rtip = tu.R_from_T(T)

        self.joint_positions = joint_positions
        self.joint_transforms = joint_transforms
        self.tip_pos = ptip
        self.tip_rot = Rtip

        Jv = np.zeros((3, self.dofs))
        Jw = np.zeros((3, self.dofs))
        for i in range(self.dofs):
            if type_list[i] is JointType.REVOLUTE:
                Jv[:, i] = tu.cross(n_list[i], (ptip - p_list[i]))
                Jw[:, i] = n_list[i]
            elif type_list[i] is JointType.LINEAR:
                Jv[:, i] = n_list[i]
                Jw[:, i] = tu.pzero()

        return (ptip, Rtip, Jv, Jw)

    def fkin_all(self, q, recompute=False):
        """Return cached joint positions, optionally recomputing FK.

        Arguments
        ---------
        q : np.ndarray
            Joint angles (ignored when *recompute* is False).
        recompute : bool
            If True, call fkin(q) before returning positions.

        Returns
        -------
        dict[str, np.ndarray]
            Mapping of joint name to 3D world position.
        """
        if recompute:
            self.fkin(q)
        return self.joint_positions

    def ikin(self, config: IKinConfig) -> tuple[np.ndarray, np.ndarray]:
        """Dispatch to the appropriate IK solver based on config type.

        Arguments
        ---------
        config : IKinConfig
            Either NRIKinConfig (iterative) or VInvIKinConfig
            (velocity-based).

        Returns
        -------
        q : np.ndarray
            Solved joint angles.
        qd : np.ndarray
            Joint velocities (zero for NR, computed for velocity IK).
        """
        if isinstance(config, NRIKinConfig):
            if not config.fkin_packager:
                config.fkin_packager = lambda q: (self.fkin(q)[0], self.fkin(q)[2])
            return self._nr_ikin(config)
        elif isinstance(config, VInvIKinConfig):
            return self._velocity_ikin(config)
        raise ValueError(f"Unknown IKinConfig type: {type(config)}")

    # ── Private Helpers ──────────────────────────────────────────────

    def _nr_ikin(self, config: NRIKinConfig) -> tuple[np.ndarray, np.ndarray]:
        """Newton-Raphson iterative IK solver.

        Iteratively updates joint angles q by stepping along the
        damped pseudo-inverse of the Jacobian until the task-space
        error drops below config.tolerance or the iteration
        limit is reached.

        Arguments
        ---------
        config : NRIKinConfig
            Solver configuration including target, seed, and limits.

        Returns
        -------
        q : np.ndarray
            Joint angles at convergence (or best effort).
        qd : np.ndarray
            Zero velocities.
        """
        q = config.q_guess.copy()
        p, J = config.fkin_packager(q)
        e = config.x_desired - p
        iter_ = 0
        while np.linalg.norm(e) > config.tolerance:
            if iter_ > config.max_iter:
                _info(self.node, "IK did not converge")
                return q, np.zeros_like(q)
            qd = _damped_pinv(J, config.gamma) @ e
            q = q + qd * config.sim_dt
            p, J = config.fkin_packager(q)
            e = config.x_desired - p
            iter_ += 1
        return q, np.zeros_like(q)

    def _velocity_ikin(self, config: VInvIKinConfig) -> tuple[np.ndarray, np.ndarray]:
        """Velocity-based IK for real-time trajectory tracking.

        Computes joint velocities qd (joint velocity) that track a
        quintic-spline position path, with proportional feedback on
        the accumulated position error.

        Arguments
        ---------
        config : VInvIKinConfig
            Velocity IK configuration with path endpoints and gains.

        Returns
        -------
        q : np.ndarray
            Updated joint angles.
        qd : np.ndarray
            Joint velocities.
        """
        (_, _, Jv, _) = self.fkin(config.q_curr)
        p, v, _ = goto5(config.t, config.duration, config.p_init, config.p_final)
        Jpinv = _damped_pinv(Jv, config.gamma)
        qd = Jpinv @ (v + config.lambda_ * config.pos_error)
        q = config.q_curr + qd * config.dt
        config.pos_error[:] = p - self.fkin(q)[0]
        return q, qd


# ── MultipleKinematicChains ──────────────────────────────────────────

class MultipleKinematicChains:
    """Compose multiple kinematic chains for robots with branching topology.

    Combines several KinematicChain objects (each potentially running
    in forward or reverse direction) into a single FK call that returns
    composed position, rotation, and full-width Jacobians.

    Attributes
    ----------
    chains : list[KinematicChain]
        Component chains.
    directions : list[int]
        +1 (forward) or -1 (reverse) for each chain.
    jointnames : list[str]
        Full list of joint names across all chains.
    n_joints : int
        Total number of joints.
    index_maps : list[list[int]]
        Per-chain mapping of chain-local DOF indices to full-robot
        joint indices.
    """

    # ── Lifecycle ────────────────────────────────────────────────────

    def __init__(self, elements, directions, jointnames):
        """Initialize from component chains.

        Arguments
        ---------
        elements : list[KinematicChain]
            Component kinematic chains.
        directions : list[int]
            +1 (forward) or -1 (reverse) for each chain.
        jointnames : list[str]
            Full list of joint names for the composed robot.
        """
        self.chains = elements
        self.directions = directions
        self.jointnames = jointnames
        self.n_joints = len(jointnames)

        self.index_maps = []
        for chain in self.chains:
            chain_indices = []
            for step in chain.chain:
                if step.type != 'FIXED' and step.dof is not None:
                    try:
                        idx = self.jointnames.index(step.name)
                        chain_indices.append(idx)
                    except ValueError:
                        print(f"Error: Joint '{step.name}' not found in jointnames list.")
            self.index_maps.append(chain_indices)

    # ── Public API ───────────────────────────────────────────────────

    def fkin(self, q):
        """Compute composed forward kinematics across all sub-chains.

        Arguments
        ---------
        q : np.ndarray
            Full joint-angle vector for the composed robot.

        Returns
        -------
        p : np.ndarray
            3D tip position.
        R : np.ndarray
            3x3 tip rotation matrix.
        Jv : np.ndarray
            3xN linear velocity Jacobian (full width).
        Jw : np.ndarray
            3xN angular velocity Jacobian (full width).
        """
        p_curr = tu.pzero()
        R_curr = tu.Reye()
        Jv_curr = np.zeros((3, self.n_joints))
        Jw_curr = np.zeros((3, self.n_joints))

        for chain, direction, indices in zip(self.chains, self.directions, self.index_maps):
            q_sub = np.array([q[k] for k in indices])
            (p_seg, R_seg, Jv_seg_local, Jw_seg_local) = chain.fkin(q_sub)

            Jv_seg = np.zeros((3, self.n_joints))
            Jw_seg = np.zeros((3, self.n_joints))
            if len(indices) > 0:
                Jv_seg[:, indices] = Jv_seg_local
                Jw_seg[:, indices] = Jw_seg_local

            if direction == -1:
                p_seg_inv = -R_seg.T @ p_seg
                R_seg_inv = R_seg.T
                Jw_seg_inv = -R_seg.T @ Jw_seg
                cross_term = np.cross(p_seg, Jw_seg, axis=0)
                Jv_seg_inv = -R_seg.T @ (Jv_seg + cross_term)

                p_seg = p_seg_inv
                R_seg = R_seg_inv
                Jv_seg = Jv_seg_inv
                Jw_seg = Jw_seg_inv

            p_next = p_curr + R_curr @ p_seg
            R_next = R_curr @ R_seg
            Jw_next = Jw_curr + R_curr @ Jw_seg
            p_rel = R_curr @ p_seg
            cross_term = np.cross(p_rel, Jw_curr, axis=0)
            Jv_next = Jv_curr + (R_curr @ Jv_seg) - cross_term

            p_curr = p_next
            R_curr = R_next
            Jv_curr = Jv_next
            Jw_curr = Jw_next

        return (p_curr, R_curr, Jv_curr, Jw_curr)

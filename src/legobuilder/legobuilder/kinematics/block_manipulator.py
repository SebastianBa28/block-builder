"""Higher-level manipulator wrapper providing FK/IK for block assembly.

Wraps KinematicChain with block-assembly-specific conventions:
gripper motor-radians-to-prismatic conversion, tilt/roll extraction
from joint angles, and a Newton-Raphson IK solver that operates in
a 4D task space [x, y, z, tilt].
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from math import pi

from legobuilder.kinematics.kinematic_chain import KinematicChain, JointType, _damped_pinv
from legobuilder.kinematics.transform_utils import T_from_Rp
from legobuilder.config import (
    GRIPPER_JOINT_INDEX, GRIPPER_MOTOR_OPEN_RAD, GRIPPER_MOTOR_CLOSED_RAD,
    TILT_JOINT_INDICES, ROLL_JOINT_INDICES,
    IK_MAX_ITERATIONS, IK_TOLERANCE, IK_SIM_DT,
    Q_READY, gripper_rad_to_meters, TILT_LIMITS
)


# ── ManipulatorState ─────────────────────────────────────────────────

@dataclass
class ManipulatorState:
    """Full state of the manipulator in task-space and joint-space.

    Captures end-effector pose (position p, orientation o, gripper),
    velocities, and the corresponding joint-space representation.

    Attributes
    ----------
    p : np.ndarray or None
        Tip position [x, y, z] in world frame.
    v : np.ndarray or None
        Tip linear velocity [vx, vy, vz].
    o : np.ndarray
        Orientation as [tilt, roll] in radians.
    gripper_open : bool
        Whether the gripper is commanded open.
    q : np.ndarray or None
        Joint angles (6D including gripper), in motor radians.
    qd : np.ndarray or None
        Joint velocities (6D).
    """

    p: Optional[np.ndarray] = None
    v: Optional[np.ndarray] = None
    o: np.ndarray = field(default_factory=lambda: np.zeros(2))
    gripper_open: bool = True
    q: Optional[np.ndarray] = None
    qd: Optional[np.ndarray] = None

    # ── Public API ───────────────────────────────────────────────────

    def get_task(self):
        """Return the 4D task-space vector [x, y, z, tilt].

        Raises
        ------
        ValueError
            If position or orientation is None, or tilt is out of
            configured limits.
        """
        if self.p is None or self.o is None:
            raise ValueError(
                "ManipulatorState is missing required fields "
                "for task-space representation."
            )

        if self.o[0] < TILT_LIMITS[0] or self.o[0] > TILT_LIMITS[1]:
            raise ValueError(
                f"Tilt angle {self.o[0]:.2f} is out of limits "
                f"{TILT_LIMITS}."
            )

        tilt = np.array([self.o[0]])
        return np.concatenate((self.p, tilt))

    def copy(self):
        """Return a deep copy of this state."""
        return ManipulatorState(
            p=self.p.copy() if self.p is not None else None,
            v=self.v.copy() if self.v is not None else None,
            o=self.o.copy() if self.o is not None else None,
            gripper_open=self.gripper_open,
            q=self.q.copy() if self.q is not None else None,
            qd=self.qd.copy() if self.qd is not None else None
        )

    def close_gripper(self):
        """Set gripper joint to closed position in q."""
        self.q[GRIPPER_JOINT_INDEX] = GRIPPER_MOTOR_CLOSED_RAD
        self.gripper_open = False

    def open_gripper(self):
        """Set gripper joint to open position in q."""
        self.q[GRIPPER_JOINT_INDEX] = GRIPPER_MOTOR_OPEN_RAD
        self.gripper_open = True

    # ── Dunder Methods ───────────────────────────────────────────────

    def __str__(self):
        """Return a human-readable string representation."""
        return (
            f"ManipulatorState(p={self.p}, v={self.v}, "
            f"o={self.o}, gripper_open={self.gripper_open}, "
            f"q={self.q}, qd={self.qd})"
        )


# ── BlockManipulator ─────────────────────────────────────────────────

class BlockManipulator:
    """Block-assembly manipulator wrapping a kinematic chain.

    Provides forward and inverse kinematics with automatic gripper
    motor-to-prismatic conversion, tilt/roll extraction, and an
    optional camera kinematic chain for end-effector camera poses.

    Attributes
    ----------
    node : Node
        ROS node (from the tip chain) used for logging.
    logger : rclpy.impl.rcutils_logger.RcutilsLogger
        Convenience reference to the node's logger.
    tilt_joint_indices : list[int]
        Joint indices that contribute to end-effector tilt.
    roll_joint_indices : list[int]
        Joint indices that contribute to end-effector roll.
    state : ManipulatorState
        Cached result from the last fkin call.
    Jv : np.ndarray
        Cached 3x6 linear velocity Jacobian.
    Jw : np.ndarray
        Cached 1x6 tilt Jacobian.
    """

    # ── Lifecycle ────────────────────────────────────────────────────

    def __init__(
        self,
        tip_chain: KinematicChain,
        camera_chain: KinematicChain = None,
    ):
        """Initialize with pre-built kinematic chains.

        Arguments
        ---------
        tip_chain : KinematicChain
            Chain from world to the gripper tip (6 DOFs).
        camera_chain : KinematicChain, optional
            Chain from world to camera_link (4 DOFs).  If provided,
            enables get_camera_transform.
        """
        self._tip_chain = tip_chain
        self._camera_chain = camera_chain
        self.node = tip_chain.node
        self.tilt_joint_indices = TILT_JOINT_INDICES
        self.roll_joint_indices = ROLL_JOINT_INDICES
        self.logger = tip_chain.node.get_logger()

        if camera_chain is not None:
            tip_names = [
                s.name for s in tip_chain.chain
                if s.type != JointType.FIXED
            ]
            cam_names = [
                s.name for s in camera_chain.chain
                if s.type != JointType.FIXED
            ]
            self._camera_dof_map = [
                tip_names.index(n) for n in cam_names
            ]

    # ── Public API ───────────────────────────────────────────────────

    def fkin_all(
        self, q: np.ndarray, recompute=False
    ) -> dict[str, np.ndarray]:
        """Return cached joint positions, optionally recomputing FK.

        Arguments
        ---------
        q : np.ndarray
            Joint angles (6D, motor radians).  Ignored when
            *recompute* is False.
        recompute : bool
            If True, run FK before returning positions.

        Returns
        -------
        dict[str, np.ndarray]
            Mapping of joint name to 3D world position.
        """
        if recompute:
            q_fk = q.copy()
            q_fk[GRIPPER_JOINT_INDEX] = gripper_rad_to_meters(
                q[GRIPPER_JOINT_INDEX]
            )
            self._tip_chain.fkin(q_fk)
        return self._tip_chain.joint_positions

    def fkin(
        self, q: np.ndarray, recompute=True
    ) -> tuple[ManipulatorState, np.ndarray, np.ndarray]:
        """Compute forward kinematics for joint configuration q.

        Converts the gripper from motor radians to URDF prismatic
        meters, runs the chain FK, and extracts tilt/roll from the
        joint angles.  Results are cached in self.state,
        self.Jv, and self.Jw.

        Arguments
        ---------
        q : np.ndarray
            Joint angles (6D including gripper), in motor radians.
        recompute : bool
            If False, return cached results without recomputing.

        Returns
        -------
        state : ManipulatorState
            Tip pose with position, orientation, and gripper state.
        Jv : np.ndarray
            3x6 linear velocity Jacobian.
        Jw : np.ndarray
            1x6 tilt Jacobian (d(tilt)/dq).
        """
        if not recompute:
            return self.state, self.Jv, self.Jw

        gripper_state = q[GRIPPER_JOINT_INDEX]

        q_fk = q.copy()
        q_fk[GRIPPER_JOINT_INDEX] = gripper_rad_to_meters(gripper_state)
        ptip, _, Jv, _ = self._tip_chain.fkin(q_fk)

        # Tilt and roll from joint angles
        # Signs account for 180-degree frame flips in URDF
        theta_tilt = q[1] - q[2] + q[3]
        theta_roll = q[0] + q[4]
        o = np.array([theta_tilt, theta_roll])

        # d(tilt)/dq = [0, +1, -1, +1, 0, 0]
        Jw = np.array([
            [0, 1, -1, 1, 0, 0],
        ])

        state = ManipulatorState(
            p=ptip,
            o=o,
            gripper_open=(gripper_state >= GRIPPER_MOTOR_OPEN_RAD),
            q=q
        )

        self.state = state
        self.Jv = Jv
        self.Jw = Jw

        self.logger

        return state, Jv, Jw

    def get_tip_transform(
        self, q: np.ndarray, recompute=False
    ) -> np.ndarray:
        """Return the 4x4 tip transform T in world coordinates.

        Arguments
        ---------
        q : np.ndarray
            Joint angles (6D).  Ignored when *recompute* is False.
        recompute : bool
            If True, run FK before returning the transform.

        Returns
        -------
        np.ndarray
            4x4 homogeneous transform T of the tip frame.
        """
        if recompute:
            q_fk = q.copy()
            q_fk[GRIPPER_JOINT_INDEX] = gripper_rad_to_meters(
                q[GRIPPER_JOINT_INDEX]
            )
            self._tip_chain.fkin(q_fk)
        return T_from_Rp(
            self._tip_chain.tip_rot, self._tip_chain.tip_pos
        )

    def get_camera_transform(
        self, q: np.ndarray, recompute=False
    ) -> np.ndarray:
        """Return the 4x4 camera_link transform T in world coordinates.

        Arguments
        ---------
        q : np.ndarray
            Joint angles (6D).  Ignored when *recompute* is False.
        recompute : bool
            If True, run FK before returning the transform.

        Returns
        -------
        np.ndarray
            4x4 homogeneous transform T of the camera link.
        """
        if recompute:
            q_cam = np.array(
                [q[i] for i in self._camera_dof_map]
            )
            self._camera_chain.fkin(q_cam)
        return T_from_Rp(
            self._camera_chain.tip_rot,
            self._camera_chain.tip_pos,
        )

    def ikin(
        self,
        desired_state: ManipulatorState,
        q_seed=None,
        visualize_elbow_down=True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Solve inverse kinematics for a desired end-effector pose.

        The gripper DOF (q[5]) is set directly from
        desired_state.gripper_open and held fixed while NR-IK
        solves for joints 0-3 in the 4D task space
        [x, y, z, tilt].  After convergence, q[4] (wrist_roll) is
        set analytically from the desired roll angle.

        Arguments
        ---------
        desired_state : ManipulatorState
            Target end-effector pose in world frame.
        q_seed : np.ndarray, optional
            6D initial joint configuration.  Defaults to Q_READY.
        visualize_elbow_down : bool
            If True and the solution has q[2] <= 0, display a 3D
            animation of the NR iteration history.

        Returns
        -------
        q : np.ndarray or None
            6D joint angles (including gripper).  None on failure.
        qd : np.ndarray or None
            6D zero velocities.  None on failure.
        """
        x = desired_state.get_task()
        gripper_q = (
            GRIPPER_MOTOR_OPEN_RAD if desired_state.gripper_open
            else GRIPPER_MOTOR_CLOSED_RAD
        )

        q = q_seed.copy() if q_seed is not None else Q_READY.copy()
        q[GRIPPER_JOINT_INDEX] = gripper_q

        p, J = self._fkin_packager(q)
        e = x - p

        history = [] if visualize_elbow_down else None

        iterations = 0
        for iterations in range(IK_MAX_ITERATIONS):
            if np.linalg.norm(e) <= IK_TOLERANCE:
                break
            if history is not None:
                history.append(
                    {k: v.copy()
                     for k, v in
                     self._tip_chain.joint_positions.items()}
                )
            qd_4 = _damped_pinv(J) @ e
            # Only update joints 0-3; gripper stays fixed
            q[:4] += qd_4 * IK_SIM_DT
            p, J = self._fkin_packager(q)
            e = x - p

        final_err = np.linalg.norm(e)
        if final_err > IK_TOLERANCE:
            self.node.get_logger().warning(
                f"IK did NOT converge in {iterations} iters: "
                f"err={final_err:.4f}"
            )
            return None, None
        else:
            self.node.get_logger().debug(
                f"IK converged in {iterations} iters: "
                f"final_err={final_err:.4f}"
            )

        if visualize_elbow_down and q[2] <= 0:
            history.append(
                {k: v.copy()
                 for k, v in
                 self._tip_chain.joint_positions.items()}
            )
            self._animate_ik(history, desired_state)

        # Enforce roll = desired_roll - base, wrapped to [-pi, pi]
        q[4] = desired_state.o[1] + q[0]
        if q[4] > pi:
            q[4] -= 2 * pi
        elif q[4] < -pi:
            q[4] += 2 * pi

        return q, np.zeros_like(q)

    # ── Private Helpers ──────────────────────────────────────────────

    def _fkin_packager(self, q):
        """Package FK output for the IK error-Jacobian loop.

        Extracts the 4D task vector [x, y, z, tilt] and the
        corresponding 4x4 Jacobian (position rows + tilt row,
        excluding wrist_roll and gripper columns) from a full FK
        call.  Used by ikin() to compute the NR error and step.

        Arguments
        ---------
        q : np.ndarray
            Joint angles (6D including gripper).  Only joints 0-3
            are used for the task-space Jacobian.

        Returns
        -------
        x : np.ndarray
            4D task vector [x, y, z, tilt].
        J : np.ndarray
            4x4 Jacobian mapping joint velocities to task-space
            velocities.
        """
        state, Jv, Jw = self.fkin(q)
        x = state.get_task()
        J = np.vstack((Jv[:, :4], Jw[:, :4]))
        return x, J

    def _animate_ik(self, history, desired_state=None):
        """Animate the NR-IK solve as a 3D arm moving through iterations.

        Arguments
        ---------
        history : list[dict[str, np.ndarray]]
            Per-iteration joint positions.
        desired_state : ManipulatorState, optional
            If provided, the target point is plotted as a red marker.
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        from matplotlib.animation import FuncAnimation

        n = len(history)
        max_frames = 200  # subsample for smooth playback
        if n > max_frames:
            indices = np.linspace(0, n - 1, max_frames, dtype=int)
            history = [history[i] for i in indices]

        joint_names = list(history[0].keys())

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        if desired_state is not None and desired_state.p is not None:
            ax.scatter(
                *desired_state.p,
                color='red', marker='x', s=100, label='Target',
            )

        line, = ax.plot(
            [], [], [], 'o-',
            color='steelblue', linewidth=2, markersize=5,
        )
        tip_marker, = ax.plot(
            [], [], [], 'o', color='green', markersize=8,
        )
        title = ax.set_title('')

        all_pts = np.array(
            [[pos for pos in frame.values()] for frame in history]
        )
        all_pts = all_pts.reshape(-1, 3)
        if desired_state is not None and desired_state.p is not None:
            all_pts = np.vstack([all_pts, desired_state.p])
        margin = 0.05
        for setter, idx in [
            (ax.set_xlim, 0), (ax.set_ylim, 1), (ax.set_zlim, 2)
        ]:
            lo = all_pts[:, idx].min() - margin
            hi = all_pts[:, idx].max() + margin
            setter(lo, hi)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_aspect('equal')

        origin = np.zeros(3)

        def update(frame_idx):
            """Update animation frame."""
            positions = history[frame_idx]
            pts = np.array(
                [origin] + [positions[name] for name in joint_names]
            )
            line.set_data(pts[:, 0], pts[:, 1])
            line.set_3d_properties(pts[:, 2])
            tip = pts[-1]
            tip_marker.set_data([tip[0]], [tip[1]])
            tip_marker.set_3d_properties([tip[2]])
            title.set_text(
                f'IK Iteration {frame_idx + 1}/{len(history)}'
            )
            return line, tip_marker, title

        anim = FuncAnimation(  # noqa: F841
            fig, update,
            frames=len(history), interval=30, blit=False,
        )
        ax.legend()
        plt.show()

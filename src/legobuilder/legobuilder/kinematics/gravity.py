"""Gravity compensation model for the 6-DOF robot arm.

Computes the torques needed at the shoulder, elbow, and wrist-tilt
joints to counteract gravitational loads on the arm's links and
end-effector assembly.  Uses a multi-body static-torque model with
precomputed moment coefficients.
"""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class WeightModel:
    """Multi-body gravity compensation model for a 5-DOF arm with gripper.

    Arm Diagram (zero position = straight up)
    ==========================================

             * ee (m_ee)
             |<-d_ee->(CoM)
      wrist  o  [M_w at joint]
             |
             |<-d2->(CoM)
             |  L2
      elbow  o  [M_e at joint]
             |
             |<-d1->(CoM)
             |  L1
    shoulder o
             |
            === base


    Variables
    ---------
    Symbol    Code field          Meaning
    ------    ----------          -------
    m1        lowerarm_mass       lowerarm link mass
    m2        upperarm_mass       upperarm link mass
    m_ee      ee_mass             end-effector assembly mass
    M_e       elbow_mass          elbow motor mass (sits at elbow joint)
    M_w       wrist_tilt_mass     wrist motor mass (sits at wrist joint)
    L1        lowerarm_length     lowerarm length
    L2        upperarm_length     upperarm length
    d1        lowerarm_com        lowerarm CoM distance from shoulder
    d2        upperarm_com        upperarm CoM distance from elbow
    d_ee      ee_com              ee CoM distance from wrist
    th1       s_th                shoulder absolute angle from vertical
    th12      s_th + e_th         upperarm absolute angle from vertical
    th123     s_th + e_th + w_th  ee absolute angle from vertical


    Derivation
    ----------
    Each joint must support everything distal to it.

    Effective supported masses (accumulate tip -> base):

        W3 = m_ee
        W2 = m2 + M_w + W3
        W1 = m1 + M_e + W2

    Each link produces two moment terms about its proximal joint:
      1. Link's own mass at its CoM:   m * g * d_com * sin(th_abs)
      2. All distal mass at link tip:  W_distal * g * L * sin(th_abs)

    Precomputed coefficients:

        C1 = g * (m1 * d1  + (W1 - m1) * L1)     <- shoulder
        C2 = g * (m2 * d2  + (W2 - m2) * L2)     <- elbow
        C3 = g * W3 * d_ee                        <- wrist

    Torques (accumulate wrist -> shoulder):

        tau_w = C3 * sin(th123)
        tau_e = C2 * sin(th12)  + tau_w
        tau_s = C1 * sin(th1)   + tau_e

    Motor command negates gravity torque:  tau_motor = -dir * tau_gravity

    Attributes
    ----------
    elbow_mass : float
        Mass of the elbow motor (X5-4), in kg.
    wrist_tilt_mass : float
        Mass of the wrist-tilt motor (X5-1), in kg.
    lowerarm_mass : float
        Mass of the lower arm link, in kg.
    upperarm_mass : float
        Mass of the upper arm link, in kg.
    ee_mass : float
        Combined mass of the end-effector assembly (bracket + gripper
        motor + depth camera), in kg.
    lowerarm_com : float
        Distance from shoulder joint to lower arm center of mass, in m.
    upperarm_com : float
        Distance from elbow joint to upper arm center of mass, in m.
    ee_com : float
        Effective CoM distance of the end-effector from the wrist, in m.
    lowerarm_length : float
        Lower arm link length from URDF, in m.
    upperarm_length : float
        Upper arm link length from URDF, in m.
    joint_directions : dict[str, int]
        Motor direction signs for each gravity-compensated joint.
    g : float
        Gravitational acceleration, in m/s^2.
    """

    # Motor masses (kg)
    elbow_mass: float = 0.360       # X5-4 motor
    wrist_tilt_mass: float = 0.330  # X5-1 motor

    # Link masses (kg)
    lowerarm_mass: float = 0.282    # ~10 g heavier than upperarm
    upperarm_mass: float = 0.272
    # 0.12 (wrist bracket + gripper) + 0.345 (X5-1 motor) + 0.12 (depth cam)
    ee_mass: float = 0.12 + 0.345 + 0.12

    # Center-of-mass distances from proximal joint (m)
    lowerarm_com: float = 0.178     # midpoint of 0.3556 m link
    upperarm_com: float = 0.152     # midpoint of 0.3048 m link
    ee_com: float = 0.1             # effective CoM of end-effector assembly

    # Link lengths from URDF (m)
    lowerarm_length: float = 0.3556
    upperarm_length: float = 0.3048

    joint_directions: dict[str, int] = field(default_factory=lambda: {
        'shoulder': 1,
        'elbow': -1,
        'wrist_tilt': 1
    })

    g: float = 9.81  # m/s^2

    # Precomputed moment coefficients (set in __post_init__)
    _C1: float = field(init=False, repr=False)
    _C2: float = field(init=False, repr=False)
    _C3: float = field(init=False, repr=False)

    def __post_init__(self):
        """Precompute moment coefficients from mass/length parameters."""
        self.wrist_tilt_support = self.ee_mass
        # Gap: 0.2 kg fudge factor compensates for unmodeled mass
        # (mounting hardware?). Empirically tuned -- removing it causes
        # gravity compensation undershoot. Original comment said 0.15
        # but code uses 0.2.
        self.elbow_support = (self.upperarm_mass + self.wrist_tilt_mass +
                              self.wrist_tilt_support + 0.3)
        self.shoulder_support = (self.lowerarm_mass + self.elbow_mass +
                                 self.elbow_support)

        distal_to_shoulder = self.shoulder_support - self.lowerarm_mass
        distal_to_elbow = self.elbow_support - self.upperarm_mass

        self._C1 = self.g * (self.lowerarm_mass * self.lowerarm_com
                             + distal_to_shoulder * self.lowerarm_length)
        self._C2 = self.g * (self.upperarm_mass * self.upperarm_com
                             + distal_to_elbow * self.upperarm_length)
        self._C3 = self.g * self.wrist_tilt_support * self.ee_com

    def compute_gravity_torques(
        self,
        q: np.ndarray,
        logger=None
    ) -> np.ndarray:
        """Compute gravity compensation torques for the full joint vector.

        Only the shoulder (index 1), elbow (index 2), and wrist-tilt
        (index 3) entries are non-zero; the remaining joints (base,
        wrist-roll, gripper) have no gravitational load in this model.

        Arguments
        ---------
        q : np.ndarray
            Joint angle array [base, shoulder, elbow, wrist_tilt,
            wrist_roll, gripper].
        logger : optional
            ROS logger for debug output.  If None, no logging occurs.

        Returns
        -------
        np.ndarray
            Torque array of same length as q.
        """
        s_d = self.joint_directions['shoulder']
        e_d = self.joint_directions['elbow']
        w_d = self.joint_directions['wrist_tilt']

        s_th = s_d * q[1]
        e_th = e_d * q[2]
        w_th = w_d * q[3]

        # Only sine matters because zero angle = straight up (vertical)
        sin_s = np.sin(s_th)
        sin_se = np.sin(s_th + e_th)
        sin_sew = np.sin(s_th + e_th + w_th)

        grav_wrist = self._C3 * sin_sew
        grav_elbow = self._C2 * sin_se + grav_wrist
        grav_shoulder = self._C1 * sin_s + grav_elbow

        tau_shoulder = -s_d * grav_shoulder
        tau_elbow = -e_d * grav_elbow
        tau_wrist = -w_d * grav_wrist

        if logger is not None:
            logger.info(
                f"th3={s_th + e_th + w_th:.2f}, tau3={tau_wrist:.2f}, "
                f"th2={s_th + e_th:.2f}, tau2={tau_elbow:.2f}, "
                f"th1={s_th:.2f}, tau1={tau_shoulder:.2f}"
            )

        tau = np.zeros(len(q))
        tau[1] = tau_shoulder
        tau[2] = tau_elbow
        tau[3] = tau_wrist
        return tau

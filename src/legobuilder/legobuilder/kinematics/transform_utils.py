"""Pure math utilities for 3D transforms, rotations, and ROS message conversion.

Provides factory and conversion functions for rotation matrices R (3x3),
homogeneous transforms T (4x4), quaternions (stored [x, y, z, w]),
axis-angle representations, and position/velocity vectors.  Also includes
bidirectional conversion between numpy arrays and ROS geometry_msgs types.
"""

import numpy as np

from geometry_msgs.msg import Point, Vector3, Quaternion, Pose, Transform, Twist


# ── Cross Products ────────────────────────────────────────────────────

def cross(a, b):
    """Cross product of two 3-vectors."""
    return crossmat(a) @ b


def crossmat(a):
    """Skew-symmetric cross-product matrix for 3-vector a."""
    return np.array([
        [0.0, -a[2], a[1]],
        [a[2], 0.0, -a[0]],
        [-a[1], a[0], 0.0]
    ])


# ── Position Vectors ─────────────────────────────────────────────────

def pzero():
    """Zero position vector p (3D)."""
    return np.zeros(3)


def pxyz(x, y, z):
    """Position vector p from scalar components."""
    return np.array([x, y, z])


# ── Unit Axis Vectors ────────────────────────────────────────────────

def nx():
    """Return the unit vector along the x-axis."""
    return nxyz(1.0, 0.0, 0.0)


def ny():
    """Return the unit vector along the y-axis."""
    return nxyz(0.0, 1.0, 0.0)


def nz():
    """Return the unit vector along the z-axis."""
    return nxyz(0.0, 0.0, 1.0)


def nxyz(x, y, z):
    """Return a normalized unit vector from components."""
    return np.array([x, y, z]) / np.sqrt(x * x + y * y + z * z)


# ── Generic Vectors ──────────────────────────────────────────────────

def vzero():
    """Zero 3-vector."""
    return np.zeros(3)


def vxyz(x, y, z):
    """3D vector from scalar components."""
    return np.array([x, y, z])


# ── Rotation Matrices ────────────────────────────────────────────────

def Reye():
    """Identity rotation matrix R (3x3)."""
    return np.eye(3)


def Rotx(alpha):
    """Rotation matrix R about the x-axis by angle alpha (radians)."""
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(alpha), -np.sin(alpha)],
        [0.0, np.sin(alpha), np.cos(alpha)]
    ])


def Roty(alpha):
    """Rotation matrix R about the y-axis by angle alpha (radians)."""
    return np.array([
        [np.cos(alpha), 0.0, np.sin(alpha)],
        [0.0, 1.0, 0.0],
        [-np.sin(alpha), 0.0, np.cos(alpha)]
    ])


def Rotz(alpha):
    """Rotation matrix R about the z-axis by angle alpha (radians)."""
    return np.array([
        [np.cos(alpha), -np.sin(alpha), 0.0],
        [np.sin(alpha), np.cos(alpha), 0.0],
        [0.0, 0.0, 1.0]
    ])


def Rotn(n, alpha):
    """Rotation matrix about arbitrary unit axis n by angle alpha.

    Uses the Rodrigues rotation formula.
    """
    nx = crossmat(n)
    return np.eye(3) + np.sin(alpha) * nx + (1.0 - np.cos(alpha)) * nx @ nx


# ── Interpolation ────────────────────────────────────────────────────

def pmid(p0, p1):
    """Midpoint between two position vectors."""
    return 0.5 * (p0 + p1)


def Rmid(R0, R1):
    """Rotation midpoint (interpolate halfway between R0 and R1)."""
    return Rinter(R0, R1, 0.5)


def pinter(p0, p1, s):
    """Linearly interpolate position: p0 + s * (p1 - p0)."""
    return p0 + (p1 - p0) * s


def vinter(p0, p1, sdot):
    """Velocity for linear position interpolation."""
    return (p1 - p0) * sdot


def Rinter(R0, R1, s):
    """Interpolate rotation from R0 toward R1 by factor s in [0, 1]."""
    (axis, angle) = axisangle_from_R(R0.T @ R1)
    return R0 @ Rotn(axis, s * angle)


def winter(R0, R1, sdot):
    """Angular velocity for rotation interpolation."""
    (axis, angle) = axisangle_from_R(R0.T @ R1)
    return R0 @ axis * angle * sdot


# ── Error Vectors ────────────────────────────────────────────────────

def ep(pd, p):
    """Position error vector (desired minus actual)."""
    return pd - p


def eR(Rd, R):
    """Rotation error vector from column-wise cross products of R and Rd."""
    return 0.5 * (cross(R[0:3, 0], Rd[0:3, 0])
                  + cross(R[0:3, 1], Rd[0:3, 1])
                  + cross(R[0:3, 2], Rd[0:3, 2]))


# ── Homogeneous Transforms (4x4) ────────────────────────────────────

def Teye():
    """Identity homogeneous transform T (4x4)."""
    return np.eye(4)


def T_from_Rp(R, p):
    """Build 4x4 homogeneous transform T from rotation R and position p."""
    return np.vstack((
        np.hstack((R, p.reshape((3, 1)))),
        np.array([0.0, 0.0, 0.0, 1.0])
    ))


def p_from_T(T):
    """Extract position vector p (3D) from transform T."""
    return T[0:3, 3]


def R_from_T(T):
    """Extract rotation matrix R (3x3) from transform T."""
    return T[0:3, 0:3]


# ── Quaternions ([x, y, z, w]) ───────────────────────────────────────

def quateye():
    """Identity quaternion [x, y, z, w]."""
    return np.array([0, 0, 0, 1])


def quat_from_xyzw(x, y, z, w):
    """Quaternion array from scalar components."""
    return np.array([x, y, z, w])


def quat_from_R(R):
    """Convert rotation matrix R to quaternion [x, y, z, w].

    Uses the numerically stable Shepperd method: picks the
    largest diagonal element to avoid division-by-zero.
    """
    A = [
        1.0 + R[0][0] + R[1][1] + R[2][2],
        1.0 + R[0][0] - R[1][1] - R[2][2],
        1.0 - R[0][0] + R[1][1] - R[2][2],
        1.0 - R[0][0] - R[1][1] + R[2][2]
    ]
    i = A.index(max(A))
    A = A[i]
    c = 0.5 / np.sqrt(A)
    if i == 0:
        q = c * np.array([R[2][1] - R[1][2], R[0][2] - R[2][0], R[1][0] - R[0][1], A])
    elif i == 1:
        q = c * np.array([A, R[1][0] + R[0][1], R[0][2] + R[2][0], R[2][1] - R[1][2]])
    elif i == 2:
        q = c * np.array([R[1][0] + R[0][1], A, R[2][1] + R[1][2], R[0][2] - R[2][0]])
    else:
        q = c * np.array([R[0][2] + R[2][0], R[2][1] + R[1][2], A, R[1][0] - R[0][1]])
    return q


def R_from_quat(quat):
    """Convert quaternion [x, y, z, w] to rotation matrix R (3x3)."""
    norm2 = np.inner(quat, quat)
    v = quat[0:3]
    w = quat[3]
    R = (2 / norm2) * (np.outer(v, v) + w * w * Reye() + w * crossmat(v)) - Reye()
    return R


# ── Axis-Angle ───────────────────────────────────────────────────────

def axisangle_from_R(R):
    """Extract (axis, angle) from rotation matrix R.

    Converts via quaternion for numerical stability.  Returns a zero
    axis vector when the angle is zero (identity rotation).

    Returns
    -------
    axis : np.ndarray
        Unit rotation axis (3D), or zeros if angle is zero.
    angle : float
        Rotation angle in radians.
    """
    quat = quat_from_R(R)
    v = quat[0:3]
    w = quat[3]
    n = np.sqrt(np.inner(v, v))
    angle = 2.0 * np.arctan2(n, w)
    if n == 0:
        axis = np.zeros(3)
    else:
        axis = np.array(v) / n
    return (axis, angle)


# ── Euler Angles (RPY) ──────────────────────────────────────────────

def R_from_RPY(roll, pitch, yaw):
    """Rotation matrix from extrinsic Tait-Bryan roll/pitch/yaw angles."""
    return Rotz(yaw) @ Roty(pitch) @ Rotx(roll)


# ── URDF Parsing Helpers ─────────────────────────────────────────────

def p_from_URDF_xyz(xyz):
    """Position vector p from URDF xyz attribute list."""
    return np.array(xyz)


def R_from_URDF_rpy(rpy):
    """Rotation matrix R from URDF rpy attribute list."""
    return R_from_RPY(rpy[0], rpy[1], rpy[2])


def T_from_URDF_origin(origin):
    """Homogeneous transform T from URDF origin element."""
    return T_from_Rp(R_from_URDF_rpy(origin.rpy), p_from_URDF_xyz(origin.xyz))


def n_from_URDF_axis(axis):
    """Joint axis vector from URDF axis element."""
    return np.array(axis)


# ── ROS Messages to NumPy ───────────────────────────────────────────

def p_from_Point(point):
    """Position vector p from a geometry_msgs/Point."""
    return pxyz(point.x, point.y, point.z)


def p_from_Vector3(vector3):
    """Position vector p from a geometry_msgs/Vector3."""
    return pxyz(vector3.x, vector3.y, vector3.z)


def quat_from_Quaternion(quaternion):
    """Quaternion array [x,y,z,w] from a geometry_msgs/Quaternion."""
    return np.array([quaternion.x, quaternion.y, quaternion.z, quaternion.w])


def R_from_Quaternion(quaternion):
    """Rotation matrix R from a geometry_msgs/Quaternion."""
    return R_from_quat(quat_from_Quaternion(quaternion))


def T_from_Pose(pose):
    """Homogeneous transform T from a geometry_msgs/Pose."""
    return T_from_Rp(R_from_Quaternion(pose.orientation), p_from_Point(pose.position))


def T_from_Transform(transform):
    """Homogeneous transform T from a geometry_msgs/Transform."""
    return T_from_Rp(R_from_Quaternion(transform.rotation), p_from_Vector3(transform.translation))


# ── NumPy to ROS Messages ───────────────────────────────────────────

def Point_from_p(p):
    """Geometry_msgs/Point from position vector p."""
    return Point(x=p[0], y=p[1], z=p[2])


def Vector3_from_v(v):
    """Geometry_msgs/Vector3 from 3-vector v."""
    return Vector3(x=v[0], y=v[1], z=v[2])


def Quaternion_from_quat(quat):
    """Geometry_msgs/Quaternion from quaternion array [x,y,z,w]."""
    return Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3])


def Quaternion_from_R(R):
    """Geometry_msgs/Quaternion from rotation matrix R."""
    return Quaternion_from_quat(quat_from_R(R))


def Pose_from_Rp(R, p):
    """Geometry_msgs/Pose from rotation matrix R and position vector p."""
    return Pose(position=Point_from_p(p), orientation=Quaternion_from_R(R))


def Pose_from_T(T):
    """Geometry_msgs/Pose from homogeneous transform T."""
    return Pose_from_Rp(R_from_T(T), p_from_T(T))


def Transform_from_Rp(R, p):
    """Geometry_msgs/Transform from rotation R and position p."""
    return Transform(translation=Vector3_from_v(p), rotation=Quaternion_from_R(R))


def Transform_from_T(T):
    """Geometry_msgs/Transform from homogeneous transform T."""
    return Transform_from_Rp(R_from_T(T), p_from_T(T))


def Twist_from_vw(v, w):
    """Geometry_msgs/Twist from linear velocity v and angular velocity w."""
    return Twist(linear=Vector3_from_v(v), angular=Vector3_from_v(w))

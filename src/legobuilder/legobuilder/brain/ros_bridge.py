"""
Pure-function ROS message serialization helpers for the Brain node.

Every function here takes plain data in and returns a ROS message object.
No ROS node reference, no publishers, no mutable state.
"""

import numpy as np

from geometry_msgs.msg import Point, Vector3
from std_msgs.msg import Header

from legobuilder_interfaces.msg import (
    IKRequestMsg,
    TrajectoryStateMsg,
    TrajectoryCommandMsg,
    GripCheckRequestMsg,
    ConnectionCheckRequestMsg,
)

from legobuilder_interfaces.msg import ContourInfoMsg

from legobuilder.kinematics.trajectory import TrajectoryState
from legobuilder.schemas import ObjectType, Object
from legobuilder.config import NUM_DOFS, JOINT_NAMES


# ── IK request / response helpers ────────────────────────────────────

def build_ik_request(
    trajectory_states: list[TrajectoryState],
    request_type: int,
    request_id: str,
    stamp,
    in_calibration: bool = False,
) -> IKRequestMsg:
    """Build an IKRequestMsg from trajectory states.

    Arguments
    ---------
    trajectory_states : list[TrajectoryState]
        Desired waypoints for the IK solver.
    request_type : int
        One of IKRequestMsg.REQUEST_PRIMARY, etc.
    request_id : str
        Unique string identifier for this request.
    stamp
        A ROS builtin_interfaces/Time message (e.g. from
        clock.now().to_msg()).

    Returns
    -------
    IKRequestMsg
        A fully populated message ready to publish.
    """
    msg = IKRequestMsg()
    msg.header.stamp = stamp
    msg.request_id = request_id
    msg.request_type = request_type

    positions = []
    tilts = []
    rolls = []
    gripper_open = []
    q_preset = []
    qd_preset = []
    needs_ik = []
    min_durations = []
    delay_befores = []
    delay_afters = []

    for state in trajectory_states:
        ms = state.final_state
        if ms.q is not None:
            needs_ik.append(False)
            q_preset.extend(ms.q.tolist())
            qd_preset.extend(
                ms.qd.tolist() if ms.qd is not None else [0.0] * NUM_DOFS
            )
            positions.extend([0.0, 0.0, 0.0])
            tilts.append(0.0)
            rolls.append(0.0)
            gripper_open.append(ms.gripper_open)
        else:
            needs_ik.append(True)
            positions.extend(ms.p.tolist())
            tilts.append(float(ms.o[0]))
            rolls.append(float(ms.o[1]))
            gripper_open.append(ms.gripper_open)
            q_preset.extend([0.0] * NUM_DOFS)
            qd_preset.extend([0.0] * NUM_DOFS)

        min_durations.append(
            float(state.min_duration) if state.min_duration is not None else 0.0
        )
        delay_befores.append(
            float(state.delay_before) if state.delay_before is not None else 0.0
        )
        delay_afters.append(
            float(state.delay_after) if state.delay_after is not None else 0.0
        )

    msg.positions = positions
    msg.tilts = tilts
    msg.rolls = rolls
    msg.gripper_open = gripper_open
    msg.q_preset = q_preset
    msg.qd_preset = qd_preset
    msg.needs_ik = needs_ik
    msg.min_durations = min_durations
    msg.delay_befores = delay_befores
    msg.delay_afters = delay_afters
    msg.in_calibration = in_calibration

    return msg


def apply_ik_solutions(
    trajectory_states: list[TrajectoryState],
    q_solutions: list[float],
    num_dofs: int = NUM_DOFS,
) -> None:
    """Fill solved joint angles from an IK response into trajectory states.

    Mutates *trajectory_states* in place (same semantics as
    BrainNode._apply_ik_solutions).

    Arguments
    ---------
    trajectory_states : list[TrajectoryState]
        The states whose final_state.q / final_state.qd
        will be filled.
    q_solutions : list[float]
        Flat list of joint angles returned by the IK solver
        (length = len(trajectory_states) * num_dofs).
    num_dofs : int
        Number of degrees of freedom per state.
    """
    for i, state in enumerate(trajectory_states):
        q = np.array(q_solutions[i * num_dofs:(i + 1) * num_dofs])
        state.final_state.q = q
        if state.final_state.qd is None:
            state.final_state.qd = np.zeros(num_dofs)


# ── Trajectory command ───────────────────────────────────────────────

def build_trajectory_command(
    states: list[TrajectoryState],
    command: int,
    stamp,
) -> TrajectoryCommandMsg:
    """Build a TrajectoryCommandMsg from a list of TrajectoryState objects.

    Arguments
    ---------
    states : list[TrajectoryState]
        Waypoints to pack into the message.
    command : int
        One of TrajectoryCommandMsg.COMMAND_APPEND_FRONT,
        COMMAND_APPEND_BACK, or COMMAND_REPLACE.
    stamp
        A ROS builtin_interfaces/Time message.

    Returns
    -------
    TrajectoryCommandMsg
        A fully populated message ready to publish.
    """
    msg = TrajectoryCommandMsg()
    msg.header = Header()
    msg.header.stamp = stamp
    msg.header.frame_id = 'world'
    msg.command = command

    for state in states:
        state_msg = TrajectoryStateMsg()

        state_msg.q = (
            [float(x) for x in state.final_state.q]
            if state.final_state.q is not None
            else [0.0] * len(JOINT_NAMES)
        )
        state_msg.qd = (
            [float(x) for x in state.final_state.qd]
            if state.final_state.qd is not None
            else [0.0] * len(JOINT_NAMES)
        )

        if state.final_state.p is not None:
            state_msg.p = Point(
                x=float(state.final_state.p[0]),
                y=float(state.final_state.p[1]),
                z=float(state.final_state.p[2]),
            )
        else:
            state_msg.p = Point()

        if state.final_state.v is not None:
            state_msg.pd = Vector3(
                x=float(state.final_state.v[0]),
                y=float(state.final_state.v[1]),
                z=float(state.final_state.v[2]),
            )
        else:
            state_msg.pd = Vector3()

        state_msg.min_duration = (
            float(state.min_duration) if state.min_duration is not None else 0.0
        )
        state_msg.delay_before = (
            float(state.delay_before) if state.delay_before is not None else 0.0
        )
        state_msg.delay_after = (
            float(state.delay_after) if state.delay_after is not None else 0.0
        )

        msg.states.append(state_msg)

    return msg


# ── Grip check ───────────────────────────────────────────────────────

def build_grip_check_request(
    request_id: str,
    stamp,
) -> GripCheckRequestMsg:
    """Build a GripCheckRequestMsg.

    Arguments
    ---------
    request_id : str
        Unique identifier for this grip check.
    stamp
        A ROS builtin_interfaces/Time message.

    Returns
    -------
    GripCheckRequestMsg
        A populated message ready to publish.
    """
    msg = GripCheckRequestMsg()
    msg.header.stamp = stamp
    msg.request_id = request_id
    return msg


# ── Connection check ────────────────────────────────────────────────

def build_connection_check_request(
    request_id: str,
    stamp,
) -> ConnectionCheckRequestMsg:
    """Build a ConnectionCheckRequestMsg.

    Arguments
    ---------
    request_id : str
        Unique identifier for this connection check.
    stamp
        A ROS builtin_interfaces/Time message.

    Returns
    -------
    ConnectionCheckRequestMsg
        A populated message ready to publish.
    """
    msg = ConnectionCheckRequestMsg()
    msg.header.stamp = stamp
    msg.request_id = request_id
    return msg


# ── Contour message conversion ───────────────────────────────────────

def _color_to_object_type(color):
    """Map a ContourInfoMsg color constant to an ObjectType enum value.

    Returns None for unrecognized colors.
    """
    if color == ContourInfoMsg.YELLOW:
        return ObjectType.YELLOW_BLOCK
    elif color == ContourInfoMsg.BLUE:
        return ObjectType.BLUE_BLOCK
    elif color == ContourInfoMsg.GREEN:
        return ObjectType.GREEN_BLOCK
    elif color == ContourInfoMsg.RED:
        return ObjectType.RED_BLOCK
    else:
        return None


def convert_contour_msgs_to_objects(
    contour_msgs: list[ContourInfoMsg],
    t_now: float,
    logger
) -> list[Object]:
    """Convert ROS ContourInfoMsg list to internal Object list.

    Arguments
    ---------
    contour_msgs : list[ContourInfoMsg]
        Raw contour messages from the detector node.
    t_now : float
        Current time in seconds (used as object timestamp).

    Returns
    -------
    list[Object]
        Domain objects with world-frame positions and types.
    """
    objects = []
    for msg in contour_msgs:
        if msg.shape_type == ContourInfoMsg.CIRCLE:
            obj = Object(
                t=t_now,
                frame_idx=msg.frame_idx,
                obj_type=ObjectType.DISK,
                center_xyz=(msg.center_x, msg.center_y, msg.center_z),
            )
        elif msg.shape_type == ContourInfoMsg.SQUARE:
            obj_type = _color_to_object_type(msg.color)
            if obj_type is None:
                continue
            corner_xys = list(zip(msg.corner_world_xs, msg.corner_world_ys))
            corner_uvs = list(zip(msg.corner_world_us, msg.corner_world_vs))
            face_conns_xyzs = [
                (x, y, z) for (x, y, z) in zip(msg.face_conns_xs, msg.face_conns_ys, msg.face_conns_zs)
            ]
            obj = Object(
                t=t_now,
                frame_idx=msg.frame_idx,
                obj_type=obj_type,
                center_uv=(msg.center_u, msg.center_v),
                center_xyz=(msg.center_x, msg.center_y, msg.center_z),
                angle=msg.angle,
                corner_xys=corner_xys,
                corner_uvs=corner_uvs,
                face_conns_xyzs=face_conns_xyzs,
                quaternion=(
                    msg.quaternion.x, msg.quaternion.y,
                    msg.quaternion.z, msg.quaternion.w,
                ),
            )
        else:
            continue
        objects.append(obj)
    return objects

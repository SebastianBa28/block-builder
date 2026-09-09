"""
Brain node for legobuilder.
Central coordinator between perception (Detector) and action (Manipulator).
Focused on ROS management: subscriptions, publishers, heartbeats.
"""
    
from collections import defaultdict
import rclpy
import numpy as np
from rclpy.node import Node

from geometry_msgs.msg import Point, Vector3
from sensor_msgs.msg import JointState
from std_msgs.msg import Header, Bool

from legobuilder_interfaces.msg import (
    ContourInfoArray,
    TrajectoryStateMsg,
    TrajectoryCommandMsg,
    TrajectoryCompleteMsg,
    ContactMsg,
    GripFailureMsg,
    PointCommandMsg,
    IKRequestMsg,
    IKResponseMsg,
    ScanRequestMsg,
)

from legobuilder.kinematics.trajectory import TrajectoryState
from legobuilder.kinematics.block_manipulator import ManipulatorState, BlockManipulator

from legobuilder.config import (
    Q_READY,
    JOINT_NAMES,
    NUM_DOFS,
    DURATION,
    COLLISION_WAIT_DURATION,
    HEARTBEAT_RATE,
    HEARTBEAT_TIMEOUT,
    GRIPPER_JOINT_INDEX,
    GRIPPER_MOTOR_CLOSED_RAD,
    GRIPPER_MOTOR_OPEN_RAD,
    DEMO_STRUCTURES,
    DEMO_STRUCTURE_INIT_HEIGHT,
    PILE_X_MIN,
    MAX_PENDING_OBJECTS,
    QUEUE_MOVEMENT_THRESHOLD,
    ARM_OCCLUSION_RADII,
    GRIP_FAILURE_RECOVERY,
    STRUCTURE_BUILD_DEMO,
    EE_DEPTH_SCAN,
    SCAN_INTERVAL,
    T_CAMERA_TIP_TRANSLATION,
    T_CAMERA_TIP_RPY,
)
from legobuilder.kinematics.transform_utils import T_from_Rp, R_from_RPY

from legobuilder.brain.process import BrainProcessor
from legobuilder.brain.block_structure import BlockStructure
from legobuilder.brain.structure_planner import StructureBuildPlanner
from legobuilder.schemas import ObjectType, Object


class BrainNode(Node):
    """
    Brain node that coordinates perception and action.
    Handles ROS communication; delegates processing to BrainProcessor.
    """

    def __init__(self, name):
        super().__init__(name)

        self.logger = self.get_logger()
        self.clock = self.get_clock()
        self.start_time = self.clock.now()

        # Initialize kinematic chain for IK computation
        self.block_manipulator = BlockManipulator(self, "world", "tip", JOINT_NAMES)
        
        # Initialize processor (handles all processing logic)
        self.processor = BrainProcessor(
            logger=self.logger,
            get_t_func=self.get_t,
            block_manipulator=self.block_manipulator
        )

        # Structure build demo
        self.structures: dict[ObjectType, BlockStructure] = {}
        if STRUCTURE_BUILD_DEMO:
            for obj_type, cfg in DEMO_STRUCTURES.items():
                s = BlockStructure(1, 1, 8, origin=cfg['origin'], roll=cfg['roll_deg'] * np.pi / 180)
                for _ in range(DEMO_STRUCTURE_INIT_HEIGHT):
                    s.place_top_block(0, 0, obj_type)
                self.structures[obj_type] = s
                self.logger.info(f"Initialized {obj_type.name} structure at {cfg['origin']}, roll={cfg['roll_deg']}deg, height={s.get_height(0, 0)}")

        self.structure_planner = StructureBuildPlanner(self.logger) if STRUCTURE_BUILD_DEMO else None
        self.pending_placement: tuple[ObjectType, BlockStructure] | None = None

        # State tracking
        self.executing_trajectory = False
        self.pending_objects: list[Object] = []
        self.total_objects_in_batch = 0
        self.precomputed: tuple[Object, list[TrajectoryState]] | None = None
        self.occlusion_footprint: list[tuple[float, float, float]] = []  # [(x, y, radius), ...]
        self._validation_grace_until: float = 0.0

        # Async IK state tracking
        self._ik_request_counter = 0
        self._primary_ik_id: str | None = None       # request_id of pending primary IK
        self._primary_ik_data: dict | None = None     # {obj, trajectory_states}
        self._precompute_ik_id: str | None = None     # request_id of pending precompute IK
        self._precompute_ik_data: dict | None = None  # {obj, trajectory_states}

        # Node online status tracking
        self.is_manipulator_online = False
        self.is_detector_online = False
        self.manipulator_last_seen = None
        self.detector_last_seen = None

        # Heartbeat publisher
        self.pub_heartbeat = self.create_publisher(Bool, name + '/heartbeat', 10)

        # Heartbeat subscribers
        self.create_subscription(Bool, 'manipulator/heartbeat', self.recv_manipulator_heartbeat, 10)
        self.create_subscription(Bool, 'detector/heartbeat', self.recv_detector_heartbeat, 10)

        # Heartbeat timer
        self.create_timer(1.0 / HEARTBEAT_RATE, self.heartbeat_tick)

        # Publishers
        self.pub_trajectory_command = self.create_publisher(
            TrajectoryCommandMsg,
            name + '/trajectory_command',
            10
        )
        self.pub_ik_request = self.create_publisher(
            IKRequestMsg, 'brain/ik_request', 10
        )

        # Subscribers
        self.create_subscription(
            IKResponseMsg, 'ik_solver/ik_response', self.recv_ik_response, 10
        )
        self.create_subscription(
            ContourInfoArray,
            'detector/contour_info',
            self.recv_contours,
            10
        )

        self.create_subscription(
            TrajectoryCompleteMsg,
            'manipulator/trajectory_complete',
            self.recv_trajectory_complete,
            10
        )

        self.create_subscription(
            ContactMsg,
            'manipulator/contact',
            self.recv_contact,
            10
        )

        self.create_subscription(
            GripFailureMsg,
            'manipulator/grip_failure',
            self.recv_grip_failure,
            10
        )

        # Point command subscriber
        self.create_subscription(PointCommandMsg, '/point', self.recvpoint, 10)

        # ── 3D World Mapping (scan triggering) ───────────────────────
        if EE_DEPTH_SCAN:
            self._current_q = None
            self._scan_count = 0
            self._T_camera_tip = T_from_Rp(
                R_from_RPY(*T_CAMERA_TIP_RPY),
                np.array(T_CAMERA_TIP_TRANSLATION, dtype=float),
            )

            self.create_subscription(JointState, '/joint_states', self._recv_joint_states, 1)
            self.pub_scan_request = self.create_publisher(ScanRequestMsg, name + '/scan_request', 10)
            self.create_timer(SCAN_INTERVAL, self._scan_callback)

        self.logger.info("Brain node initialized and running...")

    def recv_manipulator_heartbeat(self, msg: Bool):
        """Callback for manipulator heartbeat."""
        was_online = self.is_manipulator_online
        self.is_manipulator_online = True
        self.manipulator_last_seen = self.get_t()
        self.logger.debug(
            f"[RECV manipulator/heartbeat]"
            f" t={self.manipulator_last_seen:.4f}"
        )
        if not was_online:
            self.logger.info("Manipulator is now online")

    def recv_detector_heartbeat(self, msg: Bool):
        """Callback for detector heartbeat."""
        was_online = self.is_detector_online
        self.is_detector_online = True
        self.detector_last_seen = self.get_t()
        self.logger.debug(
            f"[RECV detector/heartbeat]"
            f" t={self.detector_last_seen:.4f}"
        )
        if not was_online:
            self.logger.info("Detector is now online")

    def heartbeat_tick(self):
        """Publish heartbeat and check for timed-out nodes."""
        t = self.get_t()

        # Publish our heartbeat
        self.pub_heartbeat.publish(Bool(data=True))
        self.logger.debug(f"[PUB brain/heartbeat] t={t:.4f}")

        # Check for timed-out nodes
        if self.is_manipulator_online and self.manipulator_last_seen is not None:
            if t - self.manipulator_last_seen > HEARTBEAT_TIMEOUT:
                self.logger.warn("Manipulator went offline")
                self.is_manipulator_online = False

        if self.is_detector_online and self.detector_last_seen is not None:
            if t - self.detector_last_seen > HEARTBEAT_TIMEOUT:
                self.logger.warn("Detector went offline")
                self.is_detector_online = False

    # ── 3D World Mapping ──────────────────────────────────────────────

    def _recv_joint_states(self, msg: JointState):
        """Cache current joint angles for FK-based camera pose computation."""
        self._current_q = np.array(msg.position)

    def _scan_callback(self):
        """
        Timer callback: trigger the detector to integrate a depth frame.

        Computes the camera-to-world transform from FK + mounting offset,
        and publishes a ScanRequestMsg to the detector.
        """
        # Skip if arm is moving (depth image would have motion blur)
        if self.executing_trajectory:
            return
        if self._current_q is None:
            return
        
        self.logger.info("Triggering depth scan from end-effector camera")

        T_world_tip = self.block_manipulator.get_tip_transform(self._current_q)
        T_world_camera = T_world_tip @ self._T_camera_tip

        msg = ScanRequestMsg()
        msg.request_id = str(self._scan_count)
        msg.camera_to_world = T_world_camera.flatten().tolist()
        self.pub_scan_request.publish(msg)

        self._scan_count += 1
        self.logger.debug(
            f"[PUB brain/scan_request] id={msg.request_id}"
        )

    def recv_trajectory_complete(self, msg: TrajectoryCompleteMsg):
        """Callback for receiving trajectory completion from manipulator.
        Chains to the next pending object if any remain."""
        self.logger.info(
            f"Trajectory completed. States executed: {msg.states_executed}"
        )
        self.logger.debug(
            f"[RECV manipulator/trajectory_complete]"
            f" states_executed={msg.states_executed}"
            f" t={self.get_t():.4f}"
        )

        # Update structure after successful placement
        if self.pending_placement is not None:
            obj_type, structure = self.pending_placement
            if structure is not None:
                structure.place_top_block(0, 0, obj_type)
                self.logger.info(
                    f"Updated {obj_type.name} structure: height now {structure.get_height(0, 0)}"
                )
            self.pending_placement = None

        # Clear stale precompute IK if a new primary request will be sent
        self._precompute_ik_id = None
        self._precompute_ik_data = None

        self._process_next_object()

    def recv_contact(self, msg: ContactMsg):
        """Callback for receiving contact/collision from manipulator."""
        self.logger.info(f"Contact detected: type={msg.contact_type}")
        self.logger.info(f"Position error: {list(msg.position_error)}")
        self.logger.info(f"States remaining: {msg.states_remaining}")

        # Clear pending objects, placement, and async IK — don't continue after collision
        self.pending_objects = []
        self.pending_placement = None
        self.precomputed = None
        self.occlusion_footprint = []
        self._primary_ik_id = None
        self._primary_ik_data = None
        self._precompute_ik_id = None
        self._precompute_ik_data = None

        # Send recovery trajectory to ready position
        recovery_state = TrajectoryState(
            final_state=ManipulatorState(q=Q_READY.copy(), qd=np.zeros(len(JOINT_NAMES))),
            min_duration=DURATION,
            delay_before=COLLISION_WAIT_DURATION,
            delay_after=0.0,
        )
        self._publish_trajectory_command([recovery_state], TrajectoryCommandMsg.COMMAND_REPLACE)
        self.executing_trajectory = True

    def recv_grip_failure(self, msg: GripFailureMsg):
        """Callback for grip failure from manipulator."""
        self.logger.info(
            f"Grip failure detected: gripper_pos={msg.gripper_position:.4f}"
            f" states_remaining={msg.states_remaining}"
        )

        if not GRIP_FAILURE_RECOVERY:
            self.logger.info("GRIP_FAILURE_RECOVERY disabled, ignoring.")
            return

        # Clear pending placement so structure height is NOT incremented
        self.pending_placement = None

        # Clear pending objects — positions may be stale
        self.pending_objects = []

        # Clear all async IK state
        self.precomputed = None
        self.occlusion_footprint = []
        self._primary_ik_id = None
        self._primary_ik_data = None
        self._precompute_ik_id = None
        self._precompute_ik_data = None

        # Send recovery trajectory to Q_READY (replaces current trajectory)
        recovery_state = TrajectoryState(
            final_state=ManipulatorState(
                q=Q_READY.copy(),
                qd=np.zeros(len(JOINT_NAMES)),
            ),
            min_duration=DURATION,
            delay_before=0.0,
            delay_after=0.0,
        )
        self._publish_trajectory_command(
            [recovery_state], TrajectoryCommandMsg.COMMAND_REPLACE
        )
        self.executing_trajectory = True
        self.logger.info("Grip failure recovery: returning to Q_READY")

    def get_t(self):
        """Get current time in seconds since node start."""
        now = self.clock.now()
        return (now - self.start_time).nanoseconds * 1e-9

    def recv_contours(self, msg: ContourInfoArray):
        """
        Callback for receiving contour info from detector.
        Always runs perception. During execution, validates the queue.
        When idle, populates the queue and starts processing.
        """
        self.logger.debug(
            f"[RECV detector/contour_info]"
            f" n_contours={len(msg.contours)}"
            f" executing={self.executing_trajectory}"
            f" manip_online={self.is_manipulator_online}"
        )
        if not self.is_manipulator_online:
            return

        # Always run perception — never gate on executing_trajectory
        objects = self.processor.convert_contour_msgs_to_objects(msg.contours)
        detected_objects = self.processor.process_objects(objects)

        if not STRUCTURE_BUILD_DEMO:
            return

        if not detected_objects:
            if self.executing_trajectory and self.get_t() >= self._validation_grace_until:
                self._validate_queue([])
            return

        # Filter to only pile blocks (right side of robot)
        detected_pile = [
            obj for obj in detected_objects
            if obj.center_xyz[0] > PILE_X_MIN and obj.obj_type.is_block()
        ]

        if self.executing_trajectory or self._primary_ik_id is not None:
            # Actively executing or waiting for IK — validate queue after grace period
            if self.get_t() >= self._validation_grace_until:
                self._validate_queue(detected_pile)
        else:
            if not detected_pile:
                return

            self.logger.info(f"Detected {len(detected_pile)} pile blocks")
            type_counts = defaultdict(int)
            for obj in detected_pile:
                type_counts[obj.obj_type] += 1
            for obj_type, count in type_counts.items():
                self.logger.info(f"  {obj_type.name}: {count}")

            self.pending_objects = detected_pile[:MAX_PENDING_OBJECTS]
            self.total_objects_in_batch = len(self.pending_objects)
            self._process_next_object()

    def _process_next_object(self):
        """
        Process the next pending object. Uses precomputed trajectory if valid,
        otherwise plans trajectory and sends an async IK request (returns immediately).
        """
        if not self.pending_objects:
            self.executing_trajectory = False
            self.occlusion_footprint = []
            return

        # If a precomputed trajectory is ready, use it immediately
        if self.precomputed is not None:
            obj, trajectory_states = self.precomputed
            if obj in self.pending_objects:
                self.pending_objects.remove(obj)
            self.precomputed = None
            self.logger.info(
                f"Using precomputed trajectory for {obj.obj_type.name}"
                f" at ({obj.center_xyz[0]:.3f}, {obj.center_xyz[1]:.3f})"
            )
            self._finalize_and_publish(obj, trajectory_states)
            return

        # Plan trajectory (fast — no IK), then send async IK request
        obj = self.pending_objects.pop(0)
        self.logger.info(
            f"Processing {obj.obj_type.name}"
            f" at ({obj.center_xyz[0]:.3f}, {obj.center_xyz[1]:.3f})"
        )

        trajectory_states = self.structure_planner.plan_pick_and_place(obj, self.structures)
        if not trajectory_states:
            self.logger.info("No trajectory planned for this object, skipping.")
            self._process_next_object()
            return

        # Send async IK request — returns immediately, no blocking
        request_id = self._send_ik_request(
            trajectory_states, IKRequestMsg.REQUEST_PRIMARY
        )
        self._primary_ik_id = request_id
        self._primary_ik_data = {
            'obj': obj,
            'trajectory_states': trajectory_states,
        }
        self.logger.info(f"Sent primary IK request {request_id} for {obj.obj_type.name}")

    def _finalize_and_publish(self, obj: Object, trajectory_states: list[TrajectoryState]):
        """Publish a fully solved trajectory and kick off precompute."""
        self.pending_placement = (obj.obj_type, self.structures.get(obj.obj_type))
        self._publish_trajectory_command(trajectory_states, TrajectoryCommandMsg.COMMAND_REPLACE)
        self.executing_trajectory = True
        # Grace period covers pick phase + transit to structure (states 1–5 of 9).
        # After transit, the arm is far from the pile and detection is reliable.
        n_grace = (len(trajectory_states) + 1) // 2  # 5 for 9 states
        grace_duration = sum(
            (s.min_duration or 0) + s.delay_before + s.delay_after
            for s in trajectory_states[:n_grace]
        )
        self._validation_grace_until = self.get_t() + grace_duration

        self.logger.info(f"Published {len(trajectory_states)} trajectory states")

        # Compute occlusion footprint from published trajectory
        self._compute_occlusion_footprint(trajectory_states)

        # Speculatively precompute next
        self._precompute_next()

    def _send_ik_request(
        self,
        trajectory_states: list[TrajectoryState],
        request_type: int,
    ) -> str:
        """
        Build and publish an IKRequestMsg for the given trajectory states.
        Returns the request_id assigned to this request.
        """
        self._ik_request_counter += 1
        request_id = f"ik_{self._ik_request_counter}"

        msg = IKRequestMsg()
        msg.header.stamp = self.clock.now().to_msg()
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
                # Already has joint solution — pass through
                needs_ik.append(False)
                q_preset.extend(ms.q.tolist())
                qd_preset.extend(
                    ms.qd.tolist() if ms.qd is not None else [0.0] * NUM_DOFS
                )
                # Fill placeholders for Cartesian fields
                positions.extend([0.0, 0.0, 0.0])
                tilts.append(0.0)
                rolls.append(0.0)
                gripper_open.append(ms.gripper_open)
            else:
                # Needs IK
                needs_ik.append(True)
                positions.extend(ms.p.tolist())
                tilts.append(float(ms.o[0]))
                rolls.append(float(ms.o[1]))
                gripper_open.append(ms.gripper_open)
                # Fill placeholders for preset fields
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

        self.pub_ik_request.publish(msg)
        return request_id

    def recv_ik_response(self, msg: IKResponseMsg):
        """Callback for IK solver response. Fills in joint solutions and continues pipeline."""
        req_id = msg.request_id
        req_type = msg.request_type

        if req_type == IKRequestMsg.REQUEST_PRIMARY:
            if req_id != self._primary_ik_id:
                self.logger.info(f"Discarding stale primary IK response {req_id}")
                return
            if not msg.success:
                self.logger.error(f"Primary IK request {req_id} failed")
                self._primary_ik_id = None
                self._primary_ik_data = None
                self._process_next_object()
                return

            data = self._primary_ik_data
            self._primary_ik_id = None
            self._primary_ik_data = None

            trajectory_states = data['trajectory_states']
            obj = data['obj']

            # Fill in solved joint angles
            self._apply_ik_solutions(trajectory_states, msg.q_solutions)

            self._finalize_and_publish(obj, trajectory_states)

        elif req_type == IKRequestMsg.REQUEST_PRECOMPUTE:
            if req_id != self._precompute_ik_id:
                self.logger.info(f"Discarding stale precompute IK response {req_id}")
                return
            if not msg.success:
                self.logger.error(f"Precompute IK request {req_id} failed")
                self._precompute_ik_id = None
                self._precompute_ik_data = None
                return

            data = self._precompute_ik_data
            self._precompute_ik_id = None
            self._precompute_ik_data = None

            obj = data['obj']
            trajectory_states = data['trajectory_states']

            # Check if the object is still in the queue
            if obj not in self.pending_objects:
                self.logger.info(
                    f"Precomputed object {obj.obj_type.name} no longer in queue, discarding"
                )
                return

            # Undo temporary structure height bump if we applied one
            if data.get('structure_bumped'):
                structure = self.structures.get(obj.obj_type)
                if structure is not None:
                    top_layer = structure.get_height(0, 0) - 1
                    structure.remove_block(0, 0, top_layer)

            # Fill in solved joint angles
            self._apply_ik_solutions(trajectory_states, msg.q_solutions)

            self.precomputed = (obj, trajectory_states)
            self.logger.info(
                f"Precomputed {len(trajectory_states)} states for {obj.obj_type.name}"
            )
        else:
            self.logger.warn(f"Unknown IK response type {req_type}, discarding")

    def _apply_ik_solutions(
        self,
        trajectory_states: list[TrajectoryState],
        q_solutions: list[float],
    ):
        """Fill solved joint angles from IK response into trajectory states."""
        for i, state in enumerate(trajectory_states):
            q = np.array(q_solutions[i * NUM_DOFS:(i + 1) * NUM_DOFS])
            state.final_state.q = q
            if state.final_state.qd is None:
                state.final_state.qd = np.zeros(NUM_DOFS)

    def _validate_queue(self, current_detections: list[Object], verbose: bool = False):
        """
        Validate queued objects against current detections.
        Removes objects that have moved or disappeared (unless occluded by the arm).
        """
        to_remove = []
        if verbose:
            self.logger.info(
                f"Queue validation: {len(self.pending_objects)} queued, "
                f"{len(current_detections)} pile detections"
            )
        for queued_obj in self.pending_objects:
            qxy = queued_obj.center_xyz[:2]
            same_type = [d for d in current_detections
                         if d.obj_type == queued_obj.obj_type]

            if same_type:
                dists = [np.linalg.norm(
                    np.array(d.center_xyz[:2]) - np.array(qxy)
                ) for d in same_type]
                min_dist = min(dists)
                if min_dist > QUEUE_MOVEMENT_THRESHOLD:
                    occluded = self._is_in_occlusion_zone(qxy)
                    if not occluded:
                        to_remove.append(queued_obj)
                        if verbose:
                            self.logger.info(
                                f"  REMOVE {queued_obj.obj_type.name} ({qxy[0]:.3f}, {qxy[1]:.3f}): "
                                f"{len(same_type)} same-type detected, nearest={min_dist:.3f}m "
                                f"> threshold={QUEUE_MOVEMENT_THRESHOLD}m, not occluded"
                            )
                    else:
                        if verbose:
                            self.logger.info(
                                f"  KEEP {queued_obj.obj_type.name} ({qxy[0]:.3f}, {qxy[1]:.3f}): "
                                f"{len(same_type)} same-type detected, nearest={min_dist:.3f}m "
                                f"> threshold, but OCCLUDED by arm"
                            )
                else:
                    if verbose:
                        self.logger.debug(
                            f"  KEEP {queued_obj.obj_type.name} ({qxy[0]:.3f}, {qxy[1]:.3f}): "
                            f"matched at {min_dist:.3f}m <= {QUEUE_MOVEMENT_THRESHOLD}m"
                        )
            else:
                occluded = self._is_in_occlusion_zone(qxy)
                if not occluded:
                    to_remove.append(queued_obj)
                    if verbose:
                        self.logger.info(
                            f"  REMOVE {queued_obj.obj_type.name} ({qxy[0]:.3f}, {qxy[1]:.3f}): "
                            f"no same-type detected, not occluded"
                        )
                else:
                    if verbose:
                        self.logger.info(
                            f"  KEEP {queued_obj.obj_type.name} ({qxy[0]:.3f}, {qxy[1]:.3f}): "
                            f"no same-type detected, but OCCLUDED by arm"
                        )

        precomputed_invalidated = False
        for obj in to_remove:
            self.pending_objects.remove(obj)
            if self.precomputed is not None and obj is self.precomputed[0]:
                self.precomputed = None
                precomputed_invalidated = True
            if (self._precompute_ik_data is not None
                    and obj is self._precompute_ik_data.get('obj')):
                self._precompute_ik_id = None
                self._precompute_ik_data = None
                precomputed_invalidated = True

        if precomputed_invalidated and self.pending_objects:
            self._precompute_next()

    def _precompute_next(self):
        """
        Speculatively precompute trajectory + IK for the next queued object.
        Sends an async IK request — does not block. Result arrives in recv_ik_response.
        """
        if not self.pending_objects:
            return

        obj = self.pending_objects[0]
        self.logger.info(
            f"Precomputing trajectory for {obj.obj_type.name}"
            f" at ({obj.center_xyz[0]:.3f}, {obj.center_xyz[1]:.3f})"
        )

        # Fix: if the currently executing block is the same type as the next,
        # the structure height hasn't been incremented yet (happens in
        # recv_trajectory_complete). Temporarily bump it for correct place height.
        structure_bumped = False
        if self.pending_placement is not None:
            exec_type, _ = self.pending_placement
            if exec_type == obj.obj_type:
                structure = self.structures.get(obj.obj_type)
                if structure is not None:
                    structure.place_top_block(0, 0, obj.obj_type)
                    structure_bumped = True
                    self.logger.info(
                        f"Precompute: temporarily bumped {obj.obj_type.name} "
                        f"structure to height {structure.get_height(0, 0)}"
                    )

        trajectory_states = self.structure_planner.plan_pick_and_place(obj, self.structures)

        # Undo temporary bump immediately after planning (before IK)
        if structure_bumped:
            structure = self.structures.get(obj.obj_type)
            if structure is not None:
                top_layer = structure.get_height(0, 0) - 1
                structure.remove_block(0, 0, top_layer)

        if not trajectory_states:
            self.logger.info("Precompute: no trajectory planned, skipping.")
            self.precomputed = None
            return

        request_id = self._send_ik_request(
            trajectory_states, IKRequestMsg.REQUEST_PRECOMPUTE
        )
        self._precompute_ik_id = request_id
        self._precompute_ik_data = {
            'obj': obj,
            'trajectory_states': trajectory_states,
            'structure_bumped': False,  # already undone above
        }
        self.logger.info(f"Sent precompute IK request {request_id} for {obj.obj_type.name}")

    def _compute_occlusion_footprint(self, trajectory_states: list[TrajectoryState]):
        """Precompute arm XY occlusion zones from all trajectory states."""
        self.occlusion_footprint = []
        seen = set()

        for state in trajectory_states:
            q = state.final_state.q
            if q is None:
                continue
            joint_positions = self.block_manipulator.fkin_all(q)

            for name, pos in joint_positions.items():
                x, y = float(pos[0]), float(pos[1])
                r = ARM_OCCLUSION_RADII.get(name, 0.025)
                key = (round(x, 3), round(y, 3))
                if key not in seen:
                    self.occlusion_footprint.append((x, y, r))
                    seen.add(key)

        self.logger.info(f"Occlusion footprint: {len(self.occlusion_footprint)} circles")

    def _is_in_occlusion_zone(self, xy) -> bool:
        """Check if a 2D point is within any arm occlusion circle."""
        bx, by = xy[0], xy[1]
        for (cx, cy, r) in self.occlusion_footprint:
            if (bx - cx)**2 + (by - cy)**2 < r**2:
                return True
        return False

    def recvpoint(self, pointmsg):
        x = pointmsg.x
        y = pointmsg.y
        z = pointmsg.z
        tilt = pointmsg.t * np.pi / 180
        roll = pointmsg.r * np.pi / 180

        self.logger.info(
            f"Received point ({x}, {y}, {z})"
            f" tilt={pointmsg.t} roll={pointmsg.r}"
        )

        above_block_state = ManipulatorState(
            p=np.array([x, y, 0.1]),
            o=np.array([tilt, roll]),
        )
        q, qd = self.block_manipulator.ikin(above_block_state)
        above_block_state.q = q
        above_block_state.qd = qd

        surround_block_state = ManipulatorState(
            p=np.array([x, y, 0.02]),
            o=np.array([tilt, roll]),
        )
        q, qd = self.block_manipulator.ikin(surround_block_state)
        surround_block_state.q = q
        surround_block_state.qd = qd
        
        gripping_block_state = surround_block_state.copy()
        gripping_block_state.close_gripper()
        
        q_ready_with_close_grip = Q_READY.copy()
        q_ready_with_close_grip[GRIPPER_JOINT_INDEX] = GRIPPER_MOTOR_CLOSED_RAD
        up_state = ManipulatorState(
            q=q_ready_with_close_grip, qd=np.zeros(len(JOINT_NAMES))
        )

        self._publish_trajectory_command(
            [
                TrajectoryState(
                    final_state=above_block_state,
                    min_duration=4.0,
                ),
                TrajectoryState(
                    final_state=surround_block_state,
                    min_duration=2.0,
                    delay_before=1.0
                ),
                TrajectoryState(
                    final_state=gripping_block_state,
                    min_duration=2.0,
                    delay_before=1.0
                ),
                TrajectoryState(
                    final_state=up_state,
                    min_duration=4.0,
                    delay_before=1.0,
                ),
                TrajectoryState(
                    final_state=ManipulatorState(q=Q_READY.copy(), qd=np.zeros(len(JOINT_NAMES))),
                    min_duration=1.0,
                    delay_before=5.0
                )
            ],
            TrajectoryCommandMsg.COMMAND_REPLACE
        )
        self.executing_trajectory = True

    def _publish_trajectory_command(
        self,
        states: list[TrajectoryState],
        command: int = TrajectoryCommandMsg.COMMAND_REPLACE
    ):
        """
        Publish trajectory command to manipulator.

        Args:
            states: List of TrajectoryState to send
            command: One of COMMAND_APPEND_FRONT, COMMAND_APPEND_BACK, COMMAND_REPLACE
        """
        msg = TrajectoryCommandMsg()
        msg.header = Header()
        msg.header.stamp = self.clock.now().to_msg()
        msg.header.frame_id = 'world'
        msg.command = command

        for state in states:
            state_msg = TrajectoryStateMsg()
            # state_msg.mode = TrajectoryStateMsg.MODE_Q if state.mode == P_OR_Q.Q else TrajectoryStateMsg.MODE_P

            # Joint space (ensure float values for ROS message)
            state_msg.q = [float(x) for x in state.final_state.q] if state.final_state.q is not None else [0.0] * len(JOINT_NAMES)
            state_msg.qd = [float(x) for x in state.final_state.qd] if state.final_state.qd is not None else [0.0] * len(JOINT_NAMES)

            # Cartesian space
            if state.final_state.p is not None:
                state_msg.p = Point(x=float(state.final_state.p[0]), y=float(state.final_state.p[1]), z=float(state.final_state.p[2]))
            else:
                state_msg.p = Point()

            if state.final_state.v is not None:
                state_msg.pd = Vector3(x=float(state.final_state.v[0]), y=float(state.final_state.v[1]), z=float(state.final_state.v[2]))
            else:
                state_msg.pd = Vector3()

            # Timing (ensure float values for ROS message)
            state_msg.min_duration = float(state.min_duration) if state.min_duration is not None else 0.0
            state_msg.delay_before = float(state.delay_before) if state.delay_before is not None else 0.0
            state_msg.delay_after = float(state.delay_after) if state.delay_after is not None else 0.0

            msg.states.append(state_msg)

        cmd_name = {0: 'APPEND_FRONT', 1: 'APPEND_BACK', 2: 'REPLACE'}.get(
            command, f'UNKNOWN({command})'
        )
        self.logger.info(
            f"[PUB brain/trajectory_command]"
            f" cmd={cmd_name} n_states={len(states)}"
        )
        for i, state in enumerate(states):
            q_str = (
                [round(v, 4) for v in state.final_state.q]
                if state.final_state.q is not None else None
            )
            self.logger.debug(
                f"  state[{i}]: q={q_str}"
                f" dur={state.min_duration}"
            )
        self.pub_trajectory_command.publish(msg)

    def shutdown(self):
        """Shutdown the node."""
        self.destroy_node()


def main(args=None):
    rclpy.init(args=args)

    node = BrainNode('brain')

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
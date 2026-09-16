"""3-phase build state machine extracted from BrainNode.

Coordinates the grasp -> approach -> place pipeline using injected
callback functions for ROS operations (publishing, timers, FK), keeping
the pipeline fully decoupled from the ROS node layer.
"""

from collections import defaultdict
from typing import Callable, Optional

import numpy as np

from legobuilder.kinematics.trajectory import TrajectoryState
from legobuilder.kinematics.block_manipulator import ManipulatorState

from legobuilder.brain.block_structure import (
    BlockStructure, is_point_reachable, _make_virtual_object,
)
from legobuilder.brain.gestures import GestureManager
from legobuilder.brain.ros_bridge import apply_ik_solutions
from legobuilder.schemas import ObjectType, Object, Color

from legobuilder.config import (
    Q_READY,
    Q_SCAN,
    JOINT_NAMES,
    DURATION,
    COLLISION_WAIT_DURATION,
    PILE_X_MIN,
    PILE_CENTER,
    ARM_OCCLUSION_RADII,
    GRIP_FAILURE_RECOVERY,
    VISUAL_GRIP_CHECK,
    PLACEMENT_VERIFICATION,
    GRIP_CONNECTION_DETECTION,
    CONNECTION_DETECTION_TIMEOUT,
    CONNECTION_DROP_HEIGHT,
    IDLE_CHOMP_INITIAL_DELAY,
    Q_CONNECTION_CHECK,
    PICK_TILT,
    BLOCK_SIZE,
    BLOCK_TOP_OFFSET,
    GRIPPER_MOTOR_CLOSED_RAD,
    grip_check_roll as get_grip_check_roll,
)

from legobuilder_interfaces.msg import (
    IKRequestMsg,
    TrajectoryCommandMsg,
)

MAX_CELL_REQUEUES = 2


class BuildPipeline:
    """3-phase build state machine: grasp -> approach -> place.

    Receives callback functions for all ROS operations so it remains
    decoupled from the ROS node.  The pipeline tracks its current
    build phase, IK requests, and occlusion state.  Queue management
    and speculative IK precomputation are delegated to
    PrecomputeManager.

    Attributes
    ----------
    structure : BlockStructure or None
        The target block structure being assembled.
    logger
        ROS logger instance.
    executing_trajectory : bool
        True while the manipulator is executing a trajectory.
    occlusion_footprint : list[tuple[float, float, float]]
        Precomputed (x, y, radius) circles for arm occlusion checks.
    pending_placement : ObjectType or None
        The block type currently being placed (set during place phase).
    """

    # ── Lifecycle ─────────────────────────────────────────────────────

    def __init__(
        self,
        # --- Callback functions for ROS operations ---
        send_ik_request_fn: Callable[[list, int], str],
        publish_trajectory_fn: Callable[[list, int], None],
        request_grip_check_fn: Callable[[], str],
        create_timer_fn: Callable[[float, Callable], object],
        destroy_timer_fn: Callable[[object], None],
        get_t_fn: Callable[[], float],
        fkin_all_fn: Callable[[np.ndarray], dict],
        request_connection_check_fn: Callable[[], str],
        # --- Domain objects ---
        structure: Optional[BlockStructure],
        logger,
        recovery_handler=None,
        precompute_manager=None,
        scanner=None,
        post_event_fn=None,
    ):
        """Initialize the build pipeline with injected callbacks.

        Arguments
        ---------
        send_ik_request_fn : callable
            (states, request_type) -> request_id.
        publish_trajectory_fn : callable
            (states, command) -> None.
        request_grip_check_fn : callable
            () -> request_id or None.
        create_timer_fn : callable
            (period, callback) -> timer.
        destroy_timer_fn : callable
            (timer) -> None.
        get_t_fn : callable
            () -> float returning seconds since node start.
        fkin_all_fn : callable
            (q) -> dict returning joint positions by name.
        structure : BlockStructure or None
            Target structure (None disables build pipeline).
        logger
            ROS logger instance.
        recovery_handler : RecoveryHandler or None
            Grip failure recovery handler (injected by BrainNode).
        precompute_manager : PrecomputeManager or None
            Queue and speculative IK manager (injected by BrainNode).
        scanner : StructureScanner or None
            Structure scanner for scan-first loop (injected by BrainNode).
        """
        self._send_ik_request = send_ik_request_fn
        self._publish_trajectory = publish_trajectory_fn
        self._request_grip_check = request_grip_check_fn
        self._create_timer = create_timer_fn
        self._destroy_timer = destroy_timer_fn
        self._get_t = get_t_fn
        self._fkin_all = fkin_all_fn
        self._request_connection_check = request_connection_check_fn

        self.structure = structure
        self.logger = logger
        self._recovery = recovery_handler
        self._precompute_manager = precompute_manager
        self._scanner = scanner
        self._post_event = post_event_fn or (lambda e: None)

        self._gesture_manager = GestureManager(
            publish_trajectory_fn=publish_trajectory_fn,
            create_timer_fn=create_timer_fn,
            destroy_timer_fn=destroy_timer_fn,
            get_t_fn=get_t_fn,
            scanner=scanner,
            logger=logger,
            scan_result_callback=self._on_idle_scan_result_from_gesture,
            unreachable_tracker=(
                scanner._unreachable_tracker if scanner is not None else None
            ),
            structure=structure,
            on_nod_complete_callback=self._on_nod_complete,
            post_event_fn=self._post_event,
        )

        self.executing_trajectory: bool = False
        self.occlusion_footprint: list[tuple[float, float, float]] = []
        self._validation_grace_until: float = 0.0

        self._build_phase: str | None = None
        self._pickup_context: dict | None = None
        self.pending_placement: ObjectType | None = None

        self._cell_failure_counts: dict[tuple, int] = {}
        self._last_q: np.ndarray | None = None

        # Scan-driven state
        self._next_target_cell: tuple | None = None
        self._next_block_type: ObjectType | None = None
        self._scan_unassigned: list | None = None
        self._has_done_initial_scan: bool = False
        self._detected_pile: list | None = None

        self._primary_ik_id: str | None = None
        self._primary_ik_data: dict | None = None

        self._grip_check_counter: int = 0
        self._pending_grip_check_id: str | None = None
        self._grip_check_timer = None
        self._connection_check_counter: int = 0
        self._pending_connection_check_id: str | None = None
        self._connection_detection_timer = None

        self._idle_since: float | None = None
        self._pending_nod_pickup = None  # Object deferred during nod

    # ── Dashboard events ─────────────────────────────────────────────

    def _emit(self, event_type: str, data: dict) -> None:
        """Post an event to the dashboard server."""
        self._post_event({
            'type': event_type,
            'source': 'pipeline',
            'data': data,
            'timestamp': self._get_t(),
        })

    def _emit_phase(self, phase: str, **extra) -> None:
        """Emit a state_change event for the current build phase."""
        data = {'phase': phase}
        if self._next_target_cell is not None:
            data['target_cell'] = str(self._next_target_cell)
        if self._next_block_type is not None:
            data['block_type'] = self._next_block_type.name
        data.update(extra)
        self._emit('state_change', data)

    # ── Idle chomp animation ──────────────────────────────────────────

    def _mark_idle(self) -> None:
        """Track idle state; start gesture cycle after initial delay."""
        if self._idle_since is None:
            self._idle_since = self._get_t()
            return
        elapsed = self._get_t() - self._idle_since
        if elapsed >= IDLE_CHOMP_INITIAL_DELAY:
            self._gesture_manager.start_idle()

    def _needed_type_in_pile(self, detected_pile: list) -> bool:
        """Check if the next needed block type is reachable in the pile."""
        if self.structure is None:
            return False
        # Use scan-driven target if available
        if self._next_block_type is not None:
            return any(
                obj.obj_type == self._next_block_type
                for obj in detected_pile
            )
        if not self.structure.blocks_to_place:
            return False
        row, col, layer = self.structure.blocks_to_place[0]
        target = self.structure.get_block(row, col, layer)
        return any(
            obj.obj_type == target
            for obj in detected_pile
        )

    def _on_idle_scan_result_from_gesture(self, result) -> None:
        """Handle idle scan result relayed from GestureManager.

        Updates placed_grid and target info.  If an unassigned block of
        the needed type is found, stops idle gestures and triggers pickup.
        Otherwise the arm continues the idle scan-chomp cycle.
        """
        self.structure.placed_grid = result.placed_grid
        self._emit('structure_update', {
            'placed_grid': self.structure.placed_grid.tolist(),
        })
        self.pending_placement = None
        self._pickup_context = None
        self._precompute_manager.clear()
        self._build_phase = None
        self._emit_phase('idle')

        if result.target_cell is None:
            self.logger.info("Idle scan: build complete — all cells filled")
            return

        # Wrong-color block on the structure — remove it first
        if result.is_mistake:
            self.logger.info(
                f"Idle scan: mistake detected: {result.block_type} at "
                f"{result.target_cell} — initiating removal"
            )
            self._gesture_manager.stop_idle()
            self._idle_since = None
            self._pending_nod_pickup = None
            self._start_mistake_removal(result.target_cell, result.block_type)
            return

        self._next_target_cell = result.target_cell
        self._next_block_type = result.block_type
        self._scan_unassigned = result.unassigned
        self._precompute_manager.target_cell = result.target_cell
        self._precompute_manager.block_type = result.block_type

        self.logger.info(
            f"Idle scan: next target {result.target_cell}, "
            f"type {result.block_type}"
        )

        # Pick up an unassigned block matching the needed type
        matching_unassigned = None
        if result.unassigned and result.block_type is not None:
            for obj in result.unassigned:
                if obj.obj_type == result.block_type:
                    matching_unassigned = obj
                    break

        if matching_unassigned is not None:
            # Check if unreachable blocks exist (nod will take priority)
            tracker = getattr(self._scanner, '_unreachable_tracker', None)
            has_nod_target = False
            if tracker is not None:
                statuses = tracker.get_all_statuses()
                has_nod_target = any(
                    not s.permanently_ignored for s in statuses.values()
                )

            if has_nod_target:
                # Nod takes priority -- store pending pick for after nod
                self._pending_nod_pickup = matching_unassigned
                self.logger.info(
                    f"Idle scan: deferring pickup of "
                    f"{matching_unassigned.obj_type.name} until nod completes"
                )
            else:
                self.logger.info(
                    f"Idle scan: picking unassigned "
                    f"{matching_unassigned.obj_type.name}"
                )
                self._gesture_manager.stop_idle()
                self._idle_since = None
                self._precompute_manager.pending_objects = [matching_unassigned]
                self.process_next_object()

    def _on_nod_complete(self, has_pickable_block: bool) -> None:
        """Handle GestureManager nod completion.

        Called by GestureManager after nod + pause. If a pickable block
        was deferred during the nod, initiate pickup now. Otherwise
        GestureManager will resume scan-chomp cycle automatically.

        Parameters
        ----------
        has_pickable_block
            True if the scan that triggered the nod also found a pickable block.
        """
        if self._pending_nod_pickup is not None:
            pickup = self._pending_nod_pickup
            self._pending_nod_pickup = None
            self.logger.info(
                f"Nod complete: proceeding to pick "
                f"{pickup.obj_type.name}"
            )
            self._gesture_manager.stop_idle()
            self._idle_since = None
            self._precompute_manager.pending_objects = [pickup]
            self.process_next_object()
        else:
            self.logger.info("Nod complete: no pending pickup, resuming idle")
            # GestureManager handles transition back to scan-chomp

    # ── Detection entry point ─────────────────────────────────────────

    def on_new_detections(
        self,
        detected_objects: list[Object],
        is_manipulator_online: bool,
    ) -> None:
        """Entry point after the perception pipeline runs.

        Receives all detected objects (before pile filtering) and
        handles: manipulator online check, empty check, structure
        None check, pile filtering, queue management, and dispatch.

        Arguments
        ---------
        detected_objects : list[Object]
            All objects from the current detection cycle.
        is_manipulator_online : bool
            True if both manipulator and IK solver are online.
        """
        if self._build_phase in ('scan',):
            return

        if not is_manipulator_online:
            return

        # Initial scan: trigger immediately once online, don't wait for pile
        if (self._scanner is not None
                and not self._has_done_initial_scan
                and self.structure is not None):
            self._has_done_initial_scan = True
            self._gesture_manager.stop_idle()
            self._idle_since = None
            self.logger.info(
                "Manipulator online — triggering initial structure scan"
            )
            self._build_phase = 'scan'
            self._emit_phase('scan', detail='initial_scan')
            self._scanner.start_scan(
                on_result=self._on_scan_result,
                skip_orbit=True,
            )
            return

        if not detected_objects:
            if (self.executing_trajectory
                    and self._get_t() >= self._validation_grace_until):
                self._precompute_manager.validate_queue(
                    [], self.is_in_occlusion_zone,
                )
            elif not self.executing_trajectory and self._primary_ik_id is None:
                self._mark_idle()
            return

        if self.structure is None:
            return

        detected_pile = [
            obj for obj in detected_objects
            if obj.center_xyz[0] > PILE_X_MIN and obj.obj_type.is_block() and is_point_reachable(obj.center_xyz)
        ]

        # Store detected pile for process_next_object to use later
        self._detected_pile = detected_pile

        if self.executing_trajectory or self._primary_ik_id is not None:
            if self._get_t() >= self._validation_grace_until:
                self._precompute_manager.validate_queue(
                    detected_pile, self.is_in_occlusion_zone,
                )
        else:
            if not detected_pile:
                self._mark_idle()
                return

            if not self._needed_type_in_pile(detected_pile):
                self._mark_idle()
                return

            # Let idle gesture cycle complete before starting a pickup
            if self._gesture_manager.state in (
                self._gesture_manager.MOVE_TO_SCAN,
                self._gesture_manager.SCANNING,
                self._gesture_manager.MOVE_TO_NOD,
                self._gesture_manager.NODDING,
                self._gesture_manager.POST_NOD_PAUSE,
            ):
                return

            self._gesture_manager.stop_idle()
            self._idle_since = None

            # Prefer unassigned blocks from last scan over pile
            if self._scan_unassigned and self._next_block_type is not None:
                matching = [
                    obj for obj in self._scan_unassigned
                    if obj.obj_type == self._next_block_type
                ]
                if matching:
                    self._precompute_manager.pending_objects = matching
                    self.process_next_object()
                    return

            self.logger.debug(
                f"Detected {len(detected_pile)} pile blocks"
            )
            type_counts = defaultdict(int)
            for obj in detected_pile:
                type_counts[obj.obj_type] += 1
            for obj_type, count in type_counts.items():
                self.logger.info(f"  {obj_type.name}: {count}")

            self._precompute_manager.select_and_queue(detected_pile)
            self.process_next_object()

    # ── Object processing ─────────────────────────────────────────────

    def process_next_object(self) -> None:
        """Process the next pending object via the 3-phase build flow.

        Checks the PrecomputeManager for a precomputed result first.
        If none is available, plans the grasp phase and sends an async
        IK request, then returns immediately.
        """
        if not self._precompute_manager.pending_objects:
            self.executing_trajectory = False
            self.occlusion_footprint = []
            return

        obj, precomputed_data = self._precompute_manager.pop_next()
        if obj is None:
            self.executing_trajectory = False
            self.occlusion_footprint = []
            return

        if precomputed_data is not None:
            grasp_states, context = precomputed_data
            self.logger.info(
                f"Using precomputed trajectory"
                f" for {obj.obj_type.name}"
                f" at ({obj.center_xyz[0]:.3f},"
                f" {obj.center_xyz[1]:.3f})"
            )
            context['obj'] = obj
            self._pickup_context = context
            if self._recovery is not None and 'p_pick' in context:
                self._recovery.reset(context['p_pick'])

            self._build_phase = 'grasp'
            self._emit_phase('grasp', detail='precomputed')
            self._publish_trajectory(
                grasp_states,
                TrajectoryCommandMsg.COMMAND_REPLACE,
            )
            self.executing_trajectory = True

            grace_duration = sum(
                (s.min_duration or 0)
                + s.delay_before + s.delay_after
                for s in grasp_states
            )
            self._validation_grace_until = (
                self._get_t() + grace_duration
            )
            self.compute_occlusion_footprint(grasp_states)

            self.logger.debug(
                f"Grasp phase (precomputed): published "
                f"{len(grasp_states)} states"
            )
            self._precompute_manager.precompute_next()
            return

        # Normal path: plan grasp and send IK request
        # Put the object back for plan_grasp to select from
        self._precompute_manager.pending_objects.insert(0, obj)
        result = self.structure.plan_grasp(
            detected_objects=(
                self._precompute_manager.pending_objects
            ),
            target_cell=self._next_target_cell,
            block_type=self._next_block_type,
        )
        obj, grasp_states, context = result
        if obj is None or grasp_states is None:
            # plan_grasp failed — remove the object we re-inserted so
            # the pipeline doesn't retry the same unreachable block.
            failed = self._precompute_manager.pending_objects.pop(0)
            if self._scan_unassigned:
                try:
                    self._scan_unassigned.remove(failed)
                except ValueError:
                    pass
            self.executing_trajectory = False
            self.occlusion_footprint = []
            return

        if obj in self._precompute_manager.pending_objects:
            self._precompute_manager.pending_objects.remove(obj)

        self.logger.debug(
            f"Processing {obj.obj_type.name}"
            f" at ({obj.center_xyz[0]:.3f},"
            f" {obj.center_xyz[1]:.3f})"
        )

        context['obj'] = obj
        self._pickup_context = context
        if self._recovery is not None and 'p_pick' in context:
            self._recovery.reset(context['p_pick'])

        request_id = self._send_ik_request(
            grasp_states, IKRequestMsg.REQUEST_PRIMARY
        )
        self._primary_ik_id = request_id
        self._primary_ik_data = {
            'obj': obj,
            'trajectory_states': grasp_states,
            'phase': 'grasp',
        }
        self.logger.info(
            f"Sent grasp IK request {request_id} "
            f"for {obj.obj_type.name}"
        )

    # ── Trajectory completion dispatch ────────────────────────────────

    def on_trajectory_complete(self, states_executed: int) -> None:
        """Dispatch trajectory completion to the current phase handler.

        Arguments
        ---------
        states_executed : int
            Number of trajectory states the manipulator executed.
        """
        self.logger.debug(
            f"Trajectory completed. States executed: {states_executed}"
            f" build_phase={self._build_phase}"
        )

        if self._build_phase == 'grasp':
            self.on_grasp_complete()
        elif self._build_phase == 'grip_connection_detection':
            self.run_connection_detection_check()
        elif self._build_phase == 'approach':
            self.on_approach_complete()
        elif self._build_phase == 'place':
            self.on_place_complete()
        elif self._build_phase == 'scan' or (
                self._scanner is not None and self._scanner.is_active):
            self._scanner.on_trajectory_complete()
        elif self._build_phase == 'clearing_drop':
            self._pickup_context = None
            self._build_phase = 'scan'
            self.executing_trajectory = False
            self.logger.info(
                "Clearing drop complete, triggering rescan"
            )
            self._scanner.start_scan(
                on_result=self._on_scan_result,
            )
        elif self._build_phase == 'connection_drop':
            self._pickup_context = None
            self._build_phase = 'scan'
            self.executing_trajectory = False
            self.logger.info(
                "Connection drop complete, triggering rescan"
            )
            self._scanner.start_scan(
                on_result=self._on_scan_result,
            )
        elif self._build_phase == 'mistake_drop':
            self._pickup_context = None
            self._build_phase = 'scan'
            self.executing_trajectory = False
            self.logger.info(
                "Mistake drop complete, triggering rescan"
            )
            self._scanner.start_scan(
                on_result=self._on_scan_result,
            )
        elif self._build_phase == 'grip_recovery':
            self._pickup_context = None
            self.executing_trajectory = False
            if self._scanner is not None:
                self._build_phase = 'scan'
                self.logger.info(
                    "Grip recovery complete, triggering rescan"
                )
                self._scanner.start_scan(
                    on_result=self._on_scan_result,
                )
            else:
                self._build_phase = None
                self.process_next_object()
        else:
            self.process_next_object()

    # ── Phase 1: Grasp ────────────────────────────────────────────────

    def on_grasp_complete(self) -> None:
        """Handle grasp phase completion.

        If this was a clearing grasp, drops the block and returns
        to idle for rescan.  Otherwise, if visual grip checking is
        enabled, sends a grip check request.  Otherwise, proceeds
        directly to the approach phase.
        """
        ctx = self._pickup_context
        if ctx and ctx.get('clearing'):
            self._complete_clearing_drop()
            return

        if ctx and ctx.get('mistake_removal'):
            self._complete_mistake_drop()
            return

        if VISUAL_GRIP_CHECK:
            request_id = self._request_grip_check()
            if request_id is not None:
                self._pending_grip_check_id = request_id
                self.logger.debug(
                    f"Sent visual grip check request {request_id}"
                )

                from legobuilder.config import GRIP_CHECK_TIMEOUT
                self._grip_check_timer = self._create_timer(
                    GRIP_CHECK_TIMEOUT, self.on_grip_check_timeout
                )
                return

        if GRIP_CONNECTION_DETECTION:
            self.proceed_to_connection_detection()
        else:
            self.proceed_to_approach()

    def _plan_gentle_place_states(
        self, drop_xy: np.ndarray,
    ) -> list[TrajectoryState]:
        """Plan a gentle place trajectory: move above, descend, release, lift.

        Parameters
        ----------
        drop_xy : np.ndarray
            2D [x, y] target position for placing the block.

        Returns
        -------
        list[TrajectoryState]
            Trajectory states for the place sequence.
        """
        approach_height = self.structure.APPROACH_HEIGHT
        place_z = BLOCK_SIZE / 2 + BLOCK_TOP_OFFSET
        place_p = np.array([drop_xy[0], drop_xy[1], place_z])
        place_o = np.array([PICK_TILT, get_grip_check_roll(place_p)])

        return [
            # 1. Move above drop position
            TrajectoryState(
                final_state=ManipulatorState(
                    p=place_p + np.array([0, 0, approach_height]),
                    o=place_o,
                    gripper_open=False,
                ),
                min_duration=DURATION/2,
            ),
            # 2. Descend to surface
            TrajectoryState(
                final_state=ManipulatorState(
                    p=place_p,
                    o=place_o,
                    gripper_open=False,
                ),
                min_duration=DURATION / 4,
            ),
            # 3. Release
            TrajectoryState(
                final_state=ManipulatorState(
                    p=place_p,
                    o=place_o,
                    gripper_open=True,
                ),
                min_duration=self.structure.GRIPPER_DELAY,
                delay_before=self.structure.GRIPPER_DELAY,
            ),
            # 4. Lift away
            TrajectoryState(
                final_state=ManipulatorState(
                    p=place_p + np.array([0, 0, approach_height]),
                    o=place_o,
                    gripper_open=True,
                ),
                min_duration=DURATION / 4,
            ),
        ]

    def _complete_clearing_drop(self) -> None:
        """Gently place a cleared neighbor block in the least occupied
        region of the pile."""
        self.logger.info(
            "Clearing place: setting block down in pile"
        )

        p_pick = self._pickup_context['p_pick']
        drop_xy = self._find_pile_drop_xy(p_pick[:2].copy())

        place_states = self._plan_gentle_place_states(drop_xy)

        request_id = self._send_ik_request(
            place_states, IKRequestMsg.REQUEST_PRIMARY,
        )
        self._primary_ik_id = request_id
        self._primary_ik_data = {
            'trajectory_states': place_states,
            'phase': 'clearing_drop',
        }
        self._build_phase = 'clearing_drop'
        self._emit_phase('clearing_drop')
        
    def _find_pile_drop_xy(self, reference_xy: np.ndarray) -> np.ndarray:
        """Find an XY position in the pile that maximizes distance from
        other detected blocks.

        Evaluates four candidate offsets around *reference_xy*, filters
        to positions within the pile area, and returns the one whose
        nearest neighbor is farthest away.

        Parameters
        ----------
        reference_xy : np.ndarray
            2D [x, y] origin for candidate generation.

        Returns
        -------
        np.ndarray
            2D [x, y] of the best drop position.
        """
        detected_objects = self._detected_pile or []

        best_xy = reference_xy.copy()
        # Ensure the fallback is within the pile area
        best_xy[0] = max(best_xy[0], PILE_X_MIN + 0.05)

        max_min_dist = 0.0
        candidates = [
            reference_xy + np.array([0.05, 0]),
            reference_xy + np.array([-0.05, 0]),
            reference_xy + np.array([0, 0.05]),
            reference_xy + np.array([0, -0.05]),
        ]
        for cand in candidates:
            if cand[0] <= PILE_X_MIN or cand[1] <= 0:
                continue
            if not is_point_reachable(np.array([cand[0], cand[1], 0.0])):
                continue
            min_dist = float('inf')
            for det_obj in detected_objects:
                d = np.linalg.norm(cand - np.array(det_obj.center_xyz[:2]))
                min_dist = min(min_dist, d)
            if min_dist > max_min_dist:
                max_min_dist = min_dist
                best_xy = cand

        return best_xy

    def _complete_mistake_drop(self) -> None:
        """Gently place a wrong-color block in the least occupied region
        of the pile, then rescan."""
        ctx = self._pickup_context
        cell = ctx['target_cell']
        wrong_type = ctx['wrong_type']

        self.logger.info(
            f"Mistake place: setting {wrong_type.name} from cell {cell} "
            f"down in pile"
        )

        # Clear the cell from placed_grid since we're removing it
        row, col, layer = cell
        self.structure.placed_grid[row, col, layer] = 0

        # Choose the center of the pile as the reference xy for the drop
        drop_xy = self._find_pile_drop_xy(PILE_CENTER[:2].copy())

        place_states = self._plan_gentle_place_states(drop_xy)

        request_id = self._send_ik_request(
            place_states, IKRequestMsg.REQUEST_PRIMARY,
        )
        self._primary_ik_id = request_id
        self._primary_ik_data = {
            'trajectory_states': place_states,
            'phase': 'mistake_drop',
        }
        self._build_phase = 'mistake_drop'

    def on_grip_check_response(
        self,
        quality: int,
        diagonal_score: float,
        height_score: float,
        GRIP_NO_FRAME: int,
        GRIP_DIAGONAL: int,
        GRIP_PARTIAL: int,
        GRIP_LOW: int,
        GRIP_HIGH: int,
        request_id: str,
    ) -> None:
        """Handle grip quality response from the detector.

        The GripCheckResponseMsg constants are passed as parameters
        to avoid importing ROS message types into pipeline logic.

        Arguments
        ---------
        quality : int
            Grip quality classification constant.
        diagonal_score : float
            Diagonal alignment score from visual check.
        height_score : float
            Height alignment score from visual check.
        GRIP_NO_FRAME : int
            Constant: no end-effector frame available.
        GRIP_DIAGONAL : int
            Constant: diagonal grip detected.
        GRIP_PARTIAL : int
            Constant: partial grip detected.
        GRIP_LOW : int
            Constant: grip too low on block.
        GRIP_HIGH : int
            Constant: grip too high on block.
        request_id : str
            ID of the grip check request this responds to.
        """
        if request_id != self._pending_grip_check_id:
            return
        self._pending_grip_check_id = None

        if self._grip_check_timer is not None:
            self._grip_check_timer.cancel()
            self._destroy_timer(self._grip_check_timer)
            self._grip_check_timer = None

        self.logger.debug(
            f"Grip check response: quality={quality} "
            f"diag={diagonal_score:.2f} height={height_score:.2f}"
        )

        quality_names = {
            GRIP_NO_FRAME: 'NO_FRAME', GRIP_DIAGONAL: 'DIAGONAL',
            GRIP_PARTIAL: 'PARTIAL', GRIP_LOW: 'LOW', GRIP_HIGH: 'HIGH',
        }
        self._emit('detection', {
            'grip_quality': quality_names.get(quality, 'GOOD'),
            'diagonal_score': diagonal_score,
            'height_score': height_score,
        })

        if quality == GRIP_NO_FRAME:
            self.logger.warn(
                "No EE frame for visual grip check, "
                "proceeding anyway"
            )
            if GRIP_CONNECTION_DETECTION:
                self.proceed_to_connection_detection()
            else:
                self.proceed_to_approach()
            return

        if quality == GRIP_LOW:
            self._dispatch_recovery(
                self._recovery.handle_low_grip,
                f"visual check (quality={quality})",
            )
            return

        is_bad = quality in (GRIP_DIAGONAL, GRIP_PARTIAL, GRIP_HIGH)
        if is_bad:
            self._dispatch_recovery(
                self._recovery.handle_bad_grip,
                f"visual check (quality={quality})",
            )
            return

        if GRIP_CONNECTION_DETECTION:
            self.proceed_to_connection_detection()
        else:
            self.proceed_to_approach()

    def on_grip_check_timeout(self) -> None:
        """Timer callback: visual grip check timed out."""
        if self._pending_grip_check_id is not None:
            self.logger.warn(
                f"Grip check {self._pending_grip_check_id} "
                f"timed out, proceeding anyway"
            )
            self._pending_grip_check_id = None
        if self._grip_check_timer is not None:
            self._grip_check_timer.cancel()
            self._destroy_timer(self._grip_check_timer)
            self._grip_check_timer = None
        if GRIP_CONNECTION_DETECTION:
            self.proceed_to_connection_detection()
        else:
            self.proceed_to_approach()

    # -- Intermediary Phase -- Grip connection detection before approach --

    def proceed_to_connection_detection(self) -> None:
        """Send trajectory to detection pose for grip connection check."""
        self._build_phase = 'grip_connection_detection'
        self._emit_phase('grip_connection_detection')
        state_q = Q_CONNECTION_CHECK.copy()
        states = [
            TrajectoryState(
                final_state=ManipulatorState(
                    q=state_q,
                    qd=np.zeros_like(state_q),
                ),
                min_duration=DURATION / 2,
            ),
        ]

        self._publish_trajectory(
            states,
            TrajectoryCommandMsg.COMMAND_REPLACE,
        )
        self.executing_trajectory = True

    def run_connection_detection_check(self) -> None:
        """Send a connection check request to the detector."""
        self.logger.debug(
            "Running grip connection detection check"
        )
        request_id = self._request_connection_check()
        if request_id is not None:
            self._pending_connection_check_id = request_id
            self._connection_detection_timer = self._create_timer(
                CONNECTION_DETECTION_TIMEOUT,
                self._on_connection_detection_timeout,
            )
        else:
            self.logger.warn(
                "Connection check unavailable, proceeding to approach"
            )
            self.proceed_to_approach()

    def on_connection_check_response(
        self, request_id, nearby_count, detected_color='',
    ):
        """Handle a connection check response from the detector.

        Arguments
        ---------
        request_id : str
            Correlation ID from the request.
        nearby_count : int
            Number of blocks detected near the connection position.
        detected_color : str
            Color name of the detected block (when nearby_count == 1).
        """
        if request_id != self._pending_connection_check_id:
            return
        self._pending_connection_check_id = None
        if self._connection_detection_timer is not None:
            self._connection_detection_timer.cancel()
            self._destroy_timer(self._connection_detection_timer)
            self._connection_detection_timer = None

        self.logger.info(
            f"Connection check: {nearby_count} block(s) near gripper"
            f", detected_color={detected_color!r}"
        )

        if nearby_count != 1:
            self.logger.info(
                f"Connection check failed: expected 1 block, got {nearby_count}"
            )
            self._complete_connection_drop()
            return

        # Verify the detected block color matches the expected type
        if detected_color and self._next_block_type is not None:
            try:
                detected_type = ObjectType.block_from_color(
                    Color.from_str(detected_color))
            except (ValueError, KeyError):
                self.logger.warn(
                    f"Connection check: unrecognized color {detected_color!r}"
                    ", proceeding to approach"
                )
                self.proceed_to_approach()
                return

            if detected_type != self._next_block_type:
                self.logger.info(
                    f"Connection check failed: detected {detected_type.name}"
                    f" but expected {self._next_block_type.name}"
                )
                self._complete_connection_drop()
                return

        self.proceed_to_approach()

    def _on_connection_detection_timeout(self) -> None:
        """Handle timeout waiting for connection check response — proceed (fail-open)."""
        self.logger.warn(
            "Connection detection timed out, proceeding to approach"
        )
        self._pending_connection_check_id = None
        if self._connection_detection_timer is not None:
            self._connection_detection_timer.cancel()
            self._destroy_timer(self._connection_detection_timer)
            self._connection_detection_timer = None
        self.proceed_to_approach()

    def _complete_connection_drop(self) -> None:
        """Drop connected blocks in the most spacious pile area.

        Uses _find_pile_drop_xy to find the least occupied region
        of the pile, regardless of where the block was picked from.
        """
        from legobuilder.config import grip_check_roll as get_grip_check_roll

        self.logger.info(
            "Connection check failed, dropping block in pile"
        )

        drop_xy = self._find_pile_drop_xy(PILE_CENTER[:2].copy())

        drop_p = np.array([
            drop_xy[0], drop_xy[1], CONNECTION_DROP_HEIGHT
        ])
        drop_o = np.array([
            PICK_TILT,
            get_grip_check_roll(drop_p),
        ])

        drop_states = [
            TrajectoryState(
                final_state=ManipulatorState(
                    p=drop_p,
                    o=drop_o,
                    gripper_open=False,
                ),
                min_duration=DURATION,
                delay_before=1,
            ),
            TrajectoryState(
                final_state=ManipulatorState(
                    p=drop_p,
                    o=drop_o,
                    gripper_open=True,
                ),
                min_duration=1,
                delay_before=0.5,
            ),
        ]

        request_id = self._send_ik_request(
            drop_states, IKRequestMsg.REQUEST_PRIMARY,
        )
        self._primary_ik_id = request_id
        self._primary_ik_data = {
            'trajectory_states': drop_states,
            'phase': 'connection_drop',
        }
        self._build_phase = 'connection_drop'
        self._emit_phase('connection_drop')


    # ── Phase 2: Approach ─────────────────────────────────────────────

    def proceed_to_approach(self) -> None:
        """Compute placement roll and send approach IK request.

        Called after a successful grip check (or when visual check
        is disabled).  Determines which mating faces the target cell
        has and selects a placement roll that avoids obstruction.
        """
        ctx = self._pickup_context
        row, col, layer = ctx['target_cell']

        mating_faces = self.structure.get_horizontal_mating_faces(
            row, col, layer, use_placed_grid=True,
        )
        placement_roll = self.structure.compute_placement_roll(
            mating_faces, ctx['pickup_roll']
        )
        ctx['placement_roll'] = placement_roll

        if mating_faces:
            self.logger.debug(
                f"Mating faces at ({row},{col},{layer}): "
                f"{mating_faces}, "
                f"placement_roll={placement_roll:.3f}"
            )

        self.logger.debug(
            f"Grasp OK, proceeding to approach phase "
            f"at roll {placement_roll:.3f}"
        )
        approach_states = self.structure.plan_approach(
            ctx['target_cell'], placement_roll
        )
        request_id = self._send_ik_request(
            approach_states, IKRequestMsg.REQUEST_PRIMARY
        )
        self._primary_ik_id = request_id
        self._primary_ik_data = {
            'obj': ctx.get('obj'),
            'trajectory_states': approach_states,
            'phase': 'approach',
        }

    def on_approach_complete(self) -> None:
        """Handle approach completion: plan the place phase."""
        ctx = self._pickup_context
        row, col, layer = ctx['target_cell']
        placement_roll = ctx['placement_roll']

        will_verify = (
            PLACEMENT_VERIFICATION
            and self._scanner is not None
        )

        place_states = self.structure.plan_placement(
            row, col, layer, placement_roll,
            return_to_ready=not will_verify,
        )
        request_id = self._send_ik_request(
            place_states, IKRequestMsg.REQUEST_PRIMARY
        )
        self._primary_ik_id = request_id
        self._primary_ik_data = {
            'obj': ctx.get('obj'),
            'trajectory_states': place_states,
            'phase': 'place',
        }

    # ── Phase 3: Place ────────────────────────────────────────────────

    def on_place_complete(self) -> None:
        """Handle place completion: scan structure or confirm placement."""
        self.logger.debug(
            "Place phase complete"
        )

        if (PLACEMENT_VERIFICATION
                and self._scanner is not None):
            self._build_phase = 'scan'
            self._emit_phase('scan', detail='placement_verification')
            self.executing_trajectory = False
            self._scanner.start_scan(
                on_result=self._on_scan_result,
                skip_orbit=True,
            )
            return

        self._confirm_and_advance()

    def _confirm_and_advance(self) -> None:
        """Confirm placement and advance to next object."""
        self.logger.debug("Confirming block placement")
        self.structure.confirm_block_placed()
        self._emit('structure_update', {
            'placed_grid': self.structure.placed_grid.tolist(),
        })
        self.pending_placement = None
        self._build_phase = None
        self._emit_phase('idle', detail='placement_confirmed')
        self._pickup_context = None
        self._precompute_manager.clear()
        self.process_next_object()

    def _on_scan_result(self, result) -> None:
        """Handle scan result from StructureScanner.

        Updates placed_grid, stores target info, and triggers
        the next grasp from the pile (or misplaced block re-grasp).
        If a wrong-color block is detected on the structure, initiates
        a removal sequence: pick it up, drop it in the pile, and rescan.

        Arguments
        ---------
        result : ScanResult
            Contains placed_grid, target_cell, block_type,
            unassigned, and is_mistake.
        """
        # Write rebuilt state back to structure
        self.structure.placed_grid = result.placed_grid
        self._emit('structure_update', {
            'placed_grid': self.structure.placed_grid.tolist(),
        })

        self.pending_placement = None
        self._pickup_context = None
        self._precompute_manager.clear()

        if result.target_cell is None:
            # Build complete
            self.logger.info("Build complete — all cells filled")
            self._build_phase = None
            self._emit_phase('idle', detail='build_complete')
            return

        # Wrong-color block on the structure — remove it first
        if result.is_mistake:
            self.logger.info(
                f"Mistake detected: {result.block_type} at "
                f"{result.target_cell} — initiating removal"
            )
            self._start_mistake_removal(result.target_cell, result.block_type)
            return

        # Store target for grasp planning
        self._next_target_cell = result.target_cell
        self._next_block_type = result.block_type
        self._scan_unassigned = result.unassigned
        self._build_phase = None
        self._emit_phase('idle', detail='scan_complete')

        # Prefer grasping an unassigned block matching the target type
        matching_unassigned = None
        if result.unassigned and result.block_type is not None:
            for obj in result.unassigned:
                if obj.obj_type == result.block_type:
                    matching_unassigned = obj
                    break

        self._precompute_manager.target_cell = result.target_cell
        self._precompute_manager.block_type = result.block_type

        if matching_unassigned is not None:
            self._precompute_manager.pending_objects = [matching_unassigned]

        # Trigger pick from pile (or re-grasp misplaced block)
        self.process_next_object()

    # ── Mistake removal ───────────────────────────────────────────────

    def _start_mistake_removal(
        self,
        cell: tuple[int, int, int],
        wrong_type: ObjectType,
    ) -> None:
        """Pick up a wrong-color block from the structure and drop it.

        Computes the world position of *cell*, plans a grasp trajectory
        to that position, and sets the build phase to
        ``'mistake_removal'``.  After the grasp completes, the block is
        dropped in the pile area and a rescan is triggered.

        Parameters
        ----------
        cell : tuple[int, int, int]
            (row, col, layer) of the misplaced block.
        wrong_type : ObjectType
            The type of block currently occupying *cell*.
        """
        row, col, layer = cell
        p_block = self.structure.get_world_position(row, col, layer)
        p_pick = p_block + np.array([0, 0, BLOCK_SIZE / 4 + BLOCK_TOP_OFFSET])

        # Build virtual Object + neighbor detections from grid data so
        # compute_pickup_roll can choose a roll avoiding obstructions.
        virtual_obj = _make_virtual_object(
            wrong_type.value,
            (p_block[0], p_block[1], p_block[2]),
            angle=self.structure.roll,
        )
        neighbor_detections = []
        for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nr, nc = row + dr, col + dc
            pg = self.structure.placed_grid
            if (0 <= nr < pg.shape[0] and 0 <= nc < pg.shape[1]
                    and pg[nr, nc, layer] != 0):
                npos = self.structure.get_world_position(nr, nc, layer)
                neighbor_detections.append(_make_virtual_object(
                    int(pg[nr, nc, layer]),
                    (npos[0], npos[1], npos[2]),
                    angle=self.structure.roll,
                ))

        pickup_roll = self.structure.compute_pickup_roll(
            virtual_obj, neighbor_detections, alternate_grip_convention=True
        )
        # actual_obj = None
        # min_dist = float('inf')
        # for obj in self.structure.detected_grid_objects:
        #     arr1 = np.array(obj.center_xyz)
        #     arr2 = np.array(virtual_obj.center_xyz)
        #     dist = np.linalg.norm(arr1 - arr2)
        #     if dist < min_dist:
        #         actual_obj = obj
        #         min_dist = dist
        # pickup_roll = self.structure.compute_pickup_roll(
        #     actual_obj, self.structure.detected_grid_objects, alternate_grip_convention=True
        # )

        pick_o = np.array([PICK_TILT, pickup_roll])

        approach_height = self.structure.APPROACH_HEIGHT
        pick_duration = self.structure.PICK_DURATION
        gripper_delay = self.structure.GRIPPER_DELAY

        grasp_states = [
            # 1. Approach above block
            TrajectoryState(
                final_state=ManipulatorState(
                    p=p_pick + np.array([0, 0, approach_height]),
                    o=pick_o,
                    gripper_open=True,
                ),
                min_duration=pick_duration,
            ),
            # 2. Descend to block
            TrajectoryState(
                final_state=ManipulatorState(
                    p=p_pick,
                    o=pick_o,
                    gripper_open=True,
                ),
                min_duration=pick_duration / 2,
            ),
            # 3. Close gripper
            TrajectoryState(
                final_state=ManipulatorState(
                    p=p_pick,
                    o=pick_o,
                    gripper_open=False,
                ),
                min_duration=gripper_delay,
                delay_before=gripper_delay,
            ),
            # 4. Lift
            TrajectoryState(
                final_state=ManipulatorState(
                    p=p_pick + np.array([0, 0, approach_height]),
                    o=pick_o,
                    gripper_open=False,
                ),
                min_duration=pick_duration / 2,
            ),
        ]

        self._pickup_context = {
            'target_cell': cell,
            'pickup_roll': pickup_roll,
            'p_pick': p_pick,
            'mistake_removal': True,
            'wrong_type': wrong_type,
        }

        # Transition out of 'scan' (or whatever prior phase) so the IK
        # response is routed to the pipeline, not the scanner.
        self._build_phase = 'mistake_removal'

        request_id = self._send_ik_request(
            grasp_states, IKRequestMsg.REQUEST_PRIMARY,
        )
        self._primary_ik_id = request_id
        self._primary_ik_data = {
            'trajectory_states': grasp_states,
            'phase': 'grasp',
        }
        self.logger.info(
            f"Sent mistake-removal grasp IK for {wrong_type.name} "
            f"at cell {cell}"
        )

    # ── IK response handling ──────────────────────────────────────────

    def on_ik_response(
        self,
        request_id: str,
        request_type: int,
        success: bool,
        q_solutions: list[float],
    ) -> None:
        """Handle IK solutions and dispatch by build phase.

        Fills solved joint angles into trajectory states, then
        publishes the trajectory for the current phase.

        Arguments
        ---------
        request_id : str
            The IK request identifier.
        request_type : int
            IKRequestMsg.REQUEST_PRIMARY or similar.
        success : bool
            Whether the IK solver found a valid solution.
        q_solutions : list[float]
            Flat list of solved joint angles.
        """
        # Route IK responses to scanner whenever it's active
        if (self._scanner is not None and self._scanner.is_active):
            self._scanner.on_ik_response(
                request_id, success, q_solutions,
            )
            return

        if request_type == IKRequestMsg.REQUEST_PRECOMPUTE:
            self._precompute_manager.on_precompute_response(
                request_id, success, q_solutions,
            )
            return

        if request_type == IKRequestMsg.REQUEST_PRIMARY:
            if request_id != self._primary_ik_id:
                self.logger.info(
                    f"Discarding stale primary IK response "
                    f"{request_id}"
                )
                return
            if not success:
                self.logger.error(
                    f"Primary IK request {request_id} failed"
                )
                self._primary_ik_id = None
                self._primary_ik_data = None
                self.clear_build_state()
                self.process_next_object()
                return

            data = self._primary_ik_data
            self._primary_ik_id = None
            self._primary_ik_data = None

            trajectory_states = data['trajectory_states']
            obj = data.get('obj')
            phase = data.get('phase')
            self.logger.debug(f"Processing IK solutions for phase: {phase}")

            apply_ik_solutions(trajectory_states, q_solutions)

            if phase == 'grasp':
                self._build_phase = 'grasp'
                self._emit_phase('grasp')
                self._publish_trajectory(
                    trajectory_states,
                    TrajectoryCommandMsg.COMMAND_REPLACE,
                )
                self.executing_trajectory = True
                
                grace_duration = sum(
                    (s.min_duration or 0)
                    + s.delay_before + s.delay_after
                    for s in trajectory_states
                )
                self._validation_grace_until = (
                    self._get_t() + grace_duration
                )
                self.compute_occlusion_footprint(trajectory_states)

                self.logger.info(
                    f"Grasp phase: published "
                    f"{len(trajectory_states)} states"
                )
                self._precompute_manager.precompute_next()

            elif phase == 'approach':
                self._build_phase = 'approach'
                self._emit_phase('approach')
                self._publish_trajectory(
                    trajectory_states,
                    TrajectoryCommandMsg.COMMAND_REPLACE,
                )
                self.executing_trajectory = True
                self.logger.debug(
                    f"Approach phase: published "
                    f"{len(trajectory_states)} states"
                )

            elif phase == 'place':
                self._build_phase = 'place'
                self._emit_phase('place')
                self.pending_placement = (
                    obj.obj_type if obj is not None else None
                )
                self._publish_trajectory(
                    trajectory_states,
                    TrajectoryCommandMsg.COMMAND_REPLACE,
                )
                self.executing_trajectory = True
                self.logger.debug(
                    f"Place phase: published "
                    f"{len(trajectory_states)} states"
                )

            elif phase == 'clearing_drop':
                self._build_phase = 'clearing_drop'
                self._emit_phase('clearing_drop')
                self._publish_trajectory(
                    trajectory_states,
                    TrajectoryCommandMsg.COMMAND_REPLACE,
                )
                self.executing_trajectory = True

            elif phase == 'connection_drop':
                self._build_phase = 'connection_drop'
                self._emit_phase('connection_drop')
                self._publish_trajectory(
                    trajectory_states,
                    TrajectoryCommandMsg.COMMAND_REPLACE,
                )
                self.executing_trajectory = True

            elif phase == 'mistake_drop':
                self._build_phase = 'mistake_drop'
                self._publish_trajectory(
                    trajectory_states,
                    TrajectoryCommandMsg.COMMAND_REPLACE,
                )
                self.executing_trajectory = True
        else:
            self.logger.warn(
                f"Unknown IK response type {request_type}, "
                f"discarding"
            )

    # ── State management ──────────────────────────────────────────────

    def clear_build_state(self) -> None:
        """Reset all 3-phase build state for recovery or restart."""
        self._build_phase = None
        self._emit_phase('idle', detail='state_cleared')
        self._pickup_context = None
        self.pending_placement = None
        self._precompute_manager.clear()
        self.occlusion_footprint = []
        self._primary_ik_id = None
        self._primary_ik_data = None
        self._pending_connection_check_id = None
        if self._connection_detection_timer is not None:
            self._connection_detection_timer.cancel()
            self._destroy_timer(self._connection_detection_timer)
            self._connection_detection_timer = None
        self._next_target_cell = None
        self._next_block_type = None
        self._scan_unassigned = None
        if self._scanner is not None and self._scanner.is_active:
            self._scanner.reset()
        self._gesture_manager.stop_idle()
        self._idle_since = None
        self._pending_nod_pickup = None

    # ── Contact / grip failure recovery ───────────────────────────────

    def on_contact(self, current_q=None) -> None:
        """Clear state and send recovery trajectory on collision.

        Arguments
        ---------
        current_q : np.ndarray or None
            Current joint angles (used to keep base orientation
            during recovery).
        """
        self._last_q = current_q
        if self._build_phase == 'scan' and self._scanner is not None:
            # Scan was in progress — just cancel it
            pass
        self.clear_build_state()

        recovery_states = self._build_recovery_states(
            current_q, delay_before=COLLISION_WAIT_DURATION,
        )
        self._publish_trajectory(
            recovery_states, TrajectoryCommandMsg.COMMAND_REPLACE
        )
        self.executing_trajectory = True

    def on_grip_failure(
        self, gripper_position: float, current_q=None,
    ) -> None:
        """Handle grip failure: clear state and return to Q_SCAN.

        Only acts if GRIP_FAILURE_RECOVERY is enabled.

        Arguments
        ---------
        gripper_position : float
            Current gripper motor position (radians).
        current_q : np.ndarray or None
            Current joint angles.
        """
        if not GRIP_FAILURE_RECOVERY:
            self.logger.debug(
                "GRIP_FAILURE_RECOVERY disabled, ignoring."
            )
            return

        self._last_q = current_q
        if self._build_phase == 'scan' and self._scanner is not None:
            # Scan was in progress — just cancel and recover
            pass
        self.clear_build_state()
        self._build_phase = 'grip_recovery'

        recovery_states = self._build_recovery_states(current_q)
        self._publish_trajectory(
            recovery_states, TrajectoryCommandMsg.COMMAND_REPLACE
        )
        self.executing_trajectory = True
        self.logger.info(
            "Grip failure recovery: returning to Q_SCAN"
        )

    # ── Occlusion computation ─────────────────────────────────────────

    def compute_occlusion_footprint(
        self, trajectory_states: list[TrajectoryState],
    ) -> None:
        """Precompute arm XY occlusion zones from trajectory states.

        Runs forward kinematics on each waypoint to collect joint
        positions, then stores an (x, y, radius) circle for each
        unique joint location.

        Arguments
        ---------
        trajectory_states : list[TrajectoryState]
            Waypoints whose joint solutions define the arm path.
        """
        self.occlusion_footprint = []
        seen = set()

        for state in trajectory_states:
            q = state.final_state.q
            if q is None:
                continue
            joint_positions = self._fkin_all(q)

            for name, pos in joint_positions.items():
                x, y = float(pos[0]), float(pos[1])
                r = ARM_OCCLUSION_RADII.get(name, 0.025)
                key = (round(x, 3), round(y, 3))
                if key not in seen:
                    self.occlusion_footprint.append((x, y, r))
                    seen.add(key)

        self.logger.debug(
            f"Occlusion footprint: "
            f"{len(self.occlusion_footprint)} circles"
        )

    def is_in_occlusion_zone(self, xy) -> bool:
        """Check if a 2D point falls within any arm occlusion circle.

        Arguments
        ---------
        xy : array-like
            2D point [x, y].

        Returns
        -------
        bool
            True if the point is occluded by the arm.
        """
        bx, by = xy[0], xy[1]
        for (cx, cy, r) in self.occlusion_footprint:
            if (bx - cx) ** 2 + (by - cy) ** 2 < r ** 2:
                return True
        return False

    # ── Private helpers ───────────────────────────────────────────────

    def _dispatch_recovery(self, handler_fn, reason: str) -> None:
        """Dispatch a grip recovery attempt and update pipeline state.

        Calls the given RecoveryHandler method, then either updates
        IK request state on retry or aborts to Q_SCAN if recovery
        is exhausted.

        Arguments
        ---------
        handler_fn : callable
            RecoveryHandler.handle_bad_grip or handle_low_grip.
        reason : str
            Human-readable description of the grip failure.
        """
        result = handler_fn(
            self._pickup_context,
            reason,
            IKRequestMsg.REQUEST_PRIMARY,
        )
        if result is not None:
            request_id, ik_data = result
            self._primary_ik_id = request_id
            self._primary_ik_data = ik_data
        else:
            self.clear_build_state()
            recovery_states = self._build_recovery_states(
                self._last_q
            )
            self._publish_trajectory(
                recovery_states,
                TrajectoryCommandMsg.COMMAND_REPLACE,
            )
            self.executing_trajectory = True

    def _build_recovery_states(self, current_q, delay_before=0.0):
        """Build a recovery trajectory: lift in place, then Q_SCAN.

        Keeps the current base angle during the lift to avoid
        sweeping the arm across the workspace.

        Arguments
        ---------
        current_q : np.ndarray or None
            Current joint angles.  If None, lifts from Q_SCAN.
        delay_before : float
            Delay before the first recovery waypoint (seconds).

        Returns
        -------
        list[TrajectoryState]
            Two-state recovery trajectory.
        """
        q_lifted = Q_SCAN.copy()
        if (current_q is not None
                and len(current_q) == len(JOINT_NAMES)):
            q_lifted[0] = current_q[0]

        # Lift with gripper still closed to avoid destroying the structure
        q_lifted_closed = q_lifted.copy()
        q_lifted_closed[5] = GRIPPER_MOTOR_CLOSED_RAD

        return [
            # 1. Lift to safe height with gripper closed
            TrajectoryState(
                final_state=ManipulatorState(
                    q=q_lifted_closed,
                    qd=np.zeros(len(JOINT_NAMES)),
                ),
                min_duration=DURATION / 2,
                delay_before=delay_before,
            ),
            # 2. Open gripper at safe height
            TrajectoryState(
                final_state=ManipulatorState(
                    q=q_lifted,
                    qd=np.zeros(len(JOINT_NAMES)),
                ),
                min_duration=self.structure.GRIPPER_DELAY,
            ),
            # 3. Return to Q_SCAN
            TrajectoryState(
                final_state=ManipulatorState(
                    q=Q_SCAN.copy(),
                    qd=np.zeros(len(JOINT_NAMES)),
                ),
                min_duration=DURATION / 2,
            ),
        ]

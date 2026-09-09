"""Speculative IK precomputation and queue management.

Owns the pending object queue, validates it against current detections,
and speculatively precomputes grasp IK for the next queued object while
the arm is in motion.  Follows the callback injection pattern: no ROS
imports, all side effects go through injected callables.
"""

from __future__ import annotations

from typing import Callable, Optional, TYPE_CHECKING


import numpy as np

from legobuilder.config import QUEUE_MOVEMENT_THRESHOLD
from legobuilder.schemas import Object

if TYPE_CHECKING:
    from legobuilder.brain.block_structure import BlockStructure

# IKRequestMsg.REQUEST_PRECOMPUTE == 1 (avoid ROS msg import)
_REQUEST_PRECOMPUTE = 1


class PrecomputeManager:
    """Speculative IK precomputation and pending-object queue owner.

    Manages the queue of pile objects waiting to be processed by
    BuildPipeline.  When enabled, speculatively precomputes grasp IK
    for the next queued object while the arm is executing a trajectory,
    so the pipeline can skip the IK round-trip on the next cycle.

    Follows the callback injection pattern: receives send_ik_request_fn
    as a constructor parameter and has no ROS imports.

    Attributes
    ----------
    pending_objects : list[Object]
        Queue of pile objects waiting to be processed.
    precomputed : tuple or None
        (obj, trajectory_states, context) when a precompute succeeds.
    structure : BlockStructure
        Reference to the target block structure (for plan_grasp).
    logger
        Logger instance for status messages.
    """

    # ── Lifecycle ─────────────────────────────────────────────────────

    def __init__(
        self,
        send_ik_request_fn: Callable,
        structure: Optional[BlockStructure],
        logger,
        enable_precompute: bool = True,
    ):
        """Initialize the precompute manager with injected callbacks.

        Arguments
        ---------
        send_ik_request_fn : callable
            (states, request_type) -> request_id.  Same callback used
            by BuildPipeline to send IK requests.
        structure : BlockStructure or None
            Target block structure (provides plan_grasp).
        logger
            Logger instance for info/debug messages.
        enable_precompute : bool
            When False, queue validation still runs but speculative
            IK is a no-op.
        """
        self._send_ik_request = send_ik_request_fn
        self.structure = structure
        self.logger = logger
        self._enable_precompute = enable_precompute

        self.pending_objects: list[Object] = []
        self._precompute_ik_id: str | None = None
        self._precompute_ik_data: dict | None = None
        self.precomputed: tuple | None = None
        self.target_cell: tuple | None = None
        self.block_type = None

    # ── Queue management ──────────────────────────────────────────────

    def select_and_queue(self, detected_pile: list[Object]) -> None:
        """Set queue from new pile detections.

        Called when the pipeline is idle and new detections arrive.

        Arguments
        ---------
        detected_pile : list[Object]
            Filtered pile objects from the current detection cycle.
        """
        self.pending_objects = list(detected_pile)

    def pop_next(self) -> tuple:
        """Pop and return the next object to process.

        If a precomputed result is available and its object matches the
        front of the queue, returns the precomputed trajectory data so
        the pipeline can skip the IK round-trip.

        Returns
        -------
        tuple
            (obj, precomputed_data) where precomputed_data is
            (trajectory_states, context) if precompute was valid, or
            None if the pipeline should plan normally.
        """
        if not self.pending_objects:
            return None, None

        if (self.precomputed is not None
                and len(self.pending_objects) > 0
                and self.precomputed[0] is self.pending_objects[0]):
            obj, trajectory_states, context = self.precomputed
            self.pending_objects.remove(obj)
            self.precomputed = None
            return obj, (trajectory_states, context)

        obj = self.pending_objects.pop(0)
        return obj, None

    # ── Speculative precompute ────────────────────────────────────────

    def precompute_next(self) -> None:
        """Speculatively precompute grasp IK for the next queued object.

        Sends an async IK request for the first pending object's grasp
        trajectory.  Does nothing if precompute is disabled, the queue
        is empty, or the structure is unavailable.
        """
        if not self._enable_precompute:
            return
        if not self.pending_objects:
            return
        if self.structure is None:
            return

        next_obj = self.pending_objects[0]
        self.logger.info(
            f"Precomputing trajectory for {next_obj.obj_type.name}"
            f" at ({next_obj.center_xyz[0]:.3f},"
            f" {next_obj.center_xyz[1]:.3f})"
        )
        
        result = self.structure.plan_grasp(
            detected_objects=self.pending_objects,
            target_cell=self.target_cell,
            block_type=self.block_type,
        )
        obj, grasp_states, context = result
        if obj is None or grasp_states is None:
            self.logger.debug(
                "Precompute: plan_grasp returned None, skipping"
            )
            return

        request_id = self._send_ik_request(grasp_states, _REQUEST_PRECOMPUTE)
        self._precompute_ik_id = request_id
        self._precompute_ik_data = {
            'obj': obj,
            'trajectory_states': grasp_states,
            'context': context,
            'request_id': request_id,
        }
        self.logger.info(
            f"Sent precompute IK request {request_id}"
            f" for {obj.obj_type.name}"
        )

    def on_precompute_response(
        self,
        request_id: str,
        success: bool,
        q_solutions: list[float],
    ) -> None:
        """Handle PRECOMPUTE IK response from solver.

        Discards stale or failed responses silently.  On success,
        applies IK solutions to the precomputed trajectory states and
        stores the result for pop_next to use.

        Arguments
        ---------
        request_id : str
            The IK request identifier from the response.
        success : bool
            Whether the IK solver found a valid solution.
        q_solutions : list[float]
            Flat list of solved joint angles.
        """
        if request_id != self._precompute_ik_id:
            self.logger.info(
                f"Discarding stale precompute response {request_id}"
            )
            return

        if not success:
            self.logger.info(
                f"Precompute IK request {request_id} failed"
            )
            self._precompute_ik_id = None
            self._precompute_ik_data = None
            return

        data = self._precompute_ik_data
        self._precompute_ik_id = None
        self._precompute_ik_data = None

        obj = data['obj']
        trajectory_states = data['trajectory_states']
        context = data['context']

        # Check that the precomputed object is still in the queue
        if obj not in self.pending_objects:
            self.logger.info(
                f"Precomputed object {obj.obj_type.name}"
                f" no longer in queue, discarding"
            )
            return

        # Apply IK solutions to trajectory states
        from legobuilder.brain.ros_bridge import apply_ik_solutions
        apply_ik_solutions(trajectory_states, q_solutions)

        self.precomputed = (obj, trajectory_states, context)
        self.logger.info(
            f"Precomputed {len(trajectory_states)} states"
            f" for {obj.obj_type.name}"
        )

    # ── Queue validation ──────────────────────────────────────────────

    def validate_queue(
        self,
        current_detections: list[Object],
        occlusion_check_fn: Callable,
        verbose: bool = False,
    ) -> None:
        """Validate queued objects against current detections.

        Removes objects that have moved or disappeared, unless they
        are occluded by the arm's current trajectory footprint.
        Invalidates precomputed data when the affected object is
        removed from the queue.

        Arguments
        ---------
        current_detections : list[Object]
            Pile blocks visible in the current frame.
        occlusion_check_fn : callable
            (xy) -> bool, returns True if the point is occluded by
            the arm.  Typically BuildPipeline.is_in_occlusion_zone.
        verbose : bool
            If True, log detailed per-object validation results.
        """
        to_remove = []
        if verbose:
            self.logger.info(
                f"Queue validation:"
                f" {len(self.pending_objects)} queued,"
                f" {len(current_detections)} pile detections"
            )
        for queued_obj in self.pending_objects:
            qxy = queued_obj.center_xyz[:2]
            same_type = [
                d for d in current_detections
                if d.obj_type == queued_obj.obj_type
            ]

            if same_type:
                dists = [
                    np.linalg.norm(
                        np.array(d.center_xyz[:2]) - np.array(qxy)
                    )
                    for d in same_type
                ]
                min_dist = min(dists)
                if min_dist > QUEUE_MOVEMENT_THRESHOLD:
                    occluded = occlusion_check_fn(qxy)
                    if not occluded:
                        to_remove.append(queued_obj)
                        if verbose:
                            self.logger.info(
                                f"  REMOVE "
                                f"{queued_obj.obj_type.name} "
                                f"({qxy[0]:.3f}, {qxy[1]:.3f}):"
                                f" {len(same_type)} same-type "
                                f"detected, "
                                f"nearest={min_dist:.3f}m "
                                f"> threshold="
                                f"{QUEUE_MOVEMENT_THRESHOLD}m"
                                f", not occluded"
                            )
                    else:
                        if verbose:
                            self.logger.info(
                                f"  KEEP "
                                f"{queued_obj.obj_type.name} "
                                f"({qxy[0]:.3f}, {qxy[1]:.3f}):"
                                f" {len(same_type)} same-type "
                                f"detected, "
                                f"nearest={min_dist:.3f}m "
                                f"> threshold, but "
                                f"OCCLUDED by arm"
                            )
                else:
                    if verbose:
                        self.logger.debug(
                            f"  KEEP "
                            f"{queued_obj.obj_type.name} "
                            f"({qxy[0]:.3f}, {qxy[1]:.3f}): "
                            f"matched at {min_dist:.3f}m "
                            f"<= {QUEUE_MOVEMENT_THRESHOLD}m"
                        )
            else:
                occluded = occlusion_check_fn(qxy)
                if not occluded:
                    to_remove.append(queued_obj)
                    if verbose:
                        self.logger.info(
                            f"  REMOVE "
                            f"{queued_obj.obj_type.name} "
                            f"({qxy[0]:.3f}, {qxy[1]:.3f}): "
                            f"no same-type detected, "
                            f"not occluded"
                        )
                else:
                    if verbose:
                        self.logger.info(
                            f"  KEEP "
                            f"{queued_obj.obj_type.name} "
                            f"({qxy[0]:.3f}, {qxy[1]:.3f}): "
                            f"no same-type detected, "
                            f"but OCCLUDED by arm"
                        )

        precomputed_invalidated = False
        for obj in to_remove:
            self.pending_objects.remove(obj)
            if (self.precomputed is not None
                    and obj is self.precomputed[0]):
                self.precomputed = None
                precomputed_invalidated = True
            if (self._precompute_ik_data is not None
                    and obj is self._precompute_ik_data.get('obj')):
                self._precompute_ik_id = None
                self._precompute_ik_data = None
                precomputed_invalidated = True

        if precomputed_invalidated:
            self.invalidate()

    # ── State management ──────────────────────────────────────────────

    def invalidate(self) -> None:
        """Invalidate current precompute state.

        Called when the queue changes during execution.  Clears any
        pending or completed precompute data and re-triggers precompute
        if the queue still has objects and precompute is enabled.
        """
        self.precomputed = None
        self._precompute_ik_id = None
        self._precompute_ik_data = None
        if self.pending_objects and self._enable_precompute:
            self.precompute_next()

    def clear(self) -> None:
        """Clear all state (queue, precompute).

        Called on pipeline reset (contact, grip failure, cycle end).
        """
        self.pending_objects = []
        self.precomputed = None
        self._precompute_ik_id = None
        self._precompute_ik_data = None
        self.target_cell = None
        self.block_type = None

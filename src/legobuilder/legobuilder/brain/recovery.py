"""Grip failure recovery logic extracted from BuildPipeline.

Handles diagonal and partial grip re-attempts via wrist rotation
(bad grip) and radial nudging from the original pick position (low grip).
Follows the callback injection pattern -- no ROS imports.
"""

from math import pi
from typing import Callable, Optional

import numpy as np

from legobuilder.config import (
    BASE_MOTOR_POS,
    DIAGONAL_GRIP_MAX_RETRIES,
    wrap_roll,
)
from legobuilder.brain.block_structure import BlockStructure


class RecoveryHandler:
    """Grip failure recovery with wrist rotation and radial nudging.

    Handles two failure modes detected by the visual grip checker:
    bad grip (diagonal/partial) where the block is misaligned
    in the gripper, and low grip where the gripper closed too low
    on the block.  Bad grip recovery rotates the wrist by 45 degrees
    and re-grips; low grip recovery nudges the pick position radially
    outward from the base motor.

    Follows the callback injection pattern: receives send_ik_request_fn
    as a constructor arg and has no ROS imports.

    Attributes
    ----------
    structure : BlockStructure
        Reference to the target block structure for plan_regrip calls.
    logger
        Logger instance for status messages.
    max_retries : int
        Maximum re-grip attempts before giving up.
    """

    # ── Lifecycle ─────────────────────────────────────────────────────

    def __init__(
        self,
        send_ik_request_fn: Callable,
        structure: Optional[BlockStructure],
        logger,
        max_retries: int = DIAGONAL_GRIP_MAX_RETRIES,
        post_event_fn=None,
    ):
        """Initialize the recovery handler with injected callbacks.

        Arguments
        ---------
        send_ik_request_fn : callable
            (states, request_type) -> request_id.  Same callback used
            by BuildPipeline to send IK requests.
        structure : BlockStructure or None
            Target block structure (provides plan_regrip).
        logger
            Logger instance for info/warning messages.
        max_retries : int
            Maximum number of re-grip attempts before aborting.
        post_event_fn : callable or None
            Dashboard event callback.
        """
        self._send_ik_request = send_ik_request_fn
        self.structure = structure
        self.logger = logger
        self.max_retries = max_retries
        self._post_event = post_event_fn or (lambda e: None)

        self._retry_count: int = 0
        self._original_p_pick: Optional[np.ndarray] = None

    # ── Public API ────────────────────────────────────────────────────

    def reset(self, p_pick: np.ndarray) -> None:
        """Reset retry state for a new grasp attempt.

        Must be called when a new grasp phase starts so that
        handle_low_grip computes fresh offsets from the original
        pick position.

        Arguments
        ---------
        p_pick : np.ndarray
            Original pick position (copied internally).
        """
        self._retry_count = 0
        self._original_p_pick = p_pick.copy()

    def handle_bad_grip(
        self,
        ctx: dict,
        reason: str,
        request_type: int,
    ) -> Optional[tuple]:
        """Attempt re-grip by rotating the wrist 45 degrees.

        Rotates the pickup roll by pi/4 and plans a regrip trajectory.
        Returns IK request info so the caller can update pipeline state.
        After max_retries failures, returns None to signal that recovery
        is exhausted and the caller should abort.

        Arguments
        ---------
        ctx : dict
            Pipeline pickup context with p_pick, pickup_roll, obj, etc.
        reason : str
            Human-readable description of the grip failure.
        request_type : int
            IK request type constant (e.g. IKRequestMsg.REQUEST_PRIMARY),
            passed in to avoid importing ROS message types.

        Returns
        -------
        tuple or None
            (request_id, ik_data_dict) on retry attempt, or None if
            max retries exceeded.
        """
        if self._retry_count < self.max_retries:
            self._retry_count += 1
            new_roll = wrap_roll(ctx['pickup_roll'] + pi / 4)
            ctx['pickup_roll'] = new_roll
            self._post_event({
                'type': 'error',
                'source': 'recovery',
                'data': {
                    'message': f'Bad grip: {reason}',
                    'retry': self._retry_count,
                    'max_retries': self.max_retries,
                },
                'timestamp': 0,
            })
            self.logger.info(
                f"Bad grip detected ({reason}), "
                f"re-gripping attempt "
                f"{self._retry_count}/{self.max_retries}"
            )
            regrip_states = self.structure.plan_regrip(
                ctx['p_pick'], new_roll,
            )
            request_id = self._send_ik_request(
                regrip_states, request_type
            )
            ik_data = {
                'obj': ctx.get('obj'),
                'trajectory_states': regrip_states,
                'phase': 'grasp',
            }
            return (request_id, ik_data)

        self.logger.warning(
            f"Bad grip persists after "
            f"{self.max_retries} retries ({reason}), "
            f"aborting to Q_SCAN"
        )
        return None

    def handle_low_grip(
        self,
        ctx: dict,
        reason: str,
        request_type: int,
    ) -> Optional[tuple]:
        """Attempt re-grip by nudging the pick position radially outward.

        Computes a fresh 1 cm radial nudge from the original pick
        position on each retry, fixing the mutation bug where
        incremental nudges would stack on the already-nudged position.

        Arguments
        ---------
        ctx : dict
            Pipeline pickup context with p_pick, pickup_roll, obj, etc.
        reason : str
            Human-readable description of the grip failure.
        request_type : int
            IK request type constant (e.g. IKRequestMsg.REQUEST_PRIMARY),
            passed in to avoid importing ROS message types.

        Returns
        -------
        tuple or None
            (request_id, ik_data_dict) on retry attempt, or None if
            max retries exceeded.
        """
        if self._retry_count < self.max_retries:
            self._retry_count += 1

            self._post_event({
                'type': 'error',
                'source': 'recovery',
                'data': {
                    'message': f'Low grip: {reason}',
                    'retry': self._retry_count,
                    'max_retries': self.max_retries,
                },
                'timestamp': 0,
            })

            # Fresh offset from the original pick position each retry,
            # not stacking incremental nudges on the mutated position.
            p_nudged = self._original_p_pick.copy()
            d = p_nudged[:2] - BASE_MOTOR_POS[:2]
            d_hat = d / np.linalg.norm(d)
            p_nudged[:2] += d_hat * 0.01 * self._retry_count
            ctx['p_pick'] = p_nudged

            self.logger.info(
                f"Low grip ({reason}), nudging 1cm outward to "
                f"({p_nudged[0]:.3f}, {p_nudged[1]:.3f}), "
                f"attempt {self._retry_count}/{self.max_retries}"
            )
            regrip_states = self.structure.plan_regrip(
                ctx['p_pick'], ctx['pickup_roll']
            )
            request_id = self._send_ik_request(
                regrip_states, request_type
            )
            ik_data = {
                'obj': ctx.get('obj'),
                'trajectory_states': regrip_states,
                'phase': 'grasp',
            }
            return (request_id, ik_data)

        self.logger.warning(
            f"Low grip persists after "
            f"{self.max_retries} retries ({reason}), "
            f"aborting to Q_SCAN"
        )
        return None

"""Stateful tracker for unreachable blocks across scan cycles.

Tracks nod attempts per grid cell and permanently ignores blocks
after MAX_NOD_ATTEMPTS failed nod gestures. Counters reset when a
block transitions from unreachable to reachable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BlockStatus:
    """Tracking state for a single unreachable block."""

    reason: str  # 'ungrippable', 'overhang', or 'both'
    block_type: int = 0  # ObjectType.value (1=Y, 2=B, 3=G, 4=R)
    nod_attempts: int = 0
    permanently_ignored: bool = False


class UnreachableTracker:
    """Track unreachable blocks across scan cycles.

    Maintains attempt counts and permanent-ignore state per grid cell.
    Re-evaluated each scan cycle: counters reset when a block's
    reachability status changes (was unreachable, now reachable).
    """

    MAX_NOD_ATTEMPTS = 4

    def __init__(self) -> None:
        self._blocks: dict[tuple[int, int, int], BlockStatus] = {}
        self._ignored: set[tuple[int, int, int]] = set()

    def update(
        self,
        current_unreachable: dict[tuple[int, int, int], tuple[str, int]],
        block_grid: np.ndarray | None = None,
        placed_grid: np.ndarray | None = None,
    ) -> None:
        """Update tracker with fresh reachability results.

        Blocks that were unreachable but are now reachable have their
        entries deleted (counter reset). Permanently ignored blocks
        are never deleted.

        Correctly placed blocks (desired != 0 and actual == desired)
        are excluded — they need no intervention even if geometrically
        unreachable.

        Parameters
        ----------
        current_unreachable
            Mapping of (row, col, layer) to (reason, block_type) for all
            currently unreachable cells from the latest scan.
        block_grid
            Design grid (desired block colors). If provided with
            placed_grid, used to filter out correctly placed blocks.
        placed_grid
            Detected grid (actual block colors).
        """
        if block_grid is not None and placed_grid is not None:
            current_unreachable = {
                cell: val
                for cell, val in current_unreachable.items()
                if not (block_grid[cell] != 0 and placed_grid[cell] == block_grid[cell])
            }

        # Blocks that were tracked but are now reachable -> delete
        previously_tracked = set(self._blocks.keys())
        now_reachable = previously_tracked - set(current_unreachable.keys()) - self._ignored
        for cell in now_reachable:
            del self._blocks[cell]

        # Update or add current unreachable blocks
        for cell, (reason, block_type) in current_unreachable.items():
            if cell in self._ignored:
                continue
            if cell in self._blocks:
                self._blocks[cell].reason = reason
                self._blocks[cell].block_type = block_type
            else:
                self._blocks[cell] = BlockStatus(
                    reason=reason, block_type=block_type,
                )
                
    def record_nod_attempt(self, cell: tuple[int, int, int]) -> None:
        """Record a nod gesture attempt for a block.

        No-op if cell is not tracked or already permanently ignored.
        After MAX_NOD_ATTEMPTS, the cell is permanently ignored for
        the remainder of the session.

        Parameters
        ----------
        cell
            Grid cell key (row, col, layer).
        """
        if cell not in self._blocks:
            return
        status = self._blocks[cell]
        if status.permanently_ignored:
            return
        status.nod_attempts += 1
        if status.nod_attempts >= self.MAX_NOD_ATTEMPTS:
            status.permanently_ignored = True
            self._ignored.add(cell)

    @property
    def ignored_cells(self) -> set[tuple[int, int, int]]:
        """Grid cells permanently ignored this session (returns a copy)."""
        return self._ignored.copy()

    @property
    def unreachable_cells(self) -> set[tuple[int, int, int]]:
        """All currently unreachable cells (active and ignored)."""
        return set(self._blocks.keys()) | self._ignored

    def get_status(self, cell: tuple[int, int, int]) -> BlockStatus | None:
        """Get tracking status for a cell, or None if not tracked."""
        return self._blocks.get(cell)

    def get_all_statuses(self) -> dict[tuple[int, int, int], BlockStatus]:
        """All tracked block statuses for dashboard display."""
        return dict(self._blocks)

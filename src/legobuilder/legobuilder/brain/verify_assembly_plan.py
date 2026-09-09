"""Assembly plan verifier for the legobuilder system.

Validates that an assembly JSON describes a physically buildable
structure before the robot attempts it.  Checks grid bounds,
duplicate positions/IDs, valid colors, build order support,
cantilever limits, ground connectivity, and optional reachability.

Usage::

    python -m legobuilder.brain.verify_assembly_plan path/to/assembly.json
    python verify_assembly_plan.py path/to/assembly.json
"""

import json
import sys
import heapq
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

# Handle both direct execution and module import
try:
    from legobuilder.schemas import Color
    from legobuilder.config import MAX_GRID_SIZE
except ImportError:
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from schemas import Color
    from config import MAX_GRID_SIZE


VALID_COLORS = {'orange', 'yellow', 'blue', 'green', 'red'}

NEIGHBORS_6 = [
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1),
]


# ── Data types ────────────────────────────────────────────────────────

@dataclass
class VerificationReport:
    """Result of assembly plan verification.

    Attributes
    ----------
    passed : bool
        True if no errors were found.
    errors : list[str]
        Error messages (any error means the plan is invalid).
    warnings : list[str]
        Warning messages (plan may still be valid).
    """

    passed: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        """Format the report as a human-readable string."""
        lines = []
        if self.passed:
            lines.append("PASSED: Assembly plan is valid.")
        else:
            lines.append("FAILED: Assembly plan has errors.")
        if self.errors:
            lines.append(f"\n  Errors ({len(self.errors)}):")
            for e in self.errors:
                lines.append(f"    - {e}")
        if self.warnings:
            lines.append(f"\n  Warnings ({len(self.warnings)}):")
            for w in self.warnings:
                lines.append(f"    - {w}")
        return '\n'.join(lines)


# ── Verifier ──────────────────────────────────────────────────────────

class AssemblyPlanVerifier:
    """Validate that an assembly JSON is physically buildable.

    Runs eight sequential checks against the block list and grid
    metadata.  Each check returns a (errors, warnings) tuple.

    Checks
    ------
    1. Grid bounds -- all positions within declared gridSize, z >= 1.
    2. Duplicate positions -- no two blocks at the same (x, y, z).
    3. Duplicate IDs -- all block IDs unique.
    4. Valid colors -- all colors map to known Color enum values.
    5. Build order support -- each placed block has a 6-connected
       neighbor among previously placed blocks (or is at z=1).
    6. Cantilever limits -- no block exceeds max_cantilever
       horizontal hops from a vertically-supported column.
    7. Ground connectivity -- all placed blocks form a single
       connected component touching ground.
    8. Reachability -- optional via user-provided callable.

    Attributes
    ----------
    json_path : str
        Path to the assembly JSON file.
    max_cantilever : int
        Maximum allowed horizontal hops from vertical support.
    is_point_reachable : callable or None
        Optional (x, y, z) -> bool reachability predicate.
    grid_size : dict
        Grid dimensions from JSON metadata.
    blocks : list[dict]
        Block definitions from JSON.
    """

    # ── Lifecycle ─────────────────────────────────────────────────────

    def __init__(
        self,
        json_path: str,
        max_cantilever: int = 2,
        is_point_reachable: (
            Callable[[tuple[int, int, int]], bool] | None
        ) = None,
    ):
        """Load and parse the assembly JSON.

        Arguments
        ---------
        json_path : str
            Path to the assembly JSON file.
        max_cantilever : int
            Maximum horizontal cantilever distance allowed.
        is_point_reachable : callable or None
            Optional reachability check for each block position.
        """
        self.json_path = json_path
        self.max_cantilever = max_cantilever
        self.is_point_reachable = is_point_reachable

        with open(json_path, 'r') as f:
            data = json.load(f)

        metadata = data.get("metadata", {})
        self.grid_size = metadata.get("gridSize", {})
        self.blocks = data.get("blocks", [])

    # ── Public API ────────────────────────────────────────────────────

    def verify(self) -> VerificationReport:
        """Run all checks and return a VerificationReport."""
        report = VerificationReport()

        checkers = [
            self._check_grid_bounds,
            self._check_duplicate_positions,
            self._check_duplicate_ids,
            self._check_valid_colors,
            self._check_build_order_support,
            self._check_cantilever_limits,
            self._check_ground_connectivity,
            self._check_reachability,
        ]

        for checker in checkers:
            errors, warnings = checker()
            report.errors.extend(errors)
            report.warnings.extend(warnings)

        report.passed = len(report.errors) == 0
        return report

    # ── Private helpers ───────────────────────────────────────────────

    def _get_pos(self, block: dict) -> tuple[int, int, int]:
        """Extract (x, y, z) position tuple from a block dict."""
        pos = block["position"]
        return (pos["x"], pos["y"], pos["z"])

    def _check_grid_bounds(self) -> tuple[list[str], list[str]]:
        """Check all positions are within declared grid and z >= 1."""
        errors = []
        length = self.grid_size.get("length", 0)
        width = self.grid_size.get("width", 0)
        height = self.grid_size.get("height", 0)

        max_l = MAX_GRID_SIZE["length"]
        max_w = MAX_GRID_SIZE["width"]
        max_h = MAX_GRID_SIZE["height"]
        if length > max_l:
            errors.append(
                f"gridSize length {length} exceeds max {max_l}"
            )
        if width > max_w:
            errors.append(
                f"gridSize width {width} exceeds max {max_w}"
            )
        if height > max_h:
            errors.append(
                f"gridSize height {height} exceeds max {max_h}"
            )

        for block in self.blocks:
            bid = block.get("id", "?")
            pos = block.get("position", {})
            x, y, z = pos.get("x"), pos.get("y"), pos.get("z")

            if x is None or y is None or z is None:
                errors.append(
                    f"Block {bid}: missing position coordinates"
                )
                continue

            if z < 1:
                errors.append(
                    f"Block {bid} at ({x},{y},{z}): "
                    f"z must be >= 1 (z=1 is ground level)"
                )
            if not (0 <= x < length):
                errors.append(
                    f"Block {bid} at ({x},{y},{z}): "
                    f"x={x} out of grid bounds [0, {length})"
                )
            if not (0 <= y < width):
                errors.append(
                    f"Block {bid} at ({x},{y},{z}): "
                    f"y={y} out of grid bounds [0, {width})"
                )
            if not (1 <= z < height):
                errors.append(
                    f"Block {bid} at ({x},{y},{z}): "
                    f"z={z} out of grid bounds [1, {height})"
                )

        return errors, []

    def _check_duplicate_positions(
        self,
    ) -> tuple[list[str], list[str]]:
        """Check that no two blocks share the same position."""
        errors = []
        seen = {}
        for block in self.blocks:
            bid = block.get("id", "?")
            pos = self._get_pos(block)
            if pos in seen:
                errors.append(
                    f"Block {bid} at {pos}: duplicate position "
                    f"(already used by block {seen[pos]})"
                )
            else:
                seen[pos] = bid
        return errors, []

    def _check_duplicate_ids(self) -> tuple[list[str], list[str]]:
        """Check that all block IDs are unique."""
        errors = []
        seen = set()
        for block in self.blocks:
            bid = block.get("id", "?")
            if bid in seen:
                errors.append(f"Block ID {bid}: duplicate ID")
            seen.add(bid)
        return errors, []

    def _check_valid_colors(self) -> tuple[list[str], list[str]]:
        """Check that all block colors are recognized."""
        errors = []
        for block in self.blocks:
            bid = block.get("id", "?")
            color = block.get("color", "")
            try:
                Color.from_str(color)
            except ValueError:
                errors.append(
                    f"Block {bid}: unknown color '{color}' "
                    f"(valid: {', '.join(sorted(VALID_COLORS))})"
                )
        return errors, []

    def _check_build_order_support(
        self,
    ) -> tuple[list[str], list[str]]:
        """Check each block has 6-connected support when placed.

        Blocks at z=1 are ground-supported.  All others must have
        at least one already-placed neighbor.
        """
        errors = []
        placed = set()

        for block in self.blocks:
            bid = block.get("id", "?")
            x, y, z = self._get_pos(block)

            if z == 1:
                placed.add((x, y, z))
                continue

            has_support = False
            for dx, dy, dz in NEIGHBORS_6:
                nx, ny, nz = x + dx, y + dy, z + dz
                if nz == 0:
                    continue
                if (nx, ny, nz) in placed:
                    has_support = True
                    break

            if not has_support:
                errors.append(
                    f"Block {bid} at ({x},{y},{z}): no adjacent "
                    f"support when placed "
                    f"(build step {len(placed) + 1})"
                )

            placed.add((x, y, z))

        return errors, []

    def _check_cantilever_limits(
        self,
    ) -> tuple[list[str], list[str]]:
        """Check no block exceeds max_cantilever horizontal hops.

        At each build step: (1) mark vertically-supported blocks
        bottom-up, (2) Dijkstra from all vertically-supported blocks
        with horizontal moves costing 1 and vertical moves costing
        0, (3) flag blocks exceeding the limit.
        """
        errors = []
        placed = set()

        for block in self.blocks:
            bid = block.get("id", "?")
            pos = self._get_pos(block)
            placed.add(pos)

            vert_supported = set()
            by_z = sorted(placed, key=lambda p: p[2])
            for (bx, by, bz) in by_z:
                if bz == 1:
                    vert_supported.add((bx, by, bz))
                elif (bx, by, bz - 1) in vert_supported:
                    vert_supported.add((bx, by, bz))

            dist = {}
            heap = []
            for vp in vert_supported:
                dist[vp] = 0
                heapq.heappush(heap, (0, vp))

            while heap:
                d, cur = heapq.heappop(heap)
                if d > dist.get(cur, float('inf')):
                    continue
                for dx, dy, dz in NEIGHBORS_6:
                    nb = (
                        cur[0] + dx, cur[1] + dy, cur[2] + dz,
                    )
                    if nb not in placed:
                        continue
                    cost = (
                        0 if (dx == 0 and dy == 0) else 1
                    )
                    nd = d + cost
                    if nd < dist.get(nb, float('inf')):
                        dist[nb] = nd
                        heapq.heappush(heap, (nd, nb))

            for p in placed:
                cantilever_dist = dist.get(p, float('inf'))
                if cantilever_dist > self.max_cantilever:
                    errors.append(
                        f"Block at {p}: cantilever distance "
                        f"{cantilever_dist} exceeds limit "
                        f"{self.max_cantilever} "
                        f"(after placing block {bid})"
                    )

        return errors, []

    def _check_ground_connectivity(
        self,
    ) -> tuple[list[str], list[str]]:
        """Check all placed blocks form a single connected component.

        Uses BFS from a z=1 block through 6-connected neighbors.
        Disconnected islands are reported as warnings.
        """
        warnings = []
        placed = set()

        for block in self.blocks:
            bid = block.get("id", "?")
            pos = self._get_pos(block)
            placed.add(pos)

            start = None
            for p in placed:
                if p[2] == 1:
                    start = p
                    break

            if start is None:
                continue

            visited = set()
            queue = deque([start])
            visited.add(start)
            while queue:
                cur = queue.popleft()
                for dx, dy, dz in NEIGHBORS_6:
                    nb = (
                        cur[0] + dx, cur[1] + dy, cur[2] + dz,
                    )
                    if nb in placed and nb not in visited:
                        visited.add(nb)
                        queue.append(nb)

            unreached = placed - visited
            if unreached:
                warnings.append(
                    f"After placing block {bid}: "
                    f"{len(unreached)} block(s) not connected "
                    f"to ground component: {sorted(unreached)}"
                )

        return [], warnings

    def _check_reachability(
        self,
    ) -> tuple[list[str], list[str]]:
        """Check each block position is reachable (if callable set)."""
        if self.is_point_reachable is None:
            return [], []

        errors = []
        for block in self.blocks:
            bid = block.get("id", "?")
            pos = self._get_pos(block)
            if not self.is_point_reachable(pos):
                errors.append(
                    f"Block {bid} at {pos}: position not "
                    f"reachable by robot arm"
                )

        return errors, []


def main():
    """Run the assembly plan verifier from the command line."""
    if len(sys.argv) < 2:
        print(
            f"Usage: {sys.argv[0]} <assembly.json> "
            f"[max_cantilever]"
        )
        sys.exit(1)

    json_path = sys.argv[1]
    max_cantilever = (
        int(sys.argv[2]) if len(sys.argv) > 2 else 2
    )

    verifier = AssemblyPlanVerifier(
        json_path, max_cantilever=max_cantilever,
    )
    report = verifier.verify()
    print(report)
    sys.exit(0 if report.passed else 1)


if __name__ == "__main__":
    main()

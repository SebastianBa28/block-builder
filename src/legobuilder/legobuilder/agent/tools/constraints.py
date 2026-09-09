"""Assembly constraint reference tool for the design agent."""

from legobuilder.config import MAX_GRID_SIZE


def get_assembly_constraints() -> str:
    """Return all assembly constraints and rules the robot enforces.

    Call this to review the JSON schema, coordinate system, valid colors,
    grid bounds, build order rules, cantilever limits, and other
    constraints before generating or editing an assembly.

    Returns:
        A string describing every constraint the assembly must satisfy.
    """
    max_l = MAX_GRID_SIZE["length"]
    max_w = MAX_GRID_SIZE["width"]
    max_h = MAX_GRID_SIZE["height"]
    return f"""\
## Assembly JSON Schema

```json
{{
  "metadata": {{
    "version": "1.0",
    "title": "<short title>",
    "description": "<what the structure looks like>",
    "gridSize": {{ "length": N, "width": N, "height": N }}
  }},
  "blocks": [
    {{ "id": 0, "color": "<color>", "position": {{ "x": INT, "y": INT, "z": INT }} }},
    ...
  ]
}}
```

## Coordinate System
- x = length axis, y = width axis, z = height axis.
- z=1 is ground level. z=0 is invalid (below ground).
- Grid bounds: 0 <= x < gridSize.length, 0 <= y < gridSize.width, 1 <= z < gridSize.height.
- Maximum grid size: {max_l}x{max_w}x{max_h} (length x width x height). The JSON gridSize must not exceed this.
- gridSize should be exactly large enough to contain the structure. No margin needed.

## Valid Colors
orange, yellow, blue, green, red

## Block IDs
- Must be unique integers.
- Typically sequential starting from 0.

## Build Order (CRITICAL)
- Blocks are placed by the robot in the order they appear in the "blocks" array.
- Each block (except those at z=1) MUST have at least one 6-connected neighbor
  (up/down/left/right/front/back) that was already placed (i.e., appears earlier
  in the array).
- Ground-level blocks (z=1) are always supported.
- Best practice: list blocks bottom-to-top, layer by layer.

## Cantilever Limit
- Maximum 2 horizontal hops from the nearest vertically-supported column.
- A vertically-supported column is a stack of blocks reaching down to z=1.
- Bridges and overhangs are allowed within this limit.

## Ground Connectivity
- All blocks must form a single connected component (via 6-connectivity)
  that touches the ground (z=1).
- Floating islands are not allowed.

## Inventory
- Call count_available_blocks() to check current stock.
- Do not exceed available block counts per color.
"""

"""ASCII visualization tool for the design agent."""

import json
from collections import defaultdict

_COLOR_INITIAL = {
    'green': 'G',
    'blue': 'B',
    'yellow': 'Y',
    'red': 'R',
    'orange': 'O',
}


def visualize_assembly(assembly_json: str) -> str:
    """Render a layer-by-layer ASCII visualization of a block assembly.

    Shows each z-level as a 2D grid with color initials
    (G=green, B=blue, Y=yellow, R=red, O=orange) and '.' for empty.

    Args:
        assembly_json: Complete assembly JSON string with metadata and blocks.

    Returns:
        Multi-line ASCII art showing the assembly layer by layer, bottom to top.
    """
    try:
        data = json.loads(assembly_json)
    except json.JSONDecodeError as e:
        return f"Cannot visualize: invalid JSON — {e}"

    grid_size = data.get("metadata", {}).get("gridSize", {})
    length = grid_size.get("length", 0)
    width = grid_size.get("width", 0)
    blocks = data.get("blocks", [])

    if not blocks:
        return "No blocks in assembly."

    # Group blocks by z-level
    by_z: dict[int, list[dict]] = defaultdict(list)
    for block in blocks:
        pos = block.get("position", {})
        by_z[pos.get("z", 0)].append(block)

    lines = []
    for z in sorted(by_z.keys()):
        lines.append(f"Layer z={z}:")

        # Build lookup: (x, y) -> block
        lookup = {}
        for block in by_z[z]:
            pos = block["position"]
            lookup[(pos["x"], pos["y"])] = block

        # Header row with x indices
        x_labels = "     " + "".join(f"{x:<4}" for x in range(length))
        lines.append(x_labels)

        for y in range(width):
            row = f"y={y:<2} "
            for x in range(length):
                if (x, y) in lookup:
                    b = lookup[(x, y)]
                    initial = _COLOR_INITIAL.get(
                        b.get("color", ""), "?",
                    )
                    row += f"{initial}{b['id']:<3}"
                else:
                    row += ".   "
            lines.append(row)

        lines.append("")

    return '\n'.join(lines)

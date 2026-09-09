"""Block inventory tool for the design agent."""

from legobuilder.config import AVAILABLE_BLOCKS


# Map ObjectType names to display color names
_TYPE_TO_COLOR = {
    'GREEN_BLOCK': 'green',
    'YELLOW_BLOCK': 'yellow',
    'BLUE_BLOCK': 'blue',
    'RED_BLOCK': 'red',
}


def count_available_blocks() -> str:
    """Return the number of physically available blocks per color.

    Reads from the robot's inventory configuration.

    Returns:
        Formatted string showing available block counts by color and total.
    """
    lines = ["Available blocks:"]
    total = 0
    for obj_type, count in AVAILABLE_BLOCKS.items():
        color_name = _TYPE_TO_COLOR.get(obj_type.name, obj_type.name)
        lines.append(f"  {color_name}: {count}")
        total += count
    lines.append(f"Total: {total}")
    return '\n'.join(lines)

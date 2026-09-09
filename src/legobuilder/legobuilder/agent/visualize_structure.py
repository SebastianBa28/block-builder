"""Interactive script to visualize assembly JSONs as 3D matplotlib plots.

Lists available assemblies, lets the user pick one by number, then opens
an interactive 3D view using BlockStructure.visualize().

Usage:
    uv run python -m legobuilder.agent.visualize_structure
"""

import logging
import sys
from pathlib import Path

from legobuilder.brain.block_structure import BlockStructure

_ASSEMBLIES_DIR = Path(__file__).resolve().parents[2] / "assemblies"

logger = logging.getLogger("visualize_structure")


def main():
    jsons = sorted(p for p in _ASSEMBLIES_DIR.glob("*.json"))

    if not jsons:
        print("No assembly JSON files found.")
        sys.exit(1)

    print("Available assemblies:\n")
    for i, path in enumerate(jsons, 1):
        print(f"  {i}. {path.name}")

    print()
    try:
        choice = int(input("Enter number to visualize: "))
    except (ValueError, EOFError):
        print("Invalid input.")
        sys.exit(1)

    if choice < 1 or choice > len(jsons):
        print(f"Out of range (1-{len(jsons)}).")
        sys.exit(1)

    selected = jsons[choice - 1]
    print(f"\nLoading {selected.name}...")

    structure = BlockStructure.from_json(str(selected), logger=logger)
    print(structure)
    structure.visualize()


if __name__ == "__main__":
    main()

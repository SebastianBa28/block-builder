#!/usr/bin/env python3
"""Simple script to exercise BlockStructure methods."""

import logging
from legobuilder.brain.block_structure import BlockStructure
from legobuilder.schemas import ObjectType

logger = logging.getLogger("test_block_structure")

s = BlockStructure(logger=logger, max_length=4, max_width=4, max_height=4)
print(s)

# Stack two blocks at (0, 0)
layer = s.place_top_block(0, 0, ObjectType.YELLOW_BLOCK)
print(f"Placed YELLOW at (0,0) -> layer {layer}")

layer = s.place_top_block(0, 0, ObjectType.BLUE_BLOCK)
print(f"Placed BLUE at (0,0) -> layer {layer}")

print(f"Height at (0,0): {s.get_height(0, 0)}")
print(f"Block at (0,0,0): {s.get_block(0, 0, 0)}")
print(f"Block at (0,0,1): {s.get_block(0, 0, 1)}")

# Magnetic side-attach: block at (0,1) layer 1 — no block under it, but neighbor at (0,0,1)
s.place_block(0, 1, 1, ObjectType.RED_BLOCK)
print("Placed RED at (0,1,1) via side-attach")

# Ground block
layer = s.place_top_block(1, 1, ObjectType.GREEN_BLOCK)
print(f"Placed GREEN at (1,1) -> layer {layer}")

# World positions
print(f"World pos (0,0,0): {s.get_world_position(0, 0, 0)}")
print(f"World pos (0,1,1): {s.get_world_position(0, 1, 1)}")

# Remove top block from (0,0)
removed = s.remove_block(0, 0, 1)
print(f"Removed from (0,0,1): {removed}")
print(f"Height at (0,0) now: {s.get_height(0, 0)}")

print(s)
s.visualize()

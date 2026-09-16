"""Unit tests for BlockStructure.

Verifies QUAL-01: the method must handle all layer configurations
without IndexError and correctly identify visible vs occluded blocks.
Also verifies get_world_position Z correctness with 1-based layers.
"""

import numpy as np

from legobuilder.config import BLOCK_SIZE


def test_simulated_grid_top_layer_no_crash(make_block_structure):
    """Verify block on topmost layer does not cause IndexError."""
    bs = make_block_structure(5, 5, 3)
    # Place a block at the very top layer (index 2 of 0..2)
    bs.placed_grid[0, 0, 2] = 1  # ObjectType.RED_BLOCK.value
    bs.block_grid[0, 0, 2] = 1

    # This must not raise IndexError
    result = bs.get_simulated_detection_grid()
    assert result[0, 0, 2] == 1


def test_simulated_grid_top_visible(make_block_structure):
    """Verify block on topmost occupied layer is visible."""
    bs = make_block_structure(5, 5, 3)
    bs.placed_grid[1, 1, 1] = 2  # GREEN_BLOCK on layer 1
    bs.block_grid[1, 1, 1] = 2

    result = bs.get_simulated_detection_grid()
    assert result[1, 1, 1] == 2


def test_simulated_grid_hidden_block(make_block_structure):
    """Verify block with another block above it is not visible."""
    bs = make_block_structure(5, 5, 3)
    # Bottom block at layer 0
    bs.placed_grid[2, 2, 0] = 3  # BLUE_BLOCK
    bs.block_grid[2, 2, 0] = 3
    # Block above it at layer 1
    bs.placed_grid[2, 2, 1] = 4  # YELLOW_BLOCK
    bs.block_grid[2, 2, 1] = 4

    result = bs.get_simulated_detection_grid()
    # Bottom block is hidden
    assert result[2, 2, 0] == 0
    # Top block is visible
    assert result[2, 2, 1] == 4


def test_simulated_grid_empty(make_block_structure):
    """Verify empty placed_grid returns all zeros."""
    bs = make_block_structure(5, 5, 3)

    result = bs.get_simulated_detection_grid()
    assert np.all(result == 0)


def test_get_world_position_layer1_z_no_double_count(make_block_structure):
    """Layer 1 Z = origin_z + (BLOCK_SIZE/2 - 0.01), no base double-counting.

    Regression: origin[2]=BLOCK_SIZE accounts for the 5x5 base grid.
    Layer 1 (ground level in JSON) must NOT add another BLOCK_SIZE.
    """
    bs = make_block_structure(5, 5, 3)
    pos = bs.get_world_position(0, 0, 1)
    expected_z = BLOCK_SIZE + (BLOCK_SIZE / 2 - 0.01)
    assert abs(pos[2] - expected_z) < 1e-9, (
        f"Layer 1 Z={pos[2]:.6f} != expected {expected_z:.6f}"
    )

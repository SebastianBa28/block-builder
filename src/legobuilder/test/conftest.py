"""Shared test fixtures for legobuilder unit tests.

Provides lightweight BlockStructure instances with controlled grids
for testing grid-based logic without ROS or hardware dependencies.
"""

from unittest.mock import MagicMock

from legobuilder.brain.block_structure import BlockStructure
from legobuilder.config import BLOCK_SIZE
import numpy as np
import pytest


@pytest.fixture
def make_block_structure():
    """Create a BlockStructure with a mock logger.

    Returns a callable (rows, cols, layers) -> BlockStructure with
    zeroed block_grid and placed_grid ready for manual population.
    """
    def _factory(rows: int = 5, cols: int = 5, layers: int = 3):
        logger = MagicMock()
        bs = BlockStructure(
            logger=logger,
            max_length=rows,
            max_width=cols,
            max_height=layers,
            origin=np.array([0.4, 0.4, BLOCK_SIZE]),
        )
        return bs
    return _factory

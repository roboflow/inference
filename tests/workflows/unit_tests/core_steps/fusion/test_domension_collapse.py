import pytest

from inference.core.workflows.core_steps.fusion.dimension_collapse import (
    DimensionCollapseBlock,
)
from inference.core.workflows.entities.base import Batch


@pytest.mark.asyncio
async def test_dimension_collapse() -> None:
    # given
    step = DimensionCollapseBlock()
    data = Batch(content=[1, 2, 3, 4], indices=[(0, 1), (0, 2), (0, 3), (0, 4)])

    # when
    result = await step.run(data=data)

    # then
    assert result == {"output": [1, 2, 3, 4]}

import pytest

from inference.core.workflows.core_steps.fusion.dimension_collapse.v1 import (
    DimensionCollapseBlockV1,
)
from inference.core.workflows.execution_engine.entities.base import Batch


def test_dimension_collapse() -> None:
    # given
    step = DimensionCollapseBlockV1()
    data = Batch(content=[1, 2, 3, 4], indices=[(0, 1), (0, 2), (0, 3), (0, 4)])

    # when
    result = step.run(data=data)

    # then
    assert result == {"output": [1, 2, 3, 4]}

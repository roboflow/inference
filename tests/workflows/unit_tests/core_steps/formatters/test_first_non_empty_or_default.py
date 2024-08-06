import pytest

from inference.core.workflows.core_steps.formatters.first_non_empty_or_default.version_1 import (
    FirstNonEmptyOrDefaultBlockV1,
)
from inference.core.workflows.execution_engine.entities.base import Batch


@pytest.mark.asyncio
async def test_block_run_when_only_empty_data_provided() -> None:
    # given
    step = FirstNonEmptyOrDefaultBlockV1()
    data = Batch(content=[None, None, None], indices=[(0,), (1,), (2,)])

    # when
    result = await step.run(data=data, default="some")

    # then
    assert result == {"output": "some"}


@pytest.mark.asyncio
async def test_block_run_when_only_empty_single_non_empty_data_element_provided() -> (
    None
):
    # given
    step = FirstNonEmptyOrDefaultBlockV1()
    data = Batch(content=[None, "non-empty", None], indices=[(0,), (1,), (2,)])

    # when
    result = await step.run(data=data, default="some")

    # then
    assert result == {"output": "non-empty"}


@pytest.mark.asyncio
async def test_block_run_when_only_empty_multiple_non_empty_data_element_provided() -> (
    None
):
    # given
    step = FirstNonEmptyOrDefaultBlockV1()
    data = Batch(
        content=[None, "non-empty", "another-non-empty"], indices=[(0,), (1,), (2,)]
    )

    # when
    result = await step.run(data=data, default="some")

    # then
    assert result == {"output": "non-empty"}

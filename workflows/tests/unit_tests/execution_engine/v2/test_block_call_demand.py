"""``block_call`` answers ``self.wants`` exactly as the engine does.

A block author can unit-test the branch that skips an unwanted output without
compiling a workflow: ``wanted`` sets the demand of the call and
``call.queried`` shows what the block asked.
"""

import pytest
from roboflow_workflows.execution_engine.v2.errors import (
    ContractError,
    EventEmissionError,
)
from roboflow_workflows.execution_engine.v2.reactions.testing import block_call

from tests.unit_tests.execution_engine.v2.m7_demand.blocks import Painter


def test_an_unwanted_output_can_be_left_out_and_the_query_is_visible():
    painter = Painter()

    with block_call(painter, wanted=("count",)) as call:
        result = painter.run(value=1.0)

    assert result == {"count": 1}
    assert call.queried == {"overlay"}


def test_without_wanted_every_output_is_wanted():
    painter = Painter()

    with block_call(painter) as call:
        result = painter.run(value=1.0)

    assert result == {"count": 1, "overlay": "paint:1"}
    assert call.queried == {"overlay"}


def test_the_helper_and_the_engine_reject_an_undeclared_name_with_one_message():
    painter = Painter()

    with block_call(painter, step_path=("painter",)):
        with pytest.raises(
            ContractError,
            match=r"\$steps\.painter \(test/m7_painter@v1\) asked wants\('nope'\), "
            r"but declares no such output",
        ):
            painter.wants("nope")


def test_wanted_must_name_declared_outputs():
    with pytest.raises(ContractError, match=r"wanted \['overlya'\]"):
        with block_call(Painter, wanted=("overlya",)):
            pass
    with pytest.raises(ContractError, match="collection of output names"):
        with block_call(Painter, wanted="count"):
            pass


def test_wants_outside_a_call_does_not_mention_events():
    with pytest.raises(EventEmissionError) as raised:
        Painter().wants("overlay")

    assert "demand answers exist only while the engine runs the block" in str(
        raised.value
    )

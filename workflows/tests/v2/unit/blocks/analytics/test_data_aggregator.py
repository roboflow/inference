"""Unit tests of ``v2/data_aggregator@v1``: statistics, intervals and checks."""

from typing import Any, Dict, List

import numpy as np
import pytest
import torch
from pydantic import ValidationError
from roboflow_workflows.v2.blocks.analytics.data_aggregator import (
    DataAggregatorBlock,
    output_names,
)
from roboflow_workflows.v2.data.kinds import (
    BOOLEAN_KIND,
    DICTIONARY_KIND,
    FLOAT_KIND,
    INTEGER_KIND,
    LIST_OF_VALUES_KIND,
)

from tests.v2.support.sinks.loopback import ManualClock

DONE = {"interval_complete": True}
OPEN = {"interval_complete": False}
ALL_MODES = [
    "sum",
    "avg",
    "max",
    "min",
    "count",
    "distinct",
    "count_distinct",
    "values_counts",
    "values_difference",
]


def _run(block: DataAggregatorBlock, value: Any, **params: Any) -> Dict[str, Any]:
    arguments: Dict[str, Any] = {
        "data_operations": {},
        "aggregation_mode": {"x": ["sum"]},
        "interval_unit": "runs",
        "interval": 1,
    }
    arguments.update(params)
    result = block.run(data={"x": value}, **arguments)

    return result


def _run_all(values: List[Any], **params: Any) -> List[Dict[str, Any]]:
    block = DataAggregatorBlock()
    results = [_run(block, value, interval=len(values), **params) for value in values]

    return results


def test_numeric_statistics_appear_only_at_the_interval_end() -> None:
    modes = ["sum", "avg", "max", "min", "values_difference", "count"]

    results = _run_all([2, 5, 1], aggregation_mode={"x": modes})

    assert results[:2] == [{**OPEN, **dict.fromkeys(f"x_{mode}" for mode in modes)}] * 2
    assert results[2] == {
        **DONE,
        "x_sum": 8,
        "x_avg": pytest.approx(8 / 3),
        "x_max": 5,
        "x_min": 1,
        "x_values_difference": 4,
        "x_count": 3,
    }


def test_sequence_statistics_expand_lists() -> None:
    modes = ["count", "distinct", "count_distinct", "values_counts"]

    results = _run_all([["car", "bus"], ["car"], []], aggregation_mode={"x": modes})

    final = results[-1]
    assert final["x_count"] == 3
    assert sorted(final["x_distinct"]) == ["bus", "car"]
    assert final["x_count_distinct"] == 2
    assert final["x_values_counts"] == {"car": 2, "bus": 1}


def test_a_string_counts_once() -> None:
    results = _run_all(["car", "truck"], aggregation_mode={"x": ["count"]})

    assert results[-1] == {**DONE, "x_count": 2}


def test_tensors_and_arrays_become_python_values() -> None:
    values = [torch.tensor(2.5), np.int64(3), torch.tensor([1, 1, 2])]

    sums = _run_all(values[:2], aggregation_mode={"x": ["sum"]})
    distinct = _run_all(values[2:], aggregation_mode={"x": ["values_counts"]})

    assert sums[-1] == {**DONE, "x_sum": 5.5}
    assert isinstance(sums[-1]["x_sum"], float)
    assert distinct[-1] == {**DONE, "x_values_counts": {1: 2, 2: 1}}


def test_state_restarts_after_each_interval() -> None:
    block = DataAggregatorBlock()

    results = [_run(block, value, interval=2) for value in (1, 2, 10, 20)]

    assert [result["x_sum"] for result in results] == [None, 3, None, 30]
    assert [result["interval_complete"] for result in results] == [
        False,
        True,
        False,
        True,
    ]


def test_empty_interval_keeps_v1_values() -> None:
    block = DataAggregatorBlock()

    result = _run(block, [], aggregation_mode={"x": ["count", "distinct"]})

    assert result == {**DONE, "x_count": 0, "x_distinct": []}


def test_time_interval_uses_the_clock() -> None:
    clock = ManualClock()
    block = DataAggregatorBlock(aggregation_clock=clock)

    clock.advance(59)
    before = _run(block, 1, interval_unit="minutes")
    clock.advance(1)
    at_end = _run(block, 2, interval_unit="minutes")
    after = _run(block, 4, interval_unit="minutes")

    assert (before, at_end, after) == (
        {**OPEN, "x_sum": None},
        {**DONE, "x_sum": 3},
        {**OPEN, "x_sum": None},
    )


def test_operations_run_before_aggregation() -> None:
    results = _run_all(
        [["a", "b"], ["c"]],
        data_operations={"x": [{"type": "SequenceLength"}]},
        aggregation_mode={"x": ["sum"]},
    )

    assert results[-1] == {**DONE, "x_sum": 3}


@pytest.mark.parametrize(
    "mode, value, message",
    [
        ("sum", "seven", "sum of `x` needs numbers, got str"),
        ("distinct", [["unhashable"]], "distinct of `x` needs hashable values"),
    ],
)
def test_unsuitable_values_fail_clearly(mode: str, value: Any, message: str) -> None:
    with pytest.raises(TypeError, match=message):
        _run(DataAggregatorBlock(), value, aggregation_mode={"x": [mode]})


def test_outputs_follow_configured_statistics_and_kinds() -> None:
    params = DataAggregatorBlock.Params(
        data={"x": "$inputs.x"}, aggregation_mode={"x": ALL_MODES}, interval=1
    )

    outputs = DataAggregatorBlock.describe_outputs(params)

    assert list(outputs) == ["interval_complete"] + [f"x_{mode}" for mode in ALL_MODES]
    assert outputs["interval_complete"].kinds == (BOOLEAN_KIND,)
    assert outputs["x_count"].kinds == (INTEGER_KIND,)
    assert outputs["x_distinct"].kinds == (LIST_OF_VALUES_KIND,)
    assert outputs["x_values_counts"].kinds == (DICTIONARY_KIND,)
    assert outputs["x_avg"].kinds == (FLOAT_KIND,)


@pytest.mark.parametrize(
    "params, message",
    [
        ({"aggregation_mode": {"y": ["sum"]}}, "aggregation_mode names `y`"),
        (
            {"aggregation_mode": {"x": ["sum", "sum"]}},
            "output `x_sum` is produced twice",
        ),
        (
            {
                "data": {"x": "$inputs.x", "x_count": "$inputs.y"},
                "aggregation_mode": {"x": ["count_distinct"], "x_count": ["distinct"]},
            },
            "output `x_count_distinct` is produced twice",
        ),
        (
            {"data_operations": {"y": [{"type": "SequenceLength"}]}},
            "data_operations names `y`",
        ),
        ({"interval": 0}, "greater than 0"),
    ],
    ids=[
        "unknown-name",
        "repeated-mode",
        "colliding-names",
        "unknown-operation",
        "interval",
    ],
)
def test_invalid_configuration_is_rejected(
    params: Dict[str, Any], message: str
) -> None:
    arguments: Dict[str, Any] = {
        "data": {"x": "$inputs.x"},
        "aggregation_mode": {"x": ["sum"]},
        "interval": 1,
    }
    arguments.update(params)

    with pytest.raises(ValidationError, match=message):
        DataAggregatorBlock.Params(**arguments)


def test_interval_complete_is_a_reserved_output_name() -> None:
    # No current mode can form it; the check guards a future "complete" mode.
    with pytest.raises(ValueError, match="`interval_complete` is reserved"):
        output_names({"interval": ["complete"]})

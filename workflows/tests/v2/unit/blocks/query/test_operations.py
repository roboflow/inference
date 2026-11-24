"""Generic UQL operations: values derived from the current V1 operation bodies."""

import json
import random
from datetime import date, datetime, timedelta, timezone

import pytest
from roboflow_workflows.v2.blocks.query import (
    QueryRuntimeError,
    compile_operations,
)

# (operations, input value, expected output). Expected types matter too.
OPERATION_CASES = [
    ([{"type": "StringToLowerCase"}], "Hello World", "hello world"),
    ([{"type": "StringToUpperCase"}], "car", "CAR"),
    # Default end=-1 drops the last character.
    ([{"type": "StringSubSequence"}], "abcdef", "abcde"),
    ([{"type": "StringSubSequence", "start": 1, "end": 3}], "abcdef", "bc"),
    ([{"type": "StringSubSequence", "start": -3, "end": 10}], "abcdef", "def"),
    # re.search: a match anywhere counts.
    ([{"type": "StringMatches", "regex": "hello"}], "say hello", True),
    ([{"type": "StringMatches", "regex": "^hello$"}], "say hello", False),
    ([{"type": "StringMatches", "regex": "^(car|truck)$"}], "truck", True),
    ([{"type": "ToString"}], 3.5, "3.5"),
    ([{"type": "ToString"}], None, "None"),
    ([{"type": "ToString"}], [1, 2], "[1, 2]"),
    # int casts go through float and truncate toward zero.
    ([{"type": "ToNumber", "cast_to": "int"}], "3.99", 3),
    ([{"type": "ToNumber", "cast_to": "int"}], "-3.99", -3),
    ([{"type": "ToNumber", "cast_to": "int"}], "1e3", 1000),
    ([{"type": "ToNumber", "cast_to": "int"}], "42", 42),
    ([{"type": "ToNumber", "cast_to": "float"}], "42", 42.0),
    ([{"type": "ToNumber", "cast_to": "float"}], " 2.5 ", 2.5),
    ([{"type": "ToNumber", "cast_to": "float"}], True, 1.0),
    ([{"type": "NumberRound", "decimal_digits": 2}], 3.14159, 3.14),
    ([{"type": "NumberRound", "decimal_digits": 0}], 2.5, 2.0),
    ([{"type": "NumberRound", "decimal_digits": 0}], 7, 7),
    ([{"type": "ToBoolean"}], 0, False),
    ([{"type": "ToBoolean"}], 0.5, True),
    ([{"type": "ToBoolean"}], "False", True),
    ([{"type": "ToBoolean"}], [], False),
    ([{"type": "Multiply", "other": 2}], 3, 6),
    ([{"type": "Multiply", "other": 4}], 0.5, 2.0),
    ([{"type": "Multiply", "other": 2}], "ab", "abab"),
    ([{"type": "Divide", "other": 2}], 7, 3.5),
    ([{"type": "Divide", "other": 3}], 6, 2.0),
    ([{"type": "LookupTable", "lookup_table": {"cat": "animal"}}], "cat", "animal"),
    (
        [{"type": "SequenceMap", "lookup_table": {"a": 1, "b": 2}}],
        ["a", "b", "a"],
        [1, 2, 1],
    ),
    ([{"type": "SequenceMap", "lookup_table": {"a": 1, "b": 2}}], "ab", [1, 2]),
    ([{"type": "NumericSequenceAggregate", "function": "min"}], [3, 1, 2], 1),
    ([{"type": "NumericSequenceAggregate", "function": "max"}], [3, 1, 2], 3),
    ([{"type": "NumericSequenceAggregate", "function": "sum"}], [1, 2, 3.5], 6.5),
    (
        [{"type": "NumericSequenceAggregate", "function": "max", "neutral_value": 0}],
        [],
        0,
    ),
    ([{"type": "NumericSequenceAggregate", "function": "sum"}], [], None),
    ([{"type": "SequenceAggregate", "mode": "first"}], [5, 6, 7], 5),
    ([{"type": "SequenceAggregate", "mode": "last"}], [5, 6, 7], 7),
    # Equal counts keep first-encounter order.
    (
        [{"type": "SequenceAggregate", "mode": "most_common"}],
        ["b", "a", "a", "b", "c"],
        "b",
    ),
    (
        [{"type": "SequenceAggregate", "mode": "least_common"}],
        ["b", "a", "a", "b", "c"],
        "c",
    ),
    ([{"type": "SequenceAggregate", "mode": "most_common"}], ["x", "y"], "x"),
    ([{"type": "SequenceAggregate", "mode": "least_common"}], ["x", "y"], "y"),
    (
        [{"type": "SequenceApply", "operations": [{"type": "StringToUpperCase"}]}],
        ["a", "b"],
        ["A", "B"],
    ),
    (
        [{"type": "SequenceApply", "operations": [{"type": "SequenceLength"}]}],
        [[1, 2], [3]],
        [2, 1],
    ),
    ([{"type": "SequenceApply", "operations": []}], (1, 2), [1, 2]),
    ([{"type": "SequenceElementsCount"}], ["a", "b", "a"], {"a": 2, "b": 1}),
    # Keys are the stringified elements, so 1, "1" and True collapse by text.
    ([{"type": "SequenceElementsCount"}], [1, "1", True], {"1": 2, "True": 1}),
    ([{"type": "SequenceElementsCount"}], ["x" * 130], {"x" * 128 + " [...]": 1}),
    ([{"type": "SequenceJoin"}], ["a", "b"], "a, b"),
    ([{"type": "SequenceJoin", "separator": "-"}], [1, 2, 3], "1-2-3"),
    ([{"type": "SequenceJoin"}], [], ""),
    ([{"type": "SequenceLength"}], [1, 2, 3], 3),
    ([{"type": "SequenceLength"}], {"a": 1}, 1),
    ([{"type": "SequenceLength"}], "abc", 3),
    ([{"type": "RandomNumber", "min_value": 5, "max_value": 5}], "ignored", 5.0),
    (
        [{"type": "ConvertDictionaryToJSON"}],
        {"some": "data", "n": [1, 2]},
        '{"some": "data", "n": [1, 2]}',
    ),
    (
        [{"type": "TimestampToISOFormat"}],
        datetime(2024, 5, 1, 12, 30, 15),
        "2024-05-01T12:30:15",
    ),
    (
        [{"type": "TimestampToISOFormat"}],
        datetime(2024, 5, 1, 12, 30, tzinfo=timezone(timedelta(hours=2))),
        "2024-05-01T12:30:00+02:00",
    ),
    # Chains: slice a plate suffix, parse it, scale it.
    (
        [
            {"type": "StringSubSequence", "start": 4, "end": 6},
            {"type": "ToNumber", "cast_to": "int"},
            {"type": "Multiply", "other": 2},
        ],
        "Car-42",
        84,
    ),
    # Chains: majority label of mapped classes, then a display name.
    (
        [
            {
                "type": "SequenceMap",
                "lookup_table": {"cat": "pet", "dog": "pet", "cow": "farm"},
            },
            {"type": "SequenceAggregate", "mode": "most_common"},
            {
                "type": "LookupTable",
                "lookup_table": {"pet": "Pets", "farm": "Farm animals"},
            },
            {"type": "StringToUpperCase"},
        ],
        ["cow", "cat", "dog"],
        "PETS",
    ),
    ([], {"unchanged": True}, {"unchanged": True}),
]

# (operations, input value, failing JSON path, responsible parameter)
OPERATION_FAILURES = [
    ([{"type": "StringToLowerCase"}], 5, "$[0]", None),
    ([{"type": "StringToUpperCase"}], None, "$[0]", None),
    ([{"type": "StringSubSequence"}], 5, "$[0]", None),
    ([{"type": "StringMatches", "regex": "a"}], 123, "$[0]", None),
    ([{"type": "ToNumber", "cast_to": "int"}], "abc", "$[0]", None),
    ([{"type": "ToNumber", "cast_to": "float"}], None, "$[0]", None),
    ([{"type": "ToNumber", "cast_to": "int"}], "inf", "$[0]", None),
    ([{"type": "ToNumber", "cast_to": "int"}], "nan", "$[0]", None),
    ([{"type": "NumberRound", "decimal_digits": 1}], "3.1", "$[0]", None),
    ([{"type": "Multiply", "other": 2}], None, "$[0]", None),
    ([{"type": "Divide", "other": 0}], 1, "$[0]", "other"),
    ([{"type": "Divide", "other": 2}], "a", "$[0]", None),
    (
        [{"type": "LookupTable", "lookup_table": {"cat": 1}}],
        "dog",
        "$[0]",
        "lookup_table",
    ),
    ([{"type": "LookupTable", "lookup_table": {"cat": 1}}], ["cat"], "$[0]", None),
    (
        [{"type": "SequenceMap", "lookup_table": {"a": 1}}],
        ["a", "b"],
        "$[0]",
        "lookup_table",
    ),
    ([{"type": "SequenceMap", "lookup_table": {"a": 1}}], 5, "$[0]", None),
    ([{"type": "NumericSequenceAggregate", "function": "sum"}], ["a", 1], "$[0]", None),
    ([{"type": "NumericSequenceAggregate", "function": "min"}], 5, "$[0]", None),
    ([{"type": "SequenceAggregate", "mode": "first"}], [], "$[0]", None),
    ([{"type": "SequenceAggregate", "mode": "most_common"}], [], "$[0]", None),
    ([{"type": "SequenceAggregate", "mode": "most_common"}], [[1], [1]], "$[0]", None),
    (
        [{"type": "SequenceApply", "operations": [{"type": "ToString"}]}],
        5,
        "$[0]",
        None,
    ),
    (
        [
            {
                "type": "SequenceApply",
                "operations": [{"type": "ToString"}, {"type": "Divide", "other": 2}],
            }
        ],
        [1],
        "$[0].operations[1]",
        None,
    ),
    ([{"type": "SequenceElementsCount"}], 5, "$[0]", None),
    ([{"type": "SequenceJoin"}], 5, "$[0]", None),
    ([{"type": "SequenceLength"}], 5, "$[0]", None),
    ([{"type": "ConvertDictionaryToJSON"}], "some", "$[0]", None),
    ([{"type": "ConvertDictionaryToJSON"}], {"some": {"a", "b"}}, "$[0]", None),
    ([{"type": "TimestampToISOFormat"}], "2024-01-01", "$[0]", None),
    ([{"type": "TimestampToISOFormat"}], date(2024, 1, 1), "$[0]", None),
    (
        [
            {"type": "ToString"},
            {"type": "Multiply", "other": 2},
            {"type": "ToBoolean"},
            {"type": "StringToUpperCase"},
        ],
        1,
        "$[3]",
        None,
    ),
]


@pytest.mark.parametrize("operations, value, expected", OPERATION_CASES)
def test_operation_matches_v1_behaviour(operations, value, expected) -> None:
    # given
    compiled = compile_operations(operations)

    # when
    result = compiled.apply(value)

    # then
    assert result == expected
    assert type(result) is type(expected)


@pytest.mark.parametrize("operations, value, path, parameter", OPERATION_FAILURES)
def test_operation_failure_reports_path_and_parameter(
    operations, value, path, parameter
) -> None:
    # given
    compiled = compile_operations(operations)

    # when
    with pytest.raises(QueryRuntimeError) as error:
        compiled.apply(value)

    # then
    failing = _operation_at(operations, path)
    assert error.value.path == path
    assert error.value.operation == failing["type"]
    assert error.value.parameter == parameter


def test_sequence_join_limit_is_exactly_one_million_characters() -> None:
    # given
    compiled = compile_operations([{"type": "SequenceJoin"}])
    at_limit = ["a" * 500_000, "b" * 499_998]
    over_limit = ["a" * 500_000, "b" * 499_999]

    # when
    joined = compiled.apply(at_limit)
    with pytest.raises(QueryRuntimeError) as error:
        compiled.apply(over_limit)

    # then
    assert len(joined) == 1_000_000
    assert "1000001" in str(error.value)


def test_random_number_draws_once_per_application_from_module_random(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # given
    draws = iter([0.25, 0.5, 0.75])
    monkeypatch.setattr(random, "random", lambda: next(draws))
    compiled = compile_operations(
        [
            {
                "type": "SequenceApply",
                "operations": [
                    {"type": "RandomNumber", "min_value": 10, "max_value": 20}
                ],
            }
        ]
    )

    # when
    result = compiled.apply(["x", "y", "z"])

    # then
    assert result == [12.5, 15.0, 17.5]


def test_random_number_offsets_from_min_value_when_range_is_reversed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # given
    monkeypatch.setattr(random, "random", lambda: 0.5)
    compiled = compile_operations(
        [{"type": "RandomNumber", "min_value": 10, "max_value": 0}]
    )

    # when
    result = compiled.apply(None)

    # then
    assert result == 15.0


def test_random_number_defaults_draw_between_zero_and_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # given
    monkeypatch.setattr(random, "random", lambda: 0.3)
    compiled = compile_operations([{"type": "RandomNumber"}])

    # when
    result = compiled.apply("ignored")

    # then
    assert result == 0.3


def test_dictionary_to_json_round_trips() -> None:
    # given
    compiled = compile_operations([{"type": "ConvertDictionaryToJSON"}])
    data = {"zone": {"count": 3, "labels": ["car", "bus"]}}

    # when
    result = compiled.apply(data)

    # then
    assert json.loads(result) == data


def test_runtime_parameters_reach_nested_operations_untouched() -> None:
    # given
    compiled = compile_operations(
        [{"type": "SequenceApply", "operations": [{"type": "ToString"}]}]
    )
    parameters = {"nested": {"deep": [1, {"x": 2}]}}

    # when
    result = compiled.apply([1], parameters=parameters)

    # then
    assert result == ["1"]


def test_static_operation_list_input_is_not_mutated() -> None:
    # given
    operations = [{"type": "SequenceJoin"}]
    value = ["a", "b"]

    # when
    compile_operations(operations).apply(value)

    # then
    assert operations == [{"type": "SequenceJoin"}]
    assert value == ["a", "b"]


def _operation_at(operations, path):
    node = {"operations": operations}
    for part in path[1:].replace("[", ".[").split("."):
        if not part:
            continue
        if part.startswith("["):
            node = node["operations"][int(part[1:-1])]
    return node

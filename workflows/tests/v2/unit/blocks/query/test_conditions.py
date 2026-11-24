"""UQL condition evaluation: V1 golden JSON, comparators and evaluation order."""

import random

import pytest
from roboflow_workflows.v2.blocks.query import (
    QueryDefinitionError,
    QueryRuntimeError,
    compile_condition,
    compile_operations,
)


def _binary(left, comparator, right, *, negate=False):
    return {
        "type": "BinaryStatement",
        "left_operand": left,
        "comparator": {"type": comparator},
        "right_operand": right,
        "negate": negate,
    }


def _unary(operand, operator, *, negate=False):
    return {
        "type": "UnaryStatement",
        "operand": operand,
        "operator": {"type": operator},
        "negate": negate,
    }


def _group(*statements, operator="or"):
    return {
        "type": "StatementGroup",
        "operator": operator,
        "statements": list(statements),
    }


def _static(value, *operations):
    return {"type": "StaticOperand", "value": value, "operations": list(operations)}


def _dynamic(name=None, *operations):
    operand = {"type": "DynamicOperand", "operations": list(operations)}
    if name is not None:
        operand["operand_name"] = name
    return operand


# continue_if@v1 multi-label conditions from
# workflows/tests/unit_tests/core_steps/control_flow/test_control_flow.py
V1_CONTINUE_IF_GOLDEN = [
    (
        _group(_binary(_static(["lion", "zebra"]), "==", _static(["lion", "zebra"]))),
        True,
    ),
    (
        _group(
            _binary(
                _static(["lion", "zebra"]), "==", _static(["lion", "zebra", "elephant"])
            )
        ),
        False,
    ),
    (
        _group(
            _binary(
                _static(["lion", "zebra"]),
                "any in (Sequence)",
                _static(["cat", "zebra", "dog"]),
            )
        ),
        True,
    ),
    (
        _group(
            _binary(
                _static(["lion", "zebra"]),
                "any in (Sequence)",
                _static(["cat", "elephant", "dog"]),
            )
        ),
        False,
    ),
    (
        _group(
            _binary(
                _static(["lion", "zebra"]),
                "all in (Sequence)",
                _static(["zebra", "lion"]),
            )
        ),
        True,
    ),
    (
        _group(
            _binary(
                _static(["lion", "dog"]),
                "all in (Sequence)",
                _static(["zebra", "lion"]),
            )
        ),
        False,
    ),
]

# Realistic nested condition: an accepted vehicle class AND (confident enough
# OR manually flagged), where the flag must be the boolean True itself.
NESTED_VEHICLE_CONDITION = _group(
    _binary(
        _dynamic("class_name", {"type": "StringToLowerCase"}),
        "in (Sequence)",
        _dynamic("accepted_classes"),
    ),
    _group(
        _binary(
            _dynamic(None, {"type": "Multiply", "other": 100}),
            "(Number) >=",
            _dynamic("threshold_percent"),
        ),
        _unary(_dynamic("flagged"), "(Boolean) is True"),
        operator="or",
    ),
    operator="and",
)
NESTED_VEHICLE_CASES = [
    ({"_": 0.91, "class_name": "Car", "flagged": False}, True),
    ({"_": 0.42, "class_name": "Car", "flagged": False}, False),
    ({"_": 0.42, "class_name": "Truck", "flagged": True}, True),
    # Truthy but not the boolean True: identity check fails.
    ({"_": 0.42, "class_name": "Truck", "flagged": 1}, False),
    ({"_": 0.99, "class_name": "Person", "flagged": True}, False),
]

# (comparator, left, right, expected)
COMPARATOR_CASES = [
    ("==", 1, 1.0, True),
    ("==", "car", "Car", False),
    ("(Number) ==", 2, 3, False),
    ("!=", "a", "b", True),
    ("(Number) !=", 1, 1, False),
    ("(Number) >", 2, 1, True),
    ("(Number) >", 1, 1, False),
    ("(Number) >=", 1, 1, True),
    ("(Number) <", 1, 2, True),
    ("(Number) <=", 2, 1, False),
    ("(Number) in range", 5, [1, 5], True),
    ("(Number) in range", 0.5, [0.6, 1.0], False),
    ("(String) startsWith", "car_front", "car", True),
    ("(String) endsWith", "car_front", "front", True),
    ("(String) contains", "car_front", "_", True),
    ("(String) contains", "abc", "d", False),
    ("in (Sequence)", "cat", ["cat", "dog"], True),
    ("in (Sequence)", "cat", {"cat": 1}, True),
    ("in (Sequence)", "xy", "wxyz", True),
    ("in (Sequence)", "cow", ["cat", "dog"], False),
    ("any in (Sequence)", [], ["a"], False),
    ("all in (Sequence)", [], ["a"], True),
]

# (operator, operand value, expected)
UNARY_CASES = [
    ("Exists", None, False),
    ("Exists", 0, True),
    ("DoesNotExist", None, True),
    ("DoesNotExist", "", False),
    ("(Boolean) is True", True, True),
    ("(Boolean) is True", 1, False),
    ("(Boolean) is True", "True", False),
    ("(Boolean) is False", False, True),
    ("(Boolean) is False", 0, False),
    ("(Boolean) is False", None, False),
    ("(Sequence) is empty", [], True),
    ("(Sequence) is empty", "", True),
    ("(Sequence) is empty", {}, True),
    ("(Sequence) is empty", [0], False),
    ("(Sequence) is not empty", [0], True),
    ("(Sequence) is not empty", {}, False),
]


@pytest.mark.parametrize("definition, expected", V1_CONTINUE_IF_GOLDEN)
def test_v1_continue_if_golden_conditions(definition, expected) -> None:
    # given
    condition = compile_condition(definition, declared_names=set())

    # when
    result = condition.evaluate({})

    # then
    assert result is expected


@pytest.mark.parametrize("values, expected", NESTED_VEHICLE_CASES)
def test_nested_condition_with_parameters_and_row_default(values, expected) -> None:
    # given
    condition = compile_condition(
        NESTED_VEHICLE_CONDITION,
        declared_names={
            "_",
            "class_name",
            "flagged",
            "accepted_classes",
            "threshold_percent",
        },
    )
    parameters = {"accepted_classes": ["car", "truck"], "threshold_percent": 80}

    # when
    result = condition.evaluate({**parameters, **values})

    # then
    assert result is expected


def test_v1_condition_over_detection_count_uses_sequence_length() -> None:
    # given: rock-paper-scissors condition; a list stands in for predictions
    condition = compile_condition(
        _group(
            _binary(
                _dynamic("prediction", {"type": "SequenceLength"}),
                "(Number) ==",
                _static(2),
            )
        )
    )

    # when
    two = condition.evaluate({"prediction": ["rock", "paper"]})
    three = condition.evaluate({"prediction": ["rock", "paper", "rock"]})

    # then
    assert two is True
    assert three is False


def test_v1_expression_condition_and_result_chain() -> None:
    # given: expression block case from test_expression.py
    condition = compile_condition(
        _group(_unary(_dynamic("input_list"), "(Sequence) is not empty"))
    )
    data_operations = compile_operations([{"type": "StringToUpperCase"}])
    result_operations = compile_operations(
        [{"type": "LookupTable", "lookup_table": {"SOME": "MUTATED"}}]
    )

    # when
    matched = condition.evaluate({"input_list": [1, 2]})
    output = result_operations.apply(data_operations.apply("some"))

    # then
    assert matched is True
    assert output == "MUTATED"


@pytest.mark.parametrize("comparator, left, right, expected", COMPARATOR_CASES)
def test_comparator(comparator, left, right, expected) -> None:
    # given
    condition = compile_condition(
        _group(_binary(_dynamic("a"), comparator, _dynamic("b")))
    )

    # when
    result = condition.evaluate({"a": left, "b": right})

    # then
    assert result is expected


@pytest.mark.parametrize("comparator, left, right, expected", COMPARATOR_CASES)
def test_negated_comparator(comparator, left, right, expected) -> None:
    # given
    condition = compile_condition(
        _group(_binary(_dynamic("a"), comparator, _dynamic("b"), negate=True))
    )

    # when
    result = condition.evaluate({"a": left, "b": right})

    # then
    assert result is (not expected)


@pytest.mark.parametrize("operator, value, expected", UNARY_CASES)
def test_unary_operator(operator, value, expected) -> None:
    # given
    condition = compile_condition(_group(_unary(_dynamic(), operator)))
    negated = compile_condition(_group(_unary(_dynamic(), operator, negate=True)))

    # when
    result = condition.evaluate({"_": value})
    negated_result = negated.evaluate({"_": value})

    # then
    assert result is expected
    assert negated_result is (not expected)


def test_boolean_identity_differs_from_to_boolean_truthiness() -> None:
    # given
    identity = compile_condition(_group(_unary(_dynamic(), "(Boolean) is True")))
    truthiness = compile_condition(
        _group(_unary(_dynamic(None, {"type": "ToBoolean"}), "(Boolean) is True"))
    )

    # when
    identity_result = identity.evaluate({"_": 1})
    truthiness_result = truthiness.evaluate({"_": 1})

    # then
    assert identity_result is False
    assert truthiness_result is True


@pytest.mark.parametrize(
    "operator, statements, expected",
    [
        ("and", [True, True], True),
        ("and", [True, False], False),
        ("or", [False, False], False),
        ("or", [False, True], True),
        ("or", [False, False, True], True),
    ],
)
def test_group_operators(operator, statements, expected) -> None:
    # given
    definition = _group(
        *[_unary(_static(value), "(Boolean) is True") for value in statements],
        operator=operator,
    )

    # when
    result = compile_condition(definition).evaluate({})

    # then
    assert result is expected


def test_group_operator_defaults_to_or() -> None:
    # given
    definition = {
        "type": "StatementGroup",
        "statements": [
            _unary(_static(False), "(Boolean) is True"),
            _unary(_static(True), "(Boolean) is True"),
        ],
    }

    # when
    result = compile_condition(definition).evaluate({})

    # then
    assert result is True


def test_every_statement_runs_after_or_result_is_known(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # given
    draws = []
    monkeypatch.setattr(random, "random", lambda: draws.append(1) or 0.5)
    definition = _group(
        _unary(_static(True), "(Boolean) is True"),
        _binary(_static(None, {"type": "RandomNumber"}), "(Number) >", _static(2)),
        operator="or",
    )

    # when
    result = compile_condition(definition).evaluate({})

    # then
    assert result is True
    assert len(draws) == 1


def test_error_after_and_result_is_known_still_raises() -> None:
    # given
    definition = _group(
        _unary(_static(False), "(Boolean) is True"),
        _binary(_static("text"), "(Number) >", _static(1)),
        operator="and",
    )
    condition = compile_condition(definition)

    # when
    with pytest.raises(QueryRuntimeError) as error:
        condition.evaluate({})

    # then
    assert error.value.path == "$.statements[1].comparator"
    assert error.value.operation == "(Number) >"


def test_left_operand_is_evaluated_before_right(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # given
    draws = iter([0.1, 0.9])
    monkeypatch.setattr(random, "random", lambda: next(draws))
    definition = _group(
        _binary(
            _static(None, {"type": "RandomNumber"}),
            "(Number) <",
            _static(None, {"type": "RandomNumber"}),
        )
    )

    # when
    result = compile_condition(definition).evaluate({})

    # then
    assert result is True


def test_dynamic_operand_defaults_to_row_name() -> None:
    # given
    condition = compile_condition(
        _group(_binary(_dynamic(), "==", _static("car"))), declared_names={"_"}
    )

    # when
    result = condition.evaluate({"_": "car"})

    # then
    assert result is True


def test_undeclared_operand_is_a_definition_error() -> None:
    # given
    definition = _group(
        _unary(_static(1), "Exists"),
        _binary(_dynamic("threshold"), "(Number) >", _dynamic()),
    )

    # when
    with pytest.raises(QueryDefinitionError) as error:
        compile_condition(definition, declared_names={"_"})

    # then
    assert error.value.path == "$.statements[1].left_operand"
    assert error.value.parameter == "operand_name"
    assert "threshold" in str(error.value)


def test_missing_runtime_value_is_a_runtime_error() -> None:
    # given
    condition = compile_condition(
        _group(_group(_unary(_dynamic("zones"), "(Sequence) is empty")))
    )

    # when
    with pytest.raises(QueryRuntimeError) as error:
        condition.evaluate({"_": 1})

    # then
    assert error.value.path == "$.statements[0].statements[0].operand"
    assert error.value.operation == "DynamicOperand"
    assert error.value.parameter == "operand_name"


@pytest.mark.parametrize(
    "statement, path, operation",
    [
        (
            _binary(_static("a"), "(Number) >", _static(1)),
            "$.statements[0].comparator",
            "(Number) >",
        ),
        (
            _binary(_static(5), "(String) startsWith", _static("a")),
            "$.statements[0].comparator",
            "(String) startsWith",
        ),
        (
            _binary(_static(1), "(Number) in range", _static(5)),
            "$.statements[0].comparator",
            "(Number) in range",
        ),
        (
            _binary(_static(1), "in (Sequence)", _static(5)),
            "$.statements[0].comparator",
            "in (Sequence)",
        ),
        (
            _unary(_static(None), "(Sequence) is empty"),
            "$.statements[0].operator",
            "(Sequence) is empty",
        ),
        (
            _binary(
                _static("x", {"type": "ToNumber", "cast_to": "int"}), "==", _static(1)
            ),
            "$.statements[0].left_operand.operations[0]",
            "ToNumber",
        ),
        (
            _binary(
                _static(1),
                "==",
                _static(
                    "x",
                    {"type": "StringMatches", "regex": "x"},
                    {"type": "StringToLowerCase"},
                ),
            ),
            "$.statements[0].right_operand.operations[1]",
            "StringToLowerCase",
        ),
    ],
)
def test_runtime_type_failures_report_path(statement, path, operation) -> None:
    # given
    condition = compile_condition(_group(statement))

    # when
    with pytest.raises(QueryRuntimeError) as error:
        condition.evaluate({})

    # then
    assert error.value.path == path
    assert error.value.operation == operation


def test_nested_runtime_parameter_values_are_allowed() -> None:
    # given
    condition = compile_condition(
        _group(
            _binary(
                _dynamic("config", {"type": "ConvertDictionaryToJSON"}),
                "==",
                _static('{"thresholds": [0.1, {"car": 0.2}]}'),
            )
        )
    )

    # when
    result = condition.evaluate({"config": {"thresholds": [0.1, {"car": 0.2}]}})

    # then
    assert result is True


def test_condition_is_reusable_across_runtime_values() -> None:
    # given
    condition = compile_condition(
        _group(_binary(_dynamic(), "(Number) >=", _static(0.5)))
    )

    # when
    results = [condition.evaluate({"_": value}) for value in (0.4, 0.5, 0.9)]

    # then
    assert results == [False, True, True]

"""Mistaking either gate for the other gives its exact catalogue identity."""

import pytest
from roboflow_workflows.v2.blocks import create_catalogue
from roboflow_workflows.v2.engine.compilation import compile_workflow
from roboflow_workflows.v2.engine.errors import ParamsValidationError


def _compile_gate(block_type: str, **parameters) -> None:
    compile_workflow(
        {
            "version": "2.0",
            "inputs": [{"type": "WorkflowParameter", "name": "value"}],
            "steps": [
                {"type": block_type, "name": "gate", "next_steps": [], **parameters}
            ],
            "outputs": [],
        },
        catalogue=create_catalogue(),
    )


@pytest.mark.parametrize(
    "condition_statement",
    [
        {
            "type": "StatementGroup",
            "statements": [
                {
                    "type": "BinaryStatement",
                    "left_operand": {"type": "StaticOperand", "value": 1},
                    "comparator": {"type": "(Number) =="},
                    "right_operand": {"type": "StaticOperand", "value": 1},
                }
            ],
        },
        "$inputs.value",
    ],
)
def test_boolean_gate_points_query_conditions_to_the_query_gate(
    condition_statement,
) -> None:
    with pytest.raises(ParamsValidationError) as error:
        _compile_gate("v2/continue_if", condition_statement=condition_statement)

    message = str(error.value)
    assert "condition_statement is a field of v2/continue_if@v1" in message
    assert "v2/continue_if takes a boolean condition" in message


@pytest.mark.parametrize("condition", [True, "$inputs.value"])
def test_query_gate_points_boolean_conditions_to_the_boolean_gate(condition) -> None:
    with pytest.raises(ParamsValidationError) as error:
        _compile_gate("v2/continue_if@v1", condition=condition)

    message = str(error.value)
    assert "condition is a field of v2/continue_if (ready boolean)" in message
    assert "v2/continue_if@v1 takes a V1 query condition_statement" in message

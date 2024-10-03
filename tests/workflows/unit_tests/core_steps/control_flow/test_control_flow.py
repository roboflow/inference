import pytest

from inference.core.workflows.core_steps.common.query_language.entities.operations import (
    StatementGroup,
)
from inference.core.workflows.core_steps.common.query_language.evaluation_engine.core import (
    build_eval_function,
)

CONTINUE_IF_MULTI_LABEL_EXACT_MATCH_PASS = {
    "type": "roboflow_core/continue_if@v1",
    "name": "exact_match",
    "condition_statement": {
        "type": "StatementGroup",
        "statements": [
            {
                "type": "BinaryStatement",
                "left_operand": {"type": "StaticOperand", "value": ["lion", "zebra"]},
                "comparator": {"type": "=="},
                "right_operand": {"type": "StaticOperand", "value": ["lion", "zebra"]},
            }
        ],
    },
    "next_steps": ["$steps.flip"],
    "evaluation_parameters": {},
}

CONTINUE_IF_MULTI_LABEL_EXACT_MATCH_FAIL = {
    "type": "roboflow_core/continue_if@v1",
    "name": "exact_match",
    "condition_statement": {
        "type": "StatementGroup",
        "statements": [
            {
                "type": "BinaryStatement",
                "left_operand": {"type": "StaticOperand", "value": ["lion", "zebra"]},
                "comparator": {"type": "=="},
                "right_operand": {
                    "type": "StaticOperand",
                    "value": ["lion", "zebra", "elephant"],
                },
            }
        ],
    },
    "next_steps": ["$steps.flip"],
    "evaluation_parameters": {},
}


CONTINUE_IF_MULTI_LABEL_ANY_MATCH_PASS = {
    "type": "roboflow_core/continue_if@v1",
    "name": "any_in",
    "condition_statement": {
        "type": "StatementGroup",
        "statements": [
            {
                "type": "BinaryStatement",
                "left_operand": {"type": "StaticOperand", "value": ["lion", "zebra"]},
                "comparator": {"type": "any in (Sequence)"},
                "right_operand": {
                    "type": "StaticOperand",
                    "value": ["cat", "zebra", "dog"],
                },
            }
        ],
    },
    "next_steps": ["$steps.flip"],
    "evaluation_parameters": {},
}

CONTINUE_IF_MULTI_LABEL_ANY_MATCH_FAIL = {
    "type": "roboflow_core/continue_if@v1",
    "name": "any_in",
    "condition_statement": {
        "type": "StatementGroup",
        "statements": [
            {
                "type": "BinaryStatement",
                "left_operand": {"type": "StaticOperand", "value": ["lion", "zebra"]},
                "comparator": {"type": "any in (Sequence)"},
                "right_operand": {
                    "type": "StaticOperand",
                    "value": ["cat", "elephant", "dog"],
                },
            }
        ],
    },
    "next_steps": ["$steps.flip"],
    "evaluation_parameters": {},
}


CONTINUE_IF_MULTI_LABEL_ALL_MATCH_PASS = {
    "type": "roboflow_core/continue_if@v1",
    "name": "all_in",
    "condition_statement": {
        "type": "StatementGroup",
        "statements": [
            {
                "type": "BinaryStatement",
                "left_operand": {"type": "StaticOperand", "value": ["lion", "zebra"]},
                "comparator": {"type": "all in (Sequence)"},
                "right_operand": {"type": "StaticOperand", "value": ["zebra", "lion"]},
            }
        ],
    },
    "next_steps": ["$steps.flip"],
    "evaluation_parameters": {"left": "$steps.multi_label_classes.predictions"},
}

CONTINUE_IF_MULTI_LABEL_ALL_MATCH_FAIL = {
    "type": "roboflow_core/continue_if@v1",
    "name": "all_in",
    "condition_statement": {
        "type": "StatementGroup",
        "statements": [
            {
                "type": "BinaryStatement",
                "left_operand": {"type": "StaticOperand", "value": ["lion", "dog"]},
                "comparator": {"type": "all in (Sequence)"},
                "right_operand": {"type": "StaticOperand", "value": ["zebra", "lion"]},
            }
        ],
    },
    "next_steps": ["$steps.flip"],
    "evaluation_parameters": {"left": "$steps.multi_label_classes.predictions"},
}


@pytest.mark.parametrize(
    "condition_statement, evaluation_parameters, expected_result",
    [
        (
            CONTINUE_IF_MULTI_LABEL_EXACT_MATCH_PASS["condition_statement"],
            CONTINUE_IF_MULTI_LABEL_EXACT_MATCH_PASS["evaluation_parameters"],
            True,
        ),
        (
            CONTINUE_IF_MULTI_LABEL_EXACT_MATCH_FAIL["condition_statement"],
            CONTINUE_IF_MULTI_LABEL_EXACT_MATCH_FAIL["evaluation_parameters"],
            False,
        ),
        (
            CONTINUE_IF_MULTI_LABEL_ANY_MATCH_PASS["condition_statement"],
            CONTINUE_IF_MULTI_LABEL_ANY_MATCH_PASS["evaluation_parameters"],
            True,
        ),
        (
            CONTINUE_IF_MULTI_LABEL_ANY_MATCH_FAIL["condition_statement"],
            CONTINUE_IF_MULTI_LABEL_ANY_MATCH_FAIL["evaluation_parameters"],
            False,
        ),
        (
            CONTINUE_IF_MULTI_LABEL_ALL_MATCH_PASS["condition_statement"],
            CONTINUE_IF_MULTI_LABEL_ALL_MATCH_PASS["evaluation_parameters"],
            True,
        ),
        (
            CONTINUE_IF_MULTI_LABEL_ALL_MATCH_FAIL["condition_statement"],
            CONTINUE_IF_MULTI_LABEL_ALL_MATCH_FAIL["evaluation_parameters"],
            False,
        ),
    ],
)
def test_continue_if_evaluation(
    condition_statement, evaluation_parameters, expected_result
):
    parsed_definition = StatementGroup.model_validate(condition_statement)
    evaluation_function = build_eval_function(definition=parsed_definition)
    evaluation_result = evaluation_function(evaluation_parameters)
    assert (
        evaluation_result == expected_result
    ), f"Expected {expected_result} for condition {condition_statement} with parameters {evaluation_parameters}, but got {evaluation_result}"

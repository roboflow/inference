import pytest

from inference.core.workflows.core_steps.common.query_language.entities.operations import (
    OperationsChain,
)
from inference.core.workflows.core_steps.common.query_language.errors import (
    EvaluationEngineError,
    UndeclaredSymbolError,
)
from inference.core.workflows.core_steps.formatters.expression.v1 import (
    CasesDefinition,
    ExpressionBlockV1,
    StaticCaseResult,
)


def test_block_run_when_no_data_provided_and_static_output_to_be_returned() -> None:
    # given
    step = ExpressionBlockV1()
    switch = CasesDefinition(
        type="CasesDefinition",
        cases=[],
        default=StaticCaseResult(type="StaticCaseResult", value="static-value"),
    )

    # when
    result = step.run(data={}, data_operations={}, switch=switch)

    # then
    assert result == {"output": "static-value"}


def test_block_run_when_data_provided_and_default_output_to_be_returned() -> None:
    # given
    step = ExpressionBlockV1()
    switch = CasesDefinition.model_validate(
        {
            "type": "CasesDefinition",
            "cases": [
                {
                    "type": "CaseDefinition",
                    "condition": {
                        "type": "StatementGroup",
                        "statements": [
                            {
                                "type": "UnaryStatement",
                                "operand": {
                                    "type": "DynamicOperand",
                                    "operand_name": "input_list",
                                },
                                "operator": {"type": "(Sequence) is empty"},
                            }
                        ],
                    },
                    "result": {"type": "StaticCaseResult", "value": "some"},
                }
            ],
            "default": {"type": "StaticCaseResult", "value": "default"},
        }
    )

    # when
    result = step.run(data={"input_list": [1, 2]}, data_operations={}, switch=switch)

    # then
    assert result == {"output": "default"}


def test_block_run_when_data_provided_and_first_matching_case_to_be_returned() -> None:
    # given
    step = ExpressionBlockV1()
    switch = CasesDefinition.model_validate(
        {
            "type": "CasesDefinition",
            "cases": [
                {
                    "type": "CaseDefinition",
                    "condition": {
                        "type": "StatementGroup",
                        "statements": [
                            {
                                "type": "UnaryStatement",
                                "operand": {
                                    "type": "DynamicOperand",
                                    "operand_name": "input_list",
                                },
                                "operator": {"type": "(Sequence) is empty"},
                            }
                        ],
                    },
                    "result": {"type": "StaticCaseResult", "value": "not-matching"},
                },
                {
                    "type": "CaseDefinition",
                    "condition": {
                        "type": "StatementGroup",
                        "statements": [
                            {
                                "type": "UnaryStatement",
                                "operand": {
                                    "type": "DynamicOperand",
                                    "operand_name": "input_list",
                                },
                                "operator": {"type": "(Sequence) is not empty"},
                            }
                        ],
                    },
                    "result": {"type": "StaticCaseResult", "value": "first-matching"},
                },
                {
                    "type": "CaseDefinition",
                    "condition": {
                        "type": "StatementGroup",
                        "statements": [
                            {
                                "type": "UnaryStatement",
                                "operand": {
                                    "type": "DynamicOperand",
                                    "operand_name": "input_list",
                                },
                                "operator": {"type": "(Sequence) is not empty"},
                            }
                        ],
                    },
                    "result": {"type": "StaticCaseResult", "value": "second-matching"},
                },
            ],
            "default": {"type": "StaticCaseResult", "value": "default"},
        }
    )

    # when
    result = step.run(data={"input_list": [1, 2]}, data_operations={}, switch=switch)

    # then
    assert result == {"output": "first-matching"}


def test_block_run_when_data_provided_and_data_operations_to_be_performed() -> None:
    # given
    step = ExpressionBlockV1()
    switch = CasesDefinition.model_validate(
        {
            "type": "CasesDefinition",
            "cases": [
                {
                    "type": "CaseDefinition",
                    "condition": {
                        "type": "StatementGroup",
                        "statements": [
                            {
                                "type": "UnaryStatement",
                                "operand": {
                                    "type": "DynamicOperand",
                                    "operand_name": "input_list",
                                },
                                "operator": {"type": "(Sequence) is not empty"},
                            }
                        ],
                    },
                    "result": {
                        "type": "DynamicCaseResult",
                        "parameter_name": "additional_param",
                    },
                }
            ],
            "default": {"type": "StaticCaseResult", "value": "default"},
        }
    )
    data_operations = {
        "additional_param": OperationsChain.model_validate(
            {"operations": [{"type": "StringToUpperCase"}]}
        ).operations
    }

    # when
    result = step.run(
        data={
            "input_list": [1, 2],
            "additional_param": "some",
        },
        data_operations=data_operations,
        switch=switch,
    )

    # then
    assert result == {"output": "SOME"}


def test_block_run_when_data_provided_and_operations_on_dynamic_result_to_be_performed() -> (
    None
):
    # given
    step = ExpressionBlockV1()
    switch = CasesDefinition.model_validate(
        {
            "type": "CasesDefinition",
            "cases": [
                {
                    "type": "CaseDefinition",
                    "condition": {
                        "type": "StatementGroup",
                        "statements": [
                            {
                                "type": "UnaryStatement",
                                "operand": {
                                    "type": "DynamicOperand",
                                    "operand_name": "input_list",
                                },
                                "operator": {"type": "(Sequence) is not empty"},
                            }
                        ],
                    },
                    "result": {
                        "type": "DynamicCaseResult",
                        "parameter_name": "additional_param",
                        "operations": OperationsChain.model_validate(
                            {
                                "operations": [
                                    {
                                        "type": "LookupTable",
                                        "lookup_table": {"SOME": "MUTATED"},
                                    }
                                ]
                            }
                        ).operations,
                    },
                }
            ],
            "default": {"type": "StaticCaseResult", "value": "default"},
        }
    )
    data_operations = {
        "additional_param": OperationsChain.model_validate(
            {"operations": [{"type": "StringToUpperCase"}]}
        ).operations
    }

    # when
    result = step.run(
        data={
            "input_list": [1, 2],
            "additional_param": "some",
        },
        data_operations=data_operations,
        switch=switch,
    )

    # then
    assert result == {"output": "MUTATED"}


def test_block_run_when_uql_error_happens() -> None:
    # given
    step = ExpressionBlockV1()
    switch = CasesDefinition.model_validate(
        {
            "type": "CasesDefinition",
            "cases": [
                {
                    "type": "CaseDefinition",
                    "condition": {
                        "type": "StatementGroup",
                        "statements": [
                            {
                                "type": "UnaryStatement",
                                "operand": {
                                    "type": "DynamicOperand",
                                    "operand_name": "input_list",
                                },
                                "operator": {"type": "(Sequence) is empty"},
                            }
                        ],
                    },
                    "result": {"type": "StaticCaseResult", "value": "some"},
                }
            ],
            "default": {"type": "StaticCaseResult", "value": "default"},
        }
    )

    # when
    with pytest.raises(EvaluationEngineError):
        _ = step.run(data={"input_list": 1}, data_operations={}, switch=switch)


def test_block_run_when_data_provided_and_missing_variable_detected_on_data_operations() -> (
    None
):
    # given
    step = ExpressionBlockV1()
    switch = CasesDefinition.model_validate(
        {
            "type": "CasesDefinition",
            "cases": [
                {
                    "type": "CaseDefinition",
                    "condition": {
                        "type": "StatementGroup",
                        "statements": [
                            {
                                "type": "UnaryStatement",
                                "operand": {
                                    "type": "DynamicOperand",
                                    "operand_name": "input_list",
                                },
                                "operator": {"type": "(Sequence) is not empty"},
                            }
                        ],
                    },
                    "result": {
                        "type": "DynamicCaseResult",
                        "parameter_name": "additional_param",
                    },
                }
            ],
            "default": {"type": "StaticCaseResult", "value": "default"},
        }
    )
    data_operations = {
        "additional_param": OperationsChain.model_validate(
            {"operations": [{"type": "StringToUpperCase"}]}
        ).operations
    }

    # when
    with pytest.raises(KeyError):
        _ = step.run(
            data={"input_list": [1, 2]}, data_operations=data_operations, switch=switch
        )


def test_block_run_when_data_provided_and_missing_variable_detected_in_output_building() -> (
    None
):
    # given
    step = ExpressionBlockV1()
    switch = CasesDefinition.model_validate(
        {
            "type": "CasesDefinition",
            "cases": [
                {
                    "type": "CaseDefinition",
                    "condition": {
                        "type": "StatementGroup",
                        "statements": [
                            {
                                "type": "UnaryStatement",
                                "operand": {
                                    "type": "DynamicOperand",
                                    "operand_name": "input_list",
                                },
                                "operator": {"type": "(Sequence) is not empty"},
                            }
                        ],
                    },
                    "result": {
                        "type": "DynamicCaseResult",
                        "parameter_name": "additional_param",
                    },
                }
            ],
            "default": {"type": "StaticCaseResult", "value": "default"},
        }
    )

    # when
    with pytest.raises(KeyError):
        _ = step.run(data={"input_list": [1, 2]}, data_operations={}, switch=switch)


def test_block_run_when_data_provided_and_missing_variable_detected_in_uql_building() -> (
    None
):
    # given
    step = ExpressionBlockV1()
    switch = CasesDefinition.model_validate(
        {
            "type": "CasesDefinition",
            "cases": [
                {
                    "type": "CaseDefinition",
                    "condition": {
                        "type": "StatementGroup",
                        "statements": [
                            {
                                "type": "UnaryStatement",
                                "operand": {
                                    "type": "DynamicOperand",
                                    "operand_name": "input_list",
                                },
                                "operator": {"type": "(Sequence) is not empty"},
                            }
                        ],
                    },
                    "result": {
                        "type": "DynamicCaseResult",
                        "parameter_name": "additional_param",
                    },
                }
            ],
            "default": {"type": "StaticCaseResult", "value": "default"},
        }
    )

    # when
    with pytest.raises(UndeclaredSymbolError):
        _ = step.run(
            data={"additional_param": "some"}, data_operations={}, switch=switch
        )

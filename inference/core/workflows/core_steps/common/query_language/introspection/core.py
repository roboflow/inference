from typing import List

from inference.core.workflows.core_steps.common.query_language.entities.introspection import (
    OperationDescription,
    OperatorDescription,
)
from inference.core.workflows.core_steps.common.query_language.entities.operations import (
    DoesNotExist,
    Equals,
    Exists,
    In,
    IsEmpty,
    IsFalse,
    IsNotEmpty,
    IsTrue,
    NotEquals,
    NumberGreater,
    NumberGreaterEqual,
    NumberLower,
    NumberLowerEqual,
    OperationsChain,
    StringContains,
    StringEndsWith,
    StringStartsWith,
)


def prepare_operations_descriptions() -> List[OperationDescription]:
    operations_chain_schema = OperationsChain.schema()
    operation_types = [
        type_definition["$ref"].split("/")[-1]
        for type_definition in operations_chain_schema["properties"]["operations"][
            "items"
        ]["oneOf"]
    ]
    type_definitions = [
        operations_chain_schema["$defs"][operation_type]
        for operation_type in operation_types
    ]
    result = []
    for type_definition in type_definitions:
        # TODO: fix simplification of type getter
        result.append(
            OperationDescription(
                operation_type=type_definition["properties"]["type"]["const"],
                compound=type_definition.get("compound", False),
                input_kind=type_definition["input_kind"],
                output_kind=type_definition["output_kind"],
                nested_operation_input_kind=type_definition.get(
                    "nested_operation_input_kind"
                ),
                nested_operation_output_kind=type_definition.get(
                    "nested_operation_output_kind"
                ),
            )
        )
    return result


def prepare_operators_descriptions() -> List[OperatorDescription]:
    operator_types = [
        In,
        StringContains,
        StringEndsWith,
        StringStartsWith,
        NumberLowerEqual,
        NumberLower,
        NumberGreaterEqual,
        NumberGreater,
        NotEquals,
        Equals,
        Exists,
        DoesNotExist,
        IsTrue,
        IsFalse,
        IsEmpty,
        IsNotEmpty,
    ]
    results = []
    for operator_type in operator_types:
        operator_schema = operator_type.schema()
        for alias in operator_schema["properties"]["type"].get("enum", []):
            results.append(
                OperatorDescription(
                    operator_type=alias,
                    operands_number=operator_schema["operands_number"],
                    operands_kinds=operator_schema["operands_kinds"],
                    description=operator_schema.get("description"),
                )
            )
    return results

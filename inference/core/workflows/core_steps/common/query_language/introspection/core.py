from typing import List, Optional

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
    NumberInRange,
    NumberLower,
    NumberLowerEqual,
    OperationsChain,
    StringContains,
    StringEndsWith,
    StringStartsWith,
)


def _ref_to_def_name(
    schema: dict,
    ref_key: str = "$ref",
) -> Optional[str]:
    """Extract definition name from a JSON Schema object's $ref (e.g. {'$ref': '#/$defs/Foo'} -> 'Foo')."""
    ref = schema.get(ref_key)
    if ref is None:
        return None

    return ref.split("/")[-1]


def _get_property_name_options(
    type_definition: dict,
    defs: dict,
) -> Optional[List[str]]:
    """Resolve property_name enum from operation schema when present (e.g. DetectionsProperty)."""
    properties = type_definition.get("properties") or {}
    property_name_schema = properties.get("property_name")

    if not property_name_schema:
        return None

    ref_name = _ref_to_def_name(property_name_schema)
    if ref_name is not None:
        resolved = defs.get(ref_name)
        if resolved and "enum" in resolved:
            return list(resolved["enum"])

    if "enum" in property_name_schema:
        return list(property_name_schema["enum"])

    return None


def prepare_operations_descriptions() -> List[OperationDescription]:
    operations_chain_schema = OperationsChain.model_json_schema()
    defs = operations_chain_schema.get("$defs", {})
    operation_type_definitions = operations_chain_schema["properties"]["operations"]["items"]["oneOf"]  # fmt: skip

    operation_types = [
        name
        for type_definition in operation_type_definitions
        if (name := _ref_to_def_name(type_definition)) is not None
    ]
    type_definitions = [defs[operation_type] for operation_type in operation_types]

    result = []
    for type_definition in type_definitions:
        property_name_options = _get_property_name_options(type_definition, defs)

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
                property_name_options=property_name_options,
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
        NumberInRange,
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
        operator_schema = operator_type.model_json_schema()
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

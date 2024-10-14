from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from inference.core.workflows.core_steps.common.query_language.entities.introspection import (
    OperationDescription,
    OperatorDescription,
)
from inference.core.workflows.execution_engine.entities.types import Kind
from inference.core.workflows.execution_engine.introspection.entities import (
    BlockDescription,
)


class WorkflowInferenceResponse(BaseModel):
    outputs: List[Dict[str, Any]] = Field(
        description="Dictionary with keys defined in workflow output and serialised values"
    )
    profiler_trace: Optional[List[dict]] = Field(
        description="Profiler events",
        default=None,
    )


class WorkflowValidationStatus(BaseModel):
    status: str = Field(description="Represents validation status")


class ExternalWorkflowsBlockSelectorDefinition(BaseModel):
    manifest_type_identifier: str = Field(description="Identifier of block")
    property_name: str = Field(description="Name of specific property")
    property_description: str = Field(description="Description for specific property")
    compatible_element: str = Field(
        description="Defines to what type of object (step_output, parameter, etc) reference may be pointing"
    )
    is_list_element: bool = Field(
        description="Boolean flag defining if list of references will be accepted"
    )
    is_dict_element: bool = Field(
        description="Boolean flag defining if dict of references will be accepted"
    )


class ExternalBlockPropertyPrimitiveDefinition(BaseModel):
    manifest_type_identifier: str = Field(description="Identifier of block")
    property_name: str = Field(description="Name of specific property")
    property_description: str = Field(description="Description for specific property")
    type_annotation: str = Field(
        description="Pythonic type annotation for property",
        examples=["Union[str, int]"],
    )


class ExternalOperationDescription(BaseModel):
    operation_type: str
    compound: bool
    input_kind: List[str]
    output_kind: List[str]
    nested_operation_input_kind: Optional[List[str]] = None
    nested_operation_output_kind: Optional[List[str]] = None
    description: Optional[str] = None

    @classmethod
    def from_internal_entity(
        cls, operation_description: OperationDescription
    ) -> "ExternalOperationDescription":
        nested_operation_input_kind, nested_operation_output_kind = None, None
        if operation_description.nested_operation_input_kind:
            nested_operation_input_kind = [
                k.name for k in operation_description.nested_operation_input_kind
            ]
        if operation_description.nested_operation_output_kind:
            nested_operation_output_kind = [
                k.name for k in operation_description.nested_operation_output_kind
            ]
        return cls(
            operation_type=operation_description.operation_type,
            compound=operation_description.compound,
            input_kind=[k.name for k in operation_description.input_kind],
            output_kind=[k.name for k in operation_description.output_kind],
            nested_operation_input_kind=nested_operation_input_kind,
            nested_operation_output_kind=nested_operation_output_kind,
            description=operation_description.description,
        )


class ExternalOperatorDescription(BaseModel):
    operator_type: str
    operands_number: int
    operands_kinds: List[List[str]]
    description: Optional[str] = None

    @classmethod
    def from_internal_entity(
        cls, operator_description: OperatorDescription
    ) -> "ExternalOperatorDescription":
        operands_kinds = [
            [k.name for k in kind] for kind in operator_description.operands_kinds
        ]
        return cls(
            operator_type=operator_description.operator_type,
            operands_number=operator_description.operands_number,
            operands_kinds=operands_kinds,
            description=operator_description.description,
        )


class UniversalQueryLanguageDescription(BaseModel):
    operations_description: List[ExternalOperationDescription]
    operators_descriptions: List[ExternalOperatorDescription]

    @classmethod
    def from_internal_entities(
        cls,
        operations_descriptions: List[OperationDescription],
        operators_descriptions: List[OperatorDescription],
    ) -> "UniversalQueryLanguageDescription":
        operations_descriptions = [
            ExternalOperationDescription.from_internal_entity(
                operation_description=operation_description
            )
            for operation_description in operations_descriptions
        ]
        operators_descriptions = [
            ExternalOperatorDescription.from_internal_entity(
                operator_description=operator_description
            )
            for operator_description in operators_descriptions
        ]
        return cls(
            operations_description=operations_descriptions,
            operators_descriptions=operators_descriptions,
        )


class WorkflowsBlocksDescription(BaseModel):
    blocks: List[BlockDescription] = Field(
        description="List of loaded blocks descriptions"
    )
    declared_kinds: List[Kind] = Field(description="List of kinds defined for blocks")
    kinds_connections: Dict[str, List[ExternalWorkflowsBlockSelectorDefinition]] = (
        Field(
            description="Mapping from kind name into list of blocks properties accepting references of that kind"
        )
    )
    primitives_connections: List[ExternalBlockPropertyPrimitiveDefinition] = Field(
        description="List defining all properties for all blocks that can be filled "
        "with primitive values in workflow definition."
    )
    universal_query_language_description: UniversalQueryLanguageDescription = Field(
        description="Definitions of Universal Query Language operations and operators"
    )
    dynamic_block_definition_schema: dict = Field(
        description="Schema for dynamic block definition"
    )


class ExecutionEngineVersions(BaseModel):
    versions: List[str]


class WorkflowsBlocksSchemaDescription(BaseModel):
    schema: dict = Field(description="Schema for validating block definitions")


class DescribeInterfaceResponse(BaseModel):
    inputs: Dict[str, List[str]] = Field(
        description="Dictionary mapping Workflow inputs to their kinds"
    )
    outputs: Dict[str, Union[List[str], Dict[str, List[str]]]] = Field(
        description="Dictionary mapping Workflow outputs to their kinds"
    )
    typing_hints: Dict[str, str] = Field(
        description="Dictionary mapping name of the kind with Python typing hint for underlying serialised object",
    )
    kinds_schemas: Dict[str, Union[dict, List[dict]]] = Field(
        description="Dictionary mapping name of the kind with OpenAPI 3.0 definitions of underlying objects. "
        "If list is given, entity should be treated as union of types."
    )

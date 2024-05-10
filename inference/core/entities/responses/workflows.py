from typing import Any, Dict, List

from pydantic import BaseModel, Field

from inference.core.workflows.entities.types import Kind
from inference.core.workflows.execution_engine.introspection.entities import (
    BlockDescription,
)


class WorkflowInferenceResponse(BaseModel):
    outputs: Dict[str, Any] = Field(
        description="Dictionary with keys defined in workflow output and serialised values"
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


class ExternalBlockPropertyPrimitiveDefinition(BaseModel):
    manifest_type_identifier: str = Field(description="Identifier of block")
    property_name: str = Field(description="Name of specific property")
    property_description: str = Field(description="Description for specific property")
    type_annotation: str = Field(
        description="Pythonic type annotation for property",
        examples=["Union[str, int]"],
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

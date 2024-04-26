from typing import Any, Dict, List

from pydantic import BaseModel, Field

from inference.enterprise.workflows.entities.types import Kind
from inference.enterprise.workflows.execution_engine.introspection.entities import (
    BlockDescription,
)


class WorkflowInferenceResponse(BaseModel):
    outputs: Dict[str, Any] = Field(
        description="Dictionary with keys defined in workflow output and serialised values"
    )


class WorkflowValidationStatus(BaseModel):
    status: str = Field(description="Represents validation status")


class WorkflowsBlockPropertyDefinition(BaseModel):
    manifest_type_identifier: str
    property_name: str
    compatible_element: str


class WorkflowsBlocksDescription(BaseModel):
    blocks: List[BlockDescription] = Field(
        description="List of loaded blocks descriptions"
    )
    declared_kinds: List[Kind] = Field(description="List of kinds defined for blocks")
    kinds_connections: Dict[str, List[WorkflowsBlockPropertyDefinition]] = Field(
        description="Mapping from kind name into list of blocks properties accepting references of that kind"
    )

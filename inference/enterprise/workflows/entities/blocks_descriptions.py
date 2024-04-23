from typing import List, Type

from pydantic import BaseModel, Field

from inference.enterprise.workflows.entities.steps import OutputDefinition
from inference.enterprise.workflows.entities.types import Kind
from inference.enterprise.workflows.prototypes.block import (
    WorkflowBlock,
    WorkflowBlockManifest,
)


class BlockDescription(BaseModel):
    manifest_class: Type[WorkflowBlockManifest]
    block_class: Type[WorkflowBlock]
    block_schema: dict = Field(
        description="OpenAPI specification of block manifest that "
        "can be used to create workflow step in JSON definition."
    )
    outputs_manifest: List[OutputDefinition] = Field(
        description="Definition of step outputs and their kinds"
    )
    fully_qualified_class_name: str = Field(
        description="Fully qualified class name of block implementation."
    )


class BlocksDescription(BaseModel):
    blocks: List[BlockDescription] = Field(
        description="List of blocks definitions that can be used to create workflow."
    )
    declared_kinds: List[Kind]

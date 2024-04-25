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
    block_source: str = Field(description="Name of source plugin that defines block")
    fully_qualified_block_class_name: str = Field(
        description="Fully qualified class name of block implementation."
    )
    human_friendly_block_name: str = Field(
        description="Field generated based on class name providing human-friendly name of the block."
    )
    manifest_type_identifier: str = Field(
        description="Field holds value that is used to recognise block manifest while "
        "parsing `workflow` JSON definition."
    )
    manifest_type_identifier_aliases: List[str] = Field(
        description="Aliases of `manifest_type_identifier` that are in use.",
    )


class BlocksDescription(BaseModel):
    blocks: List[BlockDescription] = Field(
        description="List of blocks definitions that can be used to create workflow."
    )
    declared_kinds: List[Kind]

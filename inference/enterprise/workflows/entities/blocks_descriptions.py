from typing import List

from pydantic import BaseModel, Field

from inference.enterprise.workflows.entities.steps import OutputDefinition


class BlockDescription(BaseModel):
    block_manifest: dict = Field(
        description="OpenAPI specification of block manifest that "
        "can be used to create workflow step in JSON definition."
    )
    outputs_manifest: List[OutputDefinition] = Field(
        description="Definition of step outputs and their kinds"
    )


class BlocksDescription(BaseModel):
    blocks: List[BlockDescription] = Field(
        description="List of blocks definitions that can be used to create workflow."
    )

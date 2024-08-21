from typing import List, Literal, Optional, Type

from pydantic import Field

from inference.core.workflows.execution_engine.entities.base import (
    OutputDefinition,
    VideoMetadata,
)
from inference.core.workflows.execution_engine.entities.types import (
    INTEGER_KIND,
    WorkflowVideoMetadataSelector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)


class Manifest(WorkflowBlockManifest):
    type: Literal["ExampleVideoMetadataProcessing"]
    name: str = Field(description="name field")
    metadata: WorkflowVideoMetadataSelector

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="frame_number", kind=[INTEGER_KIND])]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.1.0,<2.0.0"


class VideoMetadataBlock(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return Manifest

    def run(
        self,
        metadata: VideoMetadata,
    ) -> BlockResult:
        return {"frame_number": metadata.frame_number}


def load_blocks() -> List[Type[WorkflowBlock]]:
    return [VideoMetadataBlock]

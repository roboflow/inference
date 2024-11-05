from typing import Literal, List, Optional, Type, Union

from pydantic import Field

from inference.core.workflows.execution_engine.entities.base import OutputDefinition, Batch, WorkflowImageData
from inference.core.workflows.execution_engine.entities.types import STRING_KIND, WorkflowImageSelector, \
    StepOutputImageSelector, WorkflowParameterSelector, StepOutputSelector
from inference.core.workflows.prototypes.block import WorkflowBlockManifest, WorkflowBlock, BlockResult


class BlockManifest(WorkflowBlockManifest):
    type: Literal["secret_store_user"]
    image: Union[WorkflowImageSelector, StepOutputImageSelector] = Field(
        title="Input Image",
        description="The input image for this step.",
    )
    secret: Union[StepOutputSelector(kind=[STRING_KIND]), str]

    @classmethod
    def accepts_batch_input(cls) -> bool:
        return True

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="output", kind=[STRING_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.0.0,<2.0.0"


class SecretStoreUserBlock(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        image: Batch[WorkflowImageData],
        secret: str
    ) -> BlockResult:
        print(f"Secret: {secret}")
        return [{"output": "ok"}] * len(image)

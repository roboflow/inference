from typing import List, Literal, Optional, Type

from pydantic import Field

from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    IMAGE_KIND,
    SECRET_KIND,
    STRING_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)


class BlockAcceptingBatchesOfImagesManifest(WorkflowBlockManifest):
    type: Literal["block_accepting_batches_of_images"]
    image: Selector(kind=[IMAGE_KIND]) = Field(
        title="Input Image",
        description="The input image for this step.",
    )
    secret: Selector(kind=[SECRET_KIND])

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["image"]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="output", kind=[STRING_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.4.0,<2.0.0"


class BlockAcceptingBatchesOfImages(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockAcceptingBatchesOfImagesManifest

    def run(self, image: Batch[WorkflowImageData], secret: str) -> BlockResult:
        return [{"output": "ok"}] * len(image)


class BlockAcceptingEmptyBatchesOfImagesManifest(WorkflowBlockManifest):
    type: Literal["block_accepting_empty_batches_of_images"]
    image: Selector(kind=[IMAGE_KIND]) = Field(
        title="Input Image",
        description="The input image for this step.",
    )
    secret: Selector(kind=[SECRET_KIND])

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["image"]

    @classmethod
    def accepts_empty_values(cls) -> bool:
        return True

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="output", kind=[STRING_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.4.0,<2.0.0"


class BlockAcceptingEmptyBatchesOfImages(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockAcceptingEmptyBatchesOfImagesManifest

    def run(
        self, image: Batch[Optional[WorkflowImageData]], secret: Optional[str]
    ) -> BlockResult:
        return [{"output": "empty" if secret is None else "ok"}] * len(image)


class BlockAcceptingImagesManifest(WorkflowBlockManifest):
    type: Literal["block_accepting_images"]
    image: Selector(kind=[IMAGE_KIND]) = Field(
        title="Input Image",
        description="The input image for this step.",
    )
    secret: Selector(kind=[SECRET_KIND])

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="output", kind=[STRING_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.4.0,<2.0.0"


class BlockAcceptingImages(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockAcceptingImagesManifest

    def run(self, image: WorkflowImageData, secret: str) -> BlockResult:
        return {"output": "ok"}


class BlockAcceptingEmptyImagesManifest(WorkflowBlockManifest):
    type: Literal["block_accepting_empty_images"]
    image: Selector(kind=[IMAGE_KIND]) = Field(
        title="Input Image",
        description="The input image for this step.",
    )
    secret: Selector(kind=[SECRET_KIND])

    @classmethod
    def accepts_empty_values(cls) -> bool:
        return True

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="output", kind=[STRING_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.4.0,<2.0.0"


class BlockAcceptingEmptyImages(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockAcceptingEmptyImagesManifest

    def run(
        self, image: Optional[WorkflowImageData], secret: Optional[str]
    ) -> BlockResult:
        return {"output": "empty" if secret is None else "ok"}


class BlockAcceptingScalarsManifest(WorkflowBlockManifest):
    type: Literal["block_accepting_scalars"]
    secret: Selector(kind=[SECRET_KIND])

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="output", kind=[SECRET_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.4.0,<2.0.0"


class BlockAcceptingScalars(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockAcceptingScalarsManifest

    def run(self, secret: str) -> BlockResult:
        return {"output": secret}


class BlockAcceptingEmptyScalarsManifest(WorkflowBlockManifest):
    type: Literal["block_accepting_empty_scalars"]
    secret: Selector(kind=[SECRET_KIND])

    @classmethod
    def accepts_empty_values(cls) -> bool:
        return True

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="output", kind=[SECRET_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.4.0,<2.0.0"


class BlockAcceptingEmptyScalars(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockAcceptingEmptyScalarsManifest

    def run(self, secret: Optional[str]) -> BlockResult:
        return {"output": "modified-secret"}


def load_blocks() -> List[Type[WorkflowBlock]]:
    return [
        BlockAcceptingEmptyScalars,
        BlockAcceptingScalars,
        BlockAcceptingEmptyImages,
        BlockAcceptingImages,
        BlockAcceptingEmptyBatchesOfImages,
        BlockAcceptingBatchesOfImages,
    ]

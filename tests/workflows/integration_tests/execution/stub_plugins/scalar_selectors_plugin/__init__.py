from typing import Any, List, Literal, Optional, Type, Union
from uuid import uuid4

import numpy as np
from pydantic import Field

from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    IMAGE_KIND,
    LIST_OF_VALUES_KIND,
    STRING_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)


class SecretBlockManifest(WorkflowBlockManifest):
    type: Literal["secret_store"]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="secret", kind=[STRING_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class SecretStoreBlock(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return SecretBlockManifest

    def run(self) -> BlockResult:
        return {"secret": "my_secret"}


class BlockManifest(WorkflowBlockManifest):
    type: Literal["secret_store_user"]
    image: Selector(kind=[IMAGE_KIND]) = Field(
        title="Input Image",
        description="The input image for this step.",
    )
    secret: Selector(kind=[STRING_KIND])

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
        return ">=1.3.0,<2.0.0"


class SecretStoreUserBlock(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(self, image: Batch[WorkflowImageData], secret: str) -> BlockResult:
        return [{"output": secret}] * len(image)


class BatchSecretBlockManifest(WorkflowBlockManifest):
    type: Literal["batch_secret_store"]
    image: Selector(kind=[IMAGE_KIND]) = Field(
        title="Input Image",
        description="The input image for this step.",
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="secret", kind=[STRING_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class BatchSecretStoreBlock(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BatchSecretBlockManifest

    def run(self, image: WorkflowImageData) -> BlockResult:
        return {"secret": f"my_secret_{uuid4()}"}


class NonBatchSecretStoreUserBlockManifest(WorkflowBlockManifest):
    type: Literal["non_batch_secret_store_user"]
    secret: Selector(kind=[STRING_KIND])

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="output", kind=[STRING_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class NonBatchSecretStoreUserBlock(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return NonBatchSecretStoreUserBlockManifest

    def run(self, secret: str) -> BlockResult:
        return {"output": secret}


class BlockWithReferenceImagesManifest(WorkflowBlockManifest):
    type: Literal["reference_images_comparison"]
    image: Selector(kind=[IMAGE_KIND])
    reference_images: Union[Selector(kind=[LIST_OF_VALUES_KIND]), Any]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="similarity", kind=[LIST_OF_VALUES_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class BlockWithReferenceImagesBlock(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockWithReferenceImagesManifest

    def run(
        self, image: WorkflowImageData, reference_images: List[np.ndarray]
    ) -> BlockResult:
        similarity = []
        for ref_image in reference_images:
            similarity.append(
                (image.numpy_image == ref_image).sum() / image.numpy_image.size
            )
        return {"similarity": similarity}


def load_blocks() -> List[Type[WorkflowBlock]]:
    return [
        SecretStoreBlock,
        SecretStoreUserBlock,
        BatchSecretStoreBlock,
        NonBatchSecretStoreUserBlock,
        BlockWithReferenceImagesBlock,
    ]

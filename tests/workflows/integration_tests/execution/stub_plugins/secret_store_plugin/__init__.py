from typing import List, Literal, Optional, Type
from uuid import uuid4

from pydantic import Field

from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    IMAGE_KIND,
    STRING_KIND,
    BatchSelector,
    ScalarSelector,
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
    image: BatchSelector(kind=[IMAGE_KIND]) = Field(
        title="Input Image",
        description="The input image for this step.",
    )
    secret: ScalarSelector(kind=[STRING_KIND])

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
        return ">=1.3.0,<2.0.0"


class SecretStoreUserBlock(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(self, image: Batch[WorkflowImageData], secret: str) -> BlockResult:
        return [{"output": secret}] * len(image)


class BatchSecretBlockManifest(WorkflowBlockManifest):
    type: Literal["batch_secret_store"]
    image: BatchSelector(kind=[IMAGE_KIND]) = Field(
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
    image: BatchSelector(kind=[IMAGE_KIND]) = Field(
        title="Input Image",
        description="The input image for this step.",
    )
    secret: ScalarSelector(kind=[STRING_KIND])

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

    def run(self, image: WorkflowImageData, secret: str) -> BlockResult:
        return {"output": secret}


def load_blocks() -> List[Type[WorkflowBlock]]:
    return [
        SecretStoreBlock,
        SecretStoreUserBlock,
        BatchSecretStoreBlock,
        NonBatchSecretStoreUserBlock,
    ]

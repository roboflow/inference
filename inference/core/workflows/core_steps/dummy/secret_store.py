from typing import Literal, List, Optional, Type

from inference.core.workflows.execution_engine.entities.base import OutputDefinition
from inference.core.workflows.execution_engine.entities.types import STRING_KIND
from inference.core.workflows.prototypes.block import WorkflowBlockManifest, WorkflowBlock, BlockResult


class BlockManifest(WorkflowBlockManifest):
    type: Literal["secret_store"]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="secret", kind=[STRING_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.0.0,<2.0.0"


class SecretStoreBlock(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(self) -> BlockResult:
        return {"secret": "my_secret"}

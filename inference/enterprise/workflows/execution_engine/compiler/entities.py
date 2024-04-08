from dataclasses import dataclass
from typing import Type

from inference.enterprise.workflows.prototypes.block import (
    WorkflowBlock,
    WorkflowBlockManifest,
)


@dataclass(frozen=True)
class BlockSpecification:
    block_source: str
    identifier: str
    block_class: Type[WorkflowBlock]
    manifest_class: Type[WorkflowBlockManifest]


@dataclass(frozen=True)
class InitialisedStep:
    block_specification: BlockSpecification
    manifest: WorkflowBlockManifest
    step: WorkflowBlock

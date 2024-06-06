from dataclasses import dataclass
from typing import Dict, List, Optional, Type

import networkx as nx

from inference.core.workflows.entities.base import InputType, JsonField
from inference.core.workflows.prototypes.block import (
    WorkflowBlock,
    WorkflowBlockManifest,
)


@dataclass(frozen=True)
class BatchDimensionIdentifier:
    static: bool
    size: Optional[int] = None
    identifier: Optional[int] = None

    def __str__(self):
        if self.static:
            return f"BatchSize[{self.size}]"
        return f"BatchSize[?id={self.identifier}]>"

    def generate_next_dynamic_identifier(self) -> "BatchDimensionIdentifier":
        if self.static:
            raise ValueError("Attempted to generate dynamic from static one")
        return BatchDimensionIdentifier(
            static=self.static, identifier=self.identifier + 1
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


@dataclass(frozen=True)
class ParsedWorkflowDefinition:
    version: str
    inputs: List[InputType]
    steps: List[WorkflowBlockManifest]
    outputs: List[JsonField]


@dataclass(frozen=True)
class InputSubstitution:
    input_parameter_name: str
    step_manifest: WorkflowBlockManifest
    manifest_property: str


@dataclass(frozen=True)
class CompiledWorkflow:
    workflow_definition: ParsedWorkflowDefinition
    execution_graph: nx.DiGraph
    steps: Dict[str, InitialisedStep]
    input_substitutions: List[InputSubstitution]

from dataclasses import dataclass
from typing import List, Optional, Type

import networkx as nx

from inference.enterprise.workflows.entities.outputs import JsonField
from inference.enterprise.workflows.entities.types import Kind
from inference.enterprise.workflows.entities.workflows_specification import InputType
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


@dataclass(frozen=True)
class ParsedWorkflowDefinition:
    version: str
    inputs: List[InputType]
    steps: List[WorkflowBlockManifest]
    outputs: List[JsonField]


@dataclass(frozen=True)
class ReferenceDefinition:
    selected_element: str
    kind: List[Kind]


@dataclass(frozen=True)
class SelectorDefinition:
    step_name: str
    property: str
    index: Optional[int]
    selector: str
    allowed_references: List[ReferenceDefinition]


@dataclass(frozen=True)
class CompiledWorkflow:
    execution_graph: nx.DiGraph
    steps: List[InitialisedStep]

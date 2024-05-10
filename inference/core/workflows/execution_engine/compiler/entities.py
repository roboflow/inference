from dataclasses import dataclass
from typing import Dict, List, Type

import networkx as nx

from inference.core.workflows.entities.base import InputType, JsonField
from inference.core.workflows.prototypes.block import (
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

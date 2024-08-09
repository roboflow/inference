from abc import abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Generator, List, Optional, Set, Type, Union

import networkx as nx

from inference.core.workflows.execution_engine.entities.base import InputType, JsonField
from inference.core.workflows.execution_engine.introspection.entities import (
    ParsedSelector,
)
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
    workflow_json: Dict[str, Any]
    init_parameters: Dict[str, Any]


class NodeCategory(Enum):
    INPUT_NODE = "INPUT_NODE"
    STEP_NODE = "STEP_NODE"
    OUTPUT_NODE = "OUTPUT_NODE"


@dataclass
class ExecutionGraphNode:
    node_category: NodeCategory
    name: str
    selector: str
    data_lineage: List[str]


@dataclass
class InputNode(ExecutionGraphNode):
    input_manifest: InputType

    @property
    def dimensionality(self) -> int:
        return len(self.data_lineage)

    def is_batch_oriented(self) -> bool:
        return len(self.data_lineage) > 0


@dataclass
class OutputNode(ExecutionGraphNode):
    output_manifest: JsonField

    @property
    def dimensionality(self) -> int:
        return len(self.data_lineage)

    def is_batch_oriented(self) -> bool:
        return len(self.data_lineage) > 0


class NodeInputCategory(Enum):
    NON_BATCH_INPUT_PARAMETER = "NON_BATCH_INPUT_PARAMETER"
    BATCH_INPUT_PARAMETER = "BATCH_INPUT_PARAMETER"
    NON_BATCH_STEP_OUTPUT = "NON_BATCH_STEP_OUTPUT"
    BATCH_STEP_OUTPUT = "BATCH_STEP_OUTPUT"
    STATIC_VALUE = "STATIC_VALUE"


INPUTS_REFERENCES = {
    NodeInputCategory.NON_BATCH_INPUT_PARAMETER,
    NodeInputCategory.BATCH_INPUT_PARAMETER,
}
STEPS_OUTPUTS_REFERENCES = {
    NodeInputCategory.NON_BATCH_STEP_OUTPUT,
    NodeInputCategory.BATCH_STEP_OUTPUT,
}


@dataclass(frozen=True)
class ParameterSpecification:
    parameter_name: str
    nested_element_key: Optional[str] = None
    nested_element_index: Optional[int] = None


@dataclass(frozen=True)
class StepInputDefinition:
    parameter_specification: ParameterSpecification
    category: NodeInputCategory

    def points_to_input(self) -> bool:
        return self.category in INPUTS_REFERENCES

    def points_to_step_output(self) -> bool:
        return self.category in STEPS_OUTPUTS_REFERENCES

    def is_static_value(self) -> bool:
        return self.category is NodeInputCategory.STATIC_VALUE

    @abstractmethod
    def is_batch_oriented(self) -> bool:
        pass

    @abstractmethod
    def get_dimensionality(self) -> int:
        pass

    @classmethod
    def is_compound_input(cls) -> bool:
        return False


@dataclass(frozen=True)
class DynamicStepInputDefinition(StepInputDefinition):
    data_lineage: List[str]
    selector: str

    def is_batch_oriented(self) -> bool:
        return len(self.data_lineage) > 0

    def get_dimensionality(self) -> int:
        return len(self.data_lineage)


@dataclass(frozen=True)
class StaticStepInputDefinition(StepInputDefinition):
    value: Any

    def is_batch_oriented(self) -> bool:
        return False

    def get_dimensionality(self) -> int:
        return 0


@dataclass(frozen=True)
class CompoundStepInputDefinition:
    name: str
    nested_definitions: Union[List[StepInputDefinition], Dict[str, StepInputDefinition]]

    @classmethod
    def is_compound_input(cls) -> bool:
        return True

    def represents_list_of_inputs(self) -> bool:
        return isinstance(self.nested_definitions, list)

    @abstractmethod
    def iterate_through_definitions(self) -> Generator[StepInputDefinition, None, None]:
        pass


@dataclass(frozen=True)
class ListOfStepInputDefinitions(CompoundStepInputDefinition):
    nested_definitions: List[StepInputDefinition]

    def iterate_through_definitions(self) -> Generator[StepInputDefinition, None, None]:
        for definition in self.nested_definitions:
            yield definition


@dataclass(frozen=True)
class DictOfStepInputDefinitions(CompoundStepInputDefinition):
    nested_definitions: Dict[str, StepInputDefinition]

    def iterate_through_definitions(self) -> Generator[StepInputDefinition, None, None]:
        for definition in self.nested_definitions.values():
            yield definition


StepInputData = Dict[str, Union[StepInputDefinition, CompoundStepInputDefinition]]


@dataclass
class StepNode(ExecutionGraphNode):
    step_manifest: WorkflowBlockManifest
    input_data: StepInputData = field(default_factory=dict)
    dimensionality_reference_property: Optional[str] = None
    child_execution_branches: Dict[str, str] = field(default_factory=dict)
    execution_branches_impacting_inputs: Set[str] = field(default_factory=set)
    batch_oriented_parameters: Set[str] = field(default_factory=set)
    step_execution_dimensionality: int = 0

    def controls_flow(self) -> bool:
        if self.child_execution_branches:
            return True
        return False

    @property
    def output_dimensionality(self) -> int:
        return len(self.data_lineage)

    def is_batch_oriented(self) -> bool:
        return len(self.batch_oriented_parameters) > 0


@dataclass(frozen=True)
class PropertyPredecessorDefinition:
    predecessor_selector: str
    parsed_selector: ParsedSelector


@dataclass(frozen=True)
class InputDimensionalitySpecification:
    actual_dimensionality: int
    expected_offset: int

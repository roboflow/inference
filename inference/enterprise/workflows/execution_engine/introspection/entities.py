from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Type

from inference.enterprise.workflows.entities.types import STEP_AS_SELECTED_ELEMENT, Kind
from inference.enterprise.workflows.prototypes.block import WorkflowBlock


@dataclass(frozen=True)
class ReferenceDefinition:
    selected_element: str
    kind: List[Kind]


@dataclass(frozen=True)
class SelectorDefinition:
    property_name: str
    property_description: str
    allowed_references: List[ReferenceDefinition]
    is_list_element: bool


@dataclass(frozen=True)
class ParsedSelector:
    definition: SelectorDefinition
    step_name: str
    property_name: str
    value: str
    index: Optional[int]


@dataclass(frozen=True)
class PrimitiveTypeDefinition:
    property_name: str
    property_description: str
    type_annotation: str


@dataclass(frozen=True)
class BlockManifestMetadata:
    primitive_types: Dict[str, PrimitiveTypeDefinition]
    selectors: Dict[str, SelectorDefinition]


@dataclass(frozen=True)
class BlocksConnections:
    property_wise: Dict[Type[WorkflowBlock], Dict[str, Set[Type[WorkflowBlock]]]]
    block_wise: Dict[Type[WorkflowBlock], Set[Type[WorkflowBlock]]]


@dataclass(frozen=True)
class BlockPropertyDefinition:
    block_type: Type[WorkflowBlock]
    property_name: str
    compatible_element: str


@dataclass(frozen=True)
class DiscoveredConnections:
    input_connections: BlocksConnections
    output_connections: BlocksConnections
    kinds_connections: Dict[str, Set[BlockPropertyDefinition]]

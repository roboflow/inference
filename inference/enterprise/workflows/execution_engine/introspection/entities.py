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

    def to_type_annotation(self) -> str:
        type_annotation_chunks = set()
        for allowed_reference in self.allowed_references:
            if allowed_reference.selected_element == STEP_AS_SELECTED_ELEMENT:
                type_annotation_chunks.add("step")
                continue
            for kind in allowed_reference.kind:
                type_annotation_chunks.add(kind.name)
        type_annotation_str = ", ".join(type_annotation_chunks)
        if len(type_annotation_chunks) > 1:
            return f"Union[{type_annotation_str}]"
        return type_annotation_str


@dataclass(frozen=True)
class SelectorMetadata:
    definition: SelectorDefinition
    step_name: str
    value: str
    index: Optional[int]


@dataclass(frozen=True)
class PrimitiveTypeMetadata:
    property_name: str
    property_description: str
    type_annotation: str


@dataclass(frozen=True)
class BlockManifestMetadata:
    primitive_types: Dict[str, PrimitiveTypeMetadata]
    selectors: Dict[str, SelectorDefinition]


@dataclass(frozen=True)
class BlocksConnections:
    property_wise: Dict[Type[WorkflowBlock], Dict[str, Set[Type[WorkflowBlock]]]]
    block_wise: Dict[Type[WorkflowBlock], Set[Type[WorkflowBlock]]]


@dataclass(frozen=True)
class DiscoveredConnections:
    input_connections: BlocksConnections
    output_connections: BlocksConnections

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Type, Union

from pydantic import BaseModel, Field

from inference.core.workflows.execution_engine.entities.base import OutputDefinition
from inference.core.workflows.execution_engine.entities.types import Kind
from inference.core.workflows.prototypes.block import (
    WorkflowBlock,
    WorkflowBlockManifest,
)

PropertyName = str
KindName = str


@dataclass(frozen=True)
class ReferenceDefinition:
    selected_element: str
    kind: List[Kind]
    points_to_batch: Set[bool]


@dataclass(frozen=True)
class SelectorDefinition:
    property_name: PropertyName
    property_description: str
    allowed_references: List[ReferenceDefinition]
    is_list_element: bool
    is_dict_element: bool
    dimensionality_offset: int
    is_dimensionality_reference_property: bool


@dataclass(frozen=True)
class ParsedSelector:
    definition: SelectorDefinition
    step_name: str
    value: str
    index: Optional[int]
    key: Optional[str]


@dataclass(frozen=True)
class PrimitiveTypeDefinition:
    property_name: PropertyName
    property_description: str
    type_annotation: str


@dataclass(frozen=True)
class BlockManifestMetadata:
    primitive_types: Dict[PropertyName, PrimitiveTypeDefinition]
    selectors: Dict[PropertyName, SelectorDefinition]


@dataclass(frozen=True)
class BlocksConnections:
    property_wise: Dict[
        Type[WorkflowBlock], Dict[PropertyName, Set[Type[WorkflowBlock]]]
    ]
    block_wise: Dict[Type[WorkflowBlock], Set[Type[WorkflowBlock]]]


@dataclass(frozen=True)
class BlockPropertySelectorDefinition:
    block_type: Type[WorkflowBlock]
    manifest_type_identifier: str
    property_name: PropertyName
    property_description: str
    compatible_element: str
    is_list_element: bool
    is_dict_element: bool


@dataclass(frozen=True)
class BlockPropertyPrimitiveDefinition:
    block_type: Type[WorkflowBlock]
    manifest_type_identifier: str
    property_name: PropertyName
    property_description: str
    type_annotation: str


@dataclass(frozen=True)
class DiscoveredConnections:
    input_connections: BlocksConnections
    output_connections: BlocksConnections
    kinds_connections: Dict[KindName, Set[BlockPropertySelectorDefinition]]
    primitives_connections: List[BlockPropertyPrimitiveDefinition]


class BlockDescription(BaseModel):
    manifest_class: Union[Type[WorkflowBlockManifest], Type[BaseModel]] = Field(
        exclude=True
    )
    # Type[BaseModel] here is to let dynamic blocks being BaseModel to pass validation - but that should be
    # the only case for using this type in this field. Dynamic blocks implements the same interface, yet due
    # to dynamic nature of creation - cannot be initialised as abstract class WorkflowBlockManifest
    block_class: Type[WorkflowBlock] = Field(exclude=True)
    block_schema: dict = Field(
        description="OpenAPI specification of block manifest that "
        "can be used to create workflow step in JSON definition."
    )
    outputs_manifest: List[OutputDefinition] = Field(
        description="Definition of step outputs and their kinds"
    )
    block_source: str = Field(description="Name of source plugin that defines block")
    fully_qualified_block_class_name: str = Field(
        description="Fully qualified class name of block implementation."
    )
    human_friendly_block_name: str = Field(
        description="Field generated based on class name providing human-friendly name of the block."
    )
    manifest_type_identifier: str = Field(
        description="Field holds value that is used to recognise block manifest while "
        "parsing `workflow` JSON definition."
    )
    manifest_type_identifier_aliases: List[str] = Field(
        description="Aliases of `manifest_type_identifier` that are in use.",
        default_factory=list,
    )
    execution_engine_compatibility: Optional[str] = Field(
        description="Execution Engine versions compatible with block.",
        default=None,
    )
    input_dimensionality_offsets: Dict[str, int] = Field(
        description="Dimensionality offsets for input parameters",
    )
    dimensionality_reference_property: Optional[str] = Field(
        description="Selected dimensionality reference property provided if different dimensionality for "
        "different inputs are supported.",
    )
    output_dimensionality_offset: int = Field(
        description="Dimensionality offset for block output.",
    )


class BlocksDescription(BaseModel):
    blocks: List[BlockDescription] = Field(
        description="List of blocks definitions that can be used to create workflow."
    )
    declared_kinds: List[Kind]

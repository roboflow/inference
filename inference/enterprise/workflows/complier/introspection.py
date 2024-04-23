from copy import copy
from typing import List

from pydantic import BaseModel

from inference.enterprise.workflows.entities.blocks_descriptions import (
    BlockDescription,
    BlocksDescription,
)
from inference.enterprise.workflows.entities.steps import OutputDefinition
from inference.enterprise.workflows.entities.types import WILDCARD_KIND, Kind
from inference.enterprise.workflows.entities.workflows_specification import (
    ALL_BLOCKS_CLASSES,
)
from inference.enterprise.workflows.execution_engine.compiler.blocks_loader import (
    get_full_type_name,
)


def describe_available_blocks() -> BlocksDescription:
    all_blocks_classes = get_available_bocks_classes()
    blocks = []
    declared_kinds = []
    for block in all_blocks_classes:
        block_schema = block.schema()
        if hasattr(block, "describe_outputs"):
            outputs_manifest = block.describe_outputs()
        else:
            outputs_manifest = [OutputDefinition(name="*", kind=[WILDCARD_KIND])]
        declared_kinds.extend(get_kinds_declared_for_block(block_schema=block_schema))
        blocks.append(
            BlockDescription(
                manifest_class=type(block),
                block_class=type(block),
                block_schema=block_schema,
                outputs_manifest=outputs_manifest,
                fully_qualified_class_name=get_full_type_name(t=type(block)),
            )
        )
    declared_kinds = list(set(declared_kinds))
    return BlocksDescription(blocks=blocks, declared_kinds=declared_kinds)


def get_available_bocks_classes() -> List[BaseModel]:
    return copy(ALL_BLOCKS_CLASSES)


def get_kinds_declared_for_block(block_schema: dict) -> List[Kind]:
    result = []
    for property_name, property_definition in block_schema["properties"].items():
        union_elements = property_definition.get("anyOf", [property_definition])
        for element in union_elements:
            for raw_kind in element.get("kind", []):
                parsed_kind = Kind.model_validate(raw_kind)
                result.append(parsed_kind)
    return result

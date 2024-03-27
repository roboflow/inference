from copy import copy
from typing import List

from pydantic import BaseModel

from inference.enterprise.workflows.entities.blocks_descriptions import (
    BlockDescription,
    BlocksDescription,
)
from inference.enterprise.workflows.entities.steps import OutputDefinition
from inference.enterprise.workflows.entities.types import WILDCARD_KIND
from inference.enterprise.workflows.entities.workflows_specification import (
    ALL_BLOCKS_CLASSES,
)


def describe_available_blocks() -> BlocksDescription:
    all_blocks_classes = get_available_bocks_classes()
    result = []
    for block in all_blocks_classes:
        block_manifest = block.schema()
        if hasattr(block, "describe_outputs"):
            outputs_manifest = block.describe_outputs()
        else:
            outputs_manifest = [OutputDefinition(name="*", kind=[WILDCARD_KIND])]
        result.append(
            BlockDescription(
                block_manifest=block_manifest,
                outputs_manifest=outputs_manifest,
            )
        )
    return BlocksDescription(blocks=result)


def get_available_bocks_classes() -> List[BaseModel]:
    return copy(ALL_BLOCKS_CLASSES)

"""
This is just example, test implementation, please do not assume it being fully functional.
"""

from copy import deepcopy
from datetime import datetime
from typing import List, Literal, Type

import numpy as np
import supervision as sv
from pydantic import ConfigDict

from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import STRING_KIND
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "short_description": "",
            "long_description": "",
            "license": "Apache-2.0",
            "block_type": "dummy",
        }
    )
    type: Literal["CurrentTime"]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name="time",
                kind=[STRING_KIND],
            ),
        ]


class CurrentTimeBlock(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(self) -> BlockResult:
        return {"time": datetime.now().isoformat()}


def load_blocks() -> List[Type[WorkflowBlock]]:
    return [CurrentTimeBlock]

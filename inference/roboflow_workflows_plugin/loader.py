"""Compatibility stub for the Workflows plugin of inference 1.6.1-1.7.2.

The Roboflow-platform blocks are core blocks of ``roboflow_workflows`` again,
so an environment that still lists this plugin loads them once, from the core
loader.
"""

from typing import List, Type

from inference.core.workflows.prototypes.block import WorkflowBlock

BLOCKS_SOURCE = "workflows_core"


def load_blocks() -> List[Type[WorkflowBlock]]:
    return []

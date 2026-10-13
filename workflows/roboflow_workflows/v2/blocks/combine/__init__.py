"""Blocks that merge or combine detection rows.

``create_catalogue()`` registers every block in ``BLOCKS``.
"""

from roboflow_workflows.v2.blocks.combine.combine import (
    DetectionsCombineBlock,
    combine_predictions,
)
from roboflow_workflows.v2.blocks.combine.merge import DetectionsMergeBlock, merge_rows

BLOCKS = (
    DetectionsCombineBlock,
    DetectionsMergeBlock,
)

__all__ = [
    "BLOCKS",
    "DetectionsCombineBlock",
    "DetectionsMergeBlock",
    "combine_predictions",
    "merge_rows",
]

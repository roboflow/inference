"""Blocks that fuse predictions from several independent sources.

``pairwise`` holds the cross-set overlap helper shared by fusion blocks.
``consensus`` keeps the detections several models agree on.
``difference`` compares detections before and after a change.
``overlap_analysis`` reports asymmetric coverage in comparable coordinates.
``rollup`` maps crop predictions back into their parent frame and merges them.

``BLOCKS`` lists the blocks to register in the catalogue.
"""

from roboflow_workflows.v2.blocks.fusion.consensus import DetectionsConsensusBlock
from roboflow_workflows.v2.blocks.fusion.difference import DetectionsDifferenceBlock
from roboflow_workflows.v2.blocks.fusion.overlap_analysis import (
    DETECTIONS_OVERLAPS_KIND,
    OverlapAnalysisBlock,
)
from roboflow_workflows.v2.blocks.fusion.rollup import DetectionsListRollUpBlock

BLOCKS = (
    DetectionsConsensusBlock,
    DetectionsDifferenceBlock,
    OverlapAnalysisBlock,
    DetectionsListRollUpBlock,
)

__all__ = [
    "BLOCKS",
    "DETECTIONS_OVERLAPS_KIND",
    "DetectionsConsensusBlock",
    "DetectionsDifferenceBlock",
    "DetectionsListRollUpBlock",
    "OverlapAnalysisBlock",
]

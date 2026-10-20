"""Blocks that fuse predictions from several independent sources.

``pairwise`` holds the cross-set overlap helper shared by fusion blocks.
``consensus`` keeps the detections several models agree on.
``difference`` compares detections before and after a change.
``overlap_analysis`` reports asymmetric coverage in comparable coordinates.
``rollup`` maps crop predictions back into their parent frame and merges them.

``BLOCKS`` lists the blocks to register in the catalogue.
"""

from roboflow_workflows.v2.blocks.fusion.detections_consensus.v1 import DetectionsConsensusBlock
from roboflow_workflows.v2.blocks.fusion.detections_difference.v1 import DetectionsDifferenceBlock
from roboflow_workflows.v2.blocks.fusion.overlap_analysis.v1 import DETECTIONS_OVERLAPS_KIND, OverlapAnalysisBlock
from roboflow_workflows.v2.blocks.fusion.detections_list_rollup.v1 import DetectionsListRollUpBlock

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

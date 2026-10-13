"""Blocks that fuse predictions from several independent sources.

``pairwise`` holds the cross-set overlap helper shared by fusion blocks.
``consensus`` keeps the detections several models agree on.
``difference`` compares detections before and after a change.

``BLOCKS`` lists the blocks to register in the catalogue.
"""

from roboflow_workflows.v2.blocks.fusion.consensus import DetectionsConsensusBlock
from roboflow_workflows.v2.blocks.fusion.difference import DetectionsDifferenceBlock

BLOCKS = (DetectionsConsensusBlock, DetectionsDifferenceBlock)

__all__ = [
    "BLOCKS",
    "DetectionsConsensusBlock",
    "DetectionsDifferenceBlock",
]

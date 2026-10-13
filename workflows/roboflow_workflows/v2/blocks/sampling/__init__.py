"""Blocks that sample a stream by how unusual its embeddings are.

    v2/identify_changes@v1    unusual rate of change (EMA / SMA / Sliding Window)
    v2/identify_outliers@v1   direction unlike the recent window (vMF rank)

Both keep statistics per source by default; ``scope="global"`` shares one
state per block, as V1 did.
"""

from roboflow_workflows.v2.blocks.sampling._state import SamplingCapacityError
from roboflow_workflows.v2.blocks.sampling.changes import IdentifyChangesBlock
from roboflow_workflows.v2.blocks.sampling.outliers import IdentifyOutliersBlock

BLOCKS = (IdentifyChangesBlock, IdentifyOutliersBlock)

__all__ = [
    "BLOCKS",
    "IdentifyChangesBlock",
    "IdentifyOutliersBlock",
    "SamplingCapacityError",
]

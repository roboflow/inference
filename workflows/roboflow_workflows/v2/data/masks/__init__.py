"""Instance-segmentation mask views, providers and their operations.

``grid`` holds the current mask-grid backend of ``SegmentationView``; it is one
backend, not the universal compact-mask representation. ``providers`` puts the
grid, authoritative dense masks and native COCO RLE masks behind one binary
request with explicit fidelity and recipe.
"""

from roboflow_workflows.v2.data.masks.grid import (
    INSTANCE_SEGMENTATION_VIEW_KIND,
    MaskGridGeometry,
    SegmentationView,
)
from roboflow_workflows.v2.data.masks.providers import (
    APPROXIMATE_BINARY,
    BINARY,
    SCORE_GRID,
    BinaryMasks,
    DenseMaskProvider,
    GridMaskProvider,
    MaskProvider,
    MaskRecipe,
    RasterGeometry,
    RLEMaskProvider,
    UnsupportedMaskRequest,
    mask_provider_of,
)

__all__ = [
    "APPROXIMATE_BINARY",
    "BINARY",
    "BinaryMasks",
    "DenseMaskProvider",
    "GridMaskProvider",
    "INSTANCE_SEGMENTATION_VIEW_KIND",
    "MaskGridGeometry",
    "MaskProvider",
    "MaskRecipe",
    "RLEMaskProvider",
    "RasterGeometry",
    "SCORE_GRID",
    "SegmentationView",
    "UnsupportedMaskRequest",
    "mask_provider_of",
]

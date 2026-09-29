"""Native V2 CPU image block catalogue.

This package is imported only by callers that explicitly select the native
blocks. The generic V2 engine (`data`, `contracts`, `compiler`, `executor`)
does not import it, and the legacy V1 loader does not discover it.
"""

from roboflow_workflows.execution_engine.v2.blocks.image import (
    CROP_BLOCK_NAME,
    CROP_CONTRACT,
    CROP_REGIONS_AXIS,
    HAS_BRIGHTNESS_BLOCK_NAME,
    HAS_BRIGHTNESS_CONTRACT,
    INVERT_BLOCK_NAME,
    INVERT_CONTRACT,
    MOSAIC_BLOCK_NAME,
    MOSAIC_CONTRACT,
    CropBlock,
    HasBrightnessBlock,
    InvertBlock,
    MosaicBlock,
)
from roboflow_workflows.execution_engine.v2.blocks.kinds import (
    BOOLEAN_KIND,
    CROP_SUMMARY_KIND,
    IMAGE_KIND,
    INTEGER_KIND,
    is_boolean_payload,
    is_crop_summary_payload,
    is_image_payload,
    is_integer_payload,
)
from roboflow_workflows.execution_engine.v2.blocks.registry import (
    NATIVE_BLOCK_NAMES,
    NATIVE_KIND_NAMES,
    native_registry,
)

__all__ = [
    "BOOLEAN_KIND",
    "CROP_BLOCK_NAME",
    "CROP_CONTRACT",
    "CROP_REGIONS_AXIS",
    "CROP_SUMMARY_KIND",
    "CropBlock",
    "HAS_BRIGHTNESS_BLOCK_NAME",
    "HAS_BRIGHTNESS_CONTRACT",
    "HasBrightnessBlock",
    "IMAGE_KIND",
    "INTEGER_KIND",
    "INVERT_BLOCK_NAME",
    "INVERT_CONTRACT",
    "InvertBlock",
    "MOSAIC_BLOCK_NAME",
    "MOSAIC_CONTRACT",
    "MosaicBlock",
    "NATIVE_BLOCK_NAMES",
    "NATIVE_KIND_NAMES",
    "is_boolean_payload",
    "is_crop_summary_payload",
    "is_image_payload",
    "is_integer_payload",
    "native_registry",
]

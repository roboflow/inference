from typing import List, Literal, Optional, Type, Union

import cv2
import numpy as np
from pydantic import AliasChoices, ConfigDict, Field

from inference.core.workflows.core_steps.visualizations.common.base import (
    OUTPUT_IMAGE_KEY,
)
from inference.core.workflows.execution_engine.entities.base import (
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    IMAGE_KIND,
    INTEGER_KIND,
    STRING_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

SHORT_DESCRIPTION: str = "Apply morphological transformation to color images."
LONG_DESCRIPTION = """
Apply morphological transformations to color images using configurable operations to modify object shapes, remove noise, fill holes, detect edges, and extract features. Color images are processed with transformations applied to all channels, and the output is returned as color for compatibility with downstream processing blocks.

## How This Block Works

This block applies morphological transformations directly to color images (BGR or BGRA). The block:

1. Receives an input image to transform (works with color or grayscale images)
2. Ensures image is in color format (if grayscale, converts to BGR for consistent color output)
3. Creates a square structuring element (kernel) of specified kernel_size
4. Applies the selected morphological operation to all channels simultaneously
5. Preserves the color format throughout the transformation (including alpha channel for BGRA images)
6. Returns the morphologically transformed color image

Supported operations:

- **Erosion**: Shrinks objects by removing boundary pixels (separates objects, removes noise)
- **Dilation**: Expands objects by adding boundary pixels (connects objects, fills holes)
- **Opening**: Erosion followed by dilation (removes noise, separates objects)
- **Closing**: Dilation followed by erosion (fills holes, connects fragments)
- **Opening then Closing**: Opens then closes (specialized preprocessing for edge detection and refinement)
- **Gradient**: Dilation minus erosion (highlights object boundaries/edges)
- **Top Hat**: Input minus opening (detects small bright details)
- **Black Hat**: Closing minus input (detects small dark details)

All operations preserve the color format, making output compatible with downstream color-based blocks like Mask Edge Snap.

## Common Use Cases

- **Color Image Preprocessing**: Process color images while preserving color information for downstream blocks
- **Edge Detection and Refinement**: Use opening+closing for specialized preprocessing before edge snapping
- **Noise Removal**: Remove noise from color images while maintaining color channels
- **Feature Extraction**: Extract morphological features from color images
- **Image Conditioning**: Prepare color images for color-based downstream processing

## Connecting to Other Blocks

- **Before Mask Edge Snap**: Output color images for edge snapping and mask refinement
- **Before visualization blocks**: Display morphologically processed color images
- **In color image pipelines**: Process color images while maintaining color information
"""


class MorphologicalTransformationV2Manifest(WorkflowBlockManifest):
    type: Literal["roboflow_core/morphological_transformation@v2"]
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Morphological Transformation",
            "version": "v2",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "classical_computer_vision",
            "ui_manifest": {
                "section": "classical_cv",
                "icon": "far fa-image",
                "blockPriority": 5,
            },
        }
    )

    image: Selector(kind=[IMAGE_KIND]) = Field(
        title="Input Image",
        description="Input color image to apply morphological transformation to. Works with BGR, BGRA, or grayscale images. Grayscale images are automatically converted to BGR for consistent color output. Transformations are applied to all channels. Output is always in color format (BGR) for compatibility with downstream blocks like Mask Edge Snap.",
        examples=["$inputs.image", "$steps.cropping.crops"],
        validation_alias=AliasChoices("image", "images"),
    )

    kernel_size: Union[int, Selector(kind=[INTEGER_KIND])] = Field(
        default=5,
        description="Size of the square structuring element (kernel) used for morphological operations. Must be a positive integer. Typical values range from 3-15. Adjust based on object sizes and desired transformation scale.",
        examples=["5", "$inputs.kernel_size"],
    )

    operation: Union[
        Selector(kind=[STRING_KIND]),
        Literal[
            "Erosion",
            "Dilation",
            "Opening",
            "Closing",
            "Opening then Closing",
            "Gradient",
            "Top Hat",
            "Black Hat",
        ],
    ] = Field(
        default="Opening then Closing",
        description="Type of morphological operation to apply. 'Opening then Closing' is specifically designed for preprocessing before edge detection and mask refinement (e.g., with Mask Edge Snap block). All other operations follow standard morphological definitions.",
        examples=["Closing", "Opening then Closing", "$inputs.operation"],
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name=OUTPUT_IMAGE_KEY,
                kind=[IMAGE_KIND],
            ),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class MorphologicalTransformationBlockV2(WorkflowBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def get_manifest(cls) -> Type[MorphologicalTransformationV2Manifest]:
        return MorphologicalTransformationV2Manifest

    def run(
        self,
        image: WorkflowImageData,
        kernel_size: int = 5,
        operation: str = "Opening then Closing",
    ) -> BlockResult:
        updated_image = apply_morphological_operation(
            image.numpy_image, kernel_size, operation
        )

        output = WorkflowImageData.copy_and_replace(
            origin_image_data=image,
            numpy_image=updated_image,
        )
        return {OUTPUT_IMAGE_KEY: output}


def apply_morphological_operation(
    img: np.ndarray, kernel_size: int, operation: str
) -> np.ndarray:
    """Apply morphological operation to color image, preserving color format and alpha channel."""
    # Save alpha channel if present
    alpha = None
    if len(img.shape) == 2:
        # Grayscale to BGR
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 1:
        # Single channel to BGR
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        # BGRA: extract alpha, work on BGR only
        alpha = img[:, :, 3:4]
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    if operation == "Dilation":
        result = cv2.dilate(img, kernel, iterations=1)
    elif operation == "Erosion":
        result = cv2.erode(img, kernel, iterations=1)
    elif operation == "Opening":
        result = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    elif operation == "Closing":
        result = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    elif operation == "Opening then Closing":
        # Apply morphological open (erosion followed by dilation)
        opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        # Apply morphological close (dilation followed by erosion)
        result = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    elif operation == "Gradient":
        result = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
    elif operation == "Top Hat":
        result = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
    elif operation == "Black Hat":
        result = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
    else:
        raise ValueError(
            f"Invalid operation: {operation}. Supported operations are 'Erosion', 'Dilation', 'Opening', 'Closing', 'Opening then Closing', 'Gradient', 'Top Hat', 'Black Hat'."
        )

    # Re-attach alpha if it was present
    if alpha is not None:
        result = np.concatenate([result, alpha], axis=2)

    return result

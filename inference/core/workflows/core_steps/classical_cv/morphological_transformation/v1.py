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

SHORT_DESCRIPTION: str = "Apply morphological transformation to an image."
LONG_DESCRIPTION = """
Apply morphological transformations to images using configurable operations (erosion, dilation, opening, closing, gradient, top hat, black hat) to modify object shapes, remove noise, fill holes, detect edges, and extract features from binary or grayscale images for image processing, noise removal, and feature extraction workflows.

## How This Block Works

This block applies morphological transformations to images using structuring elements (kernels) to modify object shapes and extract features. The block:

1. Receives an input image to transform (color images are automatically converted to grayscale)
2. Converts the image to grayscale if it's in color format (morphological operations work on single-channel images)
3. Creates a square structuring element (kernel) of specified kernel_size:
   - Creates a square matrix of ones with dimensions (kernel_size, kernel_size)
   - The kernel defines the neighborhood used for morphological operations
   - Larger kernels affect larger regions, smaller kernels affect smaller regions
4. Applies the selected morphological operation:

   **For Erosion:**
   - Shrinks objects by removing pixels from object boundaries
   - Replaces each pixel with the minimum value in its kernel neighborhood
   - Thins objects and removes small protrusions
   - Useful for separating touching objects and removing small noise

   **For Dilation:**
   - Expands objects by adding pixels to object boundaries
   - Replaces each pixel with the maximum value in its kernel neighborhood
   - Thickens objects and fills small holes
   - Useful for connecting nearby objects and filling gaps

   **For Opening (Erosion followed by Dilation):**
   - First erodes then dilates the image
   - Removes small noise and separates touching objects
   - Preserves larger objects while eliminating small details
   - Useful for noise removal and object separation

   **For Closing (Dilation followed by Erosion):**
   - First dilates then erodes the image
   - Fills small holes and connects nearby objects
   - Preserves object shape while closing gaps
   - Useful for filling holes and connecting fragmented objects

   **For Gradient (Dilation minus Erosion):**
   - Computes the difference between dilated and eroded images
   - Highlights object boundaries and edges
   - Creates an outline effect showing object perimeters
   - Useful for edge detection and boundary extraction

   **For Top Hat (Input minus Opening):**
   - Computes the difference between original image and its opening
   - Highlights small bright details that were removed by opening
   - Emphasizes bright features smaller than the kernel
   - Useful for detecting small bright objects or details

   **For Black Hat (Closing minus Input):**
   - Computes the difference between closing and original image
   - Highlights small dark details that were filled by closing
   - Emphasizes dark features smaller than the kernel
   - Useful for detecting small dark objects or holes

5. Preserves the image structure and metadata
6. Returns the morphologically transformed image

Morphological operations use structuring elements (kernels) to probe and modify image structures. The kernel_size controls the scale of the transformation - larger kernels affect larger regions and have more pronounced effects. Basic operations (erosion, dilation) modify object sizes, composite operations (opening, closing) combine effects for noise removal and gap filling, and derived operations (gradient, top hat, black hat) extract specific features. The operations work best on binary or high-contrast grayscale images where object boundaries are clearly defined.

## Common Use Cases

- **Noise Removal**: Remove small noise and artifacts from binary or grayscale images (e.g., remove salt-and-pepper noise, eliminate small artifacts, clean up thresholded images), enabling noise removal workflows
- **Object Separation**: Separate touching or overlapping objects in images (e.g., separate touching objects, split connected regions, isolate individual objects), enabling object separation workflows
- **Hole Filling**: Fill small holes and gaps in objects (e.g., fill holes in objects, close gaps in shapes, complete fragmented objects), enabling hole filling workflows
- **Edge Detection**: Detect object boundaries and edges using morphological gradient (e.g., find object edges, extract boundaries, detect object outlines), enabling morphological edge detection workflows
- **Feature Extraction**: Extract specific features like small bright or dark objects (e.g., detect small bright details with top hat, find small dark objects with black hat, extract specific morphological features), enabling feature extraction workflows
- **Image Preprocessing**: Prepare binary or grayscale images for further processing (e.g., clean up thresholded images, prepare images for contour detection, preprocess images for analysis), enabling morphological preprocessing workflows

## Connecting to Other Blocks

This block receives an image and produces a morphologically transformed image:

- **After image thresholding blocks** to clean up thresholded binary images (e.g., remove noise from thresholded images, fill holes in binary images, separate objects in thresholded images), enabling thresholding-to-morphology workflows
- **After preprocessing blocks** to apply morphological operations after other preprocessing (e.g., apply morphology after filtering, clean up after thresholding, process after enhancement), enabling multi-stage preprocessing workflows
- **Before contour detection blocks** to prepare images for contour detection (e.g., clean up images before contour detection, fill holes before finding contours, separate objects before contour analysis), enabling morphology-to-contour workflows
- **Before analysis blocks** that process binary or grayscale images (e.g., analyze cleaned images, process separated objects, work with morphologically processed images), enabling morphological analysis workflows
- **Before visualization blocks** to display morphologically transformed images (e.g., visualize cleaned images, display processed results, show transformation effects), enabling morphological visualization workflows
- **In image processing pipelines** where morphological operations are part of a larger processing chain (e.g., clean images in pipelines, apply morphology in multi-stage workflows, process images in transformation chains), enabling morphological processing pipeline workflows

## Requirements

This block works on single-channel (grayscale) images. Color images are automatically converted to grayscale before processing. Morphological operations work best on binary or high-contrast grayscale images where object boundaries are clearly defined. For optimal results, use thresholded binary images or high-contrast grayscale images as input.
"""


class MorphologicalTransformationManifest(WorkflowBlockManifest):
    type: Literal["roboflow_core/morphological_transformation@v1"]
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Morphological Transformation",
            "version": "v1",
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
        description="Input image to apply morphological transformation to. Color images are automatically converted to grayscale before processing (morphological operations work on single-channel images). Works best on binary or high-contrast grayscale images where object boundaries are clearly defined. Thresholded binary images produce optimal results. The morphologically transformed image will have modified object shapes, removed noise, filled holes, or extracted features depending on the selected operation. Original image metadata is preserved in the output.",
        examples=["$inputs.image", "$steps.cropping.crops"],
        validation_alias=AliasChoices("image", "images"),
    )

    kernel_size: Union[int, Selector(kind=[INTEGER_KIND])] = Field(
        default=5,
        description="Size of the square structuring element (kernel) used for morphological operations. Must be a positive integer. The kernel is a square matrix of ones with dimensions (kernel_size, kernel_size). Larger values create larger kernels that affect larger regions and produce more pronounced effects. Smaller values create smaller kernels that affect smaller regions with subtler effects. Typical values range from 3-15: smaller values (3-5) for fine operations and small objects, medium values (5-9) for general use, larger values (11-15) for coarse operations and large objects. Default is 5, which provides a good balance. Adjust based on object sizes and desired transformation scale.",
        examples=["5", "$inputs.kernel_size"],
    )

    operation: Union[
        Selector(kind=[STRING_KIND]),
        Literal[
            "Erosion",
            "Dilation",
            "Opening",
            "Closing",
            "Gradient",
            "Top Hat",
            "Black Hat",
        ],
    ] = Field(
        default="Closing",
        description="Type of morphological operation to apply: 'Erosion' shrinks objects by removing boundary pixels (separates objects, removes noise), 'Dilation' expands objects by adding boundary pixels (connects objects, fills holes), 'Opening' (erosion then dilation) removes noise and separates objects, 'Closing' (default, dilation then erosion) fills holes and connects objects, 'Gradient' (dilation minus erosion) finds object boundaries/edges, 'Top Hat' (input minus opening) detects small bright details, or 'Black Hat' (closing minus input) detects small dark details. Default is 'Closing' which is commonly used for hole filling. Choose based on goals: erosion/dilation for size modification, opening/closing for noise removal and gap filling, gradient for edge detection, top hat/black hat for feature extraction.",
        examples=["Closing", "$inputs.type"],
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name=OUTPUT_IMAGE_KEY,
                kind=[
                    IMAGE_KIND,
                ],
            ),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class MorphologicalTransformationBlockV1(WorkflowBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def get_manifest(cls) -> Type[MorphologicalTransformationManifest]:
        return MorphologicalTransformationManifest

    def run(
        self,
        image: WorkflowImageData,
        kernel_size: int = 5,
        operation: str = "Closing",
    ) -> BlockResult:
        # Apply morphological closing to the image
        updated_image = update_image(image.numpy_image, kernel_size, operation)
        # needs needs the channel dimension, which gets stripped by cv2.COLOR_BGR2GRAY
        updated_image = updated_image.reshape(updated_image.shape + (1,))

        output = WorkflowImageData.copy_and_replace(
            origin_image_data=image,
            numpy_image=updated_image,
        )
        return {OUTPUT_IMAGE_KEY: output}


def update_image(img: np.ndarray, kernel_size: int, operation: str):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    if operation == "Dilation":
        dilation = cv2.dilate(img, kernel, iterations=1)
        return dilation
    elif operation == "Erosion":
        erosion = cv2.erode(img, kernel, iterations=1)
        return erosion
    elif operation == "Opening":
        opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        return opening
    elif operation == "Closing":
        closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        return closing
    elif operation == "Gradient":
        gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
        return gradient
    elif operation == "Top Hat":
        tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
        return tophat
    elif operation == "Black Hat":
        blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
        return blackhat
    else:
        raise ValueError(
            f"Invalid operation: {operation}. Supported operations are 'Erosion', 'Dilation', 'Opening', 'Closing', 'Gradient', 'Top Hat', 'Black Hat'."
        )

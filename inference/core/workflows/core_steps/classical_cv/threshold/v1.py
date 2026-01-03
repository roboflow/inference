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

SHORT_DESCRIPTION: str = "Apply a threshold to an image."
LONG_DESCRIPTION = """
Convert grayscale images to binary images using configurable thresholding methods (binary, binary_inv, trunc, tozero, tozero_inv, adaptive_mean, adaptive_gaussian, otsu) to separate foreground from background, isolate objects, prepare images for morphological operations, and create binary masks for segmentation, object detection, and analysis workflows.

## How This Block Works

This block applies thresholding operations to convert grayscale images into binary images where pixels are classified as either foreground (white) or background (black). The block:

1. Receives a grayscale input image (color images must be converted to grayscale first using an Image Convert Grayscale block)
2. Determines the thresholding method from the threshold_type parameter
3. Applies the selected thresholding operation:

   **For Binary Threshold:**
   - Uses a fixed threshold value (thresh_value)
   - Pixels above the threshold become max_value (white), pixels below become 0 (black)
   - Creates a binary image with clear separation between foreground and background
   - Best for images with uniform lighting and clear contrast

   **For Binary Inverse Threshold:**
   - Uses a fixed threshold value (thresh_value)
   - Pixels above the threshold become 0 (black), pixels below become max_value (white)
   - Inverts the binary result - dark objects become white, light backgrounds become black
   - Useful when dark objects need to be foreground (white) in the output

   **For Truncate Threshold:**
   - Uses a fixed threshold value (thresh_value)
   - Pixels above the threshold are set to the threshold value, pixels below remain unchanged
   - Clips bright pixels while preserving dark pixel values
   - Useful for reducing brightness in overexposed regions

   **For To Zero Threshold:**
   - Uses a fixed threshold value (thresh_value)
   - Pixels below the threshold are set to 0 (black), pixels above remain unchanged
   - Removes dark pixels while preserving bright pixel values
   - Useful for removing noise in dark regions

   **For To Zero Inverse Threshold:**
   - Uses a fixed threshold value (thresh_value)
   - Pixels above the threshold are set to 0 (black), pixels below remain unchanged
   - Removes bright pixels while preserving dark pixel values
   - Useful for removing noise in bright regions

   **For Adaptive Mean Threshold:**
   - Calculates threshold values locally using mean of neighborhood pixels
   - Adapts to local image characteristics (block size 11x11, constant 2)
   - Handles varying lighting conditions and illumination gradients
   - Best for images with non-uniform lighting

   **For Adaptive Gaussian Threshold:**
   - Calculates threshold values locally using weighted Gaussian mean of neighborhood pixels
   - Adapts to local image characteristics (block size 11x11, constant 2)
   - Handles varying lighting conditions with smoother transitions than adaptive mean
   - Best for images with non-uniform lighting requiring smoother adaptation

   **For Otsu's Threshold:**
   - Automatically calculates optimal threshold value using Otsu's method
   - Analyzes histogram to find threshold that minimizes intra-class variance
   - No manual threshold value needed (thresh_value is ignored)
   - Best for bimodal histograms with clear foreground/background separation

4. For fixed threshold methods (binary, binary_inv, trunc, tozero, tozero_inv), uses thresh_value as the threshold and max_value (typically 255) as the maximum output value
5. For adaptive methods (adaptive_mean, adaptive_gaussian), uses max_value as the maximum output value and adapts locally
6. For Otsu's method, automatically determines the optimal threshold and uses max_value as the maximum output value
7. Preserves image structure and metadata
8. Returns the thresholded binary image

Thresholding converts grayscale images to binary images by classifying pixels based on intensity values. Fixed threshold methods use a single threshold value for the entire image - simple and fast but sensitive to lighting variations. Adaptive threshold methods calculate local thresholds for each pixel neighborhood - more robust to lighting variations but computationally more expensive. Otsu's method automatically selects an optimal global threshold by analyzing the image histogram - works well for images with bimodal intensity distributions. The threshold_type controls the method, thresh_value sets the threshold for fixed methods, and max_value (typically 255) determines the white pixel value in binary outputs.

## Common Use Cases

- **Binary Image Creation**: Convert grayscale images to binary images for object detection and analysis (e.g., create binary masks, isolate objects from background, separate foreground and background), enabling binary image creation workflows
- **Object Segmentation**: Isolate objects from backgrounds for segmentation tasks (e.g., segment objects from backgrounds, create object masks, isolate regions of interest), enabling object segmentation workflows
- **Document Processing**: Extract text and content from scanned documents (e.g., binarize document images, enhance text contrast, prepare documents for OCR), enabling document processing workflows
- **Image Preprocessing**: Prepare images for morphological operations and contour detection (e.g., create binary images for morphology, prepare images for contour analysis, binarize for shape analysis), enabling preprocessing workflows
- **Noise Removal Preparation**: Create binary images for noise removal and cleaning operations (e.g., prepare images for morphological cleaning, create masks for filtering, binarize for denoising), enabling noise removal workflows
- **Feature Detection**: Prepare images for feature detection and analysis (e.g., create binary images for edge detection, prepare for feature extraction, binarize for pattern recognition), enabling feature detection workflows

## Connecting to Other Blocks

This block receives a grayscale image and produces a thresholded binary image:

- **After Image Convert Grayscale blocks** to convert color images to grayscale before thresholding (e.g., convert color to grayscale then threshold, prepare color images for binarization, grayscale before binary conversion), enabling color-to-binary workflows
- **After preprocessing blocks** that output grayscale images (e.g., apply thresholding after filtering, binarize after enhancement, threshold preprocessed images), enabling preprocessing-to-threshold workflows
- **Before morphological transformation blocks** to prepare binary images for morphological operations (e.g., clean thresholded images with morphology, apply morphology to binary images, process binary masks), enabling threshold-to-morphology workflows
- **Before contour detection blocks** to prepare binary images for contour detection (e.g., find contours in thresholded images, detect shapes in binary images, analyze binary object boundaries), enabling threshold-to-contour workflows
- **Before analysis blocks** that process binary images (e.g., analyze binary masks, process thresholded regions, work with binary object data), enabling threshold analysis workflows
- **In image processing pipelines** where thresholding is part of a larger binary image processing chain (e.g., binarize images in pipelines, create masks in workflows, process binary images in chains), enabling threshold processing pipeline workflows

## Requirements

This block requires grayscale input images. Color images must be converted to grayscale first using an Image Convert Grayscale block. For optimal results, use images with good contrast between foreground and background. Fixed threshold methods work best with uniform lighting, while adaptive methods handle non-uniform lighting better. Otsu's method works best with bimodal histograms (clear foreground/background separation).
"""


class ImageThresholdManifest(WorkflowBlockManifest):
    type: Literal["roboflow_core/threshold@v1"]
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Image Threshold",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "classical_computer_vision",
            "ui_manifest": {
                "section": "classical_cv",
                "icon": "far fa-circle-half-stroke",
                "blockPriority": 6,
                "opencv": True,
            },
        }
    )

    image: Selector(kind=[IMAGE_KIND]) = Field(
        title="Input Image",
        description="Input grayscale image to apply thresholding to. Must be a single-channel grayscale image (color images must be converted to grayscale first using an Image Convert Grayscale block). Thresholding converts grayscale images to binary images where pixels are classified as foreground (white) or background (black). The thresholded binary image will have pixels set to either 0 (black) or max_value (typically 255, white) based on the selected thresholding method. Original image metadata is preserved in the output. For optimal results, use images with good contrast between foreground and background.",
        examples=["$inputs.image", "$steps.cropping.crops"],
        validation_alias=AliasChoices("image", "images"),
    )

    threshold_type: Union[
        Selector(kind=[STRING_KIND]),
        Literal[
            "binary",
            "binary_inv",
            "trunc",
            "tozero",
            "tozero_inv",
            "adaptive_mean",
            "adaptive_gaussian",
            "otsu",
        ],
    ] = Field(
        description="Type of thresholding operation to apply: 'binary' (default, pixels above threshold become white, below become black) for uniform lighting, 'binary_inv' (inverse binary, pixels above threshold become black, below become white) for dark object isolation, 'trunc' (truncate, pixels above threshold set to threshold value) for brightness clipping, 'tozero' (to zero, pixels below threshold set to zero) for dark region removal, 'tozero_inv' (to zero inverse, pixels above threshold set to zero) for bright region removal, 'adaptive_mean' (adaptive mean, local threshold using mean of neighborhood) for non-uniform lighting, 'adaptive_gaussian' (adaptive Gaussian, local threshold using weighted Gaussian mean) for non-uniform lighting with smoother transitions, or 'otsu' (Otsu's method, automatic optimal threshold calculation) for bimodal histograms. Default is 'binary'. Fixed methods (binary, binary_inv, trunc, tozero, tozero_inv) use thresh_value, adaptive methods (adaptive_mean, adaptive_gaussian) adapt locally, and Otsu's method automatically calculates the optimal threshold.",
        default="binary",
        examples=["binary", "$inputs.threshold_type"],
    )

    thresh_value: Union[Selector(kind=[INTEGER_KIND]), int] = Field(
        description="Threshold value used for fixed threshold methods (binary, binary_inv, trunc, tozero, tozero_inv). Must be an integer between 0 and 255. Pixels above this value are treated as foreground for binary/binary_inv, or clipped/preserved for trunc/tozero/tozero_inv depending on the operation. Typical values range from 100-200: lower values (100-127) for darker images or to preserve more dark regions, medium values (127-150) for balanced separation, higher values (150-200) for brighter images or to preserve more bright regions. Default is 127 (middle gray). This parameter is ignored for adaptive methods (adaptive_mean, adaptive_gaussian) and Otsu's method (otsu) which calculate thresholds automatically. Adjust based on image brightness and desired foreground/background separation.",
        examples=[127, "$inputs.thresh_value"],
    )

    max_value: Union[Selector(kind=[INTEGER_KIND]), int] = Field(
        description="Maximum value used in thresholded binary outputs. Must be an integer, typically 255 (white pixel value in 8-bit images). This value is assigned to pixels classified as foreground in binary operations (binary, binary_inv) or used as the maximum value in adaptive thresholding (adaptive_mean, adaptive_gaussian, otsu). For truncate operations (trunc), this parameter is used but the actual output values are clipped to thresh_value. Default is 255, which creates standard binary images with white (255) foreground and black (0) background. For 8-bit grayscale images, keep at 255. For 16-bit images, use 65535. Controls the brightness/intensity of foreground pixels in the thresholded output.",
        default=255,
        examples=[255, "$inputs.max_value"],
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


class ImageThresholdBlockV1(WorkflowBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def get_manifest(cls) -> Type[ImageThresholdManifest]:
        return ImageThresholdManifest

    def run(
        self,
        image: WorkflowImageData,
        threshold_type: str,
        thresh_value: int,
        max_value: int,
        *args,
        **kwargs,
    ) -> BlockResult:
        # Apply threshold to the image
        thresholded_image = apply_thresholding(
            image.numpy_image, threshold_type, thresh_value, max_value
        )
        output = WorkflowImageData.copy_and_replace(
            origin_image_data=image,
            numpy_image=thresholded_image,
        )
        return {OUTPUT_IMAGE_KEY: output}


def apply_thresholding(
    image: np.ndarray, threshold_type: str, thresh_value: int, max_value: int
) -> np.ndarray:
    """
    Applies the specified thresholding to the image.

    Args:
        image (np.ndarray): Input image in grayscale.
        threshold_type (str): Type of thresholding ('binary', 'binary_inv', 'trunc', 'tozero', 'tozero_inv', 'adaptive_mean', 'adaptive_gaussian', 'otsu').
        thresh_value (int, optional): Threshold value.
        max_value (int, optional): Maximum value to use with the THRESH_BINARY and THRESH_BINARY_INV thresholding types.

    Returns:
        np.ndarray: Image with thresholding applied.
    """
    if threshold_type == "binary":
        _, thresh_image = cv2.threshold(
            image, thresh_value, max_value, cv2.THRESH_BINARY
        )
    elif threshold_type == "binary_inv":
        _, thresh_image = cv2.threshold(
            image, thresh_value, max_value, cv2.THRESH_BINARY_INV
        )
    elif threshold_type == "trunc":
        _, thresh_image = cv2.threshold(
            image, thresh_value, max_value, cv2.THRESH_TRUNC
        )
    elif threshold_type == "tozero":
        _, thresh_image = cv2.threshold(
            image, thresh_value, max_value, cv2.THRESH_TOZERO
        )
    elif threshold_type == "tozero_inv":
        _, thresh_image = cv2.threshold(
            image, thresh_value, max_value, cv2.THRESH_TOZERO_INV
        )
    elif threshold_type == "adaptive_mean":
        thresh_image = cv2.adaptiveThreshold(
            image, max_value, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2
        )
    elif threshold_type == "adaptive_gaussian":
        thresh_image = cv2.adaptiveThreshold(
            image,
            max_value,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2,
        )
    elif threshold_type == "otsu":
        _, thresh_image = cv2.threshold(
            image, 0, max_value, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
    else:
        raise ValueError(f"Unknown threshold type: {threshold_type}")

    return thresh_image

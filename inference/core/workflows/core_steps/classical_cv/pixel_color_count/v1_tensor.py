from typing import List, Literal, Optional, Tuple, Type, Union

import cv2
import numpy as np
from pydantic import AliasChoices, ConfigDict, Field

from inference.core.workflows.execution_engine.entities.base import (
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    IMAGE_KIND,
    INTEGER_KIND,
    RGB_COLOR_KIND,
    STRING_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

SHORT_DESCRIPTION: str = (
    "Count the number of pixels that match a specific color within a given tolerance."
)
LONG_DESCRIPTION = """
Count pixels in an image that match a target color within a specified tolerance range, using color range masking to identify matching pixels and returning the total count of pixels within the color tolerance range for color analysis, quality control, color-based measurements, and pixel-level color quantification workflows.

## How This Block Works

This block counts how many pixels in an image match a specific target color within a tolerance range, providing pixel-level color quantification. The block:

1. Receives an input image and a target color specification (hex string, RGB tuple string, or RGB tuple)
2. Converts the target color to BGR (Blue-Green-Red) format for OpenCV processing:
   - Parses hex color strings (e.g., "#431112" or "#412" shorthand)
   - Parses RGB tuple strings (e.g., "(128, 32, 64)")
   - Handles RGB tuples directly (e.g., (18, 17, 67))
   - Converts RGB to BGR format (reverses color channel order) since OpenCV uses BGR
3. Calculates color tolerance bounds:
   - Creates a lower bound by subtracting tolerance from each BGR channel of the target color
   - Creates an upper bound by adding tolerance to each BGR channel of the target color
   - Clips bounds to valid 0-255 range for each channel
   - Defines a 3D color cube in BGR space where matching pixels must fall
4. Creates a binary mask using OpenCV's inRange function:
   - Compares each pixel's BGR values against the lower and upper bounds
   - Sets mask pixel to 255 (white) if pixel color falls within the tolerance range
   - Sets mask pixel to 0 (black) if pixel color falls outside the tolerance range
   - Uses vectorized operations for efficient pixel-level comparison across the entire image
5. Counts matching pixels:
   - Counts non-zero pixels in the mask (pixels with value 255, representing matches)
   - Returns the total count of pixels that match the target color within tolerance

The block performs pixel-level color matching using a tolerance-based approach, allowing for slight color variations due to compression, lighting, or image processing. The tolerance creates a range around the target color - a tolerance of 10 means pixels can differ by up to ±10 in each BGR channel (for a total range of 21 values per channel). Lower tolerance values (e.g., 5-10) require very close color matches, while higher tolerance values (e.g., 20-30) allow more color variation. This is useful for counting pixels of a specific color when exact matches may not exist due to image artifacts or processing.

## Common Use Cases

- **Color Area Measurement**: Measure the area or coverage of specific colors in images (e.g., measure coverage of specific colors in images, quantify color distribution, assess color proportions), enabling color area quantification workflows
- **Quality Control and Inspection**: Count pixels of expected colors for quality control (e.g., verify color consistency in products, detect color defects, validate expected colors in images), enabling color-based quality control workflows
- **Color-Based Analysis**: Analyze images based on specific color presence or quantity (e.g., analyze color distribution in images, quantify color usage, measure color characteristics), enabling color quantification analysis workflows
- **Image Processing Validation**: Validate image processing results by counting expected colors (e.g., verify color transformations, validate color corrections, check color filtering results), enabling color validation workflows
- **Feature Detection and Measurement**: Detect and measure features based on color characteristics (e.g., count pixels in colored regions, measure color-based features, quantify color-defined areas), enabling color-based feature measurement workflows
- **Threshold-Based Color Detection**: Use pixel counting for threshold-based color detection (e.g., detect if enough pixels match a color, determine color presence thresholds, implement color-based triggers), enabling threshold-based color detection workflows

## Connecting to Other Blocks

This block receives an image and target color, and produces a pixel count:

- **After image input blocks** to count pixels of specific colors in input images (e.g., count color pixels in camera feeds, analyze colors in image inputs, quantify colors in images), enabling color pixel counting workflows
- **After crop blocks** to count pixels in specific image regions (e.g., count color pixels in cropped regions, analyze colors in specific areas, quantify colors in selected regions), enabling region-based color pixel counting
- **After preprocessing blocks** to count pixels after image processing (e.g., count colors after filtering, analyze colors after enhancement, quantify colors after transformations), enabling processed image color counting workflows
- **Before filtering or logic blocks** that use pixel counts for decision-making (e.g., filter based on pixel counts, make decisions based on color quantities, apply logic based on pixel counts), enabling count-based conditional workflows
- **Before data storage blocks** to store pixel count information (e.g., store color pixel counts with images, save color analysis results, record color quantification data), enabling color count metadata storage workflows
- **In quality control workflows** where pixel counting validates color characteristics (e.g., verify color quantities in quality control, validate color coverage, check color consistency), enabling color-based quality control workflows
"""


class ColorPixelCountManifest(WorkflowBlockManifest):
    type: Literal["roboflow_core/pixel_color_count@v1"]
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Pixel Color Count",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "classical_computer_vision",
            "ui_manifest": {
                "section": "classical_cv",
                "icon": "far fa-tally",
                "blockPriority": 2,
                "opencv": True,
            },
        }
    )
    image: Selector(kind=[IMAGE_KIND]) = Field(
        title="Input Image",
        description="Input image to analyze for pixel color counting. The block counts pixels in this image that match the target_color within the specified tolerance. All pixels in the image are analyzed. The image is processed in BGR format (OpenCV standard), and color matching is performed on each pixel's BGR values. Processing time depends on image size.",
        examples=["$inputs.image", "$steps.cropping.crops"],
        validation_alias=AliasChoices("image", "images"),
    )
    target_color: Union[
        Selector(kind=[STRING_KIND]),
        Selector(kind=[RGB_COLOR_KIND]),
        str,
        Tuple[int, int, int],
    ] = Field(
        description="Target color to count in the image. Can be specified in multiple formats: (1) Hex string format: '#RRGGBB' (6-digit, e.g., '#431112') or '#RGB' (3-digit shorthand, e.g., '#412'), (2) RGB tuple string format: '(R, G, B)' (e.g., '(128, 32, 64)'), or (3) RGB tuple: (R, G, B) tuple of integers (e.g., (18, 17, 67)). Values should be in RGB color space (0-255 per channel). The color is automatically converted to BGR format for OpenCV processing. Use this to specify the exact color you want to count pixels for.",
        examples=["#431112", "$inputs.target_color", (18, 17, 67)],
    )
    tolerance: Union[Selector(kind=[INTEGER_KIND]), int] = Field(
        default=10,
        description="Color matching tolerance value (0-255). Determines how much each BGR channel can vary from the target color and still be considered a match. The tolerance is applied to each color channel independently - a tolerance of 10 creates a range of ±10 for each BGR channel (total range of 21 values per channel). Lower values (e.g., 5-10) require very close color matches and are more precise but may miss slightly different shades. Higher values (e.g., 20-30) allow more color variation and match a wider range of similar colors but may include unintended colors. Default is 10, which provides a good balance. Adjust based on image quality, compression artifacts, and how strict you need the color matching to be.",
        examples=[10, "$inputs.tolerance"],
    )

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name="matching_pixels_count",
                kind=[INTEGER_KIND],
            ),
        ]


class PixelationCountBlockV1(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[ColorPixelCountManifest]:
        return ColorPixelCountManifest

    def run(
        self,
        image: WorkflowImageData,
        target_color: Union[str, tuple],
        tolerance: int,
    ) -> BlockResult:
        color_pixel_count = count_specific_color_pixels(
            image.numpy_image, target_color, tolerance
        )
        return {"matching_pixels_count": color_pixel_count}


def count_specific_color_pixels(
    image: np.ndarray,
    target_color: Union[str, Tuple[int, int, int]],
    tolerance: int,
) -> int:
    """
    Counts the number of pixels that match the target color within the given tolerance.

    Args:
        image: Input image.
        target_color (Union[str, tuple]): Target color in hex format (e.g., '#431112') or BGR tuple (e.g., (18, 17, 67)).
        tolerance (int, optional): Tolerance for color matching. Defaults to 10.

    Returns:
        int: Number of pixels that match the target color.
    """
    target_color_bgr = convert_color_to_bgr_tuple(color=target_color)
    lower_bound = np.array(target_color_bgr) - tolerance
    upper_bound = np.array(target_color_bgr) + tolerance

    # Use vectorized comparison to directly create a mask and count non-zero elements
    mask = cv2.inRange(image, lower_bound, upper_bound)

    return int(cv2.countNonZero(mask))


def convert_color_to_bgr_tuple(
    color: Union[str, Tuple[int, int, int]],
) -> Tuple[int, int, int]:
    if isinstance(color, str):
        return convert_string_color_to_bgr_tuple(color=color)
    if isinstance(color, tuple) and len(color) == 3:
        return color[::-1]
    raise ValueError(f"Invalid color format: {color}")


def convert_string_color_to_bgr_tuple(color: str) -> Tuple[int, int, int]:
    # Check if color is in hex format
    if color.startswith("#"):
        try:
            if len(color) == 7:
                return (int(color[5:7], 16), int(color[3:5], 16), int(color[1:3], 16))
            elif len(color) == 4:
                return (
                    int(color[3] * 2, 16),
                    int(color[2] * 2, 16),
                    int(color[1] * 2, 16),
                )
        except ValueError as e:
            raise ValueError(f"Invalid hex color format: {color}") from e

    # Check if color is in tuple format
    elif color.startswith("(") and color.endswith(")"):
        try:
            return tuple(map(int, color[1:-1].split(",")))[::-1]
        except ValueError as e:
            raise ValueError(f"Invalid tuple color format: {color}") from e

    raise ValueError(f"Invalid color format: {color}")

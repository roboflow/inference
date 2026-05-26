from typing import List, Literal, Optional, Tuple, Type, Union

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
    FLOAT_KIND,
    IMAGE_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

SHORT_DESCRIPTION: str = "Calculate a score to indicate how well-focused a camera is."
LONG_DESCRIPTION = """
Calculate focus quality scores using the Brenner function measure to assess image sharpness, detect blur, evaluate camera focus performance, enable auto-focus systems, perform image quality assessment, and determine optimal focus settings for camera calibration and image capture workflows.

## How This Block Works

This block calculates the Brenner focus measure, which quantifies image sharpness by measuring texture detail at fine scales. The block:

1. Receives an input image (color or grayscale, automatically converts color to grayscale)
2. Converts the image to grayscale if it's in color format (Brenner measure works on single-channel images)
3. Converts the grayscale image to 16-bit integer format for precise calculations
4. Calculates horizontal and vertical intensity differences:
   - Computes horizontal differences by comparing pixels 2 positions apart horizontally: `pixel[x+2] - pixel[x]`
   - Computes vertical differences by comparing pixels 2 positions apart vertically: `pixel[y+2] - pixel[y]`
   - Uses clipping to keep only positive differences (highlights sharp edges and details)
   - Measures rapid intensity changes that indicate fine-scale texture and sharp edges
5. Calculates the focus measure matrix:
   - Takes the maximum of horizontal and vertical differences at each pixel location
   - Squares the differences to emphasize larger variations (stronger response to sharp edges)
   - Creates a matrix where higher values indicate sharper, more focused regions
6. Normalizes and converts to visualization format:
   - Normalizes the focus measure matrix to 0-255 range for display
   - Converts to 8-bit format for visualization
   - Creates a visual representation showing focus quality across the image
7. Overlays the Brenner value text on the image:
   - Displays the mean focus measure value on the top-left of the image
   - Shows focus quality as a numerical score for easy assessment
8. Preserves image structure and metadata
9. Returns the visualization image and the mean focus measure value as a float

The Brenner focus measure quantifies image sharpness by analyzing fine-scale texture and edge detail. In-focus images contain many sharp edges and fine texture details, resulting in large intensity differences between nearby pixels and high Brenner scores. Out-of-focus images have blurred edges and lack fine detail, resulting in small intensity differences and low Brenner scores. The measure uses a 2-pixel spacing to detect fine-scale texture while being robust to noise. Higher Brenner values indicate better focus, with typical ranges varying based on image content and resolution. The visualization shows focus quality distribution across the image, helping identify well-focused and blurred regions.

## Common Use Cases

- **Auto-Focus Systems**: Assess focus quality to enable automatic camera focus adjustment (e.g., evaluate focus during auto-focus operations, detect optimal focus position, trigger focus adjustments based on Brenner scores), enabling auto-focus workflows
- **Image Quality Assessment**: Evaluate image sharpness and detect blurry images for quality control (e.g., assess image quality in capture pipelines, detect out-of-focus images, filter low-quality images), enabling quality assessment workflows
- **Camera Calibration**: Evaluate focus performance during camera setup and calibration (e.g., assess focus during camera calibration, optimize focus settings, evaluate camera performance), enabling camera calibration workflows
- **Blur Detection**: Detect blurry images in image processing pipelines (e.g., identify blurry images for rejection, detect focus issues, assess image sharpness), enabling blur detection workflows
- **Focus Optimization**: Determine optimal focus settings for image capture systems (e.g., find best focus position, optimize focus parameters, evaluate focus across settings), enabling focus optimization workflows
- **Image Analysis**: Assess image sharpness as part of image analysis workflows (e.g., evaluate image quality before processing, assess focus for analysis tasks, measure image sharpness metrics), enabling focus analysis workflows

## Connecting to Other Blocks

This block receives an image and produces a focus measure visualization image and a focus_measure float value:

- **After image capture or preprocessing blocks** to assess focus quality of captured or processed images (e.g., evaluate focus after image capture, assess sharpness after preprocessing, measure focus in image pipelines), enabling focus assessment workflows
- **Before logic blocks** like Continue If to make decisions based on focus quality (e.g., continue if focus is good, filter images based on focus scores, make decisions using focus measures), enabling focus-based decision workflows
- **Before analysis blocks** to assess image quality before analysis (e.g., evaluate focus before analysis, assess sharpness for processing, measure quality before analysis), enabling quality-based analysis workflows
- **In auto-focus systems** where focus measurement is part of a feedback loop (e.g., measure focus for auto-focus, assess focus in feedback systems, evaluate focus in control loops), enabling auto-focus system workflows
- **Before visualization blocks** to display focus quality information (e.g., visualize focus scores, display focus measures, show focus quality), enabling focus visualization workflows
- **In image quality control pipelines** where focus assessment is part of quality checks (e.g., assess focus in quality pipelines, evaluate sharpness in QC workflows, measure focus for quality control), enabling quality control workflows

## Requirements

This block works on color or grayscale input images. Color images are automatically converted to grayscale before processing (Brenner measure works on single-channel images). The block outputs both a visualization image (with focus measure displayed) and a numerical focus_measure value. Higher Brenner values indicate better focus and sharper images, while lower values indicate blur and poor focus. The focus measure is sensitive to image content and resolution, so threshold values for "good" focus should be calibrated based on specific use cases and image characteristics.
"""


class CameraFocusManifest(WorkflowBlockManifest):
    type: Literal["roboflow_core/camera_focus@v1"]
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Camera Focus",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "classical_computer_vision",
            "ui_manifest": {
                "section": "classical_cv",
                "icon": "far fa-aperture",
                "blockPriority": 8,
                "opencv": True,
            },
        }
    )

    image: Selector(kind=[IMAGE_KIND]) = Field(
        title="Input Image",
        description="Input image (color or grayscale) to calculate focus quality for. Color images are automatically converted to grayscale before processing (Brenner focus measure works on single-channel images). The block calculates the Brenner function score which measures fine-scale texture and edge detail to assess image sharpness. The output includes both a visualization image (with focus measure value displayed) and a numerical focus_measure float value. Higher Brenner values indicate better focus and sharper images (more fine-scale texture and sharp edges), while lower values indicate blur and poor focus (lacking fine detail). The focus measure uses intensity differences between pixels 2 positions apart to detect fine-scale texture. Original image metadata is preserved. Use this block to assess focus quality, detect blur, enable auto-focus systems, or perform image quality assessment.",
        examples=["$inputs.image", "$steps.cropping.crops"],
        validation_alias=AliasChoices("image", "images"),
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name=OUTPUT_IMAGE_KEY,
                kind=[IMAGE_KIND],
            ),
            OutputDefinition(
                name="focus_measure",
                kind=[
                    FLOAT_KIND,
                ],
            ),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class CameraFocusBlockV1(WorkflowBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def get_manifest(cls) -> Type[CameraFocusManifest]:
        return CameraFocusManifest

    def run(self, image: WorkflowImageData, *args, **kwargs) -> BlockResult:
        # Calculate the Brenner measure
        brenner_image, brenner_value = calculate_brenner_measure(image.numpy_image)
        output = WorkflowImageData.copy_and_replace(
            origin_image_data=image,
            numpy_image=brenner_image,
        )
        return {
            OUTPUT_IMAGE_KEY: output,
            "focus_measure": brenner_value,
        }


def calculate_brenner_measure(
    input_image: np.ndarray,
    text_color: Tuple[int, int, int] = (255, 255, 255),
    text_thickness: int = 2,
) -> Tuple[np.ndarray, float]:
    """
    Brenner's focus measure.

    Parameters
    ----------
    input_image : np.ndarray
        The input image in grayscale.
    text_color : Tuple[int, int, int], optional
        The color of the text displaying the Brenner value, in BGR format. Default is white (255, 255, 255).
    text_thickness : int, optional
        The thickness of the text displaying the Brenner value. Default is 2.

    Returns
    -------
    Tuple[np.ndarray, float]
        The Brenner image and the Brenner value.
    """
    # Convert image to grayscale if it has 3 channels
    if len(input_image.shape) == 3:
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    # Convert image to 16-bit integer format
    converted_image = input_image.astype(np.int16)

    # Get the dimensions of the image
    height, width = converted_image.shape

    # Initialize two matrices for horizontal and vertical focus measures
    horizontal_diff = np.zeros((height, width))
    vertical_diff = np.zeros((height, width))

    # Calculate horizontal and vertical focus measures
    horizontal_diff[:, : width - 2] = np.clip(
        converted_image[:, 2:] - converted_image[:, :-2], 0, None
    )
    vertical_diff[: height - 2, :] = np.clip(
        converted_image[2:, :] - converted_image[:-2, :], 0, None
    )

    # Calculate final focus measure
    focus_measure = np.max((horizontal_diff, vertical_diff), axis=0) ** 2

    # Convert focus measure matrix to 8-bit for visualization
    focus_measure_image = ((focus_measure / focus_measure.max()) * 255).astype(np.uint8)

    # Display the Brenner value on the top left of the image
    cv2.putText(
        focus_measure_image,
        f"Focus value: {focus_measure.mean():.2f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        text_color,
        text_thickness,
    )

    return focus_measure_image, focus_measure.mean()

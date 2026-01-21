from typing import List, Literal, Optional, Type, Union

import cv2
import numpy as np
from pydantic import AliasChoices, ConfigDict, Field, PositiveInt

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

SHORT_DESCRIPTION = "Resize, flip, or rotate an image."
LONG_DESCRIPTION = """
Apply geometric transformations to images including resizing to specified dimensions (with aspect ratio preservation), rotating by specified degrees (clockwise or counterclockwise), or flipping vertically, horizontally, or both, providing flexible image preprocessing for model input preparation, image orientation correction, and geometric image manipulation workflows.

## How This Block Works

This block applies one geometric transformation operation (resize, rotate, or flip) to an input image based on the selected task_type. The block:

1. Receives an input image and selects one transformation task (resize, rotate, or flip)
2. Validates task-specific parameters (width/height for resize, rotation_degrees for rotate, flip_type for flip)
3. Applies the selected transformation:

   **For resize task:**
   - Validates width and height are positive integers (greater than 0)
   - Supports aspect ratio preservation: if only width or only height is provided, calculates the missing dimension to maintain the original aspect ratio
   - If both width and height are provided, resizes to exact dimensions (may distort aspect ratio)
   - Uses OpenCV's INTER_AREA interpolation for high-quality downsampling
   - Returns resized image with specified dimensions

   **For rotate task:**
   - Validates rotation_degrees is between -360 and 360 degrees
   - Positive values rotate clockwise, negative values rotate counterclockwise
   - Calculates rotation matrix around image center
   - Automatically adjusts canvas size to contain the rotated image (no cropping)
   - Uses OpenCV's warpAffine for smooth rotation with bilinear interpolation
   - Returns rotated image with canvas sized to fit the full rotated image

   **For flip task:**
   - Validates flip_type is "vertical", "horizontal", or "both"
   - Vertical flip: flips image upside down (mirrors along horizontal axis)
   - Horizontal flip: flips image left-right (mirrors along vertical axis)
   - Both: applies both vertical and horizontal flips simultaneously (180-degree rotation equivalent)
   - Uses OpenCV's flip function for efficient mirroring
   - Returns flipped image with same dimensions as input

4. Preserves image metadata from the original image (parent metadata, image properties)
5. Returns the transformed image maintaining original image metadata structure

The block performs one transformation at a time - select resize, rotate, or flip via task_type. Each transformation is applied independently and produces a clean output. Resize supports flexible aspect ratio handling, rotation automatically adjusts canvas size to prevent cropping, and flip operations provide efficient mirroring along different axes. The transformations use OpenCV for efficient, high-quality geometric image manipulation.

## Common Use Cases

- **Model Input Preparation**: Resize images to match model input requirements (e.g., resize images to specific dimensions for object detection models, adjust image sizes for classification model inputs, normalize image dimensions for consistent model processing), enabling proper model input formatting
- **Image Orientation Correction**: Rotate images to correct orientation issues (e.g., rotate images captured in wrong orientation, correct camera rotation, adjust image orientation for proper display), enabling image orientation workflows
- **Data Augmentation**: Apply geometric transformations for data augmentation (e.g., flip images horizontally for augmentation, rotate images for training data variety, apply transformations to increase dataset diversity), enabling data augmentation workflows
- **Image Display Preparation**: Transform images for display or presentation purposes (e.g., flip images for mirror effects, resize images for display dimensions, rotate images for correct viewing orientation), enabling image presentation workflows
- **Workflow Image Standardization**: Standardize image dimensions or orientation across workflow inputs (e.g., resize all images to consistent dimensions, normalize image orientations, prepare images for uniform processing), enabling image standardization workflows
- **Image Formatting for Downstream Blocks**: Prepare images for blocks that require specific dimensions or orientations (e.g., resize before detection models, rotate for proper processing, flip for compatibility with other blocks), enabling image preparation workflows

## Connecting to Other Blocks

This block receives an image and produces a transformed image:

- **After image input blocks** to preprocess images before further processing (e.g., resize input images, correct image orientation, prepare images for workflow processing), enabling image preprocessing workflows
- **Before detection or classification models** to format images for model requirements (e.g., resize to model input dimensions, adjust orientation for proper detection, prepare images for model processing), enabling model-compatible image preparation
- **Before crop blocks** to prepare images before cropping (e.g., resize before cropping, rotate before region extraction, adjust orientation before cropping), enabling pre-crop image preparation
- **Before visualization blocks** to prepare images for display (e.g., resize for display, rotate for proper viewing, flip for presentation), enabling image display preparation workflows
- **In image processing pipelines** where geometric transformations are needed (e.g., resize in multi-stage pipelines, rotate in processing workflows, flip in transformation chains), enabling geometric transformation pipelines
- **After other transformation blocks** to apply additional geometric operations (e.g., resize after cropping, rotate after other transformations, flip after processing), enabling multi-stage geometric transformation workflows
"""


class ImagePreprocessingManifest(WorkflowBlockManifest):
    type: Literal["roboflow_core/image_preprocessing@v1"]
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Image Preprocessing",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "classical_computer_vision",
            "search_keywords": [
                "resize",
                "rotate",
                "flip",
                "scale",
                "transform",
                "mirror",
            ],
            "ui_manifest": {
                "section": "classical_cv",
                "icon": "far fa-image",
                "blockPriority": 0.1,
                "opencv": True,
            },
        }
    )
    image: Selector(kind=[IMAGE_KIND]) = Field(
        title="Input Image",
        description="Input image to transform. The image will have one geometric transformation applied (resize, rotate, or flip) based on the selected task_type. Supports images from inputs, previous workflow steps, or crop outputs. The output image maintains the original image's metadata structure.",
        examples=["$inputs.image", "$steps.cropping.crops"],
        validation_alias=AliasChoices("image", "images"),
    )
    task_type: Literal["resize", "rotate", "flip"] = Field(
        description="Type of geometric transformation to apply to the image: 'resize' to change image dimensions (requires width/height), 'rotate' to rotate the image by specified degrees (requires rotation_degrees), or 'flip' to mirror the image along axes (requires flip_type). Only one transformation is applied per block execution. Select the appropriate task type based on your preprocessing needs.",
    )
    width: Union[PositiveInt, Selector(kind=[INTEGER_KIND])] = Field(  # type: ignore
        title="Width",
        default=640,
        description="Target width in pixels for resizing. Required when task_type is 'resize'. Must be a positive integer (greater than 0). If only width is provided (height is None), the height is automatically calculated to preserve aspect ratio. If both width and height are provided, the image is resized to exact dimensions (may distort aspect ratio). Default is 640 pixels. Use this to resize images to specific dimensions for model inputs or display requirements.",
        examples=[640, "$inputs.resize_width"],
        json_schema_extra={
            "relevant_for": {
                "task_type": {
                    "values": ["resize"],
                    "required": True,
                },
            },
        },
    )
    height: Union[PositiveInt, Selector(kind=[INTEGER_KIND])] = Field(  # type: ignore
        title="Height",
        default=640,
        description="Target height in pixels for resizing. Required when task_type is 'resize'. Must be a positive integer (greater than 0). If only height is provided (width is None), the width is automatically calculated to preserve aspect ratio. If both width and height are provided, the image is resized to exact dimensions (may distort aspect ratio). Default is 640 pixels. Use this to resize images to specific dimensions for model inputs or display requirements.",
        examples=[640, "$inputs.resize_height"],
        json_schema_extra={
            "relevant_for": {
                "task_type": {
                    "values": ["resize"],
                    "required": True,
                },
            },
        },
    )
    rotation_degrees: Union[int, Selector(kind=[INTEGER_KIND])] = Field(  # type: ignore
        title="Degrees of Rotation",
        description="Rotation angle in degrees. Required when task_type is 'rotate'. Must be between -360 and 360 degrees. Positive values rotate the image clockwise, negative values rotate counterclockwise. The rotation is performed around the image center, and the canvas size is automatically adjusted to contain the full rotated image (no cropping occurs). For example, 90 rotates 90 degrees clockwise, -90 rotates 90 degrees counterclockwise, 180 rotates 180 degrees. Default is 90 degrees.",
        default=90,
        examples=[90, "$inputs.rotation_degrees"],
        json_schema_extra={
            "relevant_for": {
                "task_type": {
                    "values": ["rotate"],
                    "required": True,
                },
            }
        },
    )
    flip_type: Union[
        Selector(kind=[STRING_KIND]), Literal["vertical", "horizontal", "both"]
    ] = Field(  # type: ignore
        title="Flip Type",
        description="Type of flip operation to apply. Required when task_type is 'flip'. Options: 'vertical' flips the image upside down (mirrors along horizontal axis, top becomes bottom), 'horizontal' flips left-right (mirrors along vertical axis, left becomes right), 'both' applies both vertical and horizontal flips simultaneously (equivalent to 180-degree rotation). The image dimensions remain unchanged after flipping. Default is 'vertical'. Use this for mirroring images or data augmentation.",
        default="vertical",
        examples=["vertical", "horizontal", "$inputs.flip_type"],
        json_schema_extra={
            "relevant_for": {
                "task_type": {
                    "values": ["flip"],
                    "required": True,
                },
            }
        },
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="image", kind=[IMAGE_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class ImagePreprocessingBlockV1(WorkflowBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def get_manifest(cls) -> Type[ImagePreprocessingManifest]:
        return ImagePreprocessingManifest

    def run(
        self,
        image: WorkflowImageData,
        task_type: str,
        width: Optional[int],
        height: Optional[int],
        rotation_degrees: Optional[int],
        flip_type: Optional[str],
        *args,
        **kwargs,
    ) -> BlockResult:

        response_image = None

        if task_type == "resize":
            if width is not None and width <= 0:
                raise ValueError("Width must be greater than 0")
            if height is not None and height <= 0:
                raise ValueError("Height must be greater than 0")
            response_image = apply_resize_image(image.numpy_image, width, height)
        elif task_type == "rotate":
            if rotation_degrees is not None and not (-360 <= rotation_degrees <= 360):
                raise ValueError("Rotation degrees must be between -360 and 360")
            response_image = apply_rotate_image(image.numpy_image, rotation_degrees)
        elif task_type == "flip":
            if flip_type is not None and flip_type not in [
                "vertical",
                "horizontal",
                "both",
            ]:
                raise ValueError(
                    "Flip type must be 'vertical', 'horizontal', or 'both'"
                )
            response_image = apply_flip_image(image.numpy_image, flip_type)
        else:
            raise ValueError(f"Invalid task type: {task_type}")

        output_image = WorkflowImageData.copy_and_replace(
            origin_image_data=image, numpy_image=response_image
        )
        return {"image": output_image}


def apply_resize_image(
    np_image: np.ndarray, width: Optional[int], height: Optional[int]
) -> np.ndarray:
    if width is None and height is None:
        return np_image.copy()

    current_height, current_width = np_image.shape[:2]

    if width is None:
        aspect_ratio = height / current_height
        width = int(current_width * aspect_ratio)
    elif height is None:
        aspect_ratio = width / current_width
        height = int(current_height * aspect_ratio)

    resized_image = cv2.resize(np_image, (width, height), interpolation=cv2.INTER_AREA)
    return resized_image


def apply_rotate_image(np_image: np.ndarray, rotation_degrees: Optional[int]):
    if rotation_degrees is None or rotation_degrees == 0:
        return np_image.copy()
    existing_height, existing_width = np_image.shape[:2]
    center = (existing_width // 2, existing_height // 2)  # Corrected order
    rotation_matrix = cv2.getRotationMatrix2D(center, rotation_degrees, 1.0)

    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])
    new_width = int((existing_height * sin) + (existing_width * cos))
    new_height = int((existing_height * cos) + (existing_width * sin))

    rotation_matrix[0, 2] += (new_width / 2) - center[0]
    rotation_matrix[1, 2] += (new_height / 2) - center[1]

    rotated_image = cv2.warpAffine(np_image, rotation_matrix, (new_width, new_height))
    return rotated_image


def apply_flip_image(np_image: np.ndarray, flip_type: Optional[str]):
    if flip_type == "vertical":
        return cv2.flip(np_image, 0)
    elif flip_type == "horizontal":
        return cv2.flip(np_image, 1)
    elif flip_type == "both":
        return cv2.flip(np_image, -1)
    else:
        return np_image

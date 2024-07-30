from abc import ABC, abstractmethod
from typing import List, Literal, Optional, Type, Union

import numpy as np
from pydantic import AliasChoices, ConfigDict, Field

#### Traditional CV (Opencv) Import 
import cv2
####

from inference.core.workflows.core_steps.visualizations.utils import str_to_color
from inference.core.workflows.core_steps.visualizations.base import (
    OUTPUT_IMAGE_KEY,
)
from inference.core.workflows.entities.base import OutputDefinition, WorkflowImageData, Batch
from inference.core.workflows.entities.types import (
    BATCH_OF_INSTANCE_SEGMENTATION_PREDICTION_KIND,
    BATCH_OF_KEYPOINT_DETECTION_PREDICTION_KIND,
    BATCH_OF_OBJECT_DETECTION_PREDICTION_KIND,
    BATCH_OF_IMAGES_KIND,
    STRING_KIND,
    INTEGER_KIND,
    StepOutputImageSelector,
    StepOutputSelector,
    WorkflowImageSelector,
    WorkflowParameterSelector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

TYPE: str = "ColorPixelCount"
SHORT_DESCRIPTION: str = "Count the number of pixels that match a specific color within a given tolerance."
LONG_DESCRIPTION: str = "Count the number of pixels that match a specific color within a given tolerance."

class ColorPixelCountManifest(WorkflowBlockManifest):
    type: Literal[f"{TYPE}"]
    model_config = ConfigDict(
        json_schema_extra={
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "traditional",
        }
    )

    image: Union[WorkflowImageSelector, StepOutputImageSelector] = Field(
        title="Input Image",
        description="The input image for this step.",
        examples=["$inputs.image", "$steps.cropping.crops"],
        validation_alias=AliasChoices("image", "images"),
    )

    target_color: Union[WorkflowParameterSelector(kind=[STRING_KIND]), str, tuple] = Field(
        description="Target color to count in the image. Can be a hex string (e.g., '#431112') or a BGR tuple (e.g., (18, 17, 67)).",
        examples=["#431112", "$inputs.target_color", (18, 17, 67)]
    )

    tolerance: Union[WorkflowParameterSelector(kind=[INTEGER_KIND]), int] = Field(
        description="Tolerance for color matching.", examples=[10, "$inputs.tolerance"],
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name=OUTPUT_IMAGE_KEY,
                kind=[
                    BATCH_OF_IMAGES_KIND,
                ],
            ),
            OutputDefinition(
                name="Number of Color Pixels",
                kind=[
                    INTEGER_KIND,
                ],
            )
        ]


class PixelationCountBlock(WorkflowBlock):  # Ensure the class name matches the import
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def get_manifest(cls) -> Type[ColorPixelCountManifest]:
        return ColorPixelCountManifest
    
    def count_specific_color_pixels(self, image, target_color: Union[str, tuple], tolerance: int = 10) -> int:
        """
        Counts the number of pixels that match the target color within the given tolerance.

        Args:
            image: Input image.
            target_color (Union[str, tuple]): Target color in hex format (e.g., '#431112') or BGR tuple (e.g., (18, 17, 67)).
            tolerance (int, optional): Tolerance for color matching. Defaults to 10.

        Returns:
            int: Number of pixels that match the target color.
        """
        # Convert hex color to BGR if necessary
        if isinstance(target_color, str):
            if target_color.startswith('#') and len(target_color) == 7:
                try:
                    target_color_bgr = tuple(int(target_color[i:i+2], 16) for i in (5, 3, 1))
                except ValueError as e:
                    raise ValueError(f"Invalid hex color format: {target_color}") from e
            elif target_color.startswith('(') and target_color.endswith(')'):
                try:
                    target_color_bgr = tuple(map(int, target_color[1:-1].split(',')))
                except ValueError as e:
                    raise ValueError(f"Invalid tuple color format: {target_color}") from e
            else:
                raise ValueError(f"Invalid color format, must be hex or tuple BGR: {target_color}")
        elif isinstance(target_color, tuple) and len(target_color) == 3:
            target_color_bgr = target_color
        else:
            raise ValueError(f"Invalid color format, must be hex or tuple BGR: {target_color}")

        # Define the target color and the tolerance range
        lower_bound = np.array([target_color_bgr[0] - tolerance, target_color_bgr[1] - tolerance, target_color_bgr[2] - tolerance])
        upper_bound = np.array([target_color_bgr[0] + tolerance, target_color_bgr[1] + tolerance, target_color_bgr[2] + tolerance])
        
        # Create a mask that isolates the target color
        mask = cv2.inRange(image, lower_bound, upper_bound)
        
        # Count the number of pixels that match the target color
        color_pixels = np.sum(mask > 0)
        
        return color_pixels

    async def run(self, image: WorkflowImageData, target_color: Union[str, tuple], tolerance: int, *args, **kwargs) -> BlockResult:
        # Count the specific color pixels in the image
        color_pixel_count = self.count_specific_color_pixels(image.numpy_image, target_color, tolerance)

        output = WorkflowImageData(
            parent_metadata=image.parent_metadata,
            workflow_root_ancestor_metadata=image.workflow_root_ancestor_metadata,
            numpy_image=image.numpy_image,  # Keeping the original image as output
        )

        return {OUTPUT_IMAGE_KEY: output, "Number of Color Pixels": color_pixel_count}
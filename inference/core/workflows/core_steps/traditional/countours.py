## Required Libraries:
from abc import ABC, abstractmethod
from typing import List, Literal, Optional, Type, Union

import numpy as np
import supervision as sv
from pydantic import AliasChoices, ConfigDict, Field

#### Traditional CV (Opencv) Import 
import cv2
####

from inference.core.workflows.core_steps.visualizations.utils import str_to_color
# TODO: Is this kosher?
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

TYPE: str = "ImageContours"
SHORT_DESCRIPTION: str = "Find and draw contours on an image."
LONG_DESCRIPTION: str = "Find and draw contours on an image."

class ImageContoursManifest(WorkflowBlockManifest):
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
        title="Threshold Image",
        description="The threshold input image for this step.",
        examples=["$inputs.image", "$steps.cropping.crops"],
        validation_alias=AliasChoices("image", "images"),
    )

    # TODO: Shouldnt have to be RGB... messy.
    raw_image: Union[WorkflowImageSelector, StepOutputImageSelector] = Field(
        title="RGB Input Image",
        description="The input image for this step.",
        examples=["$inputs.image", "$steps.cropping.crops"],
        validation_alias=AliasChoices("raw_image", "raw_images"),
    )

    line_thickness: Union[WorkflowParameterSelector(kind=[INTEGER_KIND]), int] = Field(
        description="Line thickness for drawing contours.",
        examples=[3, "$inputs.line_thickness"],
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
                name="Number of Contours",
                kind=[
                    INTEGER_KIND,
                ],
            )
        ]


class ImageContoursBlock(WorkflowBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def get_manifest(cls) -> Type[ImageContoursManifest]:
        return ImageContoursManifest
    
    def find_and_draw_contours(self, image: np.ndarray, image_draw: np.ndarray, color: tuple = (255, 0, 255), thickness: int = 3) -> tuple[np.ndarray, int]:
        """
        Finds and draws contours on the image.

        Args:
            image (np.ndarray): Input thresholded image.
            color (tuple, optional): Color of the contour lines in BGR. Defaults to purple (255, 0, 255).
            thickness (int, optional): Thickness of the contour lines. Defaults to 3.

        Returns:
            tuple: Image with contours drawn and number of contours.
        """
        # Find contours
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        print('='*50)
        print(contours)
        print(len(contours))
        print('='*50)

        # Draw contours on a copy of the original image
        contour_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(contour_image, contours, -1, color, thickness)

        # Return the image with contours and the number of contours
        return contour_image, len(contours)

    # TODO: Check this is all good and robust.
    async def run(self, image: WorkflowImageData, raw_image: WorkflowImageData, line_thickness: int, *args, **kwargs) -> BlockResult:
        # Find and draw contours
        contour_image, num_contours = self.find_and_draw_contours(image.numpy_image, raw_image.numpy_image, thickness=line_thickness)

        output = WorkflowImageData(
            parent_metadata=image.parent_metadata,
            workflow_root_ancestor_metadata=image.workflow_root_ancestor_metadata,
            numpy_image=contour_image,
        )

        print('-----------')
        print(num_contours)
        print('-----------')

        return {OUTPUT_IMAGE_KEY: output, "Number of Contours": num_contours}
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

TYPE: str = "ImageBlur"
SHORT_DESCRIPTION: str = "Apply a blur to an image."
LONG_DESCRIPTION: str = "Apply a blur to an image."


class ImageBlurManifest(WorkflowBlockManifest):
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

    blur_type: Union[
        WorkflowParameterSelector(kind=[STRING_KIND]), Literal["average", "gaussian", "median", "bilateral"]
    ] = Field(
        description="Type of Blur to perform on image.", examples=["average", "$inputs.blur_type"]
    )

    kernel_size: Union[WorkflowParameterSelector(kind=[INTEGER_KIND]), int] = Field(
        description="Size of the average pooling kernel used for blurring.",
        examples=[5, "$inputs.kernel_size"],
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
        ]


class ImageBlurBlock(WorkflowBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def get_manifest(cls) -> Type[ImageBlurManifest]:
        return ImageBlurManifest
    
    # TODO: Fix image type
    def apply_blur(self, image, blur_type: str, ksize: int = 5) -> np.ndarray:
        """
        Applies the specified blur to the image.

        Args:
            image: Input image.
            blur_type (str): Type of blur ('average', 'gaussian', 'median', 'bilateral').
            ksize (int, optional): Kernel size for the blur. Defaults to 5.

        Returns:
            np.ndarray: Blurred image.
        """

        if blur_type == 'average':
            blurred_image = cv2.blur(image, (ksize, ksize))
        elif blur_type == 'gaussian':
            blurred_image = cv2.GaussianBlur(image, (ksize, ksize), 0)
        elif blur_type == 'median':
            blurred_image = cv2.medianBlur(image, ksize)
        elif blur_type == 'bilateral':
            blurred_image = cv2.bilateralFilter(image, ksize, 75, 75)
        else:
            raise ValueError(f"Unknown blur type: {blur_type}")

        return blurred_image

    # TODO: Check this is all good and robust.
    async def run(self, image: WorkflowImageData, blur_type: str, kernel_size: int, *args, **kwargs) -> BlockResult:
        # Apply blur to the image
        blurred_image = self.apply_blur(image.numpy_image, blur_type, kernel_size)

        output = WorkflowImageData(
            parent_metadata=image.parent_metadata,
            workflow_root_ancestor_metadata=image.workflow_root_ancestor_metadata,
            numpy_image=blurred_image,
        )

        return {OUTPUT_IMAGE_KEY: output}
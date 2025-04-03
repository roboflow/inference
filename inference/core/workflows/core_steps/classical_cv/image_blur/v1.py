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

SHORT_DESCRIPTION: str = "Apply a blur to an image."
LONG_DESCRIPTION: str = """
Apply a blur to an image. 
The blur type and kernel size can be specified.
"""


class ImageBlurManifest(WorkflowBlockManifest):
    type: Literal["roboflow_core/image_blur@v1"]
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Image Blur",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "classical_computer_vision",
            "ui_manifest": {
                "section": "classical_cv",
                "icon": "far fa-droplet",
                "blockPriority": 5,
                "opencv": True,
            },
        }
    )

    image: Selector(kind=[IMAGE_KIND]) = Field(
        title="Input Image",
        description="The input image for this step.",
        examples=["$inputs.image", "$steps.cropping.crops"],
        validation_alias=AliasChoices("image", "images"),
    )

    blur_type: Union[
        Selector(kind=[STRING_KIND]),
        Literal["average", "gaussian", "median", "bilateral"],
    ] = Field(
        default="gaussian",
        description="Type of Blur to perform on image.",
        examples=["average", "$inputs.blur_type"],
    )

    kernel_size: Union[Selector(kind=[INTEGER_KIND]), int] = Field(
        default=5,
        description="Size of the average pooling kernel used for blurring.",
        examples=[5, "$inputs.kernel_size"],
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


class ImageBlurBlockV1(WorkflowBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def get_manifest(cls) -> Type[ImageBlurManifest]:
        return ImageBlurManifest

    def run(
        self,
        image: WorkflowImageData,
        blur_type: str,
        kernel_size: int,
        *args,
        **kwargs,
    ) -> BlockResult:
        # Apply blur to the image
        blurred_image = apply_blur(image.numpy_image, blur_type, kernel_size)
        output = WorkflowImageData.copy_and_replace(
            origin_image_data=image,
            numpy_image=blurred_image,
        )
        return {OUTPUT_IMAGE_KEY: output}


def apply_blur(image: np.ndarray, blur_type: str, ksize: int = 5) -> np.ndarray:
    """
    Applies the specified blur to the image.

    Args:
        image: Input image.
        blur_type (str): Type of blur ('average', 'gaussian', 'median', 'bilateral').
        ksize (int, optional): Kernel size for the blur. Defaults to 5.

    Returns:
        np.ndarray: Blurred image.
    """

    if blur_type == "average":
        blurred_image = cv2.blur(image, (ksize, ksize))
    elif blur_type == "gaussian":
        blurred_image = cv2.GaussianBlur(image, (ksize, ksize), 0)
    elif blur_type == "median":
        blurred_image = cv2.medianBlur(image, ksize)
    elif blur_type == "bilateral":
        blurred_image = cv2.bilateralFilter(image, ksize, 75, 75)
    else:
        raise ValueError(f"Unknown blur type: {blur_type}")

    return blurred_image

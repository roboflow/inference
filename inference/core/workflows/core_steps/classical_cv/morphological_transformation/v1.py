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
LONG_DESCRIPTION: str = """
Apply morphological transformation to an image

See https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html

Note: This block currently only supports single-channel (grayscale) images.
Color images will be converted to grayscale before processing.

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
        description="The input image for this step.",
        examples=["$inputs.image", "$steps.cropping.crops"],
        validation_alias=AliasChoices("image", "images"),
    )

    kernel_size: Union[int, Selector(kind=[INTEGER_KIND])] = Field(
        default=5,
        description="Size of the kernel to use for morphological transformation.",
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
        description="Type of morphological operation to use.",
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

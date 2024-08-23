from typing import List, Literal, Optional, Type, Union

import cv2
import numpy as np
from pydantic import AliasChoices, ConfigDict, Field

from inference.core.workflows.execution_engine.entities.base import (
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    CONTOURS_KIND,
    INTEGER_KIND,
    NUMPY_ARRAY_KIND,
    StepOutputImageSelector,
    WorkflowImageSelector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

SHORT_DESCRIPTION: str = "Find and count the contours on an image."
LONG_DESCRIPTION: str = """
Finds the contours in an image. It returns the contours and number of contours. The input image should be thresholded before using this block.
"""


class ImageContoursDetectionManifest(WorkflowBlockManifest):
    type: Literal["roboflow_core/contours_detection@v1"]
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Image Contours",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "classical_computer_vision",
        }
    )

    image: Union[WorkflowImageSelector, StepOutputImageSelector] = Field(
        title="Input Image",
        description="The input image for this step.",
        examples=["$inputs.image", "$steps.cropping.crops"],
        validation_alias=AliasChoices("image", "images"),
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name="contours",
                kind=[
                    CONTOURS_KIND,
                ],
            ),
            OutputDefinition(
                name="hierarchy",
                kind=[
                    NUMPY_ARRAY_KIND,
                ],
            ),
            OutputDefinition(
                name="number_contours",
                kind=[
                    INTEGER_KIND,
                ],
            ),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.0.0,<2.0.0"


class ImageContoursDetectionBlockV1(WorkflowBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def get_manifest(cls) -> Type[ImageContoursDetectionManifest]:
        return ImageContoursDetectionManifest

    def run(self, image: WorkflowImageData, *args, **kwargs) -> BlockResult:
        # Find and draw contours
        contours, hierarchy = count_contours(image.numpy_image)

        return {
            "contours": contours,
            "hierarchy": hierarchy,
            "number_contours": len(contours),
        }


def count_contours(
    image: np.ndarray,
) -> tuple[np.ndarray, int]:
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
    contours, hierarchy = cv2.findContours(
        image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Return the image with contours and the number of contours
    return contours, hierarchy

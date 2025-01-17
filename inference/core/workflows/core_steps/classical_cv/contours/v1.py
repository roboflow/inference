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
    CONTOURS_KIND,
    IMAGE_KIND,
    INTEGER_KIND,
    NUMPY_ARRAY_KIND,
    Selector,
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
            "ui_manifest": {
                "section": "classical_cv",
                "icon": "far fa-border-all",
                "blockPriority": 4,
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

    line_thickness: Union[Selector(kind=[INTEGER_KIND]), int] = Field(
        description="Line thickness for drawing contours.",
        default=3,
        examples=[3, "$inputs.line_thickness"],
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
        return ">=1.3.0,<2.0.0"


class ImageContoursDetectionBlockV1(WorkflowBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def get_manifest(cls) -> Type[ImageContoursDetectionManifest]:
        return ImageContoursDetectionManifest

    def run(
        self, image: WorkflowImageData, line_thickness: int, *args, **kwargs
    ) -> BlockResult:
        # Find and draw contours
        contour_image, contours, hierarchy = find_and_draw_contours(
            image.numpy_image, thickness=line_thickness
        )
        output = WorkflowImageData.copy_and_replace(
            origin_image_data=image, numpy_image=contour_image
        )
        return {
            OUTPUT_IMAGE_KEY: output,
            "contours": contours,
            "hierarchy": hierarchy,
            "number_contours": len(contours),
        }


def find_and_draw_contours(
    image: np.ndarray, color: Tuple[int, int, int] = (255, 0, 255), thickness: int = 3
) -> Tuple[np.ndarray, int]:
    """
    Finds and draws contours on the image.

    Args:
        image (np.ndarray): Input thresholded image.
        color (tuple, optional): Color of the contour lines in BGR. Defaults to purple (255, 0, 255).
        thickness (int, optional): Thickness of the contour lines. Defaults to 3.

    Returns:
        tuple: Image with contours drawn and number of contours.
    """
    # If not in grayscale, convert to grayscale
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find contours
    contours, hierarchy = cv2.findContours(
        image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Draw contours on a copy of the original image
    contour_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(contour_image, contours, -1, color, thickness)

    # Return the image with contours and the number of contours
    return contour_image, contours, hierarchy

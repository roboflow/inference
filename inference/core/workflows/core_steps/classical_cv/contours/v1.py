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
    BATCH_OF_IMAGES_KIND,
    INTEGER_KIND,
    StepOutputImageSelector,
    WorkflowImageSelector,
    WorkflowParameterSelector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

SHORT_DESCRIPTION: str = "Find and draw contours on an image."
LONG_DESCRIPTION: str = """
Finds the contours in an image. It returns the image with the contours drawn on it as well as the number of contours. The input image should be thresholded before using this block.
"""


class ImageContoursManifest(WorkflowBlockManifest):
    type: Literal["roboflow_core/contours@v1"]
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

    raw_image: Union[WorkflowImageSelector, StepOutputImageSelector] = Field(
        title="RGB Draw Image",
        description="The image to draw the contours on.",
        examples=["$inputs.image", "$steps.cropping.crops"],
        validation_alias=AliasChoices("raw_image", "raw_images"),
    )

    line_thickness: Union[WorkflowParameterSelector(kind=[INTEGER_KIND]), int] = Field(
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
                    BATCH_OF_IMAGES_KIND,
                ],
            ),
            OutputDefinition(
                name="Number of Contours",
                kind=[
                    INTEGER_KIND,
                ],
            ),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.0.0,<2.0.0"


class ImageContoursBlockV1(WorkflowBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def get_manifest(cls) -> Type[ImageContoursManifest]:
        return ImageContoursManifest

    def find_and_draw_contours(
        self,
        image: np.ndarray,
        image_draw: np.ndarray,
        color: tuple = (255, 0, 255),
        thickness: int = 3,
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
        contours, _ = cv2.findContours(
            image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Draw contours on a copy of the original image
        contour_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(contour_image, contours, -1, color, thickness)

        # Return the image with contours and the number of contours
        return contour_image, len(contours)

    def run(
        self,
        image: WorkflowImageData,
        raw_image: WorkflowImageData,
        line_thickness: int,
        *args,
        **kwargs
    ) -> BlockResult:
        # Find and draw contours
        contour_image, num_contours = self.find_and_draw_contours(
            image.numpy_image, raw_image.numpy_image, thickness=line_thickness
        )

        output = WorkflowImageData(
            parent_metadata=image.parent_metadata,
            workflow_root_ancestor_metadata=image.workflow_root_ancestor_metadata,
            numpy_image=contour_image,
        )

        return {OUTPUT_IMAGE_KEY: output, "Number of Contours": num_contours}

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
    StepOutputImageSelector,
    WorkflowImageSelector,
    WorkflowParameterSelector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

SHORT_DESCRIPTION: str = "Apply a threshold to an image."
LONG_DESCRIPTION: str = """
Apply a threshold to an image. The image must be in grayscale.
"""


class ImageThresholdManifest(WorkflowBlockManifest):
    type: Literal["roboflow_core/threshold@v1"]
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Image Threshold",
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

    threshold_type: Union[
        WorkflowParameterSelector(kind=[STRING_KIND]),
        Literal[
            "binary",
            "binary_inv",
            "trunc",
            "tozero",
            "tozero_inv",
            "adaptive_mean",
            "adaptive_gaussian",
            "otsu",
        ],
    ] = Field(
        description="Type of Edge Detection to perform.",
        default="binary",
        examples=["binary", "$inputs.threshold_type"],
    )

    thresh_value: Union[WorkflowParameterSelector(kind=[INTEGER_KIND]), int] = Field(
        description="Threshold value.",
        examples=[127, "$inputs.thresh_value"],
    )

    max_value: Union[WorkflowParameterSelector(kind=[INTEGER_KIND]), int] = Field(
        description="Maximum value for thresholding",
        default=255,
        examples=[255, "$inputs.max_value"],
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
        return ">=1.0.0,<2.0.0"


class ImageThresholdBlockV1(WorkflowBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def get_manifest(cls) -> Type[ImageThresholdManifest]:
        return ImageThresholdManifest

    def run(
        self,
        image: WorkflowImageData,
        threshold_type: str,
        thresh_value: int,
        max_value: int,
        *args,
        **kwargs,
    ) -> BlockResult:
        # Apply threshold to the image
        thresholded_image = apply_thresholding(
            image.numpy_image, threshold_type, thresh_value, max_value
        )

        output = WorkflowImageData(
            parent_metadata=image.parent_metadata,
            workflow_root_ancestor_metadata=image.workflow_root_ancestor_metadata,
            numpy_image=thresholded_image,
        )

        return {OUTPUT_IMAGE_KEY: output}


def apply_thresholding(
    image: np.ndarray, threshold_type: str, thresh_value: int, max_value: int
) -> np.ndarray:
    """
    Applies the specified thresholding to the image.

    Args:
        image (np.ndarray): Input image in grayscale.
        threshold_type (str): Type of thresholding ('binary', 'binary_inv', 'trunc', 'tozero', 'tozero_inv', 'adaptive_mean', 'adaptive_gaussian', 'otsu').
        thresh_value (int, optional): Threshold value.
        max_value (int, optional): Maximum value to use with the THRESH_BINARY and THRESH_BINARY_INV thresholding types.

    Returns:
        np.ndarray: Image with thresholding applied.
    """
    if threshold_type == "binary":
        _, thresh_image = cv2.threshold(
            image, thresh_value, max_value, cv2.THRESH_BINARY
        )
    elif threshold_type == "binary_inv":
        _, thresh_image = cv2.threshold(
            image, thresh_value, max_value, cv2.THRESH_BINARY_INV
        )
    elif threshold_type == "trunc":
        _, thresh_image = cv2.threshold(
            image, thresh_value, max_value, cv2.THRESH_TRUNC
        )
    elif threshold_type == "tozero":
        _, thresh_image = cv2.threshold(
            image, thresh_value, max_value, cv2.THRESH_TOZERO
        )
    elif threshold_type == "tozero_inv":
        _, thresh_image = cv2.threshold(
            image, thresh_value, max_value, cv2.THRESH_TOZERO_INV
        )
    elif threshold_type == "adaptive_mean":
        thresh_image = cv2.adaptiveThreshold(
            image, max_value, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2
        )
    elif threshold_type == "adaptive_gaussian":
        thresh_image = cv2.adaptiveThreshold(
            image,
            max_value,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2,
        )
    elif threshold_type == "otsu":
        _, thresh_image = cv2.threshold(
            image, 0, max_value, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
    else:
        raise ValueError(f"Unknown threshold type: {threshold_type}")

    return thresh_image

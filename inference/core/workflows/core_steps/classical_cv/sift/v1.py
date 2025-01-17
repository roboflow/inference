from typing import List, Literal, Optional, Type

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
    IMAGE_KEYPOINTS_KIND,
    IMAGE_KIND,
    NUMPY_ARRAY_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

####

SHORT_DESCRIPTION: str = "Apply SIFT to an image."
LONG_DESCRIPTION: str = """
The Scale-Invariant Feature Transform (SIFT) algorithm is a popular method in computer vision for detecting 
and describing features (interesting parts) in images. SIFT is used to find key points in an image and 
describe them in a way that allows for recognizing the same objects or features in different images, 
even if the images are taken from different angles, distances, or lighting conditions.

Read more: https://en.wikipedia.org/wiki/Scale-invariant_feature_transform
"""


class SIFTDetectionManifest(WorkflowBlockManifest):
    type: Literal["roboflow_core/sift@v1"]
    model_config = ConfigDict(
        json_schema_extra={
            "name": "SIFT",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "classical_computer_vision",
            "ui_manifest": {
                "section": "classical_cv",
                "icon": "far fa-grid-2-plus",
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

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name=OUTPUT_IMAGE_KEY,
                kind=[IMAGE_KIND],
            ),
            OutputDefinition(
                name="keypoints",
                kind=[IMAGE_KEYPOINTS_KIND],
            ),
            OutputDefinition(
                name="descriptors",
                kind=[NUMPY_ARRAY_KIND],
            ),
        ]


class SIFTBlockV1(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[SIFTDetectionManifest]:
        return SIFTDetectionManifest

    def run(self, image: WorkflowImageData, *args, **kwargs) -> BlockResult:
        img_with_kp, keypoints, descriptors = apply_sift(image.numpy_image)
        output_image = WorkflowImageData.copy_and_replace(
            origin_image_data=image,
            numpy_image=img_with_kp,
        )
        return {
            OUTPUT_IMAGE_KEY: output_image,
            "keypoints": keypoints,
            "descriptors": descriptors,
        }


def apply_sift(image: np.ndarray) -> (np.ndarray, list, np.ndarray):
    """
    Applies SIFT to the image.
    Args:
        image: Input image.
    Returns:
        np.ndarray: Image with keypoints drawn.
        list: Keypoints detected.
        np.ndarray: Descriptors of the keypoints.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(gray, None)
    img_with_kp = cv2.drawKeypoints(gray, kp, image)
    # Convert keypoints to the desired format
    keypoints = [
        {
            "pt": (point.pt[0], point.pt[1]),
            "size": point.size,
            "angle": point.angle,
            "response": point.response,
            "octave": point.octave,
            "class_id": point.class_id,
        }
        for point in kp
    ]
    return img_with_kp, keypoints, des

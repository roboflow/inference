## Required Libraries:
from abc import ABC, abstractmethod
from typing import List, Literal, Optional, Type, Union

#### Traditional CV (Opencv) Import
import cv2
import numpy as np
import supervision as sv
from pydantic import AliasChoices, ConfigDict, Field

from inference.core.workflows.core_steps.visualizations.common.base import (
    OUTPUT_IMAGE_KEY,
)
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    BATCH_OF_IMAGES_KIND,
    BATCH_OF_KEYPOINT_DETECTION_PREDICTION_KIND,
    INTEGER_KIND,
    NUMPY_ARRAY_KIND,
    STRING_KIND,
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

####



TYPE: str = "SIFT"
SHORT_DESCRIPTION: str = "Apply SIFT to an image."
LONG_DESCRIPTION: str = "Apply SIFT to an image."


class SIFTDetectionManifest(WorkflowBlockManifest):
    type: Literal[f"{TYPE}"]
    model_config = ConfigDict(
        json_schema_extra={
            "name": "SIFT",
            "version": "v1",
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

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.0.0,<2.0.0"

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
                name="keypoints",
                kind=[
                    BATCH_OF_KEYPOINT_DETECTION_PREDICTION_KIND,
                ],
            ),
            OutputDefinition(
                name="descriptors",
                kind=[
                    NUMPY_ARRAY_KIND,
                ],
            ),
        ]


class SIFTBlockV1(WorkflowBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def get_manifest(cls) -> Type[SIFTDetectionManifest]:
        return SIFTDetectionManifest

    def apply_sift(self, image: np.ndarray) -> (np.ndarray, list, np.ndarray):
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
        # Draw keypoints with larger circles
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

    def run(self, image: WorkflowImageData, *args, **kwargs) -> BlockResult:
        # Apply SIFT to the image
        img_with_kp, keypoints, descriptors = self.apply_sift(image.numpy_image)

        output_image = WorkflowImageData(
            parent_metadata=image.parent_metadata,
            workflow_root_ancestor_metadata=image.workflow_root_ancestor_metadata,
            numpy_image=img_with_kp,
        )

        return {
            OUTPUT_IMAGE_KEY: output_image,
            "keypoints": keypoints,
            "descriptors": descriptors,
        }

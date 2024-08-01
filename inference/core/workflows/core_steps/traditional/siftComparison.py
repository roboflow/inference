import cv2
from typing import List, Literal, Optional, Type, Union
from pydantic import AliasChoices, ConfigDict, Field, PositiveInt
from inference.core.workflows.entities.base import Batch, OutputDefinition
from inference.core.workflows.core_steps.visualizations.base import (
    OUTPUT_IMAGE_KEY,
)
from inference.core.workflows.entities.base import OutputDefinition, WorkflowImageData, Batch

import numpy as np

from typing import (
    Any,
    Dict,
    Generator,
    List,
    Literal,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
)

from inference.core.workflows.entities.types import (
    BATCH_OF_KEYPOINT_DETECTION_PREDICTION_KIND,
    BATCH_OF_IMAGES_KIND,

    BOOLEAN_KIND,
    INTEGER_KIND,
    StepOutputSelector,
    WorkflowParameterSelector,
    NUMPY_ARRAY_KIND,
    IMAGE_KIND,
    StepOutputImageSelector,
    StepOutputSelector,
    WorkflowImageSelector,

)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

LONG_DESCRIPTION = """
Compare SIFT descriptors from multiple images using FLANN-based matcher.

This block is useful for determining if multiple images match based on their SIFT descriptors.
"""

SHORT_DESCRIPTION = "Compare SIFT descriptors from multiple images."

class SIFTComparisonBlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "comparison",
        }
    )
    type: Literal["SIFTComparison"]
    
    
    image1: Union[WorkflowImageSelector, StepOutputImageSelector] = Field(
        title="Input Image 1",
        description="The first input image for this step.",
        examples=["$inputs.image1", "$steps.cropping.crops1"],
        validation_alias=AliasChoices("image1", "images1"),
    )

    image2: Union[WorkflowImageSelector, StepOutputImageSelector] = Field(
        title="Input Image 2",
        description="The second input image for this step.",
        examples=["$inputs.image2", "$steps.cropping.crops2"],
        validation_alias=AliasChoices("image2", "images2"),
    )
    
    descriptor1: StepOutputSelector(
        kind=[NUMPY_ARRAY_KIND]
    ) = Field(
        description="Reference to SIFT descriptors from the first image to compare",
        examples=["$steps.sift.descriptors"],
        validation_alias=AliasChoices("descriptor1", "descriptor_1"),
    )

    descriptor2: StepOutputSelector(
        kind=[NUMPY_ARRAY_KIND]
    ) = Field(
        description="Reference to SIFT descriptors from the second image to compare",
        examples=["$steps.sift.descriptors"],
        validation_alias=AliasChoices("descriptor2", "descriptor_2"),
    )
    
    kp1: StepOutputSelector(
        kind=[BATCH_OF_KEYPOINT_DETECTION_PREDICTION_KIND]
    ) = Field(
        description="Keypoints from the first image",
        examples=["$steps.sift.keypoints1"],
        validation_alias=AliasChoices("kp1", "keypoints1"),
    )

    kp2: StepOutputSelector(
        kind=[BATCH_OF_KEYPOINT_DETECTION_PREDICTION_KIND]
    ) = Field(
        description="Keypoints from the second image",
        examples=["$steps.sift.keypoints2"],
        validation_alias=AliasChoices("kp2", "keypoints2"),
    )
    
    good_matches_threshold: Union[
        PositiveInt, WorkflowParameterSelector(kind=[INTEGER_KIND])
    ] = Field(
        default=10,
        description="Threshold for the number of good matches to consider the images as matching",
        examples=[10, "$inputs.good_matches_threshold"],
    )

    ratio_threshold: Union[
        float, WorkflowParameterSelector(kind=[INTEGER_KIND])
    ] = Field(
        default=0.7,
        description="Ratio threshold for the ratio test",
        examples=[0.7, "$inputs.ratio_threshold"],
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        """
        Define the outputs of the SIFTComparison block.
        """
        return [
            OutputDefinition(
                name="good_matches_count", kind=[INTEGER_KIND]
            ),
            OutputDefinition(
                name="images_match", kind=[BOOLEAN_KIND]
            ),
            OutputDefinition(
                name=OUTPUT_IMAGE_KEY,
                kind=[
                    BATCH_OF_IMAGES_KIND,
                ],
            ),
        ]

class SIFTComparisonBlock(WorkflowBlock):
    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return SIFTComparisonBlockManifest

    async def run(
        self,
        descriptor1: np.ndarray,
        descriptor2: np.ndarray,
        kp1: List[cv2.KeyPoint],
        kp2: List[cv2.KeyPoint],
        image1: WorkflowImageData,
        image2: WorkflowImageData,
        good_matches_threshold: int = 10,
        ratio_threshold: float = 0.7,
    ) -> BlockResult:
        # Ensure descriptors are numpy arrays and convert to float32
        # descriptor1 = np.array(descriptor1).astype(np.float32)
        # descriptor2 = np.array(descriptor2).astype(np.float32)

        # Implement the SIFT comparison logic
        flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
        matches = flann.knnMatch(descriptor1, descriptor2, k=2)

        # Apply ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < ratio_threshold * n.distance:
                good_matches.append(m)
        good_matches_count = len(good_matches)
        images_match = good_matches_count >= good_matches_threshold

        # Draw matches
        img_matches = cv2.drawMatches(
            image1.numpy_image, kp1, image2.numpy_image, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

        output_image = WorkflowImageData(
            parent_metadata=image1.parent_metadata,
            workflow_root_ancestor_metadata=image1.workflow_root_ancestor_metadata,
            numpy_image=img_matches,
        )

        return {
            OUTPUT_IMAGE_KEY: output_image,
            "images_match": images_match,
            "good_matches_count": good_matches_count,
        }
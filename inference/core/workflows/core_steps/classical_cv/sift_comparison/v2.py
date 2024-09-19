from typing import List, Literal, Optional, Type, Union
from uuid import uuid4

import cv2
import numpy as np
from pydantic import ConfigDict, Field, PositiveInt

from inference.core.workflows.execution_engine.entities.base import (
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    FLOAT_ZERO_TO_ONE_KIND,
    IMAGE_KEYPOINTS_KIND,
    IMAGE_KIND,
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

LONG_DESCRIPTION = """
Compare SIFT descriptors from multiple images using FLANN-based matcher.

This block is useful for determining if multiple images match based on their SIFT descriptors.
"""

SHORT_DESCRIPTION = "Compare SIFT descriptors from multiple images."


class SIFTComparisonBlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "SIFT Comparison",
            "version": "v2",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "classical_computer_vision",
        }
    )
    type: Literal["roboflow_core/sift_comparison@v2"]
    input_1: Union[
        WorkflowImageSelector,
        StepOutputImageSelector,
        StepOutputSelector(kind=[NUMPY_ARRAY_KIND]),
    ] = Field(
        description="Reference to Image or SIFT descriptors from the first image to compare",
        examples=["$inputs.image1", "$steps.sift.descriptors"],
    )
    input_2: Union[
        WorkflowImageSelector,
        StepOutputImageSelector,
        StepOutputSelector(kind=[NUMPY_ARRAY_KIND]),
    ] = Field(
        description="Reference to Image or SIFT descriptors from the second image to compare",
        examples=["$inputs.image2", "$steps.sift.descriptors"],
    )
    good_matches_threshold: Union[
        PositiveInt, WorkflowParameterSelector(kind=[INTEGER_KIND])
    ] = Field(
        default=50,
        description="Threshold for the number of good matches to consider the images as matching",
        examples=[50, "$inputs.good_matches_threshold"],
    )
    ratio_threshold: Union[
        float, WorkflowParameterSelector(kind=[FLOAT_ZERO_TO_ONE_KIND])
    ] = Field(
        default=0.7,
        description="Ratio threshold for the ratio test, which is used to filter out poor matches by comparing "
        "the distance of the closest match to the distance of the second closest match. A lower "
        "ratio indicates stricter filtering.",
        examples=[0.7, "$inputs.ratio_threshold"],
    )
    matcher: Union[
        Literal["FlannBasedMatcher", "BFMatcher"],
        WorkflowParameterSelector(kind=[STRING_KIND]),
    ] = Field(  # type: ignore
        default="FlannBasedMatcher",
        description="Matcher to use for comparing the SIFT descriptors",
        examples=["FlannBasedMatcher", "$inputs.matcher"],
    )
    visualize: Union[bool, WorkflowParameterSelector(kind=[BOOLEAN_KIND])] = Field(
        default=False,
        description="Whether to visualize the keypoints and matches between the two images",
        examples=[True, "$inputs.visualize"],
    )

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.0.0,<2.0.0"

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="images_match", kind=[BOOLEAN_KIND]),
            OutputDefinition(name="good_matches_count", kind=[INTEGER_KIND]),
            OutputDefinition(
                name="keypoints_1",
                kind=[IMAGE_KEYPOINTS_KIND],
            ),
            OutputDefinition(
                name="descriptors_1",
                kind=[NUMPY_ARRAY_KIND],
            ),
            OutputDefinition(
                name="keypoints_2",
                kind=[IMAGE_KEYPOINTS_KIND],
            ),
            OutputDefinition(
                name="descriptors_2",
                kind=[NUMPY_ARRAY_KIND],
            ),
            OutputDefinition(
                name="visualization_1",
                kind=[IMAGE_KIND],
            ),
            OutputDefinition(
                name="visualization_2",
                kind=[IMAGE_KIND],
            ),
            OutputDefinition(
                name="visualization_matches",
                kind=[IMAGE_KIND],
            ),
        ]


class SIFTComparisonBlockV2(WorkflowBlock):
    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return SIFTComparisonBlockManifest

    def run(
        self,
        input_1: Union[np.ndarray, WorkflowImageData],
        input_2: Union[np.ndarray, WorkflowImageData],
        good_matches_threshold: int = 50,
        ratio_threshold: float = 0.7,
        matcher: str = "FlannBasedMatcher",
        visualize: bool = False,
    ) -> BlockResult:
        if isinstance(input_1, WorkflowImageData):
            image_1 = input_1.numpy_image
            visualization_1, kp_1, keypoints_1, descriptors_1 = apply_sift(
                image_1, visualize
            )
        else:
            image_1 = None
            descriptors_1 = input_1
            kp_1 = None
            keypoints_1 = None
            visualization_1 = None

        if isinstance(input_2, WorkflowImageData):
            image_2 = input_2.numpy_image
            visualization_2, kp_2, keypoints_2, descriptors_2 = apply_sift(
                image_2, visualize
            )
        else:
            image_2 = None
            descriptors_2 = input_2
            kp_2 = None
            keypoints_2 = None
            visualization_2 = None

        # Check if both descriptor arrays have at least 2 descriptors
        if len(descriptors_1) < 2 or len(descriptors_2) < 2:
            return {
                "good_matches_count": 0,
                "images_match": False,
                "keypoints_1": keypoints_1,
                "descriptors_1": descriptors_1,
                "keypoints_2": keypoints_2,
                "descriptors_2": descriptors_2,
                "visualization_1": visualization_1,
                "visualization_2": visualization_2,
                "visualization_matches": None,
            }

        if matcher == "BFMatcher":
            bf = cv2.BFMatcher(cv2.NORM_L2)
            matches = bf.knnMatch(descriptors_1, descriptors_2, k=2)
        else:
            flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
            matches = flann.knnMatch(descriptors_1, descriptors_2, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < ratio_threshold * n.distance:
                good_matches.append([m])  # Wrap m in a list
        good_matches_count = len(good_matches)
        images_match = good_matches_count >= good_matches_threshold

        if visualization_1 is not None:
            visualization_1 = WorkflowImageData(
                parent_metadata=input_1.parent_metadata,
                workflow_root_ancestor_metadata=input_1.workflow_root_ancestor_metadata,
                numpy_image=visualization_1,
            )

        if visualization_2 is not None:
            visualization_2 = WorkflowImageData(
                parent_metadata=input_2.parent_metadata,
                workflow_root_ancestor_metadata=input_2.workflow_root_ancestor_metadata,
                numpy_image=visualization_2,
            )

        visualization_matches = None
        if visualize and image_1 is not None and image_2 is not None:
            if matcher == "BFMatcher":
                draw_params = dict(
                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
                )
            else:
                draw_params = dict(
                    matchColor=(0, 255, 0),
                    singlePointColor=(0, 0, 255),
                    flags=cv2.DrawMatchesFlags_DEFAULT,
                )

            visualization_matches = cv2.drawMatchesKnn(
                image_1,
                kp_1,
                image_2,
                kp_2,
                good_matches,
                None,
                **draw_params,
            )

            visualization_matches = WorkflowImageData(
                parent_metadata=input_1.parent_metadata,
                workflow_root_ancestor_metadata=input_1.workflow_root_ancestor_metadata,
                numpy_image=visualization_matches,
            )

        return {
            "good_matches_count": good_matches_count,
            "images_match": images_match,
            "keypoints_1": keypoints_1,
            "descriptors_1": descriptors_1,
            "keypoints_2": keypoints_2,
            "descriptors_2": descriptors_2,
            "visualization_1": visualization_1,
            "visualization_2": visualization_2,
            "visualization_matches": visualization_matches,
        }


def apply_sift(
    image: np.ndarray, visualize=False
) -> (Optional[np.ndarray], list, list, np.ndarray):
    """
    Applies SIFT to the image.
    Args:
        image: Input image.
        visualize: Whether to visualize keypoints on the image.
    Returns:
        img_with_kp: Image with keypoints drawn (if visualize is True).
        kp: List of cv2.KeyPoint objects.
        keypoints_dicts: List of keypoints as dictionaries.
        des: Descriptors of the keypoints.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(gray, None)
    img_with_kp = None
    if visualize:
        img_with_kp = cv2.drawKeypoints(gray, kp, None)
    # Convert keypoints to the desired format
    keypoints_dicts = [
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
    return img_with_kp, kp, keypoints_dicts, des

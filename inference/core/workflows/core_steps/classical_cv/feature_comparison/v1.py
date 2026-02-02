from typing import Any, List, Literal, Optional, Type, Union

import cv2
import numpy as np
from pydantic import ConfigDict, Field, PositiveInt

from inference.core.workflows.execution_engine.entities.base import OutputDefinition
from inference.core.workflows.execution_engine.entities.types import (
    FLOAT_ZERO_TO_ONE_KIND,
    IMAGE_KEYPOINTS_KIND,
    INTEGER_KIND,
    LIST_OF_VALUES_KIND,
    NUMPY_ARRAY_KIND,
    STRING_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

LONG_DESCRIPTION = """
Compare pre-computed SIFT keypoints and descriptors from two images using configurable matcher algorithms (FLANN or brute force), applying Lowe's ratio test filtering, and returning the list of good matches for downstream processing such as pose estimation, homography estimation, or custom match-based workflows.

## How This Block Works

This block compares SIFT features from two images by matching their descriptors. Unlike the SIFT Comparison block, it does not accept raw images: it expects already computed keypoints and descriptors (e.g. from SIFT blocks). The block:

1. Receives keypoints and descriptors for two images (keypoints_1, descriptors_1, keypoints_2, descriptors_2)
2. Validates that both descriptor arrays have at least 2 descriptors (required for Lowe's ratio test)
3. Selects matcher algorithm based on matcher parameter (FlannBasedMatcher or BFMatcher)
4. Performs k-nearest neighbor matching (k=2) and filters matches using Lowe's ratio test
5. Returns the list of good matches, each as keypoint_pairs (the pt coordinates (x, y) of the two matched points from image 1 and image 2) and distance

The good_matches output can be used by downstream blocks for homography estimation, drawing matches, or custom analytics.
"""

SHORT_DESCRIPTION = "Compare SIFT keypoints and descriptors; returns good matches as point pairs (pt coords) and distance."


class FeatureComparisonBlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Feature Comparison",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "classical_computer_vision",
            "ui_manifest": {
                "section": "classical_cv",
                "icon": "far fa-magnifying-glass-arrows-rotate",
                "blockPriority": 3,
                "opencv": True,
            },
        }
    )
    type: Literal["roboflow_core/feature_comparison@v1"]

    keypoints_1: Selector(kind=[IMAGE_KEYPOINTS_KIND]) = Field(
        description="SIFT keypoints from the first image (e.g. from a SIFT block). Used to output pt coordinates in keypoint_pairs.",
        examples=["$steps.sift.keypoints"],
    )
    descriptors_1: Selector(kind=[NUMPY_ARRAY_KIND]) = Field(
        description="SIFT descriptors from the first image. Numpy array of 128-dimensional descriptors.",
        examples=["$steps.sift.descriptors"],
    )
    keypoints_2: Selector(kind=[IMAGE_KEYPOINTS_KIND]) = Field(
        description="SIFT keypoints from the second image.",
        examples=["$steps.sift.keypoints"],
    )
    descriptors_2: Selector(kind=[NUMPY_ARRAY_KIND]) = Field(
        description="SIFT descriptors from the second image.",
        examples=["$steps.sift.descriptors"],
    )
    ratio_threshold: Union[float, Selector(kind=[FLOAT_ZERO_TO_ONE_KIND])] = Field(
        default=0.7,
        description="Ratio threshold for Lowe's ratio test. Matches kept when best_distance < ratio_threshold * second_best_distance.",
        examples=[0.7],
    )
    matcher: Union[
        Literal["FlannBasedMatcher", "BFMatcher"],
        Selector(kind=[STRING_KIND]),
    ] = Field(
        default="FlannBasedMatcher",
        description="Matcher algorithm: 'FlannBasedMatcher' (faster) or 'BFMatcher' (exact).",
        examples=["FlannBasedMatcher"],
    )

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="good_matches", kind=[LIST_OF_VALUES_KIND]),
            OutputDefinition(name="good_matches_count", kind=[INTEGER_KIND]),
        ]


def _match_to_keypoint_pair(
    m: cv2.DMatch,
    keypoints_1: list,
    keypoints_2: list,
) -> dict[str, Any]:
    """Convert a cv2.DMatch to a dict with keypoint_pairs (pt coords only) and distance."""
    pt1 = None
    pt2 = None
    if m.queryIdx < len(keypoints_1):
        kp1 = keypoints_1[m.queryIdx]
        pt1 = kp1.get("pt") if isinstance(kp1, dict) else None
    if m.trainIdx < len(keypoints_2):
        kp2 = keypoints_2[m.trainIdx]
        pt2 = kp2.get("pt") if isinstance(kp2, dict) else None
    return {
        "keypoint_pairs": [pt1, pt2],
        "distance": float(m.distance),
    }


class FeatureComparisonBlockV1(WorkflowBlock):
    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return FeatureComparisonBlockManifest

    def run(
        self,
        keypoints_1: list,
        descriptors_1: np.ndarray,
        keypoints_2: list,
        descriptors_2: np.ndarray,
        ratio_threshold: float = 0.7,
        matcher: str = "FlannBasedMatcher",
    ) -> BlockResult:
        if descriptors_1 is None or descriptors_2 is None:
            return {
                "good_matches": [],
                "good_matches_count": 0,
            }
        keypoints_1 = keypoints_1 or []
        keypoints_2 = keypoints_2 or []
        descriptors_1 = np.asarray(descriptors_1)
        descriptors_2 = np.asarray(descriptors_2)

        if len(descriptors_1) < 2 or len(descriptors_2) < 2:
            return {
                "good_matches": [],
                "good_matches_count": 0,
            }

        if matcher == "BFMatcher":
            bf = cv2.BFMatcher(cv2.NORM_L2)
            matches = bf.knnMatch(descriptors_1, descriptors_2, k=2)
        else:
            flann = cv2.FlannBasedMatcher(
                dict(algorithm=1, trees=5), dict(checks=50)
            )
            matches = flann.knnMatch(descriptors_1, descriptors_2, k=2)

        good_matches: List[dict[str, Any]] = []
        for m, n in matches:
            if m.distance < ratio_threshold * n.distance:
                good_matches.append(
                    _match_to_keypoint_pair(m, keypoints_1, keypoints_2)
                )

        return {
            "good_matches": good_matches,
            "good_matches_count": len(good_matches),
        }

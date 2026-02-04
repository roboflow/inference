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


def _filter_keypoints_by_mask(
    keypoints: list,
    descriptors: np.ndarray,
    mask: np.ndarray,
) -> tuple[list, np.ndarray]:
    """Keep only keypoints whose pt falls inside mask (where mask value is 1).
    Mask is indexed as mask[y, x]; keypoint pt is (x, y). Mask shape must cover
    keypoint coordinates (e.g. same as image size).
    """
    mask = np.asarray(mask)
    if mask.ndim == 3:
        mask = np.squeeze(mask)

    if mask.ndim != 2 or mask.size == 0:
        raise ValueError(
            "mask must be a non-empty 2D array (H, W) with values 0/1."
        )

    descriptors = np.asarray(descriptors)

    if len(keypoints) == 0:
        raise ValueError("No keypoints found.")

    kept_kps: list = []
    kept_descs: list = []
    h, w = mask.shape

    for i, kp in enumerate(keypoints):
        pt = kp.get("pt")
        if pt is None:
            continue
        x, y = float(pt[0]), float(pt[1])
        ix, iy = int(round(x)), int(round(y))
        if 0 <= iy < h and 0 <= ix < w and mask[iy, ix] != 0:
            kept_kps.append(kp)
            if i < len(descriptors):
                kept_descs.append(descriptors[i])

    if not kept_descs:
        raise ValueError("No valid keypoints found after filtering by mask.")

    return kept_kps, np.array(kept_descs)


LONG_DESCRIPTION = """
Compare pre-computed SIFT keypoints and descriptors from two images using configurable matcher algorithms (FLANN or brute force), applying Lowe's ratio test filtering, and returning the list of good matches for downstream processing such as pose estimation, homography estimation, or custom match-based workflows.

## How This Block Works

This block compares SIFT features from two images by matching their descriptors. Unlike the SIFT Comparison block, it does not accept raw images: it expects already computed keypoints and descriptors (e.g. from SIFT blocks). The block:

1. Receives keypoints and descriptors for two images (keypoints_1, descriptors_1, keypoints_2, descriptors_2)
2. Optionally filters keypoints to those inside binary masks (mask_1, mask_2): only keypoints whose pt lies where mask value is 1 are used; good_matches are then constrained to these regions
3. Validates that both descriptor arrays have at least 2 descriptors (required for Lowe's ratio test)
4. Selects matcher algorithm based on matcher parameter (FlannBasedMatcher or BFMatcher)
5. Performs k-nearest neighbor matching (k=2) and filters matches using Lowe's ratio test
6. Returns the list of good matches, each as keypoint_pairs (the pt coordinates (x, y) of the two matched points from image 1 and image 2) and distance

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
        description=(
            "SIFT keypoints from the first image (e.g. from a SIFT block). "
            "Used to output pt coordinates in keypoint_pairs."
        ),
        examples=["$steps.sift.keypoints"],
    )
    descriptors_1: Selector(kind=[NUMPY_ARRAY_KIND]) = Field(
        description=(
            "SIFT descriptors from the first image. "
            "Numpy array of 128-dimensional descriptors."
        ),
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
    mask_1: Optional[Union[Selector(kind=[NUMPY_ARRAY_KIND]), None]] = Field(
        default=None,
        description=(
            "Optional binary mask (values 1/0) for the first image. "
            "When set, only keypoints whose pt lies in the mask (value 1) are "
            "used; good_matches are then constrained to this region."
        ),
        examples=["$steps.mask_1.output"],
    )
    mask_2: Optional[Union[Selector(kind=[NUMPY_ARRAY_KIND]), None]] = Field(
        default=None,
        description=(
            "Optional binary mask (values 1/0) for the second image. "
            "When set, only keypoints whose pt lies in the mask (value 1) are "
            "used; good_matches are then constrained to this region."
        ),
        examples=["$steps.mask_2.output"],
    )
    ratio_threshold: Union[
        float, Selector(kind=[FLOAT_ZERO_TO_ONE_KIND])
    ] = Field(
        default=0.7,
        description=(
            "Ratio threshold for Lowe's ratio test. Matches kept when "
            "best_distance < ratio_threshold * second_best_distance."
        ),
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
    keypoints_1: list[dict[str, Any]],
    keypoints_2: list[dict[str, Any]],
) -> dict[str, Any]:
    """Convert a cv2.DMatch to a dict with keypoint_pairs (pt coords) and distance.
    keypoints_1/keypoints_2 are lists of dicts; each dict must have a 'pt' key.
    """
    kp1 = keypoints_1[m.queryIdx]
    kp2 = keypoints_2[m.trainIdx]
    return {
        "keypoint_pairs": [kp1["pt"], kp2["pt"]],
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
        mask_1: Optional[np.ndarray] = None,
        mask_2: Optional[np.ndarray] = None,
    ) -> BlockResult:
        if descriptors_1 is None:
            raise ValueError("descriptors_1 must not be None.")
        if descriptors_2 is None:
            raise ValueError("descriptors_2 must not be None.")

        keypoints_1 = keypoints_1 or []
        keypoints_2 = keypoints_2 or []
        descriptors_1 = np.asarray(descriptors_1)
        descriptors_2 = np.asarray(descriptors_2)

        if mask_1 is not None and mask_1.size > 0:
            keypoints_1, descriptors_1 = _filter_keypoints_by_mask(
                keypoints_1,
                descriptors_1,
                mask_1,
            )
        if mask_2 is not None and mask_2.size > 0:
            keypoints_2, descriptors_2 = _filter_keypoints_by_mask(
                keypoints_2,
                descriptors_2,
                mask_2,
            )

        if len(descriptors_1) < 2:
            raise ValueError(
                "At least 2 descriptors required for image 1 (after optional "
                f"mask filter). Got {len(descriptors_1)}."
            )
        if len(descriptors_2) < 2:
            raise ValueError(
                "At least 2 descriptors required for image 2 (after optional "
                f"mask filter). Got {len(descriptors_2)}."
            )

        if matcher == "BFMatcher":
            bf = cv2.BFMatcher(cv2.NORM_L2)
            matches = bf.knnMatch(descriptors_1, descriptors_2, k=2)
        else:
            flann = cv2.FlannBasedMatcher(
                dict(algorithm=1, trees=5),
                dict(checks=50),
            )
            matches = flann.knnMatch(descriptors_1, descriptors_2, k=2)

        good_matches: List[dict[str, Any]] = []
        for m, n in matches:
            if m.distance < ratio_threshold * n.distance:
                good_matches.append(
                    _match_to_keypoint_pair(m, keypoints_1, keypoints_2),
                )

        return {
            "good_matches": good_matches,
            "good_matches_count": len(good_matches),
        }

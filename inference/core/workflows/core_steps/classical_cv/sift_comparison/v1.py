from typing import List, Literal, Optional, Type, Union

import cv2
import numpy as np
from pydantic import ConfigDict, Field, PositiveInt

from inference.core.workflows.execution_engine.entities.base import OutputDefinition
from inference.core.workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    INTEGER_KIND,
    NUMPY_ARRAY_KIND,
    StepOutputSelector,
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
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "classical_computer_vision",
        }
    )
    type: Literal["roboflow_core/sift_comparison@v1"]
    descriptor_1: StepOutputSelector(kind=[NUMPY_ARRAY_KIND]) = Field(
        description="Reference to SIFT descriptors from the first image to compare",
        examples=["$steps.sift.descriptors"],
    )
    descriptor_2: StepOutputSelector(kind=[NUMPY_ARRAY_KIND]) = Field(
        description="Reference to SIFT descriptors from the second image to compare",
        examples=["$steps.sift.descriptors"],
    )
    good_matches_threshold: Union[
        PositiveInt, WorkflowParameterSelector(kind=[INTEGER_KIND])
    ] = Field(
        default=50,
        description="Threshold for the number of good matches to consider the images as matching",
        examples=[50, "$inputs.good_matches_threshold"],
    )
    ratio_threshold: Union[float, WorkflowParameterSelector(kind=[INTEGER_KIND])] = (
        Field(
            default=0.7,
            description="Ratio threshold for the ratio test, which is used to filter out poor matches by comparing "
            "the distance of the closest match to the distance of the second closest match. A lower "
            "ratio indicates stricter filtering.",
            examples=[0.7, "$inputs.ratio_threshold"],
        )
    )

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.0.0,<2.0.0"

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="good_matches_count", kind=[INTEGER_KIND]),
            OutputDefinition(name="images_match", kind=[BOOLEAN_KIND]),
        ]


class SIFTComparisonBlockV1(WorkflowBlock):
    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return SIFTComparisonBlockManifest

    def run(
        self,
        descriptor_1: np.ndarray,
        descriptor_2: np.ndarray,
        good_matches_threshold: int = 50,
        ratio_threshold: float = 0.7,
    ) -> BlockResult:
        # Check if both descriptor arrays have at least 2 descriptors
        if len(descriptor_1) < 2 or len(descriptor_2) < 2:
            return {
                "good_matches_count": 0,
                "images_match": False,
            }
        flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
        matches = flann.knnMatch(descriptor_1, descriptor_2, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < ratio_threshold * n.distance:
                good_matches.append(m)
        good_matches_count = len(good_matches)
        images_match = good_matches_count >= good_matches_threshold
        return {
            "good_matches_count": good_matches_count,
            "images_match": images_match,
        }

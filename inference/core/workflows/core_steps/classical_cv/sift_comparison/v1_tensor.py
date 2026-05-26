from typing import List, Literal, Optional, Type, Union

import cv2
import numpy as np
from pydantic import ConfigDict, Field, PositiveInt

from inference.core.workflows.execution_engine.entities.base import OutputDefinition
from inference.core.workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    INTEGER_KIND,
    NUMPY_ARRAY_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

LONG_DESCRIPTION = """
Compare SIFT (Scale Invariant Feature Transform) descriptors from two images using FLANN-based matching and Lowe's ratio test, determining image similarity by counting feature matches and returning a boolean match result based on a configurable threshold for image matching, similarity detection, duplicate detection, and feature-based image comparison workflows.

## How This Block Works

This block compares SIFT descriptors from two images to determine if they match by finding corresponding features and counting good matches. The block:

1. Receives SIFT descriptors from two images (descriptor_1 and descriptor_2) - these descriptors should come from a SIFT feature detection step that has already extracted keypoints and computed descriptors for both images
2. Validates that both descriptor arrays have at least 2 descriptors (required for ratio test filtering - needs at least 2 nearest neighbors)
3. Creates a FLANN (Fast Library for Approximate Nearest Neighbors) based matcher:
   - Uses FLANN algorithm for efficient approximate nearest neighbor search in high-dimensional descriptor space
   - Configures FLANN with algorithm parameters optimized for SIFT descriptors (algorithm=1, trees=5, checks=50)
   - FLANN is faster than brute force matching for large descriptor sets while maintaining good accuracy
4. Performs k-nearest neighbor matching (k=2) to find the 2 closest descriptor matches for each descriptor in image 1:
   - For each descriptor in descriptor_1, finds the 2 most similar descriptors in descriptor_2
   - Uses Euclidean distance in descriptor space to measure similarity
   - Returns matches with distance values indicating how similar the descriptors are
5. Filters good matches using Lowe's ratio test:
   - For each match, compares the distance to the best match (m.distance) with the distance to the second-best match (n.distance)
   - Keeps matches where m.distance < ratio_threshold * n.distance
   - This ratio test filters out ambiguous matches where multiple descriptors are similarly close
   - Lower ratio_threshold values (e.g., 0.6) require more distinct matches (stricter filtering)
   - Higher ratio_threshold values (e.g., 0.8) allow more matches (more lenient filtering)
6. Counts the number of good matches after ratio test filtering
7. Determines if images match by comparing good_matches_count to good_matches_threshold:
   - If good_matches_count >= good_matches_threshold, images_match = True
   - If good_matches_count < good_matches_threshold, images_match = False
8. Returns the count of good matches and the boolean match result

The block uses SIFT descriptors which are scale and rotation invariant, making it effective for matching images with different scales, rotations, or viewing angles. FLANN matching provides efficient approximate nearest neighbor search for fast comparison of large descriptor sets. Lowe's ratio test improves match quality by filtering ambiguous matches where the best match isn't significantly better than alternatives. The threshold-based matching allows configurable sensitivity - lower thresholds require fewer matches (more lenient), higher thresholds require more matches (stricter).

## Common Use Cases

- **Image Similarity Detection**: Determine if two images are similar or match each other (e.g., detect similar images in collections, find matching images in databases, identify duplicate images), enabling image similarity workflows
- **Duplicate Image Detection**: Identify duplicate or near-duplicate images in image collections (e.g., find duplicate images in photo libraries, detect repeated images in datasets, identify identical images with different scales or orientations), enabling duplicate detection workflows
- **Feature-Based Image Matching**: Match images based on visual features and keypoints (e.g., match images with similar content, find corresponding images across different views, identify matching images in image sequences), enabling feature-based matching workflows
- **Image Verification**: Verify if images match expected patterns or references (e.g., verify image authenticity, check if images match reference images, validate image content against templates), enabling image verification workflows
- **Image Comparison and Analysis**: Compare images to analyze similarities and differences (e.g., compare images for quality control, analyze image variations, measure image similarity scores), enabling image comparison analysis workflows
- **Content-Based Image Retrieval**: Use feature matching for content-based image search and retrieval (e.g., find similar images in databases, retrieve images by visual similarity, search images by content matching), enabling content-based retrieval workflows

## Connecting to Other Blocks

This block receives SIFT descriptors from two images and produces match results:

- **After SIFT feature detection blocks** to compare SIFT descriptors from different images (e.g., compare descriptors from multiple images, match images using SIFT features, analyze image similarity with SIFT), enabling SIFT-based image comparison workflows
- **Before filtering or logic blocks** that use match results for decision-making (e.g., filter based on image matches, make decisions based on similarity, apply logic based on match results), enabling match-based conditional workflows
- **Before data storage blocks** to store match results (e.g., store image match results, save similarity scores, record comparison data), enabling match result storage workflows
- **In image comparison pipelines** where multiple images need to be compared (e.g., compare images in sequences, analyze image similarities in workflows, process image comparisons in pipelines), enabling image comparison pipeline workflows
- **Before visualization blocks** to visualize match results (e.g., display match results, visualize similar images, show comparison outcomes), enabling match visualization workflows
- **In duplicate detection workflows** where images need to be checked for duplicates (e.g., detect duplicates in image collections, find repeated images, identify identical images), enabling duplicate detection workflows
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
            "ui_manifest": {
                "section": "classical_cv",
                "icon": "far fa-magnifying-glass-arrows-rotate",
                "blockPriority": 3,
                "opencv": True,
            },
        }
    )
    type: Literal["roboflow_core/sift_comparison@v1"]
    descriptor_1: Selector(kind=[NUMPY_ARRAY_KIND]) = Field(
        description="SIFT descriptors from the first image to compare. Should be a numpy array of SIFT descriptors (typically from a SIFT feature detection block). Each descriptor is a 128-dimensional vector describing the visual characteristics around a keypoint. The descriptors should be computed using the same SIFT parameters for both images. At least 2 descriptors are required for the ratio test to work. Use descriptors from a SIFT feature detection step that has processed the first image.",
        examples=["$steps.sift.descriptors"],
    )
    descriptor_2: Selector(kind=[NUMPY_ARRAY_KIND]) = Field(
        description="SIFT descriptors from the second image to compare. Should be a numpy array of SIFT descriptors (typically from a SIFT feature detection block). Each descriptor is a 128-dimensional vector describing the visual characteristics around a keypoint. The descriptors should be computed using the same SIFT parameters for both images. At least 2 descriptors are required for the ratio test to work. Use descriptors from a SIFT feature detection step that has processed the second image. These descriptors will be matched against descriptor_1 to determine image similarity.",
        examples=["$steps.sift.descriptors"],
    )
    good_matches_threshold: Union[PositiveInt, Selector(kind=[INTEGER_KIND])] = Field(
        default=50,
        description="Minimum number of good matches required to consider the images as matching. Must be a positive integer. If the number of good matches (after ratio test filtering) is greater than or equal to this threshold, images_match will be True. Lower values (e.g., 20-30) are more lenient and will match images with fewer feature correspondences. Higher values (e.g., 80-100) are stricter and require more feature matches. Default is 50, which provides a good balance. Adjust based on image content, expected similarity level, and false positive/negative tolerance. Use lower thresholds for images with few features, higher thresholds for images with rich texture and many features.",
        examples=[50, "$inputs.good_matches_threshold"],
    )
    ratio_threshold: Union[float, Selector(kind=[INTEGER_KIND])] = Field(
        default=0.7,
        description="Ratio threshold for Lowe's ratio test used to filter ambiguous matches. The ratio test compares the distance to the best match with the distance to the second-best match. Matches are kept only if best_match_distance < ratio_threshold * second_best_match_distance. Lower values (e.g., 0.6) require more distinct matches and are stricter (filter out more matches, leaving only high-confidence matches). Higher values (e.g., 0.8) are more lenient (allow more matches, including some ambiguous ones). Default is 0.7, which provides good balance between match quality and quantity. Typical range is 0.6-0.8. Use lower values when you need high-confidence matches only, higher values when you want more matches or have images with sparse features.",
        examples=[0.7, "$inputs.ratio_threshold"],
    )

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"

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

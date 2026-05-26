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
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

LONG_DESCRIPTION = """
Compare two images or their SIFT descriptors using configurable matcher algorithms (FLANN or brute force), automatically computing SIFT features when images are provided, applying Lowe's ratio test filtering, and optionally generating visualizations of keypoints and matches for image matching, similarity detection, duplicate detection, and feature-based image comparison workflows.

## How This Block Works

This block compares two images or their SIFT descriptors to determine if they match by finding corresponding features and counting good matches. The block:

1. Receives two inputs (input_1 and input_2) that can be either images or pre-computed SIFT descriptors
2. Processes each input based on its type:
   - **If input is an image**: Automatically computes SIFT keypoints and descriptors using OpenCV's SIFT detector
     - Converts image to grayscale
     - Detects keypoints and computes 128-dimensional SIFT descriptors
     - Optionally creates keypoint visualization if visualize=True
     - Converts keypoints to dictionary format for output
   - **If input is descriptors**: Uses the provided descriptors directly (skips SIFT computation)
3. Validates that both descriptor arrays have at least 2 descriptors (required for ratio test filtering)
4. Selects matcher algorithm based on matcher parameter:
   - **FlannBasedMatcher** (default): Uses FLANN for efficient approximate nearest neighbor search, faster for large descriptor sets
   - **BFMatcher**: Uses brute force matching with L2 norm, exact matching but slower for large descriptor sets
5. Performs k-nearest neighbor matching (k=2) to find the 2 closest descriptor matches for each descriptor in input_1:
   - For each descriptor in descriptors_1, finds the 2 most similar descriptors in descriptors_2
   - Uses Euclidean distance (L2 norm) in descriptor space to measure similarity
   - Returns matches with distance values indicating how similar the descriptors are
6. Filters good matches using Lowe's ratio test:
   - For each match, compares the distance to the best match (m.distance) with the distance to the second-best match (n.distance)
   - Keeps matches where m.distance < ratio_threshold * n.distance
   - This ratio test filters out ambiguous matches where multiple descriptors are similarly close
   - Lower ratio_threshold values (e.g., 0.6) require more distinct matches (stricter filtering)
   - Higher ratio_threshold values (e.g., 0.8) allow more matches (more lenient filtering)
7. Counts the number of good matches after ratio test filtering
8. Determines if images match by comparing good_matches_count to good_matches_threshold:
   - If good_matches_count >= good_matches_threshold, images_match = True
   - If good_matches_count < good_matches_threshold, images_match = False
9. Optionally generates visualizations if visualize=True and images were provided:
   - Creates keypoint visualizations for each image (images with keypoints drawn)
   - Creates a matches visualization showing corresponding keypoints between the two images connected by lines
10. Returns match results, keypoints, descriptors, and optional visualizations

The block provides flexibility by accepting either images (with automatic SIFT computation) or pre-computed descriptors. When images are provided, the block handles all SIFT processing internally, making it easier to use without requiring separate SIFT feature detection steps. The optional visualization feature helps debug and understand matching results by showing keypoints and matches visually. SIFT descriptors are scale and rotation invariant, making the block effective for matching images with different scales, rotations, or viewing angles.

## Common Use Cases

- **Image Similarity Detection**: Determine if two images are similar or match each other (e.g., detect similar images in collections, find matching images in databases, identify duplicate images), enabling image similarity workflows
- **Duplicate Image Detection**: Identify duplicate or near-duplicate images in image collections (e.g., find duplicate images in photo libraries, detect repeated images in datasets, identify identical images with different scales or orientations), enabling duplicate detection workflows
- **Feature-Based Image Matching**: Match images based on visual features and keypoints (e.g., match images with similar content, find corresponding images across different views, identify matching images in image sequences), enabling feature-based matching workflows
- **Image Verification**: Verify if images match expected patterns or references (e.g., verify image authenticity, check if images match reference images, validate image content against templates), enabling image verification workflows
- **Image Comparison and Analysis**: Compare images to analyze similarities and differences (e.g., compare images for quality control, analyze image variations, measure image similarity scores), enabling image comparison analysis workflows
- **Content-Based Image Retrieval**: Use feature matching for content-based image search and retrieval (e.g., find similar images in databases, retrieve images by visual similarity, search images by content matching), enabling content-based retrieval workflows

## Connecting to Other Blocks

This block receives images or SIFT descriptors and produces match results with optional visualizations:

- **After image input blocks** to compare images directly (e.g., compare input images, match images from camera feeds, analyze image similarities), enabling direct image comparison workflows
- **After SIFT feature detection blocks** to compare pre-computed SIFT descriptors (e.g., compare descriptors from different images, match images using existing SIFT features, analyze image similarity with pre-computed descriptors), enabling descriptor-based comparison workflows
- **Before filtering or logic blocks** that use match results for decision-making (e.g., filter based on image matches, make decisions based on similarity, apply logic based on match results), enabling match-based conditional workflows
- **Before data storage blocks** to store match results and visualizations (e.g., store image match results, save similarity scores, record comparison data with visualizations), enabling match result storage workflows
- **Before visualization blocks** to further process or display visualizations (e.g., display match visualizations, show keypoint images, render comparison results), enabling visualization workflow outputs
- **In image comparison pipelines** where multiple images need to be compared (e.g., compare images in sequences, analyze image similarities in workflows, process image comparisons in pipelines), enabling image comparison pipeline workflows

## Version Differences

This version (v2) includes several enhancements over v1:

- **Flexible Input Types**: Accepts both images and pre-computed SIFT descriptors as input (v1 only accepted descriptors), allowing direct image comparison without requiring separate SIFT feature detection steps
- **Automatic SIFT Computation**: Automatically computes SIFT keypoints and descriptors when images are provided, eliminating the need for separate SIFT feature detection blocks in simple workflows
- **Matcher Selection**: Added configurable matcher parameter to choose between FlannBasedMatcher (default, faster) and BFMatcher (exact, slower), providing flexibility for different performance requirements
- **Visualization Support**: Added optional visualization feature that generates keypoint visualizations and match visualizations when images are provided, helping debug and understand matching results
- **Enhanced Outputs**: Returns keypoints and descriptors for both images, plus optional visualizations (keypoint images and match visualization), providing more comprehensive output data for downstream processing
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
            "ui_manifest": {
                "section": "classical_cv",
                "icon": "far fa-magnifying-glass-arrows-rotate",
                "blockPriority": 3,
                "opencv": True,
            },
        }
    )
    type: Literal["roboflow_core/sift_comparison@v2"]
    input_1: Union[Selector(kind=[IMAGE_KIND, NUMPY_ARRAY_KIND]),] = Field(
        description="First input to compare - can be either an image or pre-computed SIFT descriptors (numpy array). If an image is provided, SIFT keypoints and descriptors will be automatically computed. If descriptors are provided, they will be used directly. Supports images from inputs or workflow steps, or descriptors from SIFT feature detection blocks. Images should be in standard image format, descriptors should be numpy arrays of 128-dimensional SIFT descriptors.",
        examples=["$inputs.image1", "$steps.sift.descriptors"],
    )
    input_2: Union[Selector(kind=[IMAGE_KIND, NUMPY_ARRAY_KIND]),] = Field(
        description="Second input to compare - can be either an image or pre-computed SIFT descriptors (numpy array). If an image is provided, SIFT keypoints and descriptors will be automatically computed. If descriptors are provided, they will be used directly. Supports images from inputs or workflow steps, or descriptors from SIFT feature detection blocks. Images should be in standard image format, descriptors should be numpy arrays of 128-dimensional SIFT descriptors. This input will be matched against input_1 to determine image similarity.",
        examples=["$inputs.image2", "$steps.sift.descriptors"],
    )
    good_matches_threshold: Union[PositiveInt, Selector(kind=[INTEGER_KIND])] = Field(
        default=50,
        description="Minimum number of good matches required to consider the images as matching. Must be a positive integer. If the number of good matches (after ratio test filtering) is greater than or equal to this threshold, images_match will be True. Lower values (e.g., 20-30) are more lenient and will match images with fewer feature correspondences. Higher values (e.g., 80-100) are stricter and require more feature matches. Default is 50, which provides a good balance. Adjust based on image content, expected similarity level, and false positive/negative tolerance. Use lower thresholds for images with few features, higher thresholds for images with rich texture and many features.",
        examples=[50, "$inputs.good_matches_threshold"],
    )
    ratio_threshold: Union[float, Selector(kind=[FLOAT_ZERO_TO_ONE_KIND])] = Field(
        default=0.7,
        description="Ratio threshold for Lowe's ratio test used to filter ambiguous matches. The ratio test compares the distance to the best match with the distance to the second-best match. Matches are kept only if best_match_distance < ratio_threshold * second_best_match_distance. Lower values (e.g., 0.6) require more distinct matches and are stricter (filter out more matches, leaving only high-confidence matches). Higher values (e.g., 0.8) are more lenient (allow more matches, including some ambiguous ones). Default is 0.7, which provides good balance between match quality and quantity. Typical range is 0.6-0.8. Use lower values when you need high-confidence matches only, higher values when you want more matches or have images with sparse features.",
        examples=[0.7, "$inputs.ratio_threshold"],
    )
    matcher: Union[
        Literal["FlannBasedMatcher", "BFMatcher"],
        Selector(kind=[STRING_KIND]),
    ] = Field(  # type: ignore
        default="FlannBasedMatcher",
        description="Matcher algorithm to use for comparing SIFT descriptors: 'FlannBasedMatcher' (default) uses FLANN for efficient approximate nearest neighbor search - faster for large descriptor sets, suitable for most use cases. 'BFMatcher' uses brute force matching with L2 norm - exact matching but slower for large descriptor sets, useful when you need exact results or have small descriptor sets. Default is 'FlannBasedMatcher' for optimal performance. Choose BFMatcher only if you need exact matching or have performance constraints that favor brute force.",
        examples=["FlannBasedMatcher", "$inputs.matcher"],
    )
    visualize: Union[bool, Selector(kind=[BOOLEAN_KIND])] = Field(
        default=False,
        description="Whether to generate visualizations of keypoints and matches. When True and images are provided as input, the block generates: (1) visualization_1 and visualization_2 showing keypoints drawn on each image, (2) visualization_matches showing corresponding keypoints between the two images connected by lines. Visualizations are only generated when images (not descriptors) are provided. Default is False. Set to True when you need to debug matching results, understand why images match or don't match, or want visual output for display or analysis purposes.",
        examples=[True, "$inputs.visualize"],
    )

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"

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
            visualization_1 = WorkflowImageData.copy_and_replace(
                origin_image_data=input_1,
                numpy_image=visualization_1,
            )

        if visualization_2 is not None:
            visualization_2 = WorkflowImageData.copy_and_replace(
                origin_image_data=input_2,
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

            visualization_matches = WorkflowImageData.copy_and_replace(
                origin_image_data=input_1,
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

from typing import List, Literal, Optional, Type, Union

import cv2 as cv
import numpy as np
from pydantic import AliasChoices, ConfigDict, Field

from inference.core.logger import logger
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    FLOAT_ZERO_TO_ONE_KIND,
    IMAGE_KIND,
    INTEGER_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

OUTPUT_KEY: str = "stitched_image"
LONG_DESCRIPTION = """
Stitch two overlapping images together into a single panoramic image using SIFT (Scale Invariant Feature Transform) feature matching and homography-based image alignment, automatically detecting common features, calculating geometric transformations, and blending images to create seamless panoramic compositions from overlapping scenes.

## How This Block Works

This block stitches two overlapping images together by detecting common features, calculating geometric transformations, and aligning the images into a single panoramic result. The block:

1. Receives two input images (image1 and image2) that contain overlapping regions with sufficient detail for feature matching
2. Detects keypoints and computes descriptors using SIFT (Scale Invariant Feature Transform) for both images:
   - Identifies distinctive feature points (keypoints) in each image that are invariant to scale and rotation
   - Computes feature descriptors (128-dimensional vectors) describing the visual characteristics around each keypoint
3. Matches keypoints between the two images using brute force matching:
   - Finds the best matching descriptors for each keypoint in image1 among all keypoints in image2
   - Uses k-nearest neighbor matching (configurable via count_of_best_matches_per_query_descriptor) to find multiple potential matches per query keypoint
4. Filters good matches using Lowe's ratio test:
   - Compares the distance to the best match with the distance to the second-best match
   - Keeps matches where the best match distance is less than 0.75 times the second-best match distance (reduces false matches)
5. Determines image ordering based on keypoint positions (identifies which image should be placed first based on spatial distribution of matched features)
6. Calculates homography transformation matrix using RANSAC (Random Sample Consensus):
   - Finds a perspective transformation matrix that maps points from one image to the other
   - Uses RANSAC to robustly estimate the transformation while filtering out outlier matches
   - Configurable maximum reprojection error (max_allowed_reprojection_error) controls which point pairs are considered inliers
7. Calculates canvas size and translation:
   - Determines the size needed to contain both images after transformation
   - Calculates translation needed to ensure both images fit within the canvas boundaries
8. Warps the second image using the homography transformation:
   - Applies perspective transformation to align the second image with the first
   - Combines homography matrix with translation matrix for correct positioning
9. Stitches images together:
   - Places the first image onto the warped second image canvas
   - Creates the final stitched panoramic image containing both input images aligned and blended
10. Returns the stitched image, or None if stitching fails (e.g., insufficient matches, transformation calculation failure)

The block uses SIFT for robust feature detection that works well with images containing sufficient detail and texture. The RANSAC-based homography calculation handles perspective distortions and ensures robust alignment even with some incorrect matches. The reprojection error threshold controls the sensitivity of the alignment - lower values require more precise matches, while higher values (useful for low-detail images) allow more tolerance for matching variations.

## Common Use Cases

- **Panoramic Image Creation**: Stitch overlapping images together to create wide panoramic views (e.g., create panoramic photos from overlapping camera shots, stitch together images from rotating cameras, combine multiple overlapping images into panoramas), enabling panoramic image generation workflows
- **Wide-Area Scene Reconstruction**: Combine multiple overlapping views of a scene into a single comprehensive image (e.g., reconstruct wide scenes from multiple camera angles, combine overlapping surveillance camera views, stitch together images from multiple viewpoints), enabling wide-area scene visualization
- **Multi-Image Mosaicking**: Create image mosaics from overlapping image tiles or sections (e.g., stitch together image tiles for large-scale mapping, combine overlapping satellite image sections, create mosaics from overlapping image captures), enabling image mosaic creation workflows
- **Scene Documentation**: Combine multiple overlapping images to document large scenes or areas (e.g., document large spaces with multiple overlapping photos, combine overlapping views for scene documentation, stitch together images for comprehensive scene capture), enabling comprehensive scene documentation
- **Video Frame Stitching**: Stitch together overlapping frames from video sequences (e.g., create panoramic views from video frames, combine overlapping frames from moving cameras, stitch together consecutive video frames), enabling video-based panoramic workflows
- **Multi-Camera View Combination**: Combine overlapping views from multiple cameras into a single unified view (e.g., stitch together overlapping camera feeds, combine multi-camera views for monitoring, merge overlapping camera perspectives), enabling multi-camera view integration workflows

## Connecting to Other Blocks

This block receives two images and produces a single stitched image:

- **After image input blocks** or **image preprocessing blocks** to stitch preprocessed images together (e.g., stitch images after preprocessing, combine images after enhancement, merge images after filtering), enabling image stitching workflows
- **After crop blocks** to stitch together cropped image regions from different sources (e.g., stitch cropped regions from different images, combine cropped sections from multiple sources, merge cropped regions into panoramas), enabling cropped region stitching workflows
- **After transformation blocks** to stitch images that have been transformed or adjusted (e.g., stitch images after perspective correction, combine images after geometric transformations, merge images after adjustments), enabling transformed image stitching workflows
- **Before detection or analysis blocks** that benefit from panoramic views (e.g., detect objects in stitched panoramic images, analyze wide-area stitched scenes, process comprehensive stitched views), enabling panoramic analysis workflows
- **Before visualization blocks** to display stitched panoramic images (e.g., visualize stitched panoramas, display wide-area stitched views, show comprehensive stitched scenes), enabling panoramic visualization outputs
- **In multi-stage image processing workflows** where images need to be stitched before further processing (e.g., stitch images before detection, combine images before analysis, merge images for comprehensive processing), enabling multi-stage panoramic processing workflows
"""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Stitch Images",
            "version": "v1",
            "short_description": "Stitch two images by common parts.",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "transformation",
            "ui_manifest": {
                "section": "transformation",
                "icon": "far fa-object-union",
                "opencv": True,
                "blockPriority": 6,
            },
        }
    )
    type: Literal["roboflow_core/stitch_images@v1"]
    image1: Selector(kind=[IMAGE_KIND]) = Field(
        title="First image to stitch",
        description="First input image to stitch. Should contain overlapping regions with image2 and sufficient detail/texture for SIFT feature detection. The images must have overlapping content for successful stitching. The block will determine the optimal positioning and alignment of this image relative to image2 during stitching. Images with rich texture and detail work best for SIFT-based feature matching.",
        examples=["$inputs.image1"],
        validation_alias=AliasChoices("image1"),
    )
    image2: Selector(kind=[IMAGE_KIND]) = Field(
        title="Second image to stitch",
        description="Second input image to stitch. Should contain overlapping regions with image1 and sufficient detail/texture for SIFT feature detection. The images must have overlapping content for successful stitching. The block will warp and align this image to match image1's perspective during stitching. Images with rich texture and detail work best for SIFT-based feature matching.",
        examples=["$inputs.image2"],
        validation_alias=AliasChoices("image2"),
    )
    max_allowed_reprojection_error: Union[Optional[float], Selector(kind=[FLOAT_ZERO_TO_ONE_KIND])] = Field(  # type: ignore
        default=3,
        description="Maximum allowed reprojection error (in pixels) to treat a point pair as an inlier during RANSAC homography calculation. This corresponds to cv.findHomography's ransacReprojThreshold parameter. Lower values require more precise matches (stricter alignment) but may fail with noisy matches. Higher values allow more tolerance for matching variations (more lenient alignment) and can improve results for low-detail images or images with imperfect feature matches. Default is 3 pixels. Increase this value (e.g., 5-10) for images with less detail or when stitching fails with default settings.",
        examples=[3, "$inputs.min_overlap_ratio_w"],
    )
    count_of_best_matches_per_query_descriptor: Union[Optional[int], Selector(kind=[INTEGER_KIND])] = Field(  # type: ignore
        default=2,
        description="Number of best matches to find per query descriptor during keypoint matching. This corresponds to cv.BFMatcher.knnMatch's k parameter. Must be greater than 0. The block finds the k nearest neighbor matches for each keypoint descriptor in image1 among all descriptors in image2. Then uses Lowe's ratio test to filter good matches (comparing best match distance with second-best match distance). Higher values provide more candidate matches but increase computation. Default is 2 (finds 2 best matches per descriptor). Typical values range from 2-5. Use higher values if you need more match candidates for difficult images.",
        examples=[2, "$inputs.k"],
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name=OUTPUT_KEY, kind=[IMAGE_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class StitchImagesBlockV1(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        image1: WorkflowImageData,
        image2: WorkflowImageData,
        count_of_best_matches_per_query_descriptor: int,
        max_allowed_reprojection_error: float,
    ) -> BlockResult:
        if count_of_best_matches_per_query_descriptor == 0:
            raise ValueError(
                "count_of_best_matches_per_query_descriptor must be greater than 0"
            )
        try:
            merged_image = stitch_images(
                image1=image1.numpy_image,
                image2=image2.numpy_image,
                count_of_best_matches_per_query_descriptor=abs(
                    int(round(count_of_best_matches_per_query_descriptor))
                ),
                max_allowed_reprojection_error=abs(max_allowed_reprojection_error),
            )
        except Exception as exc:
            logger.info("Stitching failed, %s", exc)
            return {OUTPUT_KEY: None}
        parent_metadata = ImageParentMetadata(
            parent_id=f"{image1.parent_metadata.parent_id} + {image2.parent_metadata.parent_id}"
        )
        return {
            OUTPUT_KEY: WorkflowImageData(
                parent_metadata=parent_metadata,
                numpy_image=merged_image,
            )
        }


def stitch_images(
    image1: np.ndarray,
    image2: np.ndarray,
    count_of_best_matches_per_query_descriptor: int,
    max_allowed_reprojection_error: float,
) -> np.ndarray:
    # https://docs.opencv.org/4.10.0/d7/d60/classcv_1_1SIFT.html#a4264f700a8133074fb477e30d9beb331
    sift = cv.SIFT_create()

    # https://docs.opencv.org/4.10.0/d0/d13/classcv_1_1Feature2D.html#a8be0d1c20b08eb867184b8d74c15a677
    keypoints_1, descriptors_1 = sift.detectAndCompute(image=image1, mask=None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(image=image2, mask=None)

    # https://docs.opencv.org/4.10.0/d3/da1/classcv_1_1BFMatcher.html#a02ef4d594b33d091767cbfe442aefb8a
    bf = cv.BFMatcher_create()
    # https://docs.opencv.org/4.10.0/db/d39/classcv_1_1DescriptorMatcher.html#a378f35c9b1a5dfa4022839a45cdf0e89
    matches = bf.knnMatch(
        queryDescriptors=descriptors_1,
        trainDescriptors=descriptors_2,
        k=count_of_best_matches_per_query_descriptor,
    )

    good_matches = [m[0] for m in matches if m[0].distance < 0.75 * m[1].distance]

    image1_pts = np.float32([keypoints_1[m.queryIdx].pt for m in good_matches]).reshape(
        -1, 1, 2
    )
    image2_pts = np.float32([keypoints_2[m.trainIdx].pt for m in good_matches]).reshape(
        -1, 1, 2
    )

    image1_first = np.mean([kp.pt[0] for kp in keypoints_1]) < np.mean(
        [kp.pt[0] for kp in keypoints_2]
    )
    if image1_first:
        first_image_pts = image1_pts
        second_image_pts = image2_pts
        first_image = image2
        second_image = image1
    else:
        first_image_pts = image2_pts
        second_image_pts = image1_pts
        first_image = image1
        second_image = image2

    # https://docs.opencv.org/4.10.0/d9/d0c/group__calib3d.html#ga4abc2ece9fab9398f2e560d53c8c9780
    transformation_matrix, mask = cv.findHomography(
        srcPoints=first_image_pts,
        dstPoints=second_image_pts,
        method=cv.RANSAC,
        ransacReprojThreshold=max_allowed_reprojection_error,
    )

    h1, w1 = first_image.shape[:2]
    h2, w2 = second_image.shape[:2]

    # https://docs.opencv.org/4.10.0/d2/de8/group__core__array.html#gad327659ac03e5fd6894b90025e6900a7
    warped_image_corners = cv.perspectiveTransform(
        src=np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2),
        m=transformation_matrix,
    )
    [xmin, ymin] = np.int32(warped_image_corners.min(axis=0).ravel())
    [xmax, ymax] = np.int32(warped_image_corners.max(axis=0).ravel())

    xmax = max(xmax, w1)
    ymax = max(ymax, h1)

    translation_dist = [-xmin, -ymin]

    if translation_dist[0] < 0 or translation_dist[1] < 0:
        translation_dist = [max(0, translation_dist[0]), max(0, translation_dist[1])]

    H_translation = np.array(
        [[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]]
    )

    # https://docs.opencv.org/4.10.0/da/d54/group__imgproc__transform.html#gaf73673a7e8e18ec6963e3774e6a94b87
    second_image_warped = cv.warpPerspective(
        src=second_image,
        M=H_translation @ transformation_matrix,
        dsize=(xmax - xmin, ymax - ymin),
    )

    if (
        translation_dist[0] + w1 <= second_image_warped.shape[1]
        and translation_dist[1] + h1 <= second_image_warped.shape[0]
    ):
        second_image_warped[
            translation_dist[1] : translation_dist[1] + h1,
            translation_dist[0] : translation_dist[0] + w1,
        ] = first_image

    return second_image_warped

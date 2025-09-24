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
This block combines two related scenes both containing fair amount of details.
Block is utilizing Scale Invariant Feature Transform (SIFT)
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
        description="First input image for this step.",
        examples=["$inputs.image1"],
        validation_alias=AliasChoices("image1"),
    )
    image2: Selector(kind=[IMAGE_KIND]) = Field(
        title="Second image to stitch",
        description="Second input image for this step.",
        examples=["$inputs.image2"],
        validation_alias=AliasChoices("image2"),
    )
    max_allowed_reprojection_error: Union[Optional[float], Selector(kind=[FLOAT_ZERO_TO_ONE_KIND])] = Field(  # type: ignore
        default=3,
        description="Advanced parameter overwriting cv.findHomography ransacReprojThreshold parameter."
        " Maximum allowed reprojection error to treat a point pair as an inlier."
        " Increasing value of this parameter for low details photo may yield better results.",
        examples=[3, "$inputs.min_overlap_ratio_w"],
    )
    count_of_best_matches_per_query_descriptor: Union[Optional[int], Selector(kind=[INTEGER_KIND])] = Field(  # type: ignore
        default=2,
        description="Advanced parameter overwriting cv.BFMatcher.knnMatch `k` parameter."
        " Count of best matches found per each query descriptor or less if a query descriptor has less than k possible matches in total.",
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

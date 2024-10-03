import cv2 as cv
import numpy as np

from inference.core.workflows.core_steps.transformations.stitch_images.v1 import (
    BlockManifest,
    stitch_images,
)


def test_stitch_images_validation_when_valid_manifest_is_given():
    # given
    data = {
        "type": "roboflow_core/stitch_images@v1",
        "name": "some",
        "image1": "$inputs.image1",
        "image2": "$inputs.image2",
        "max_allowed_reprojection_error": 3,
        "count_of_best_matches_per_query_descriptor": 2,
    }

    # when
    result = BlockManifest.model_validate(data)

    # then
    assert result == BlockManifest(
        type="roboflow_core/stitch_images@v1",
        name="some",
        image1="$inputs.image1",
        image2="$inputs.image2",
        max_allowed_reprojection_error=3,
        count_of_best_matches_per_query_descriptor=2,
    )


def test_stitch_images():
    # given
    contours = np.array(
        [
            [120, 10],
            [130, 15],
            [110, 20],
            [140, 25],
            [100, 30],
            [150, 35],
            [90, 40],
            [160, 45],
            [80, 50],
            [170, 55],
            [70, 60],
            [180, 65],
            [60, 70],
            [190, 75],
            [50, 80],
            [200, 85],
            [40, 90],
            [210, 95],
            [30, 90],
            [220, 95],
            [20, 90],
            [230, 95],
            [10, 90],
            [240, 95],
        ],
        dtype=int,
    )
    img1 = np.zeros((640, 640, 3), dtype=np.uint8)
    img1[:, :] = (255, 0, 0)
    img1 = cv.drawContours(
        img1,
        [contours],
        -1,
        (0, 0, 255),
        -1,
    )
    img1 = cv.line(img1, [5, 5], [5, 600], (0, 255, 0), 5, cv.LINE_8)
    img2 = np.zeros((640, 640, 3), dtype=np.uint8)
    img2[:, :] = (255, 0, 0)
    img2 = cv.drawContours(
        img2,
        [contours + np.array([300, 0])],
        -1,
        (0, 0, 255),
        -1,
    )
    img2 = cv.line(img2, [600, 5], [600, 600], (0, 150, 150), 5, cv.LINE_8)
    # when
    res = stitch_images(
        image1=img1,
        image2=img2,
        count_of_best_matches_per_query_descriptor=3,
        max_allowed_reprojection_error=5,
    )

    expected_res = np.zeros((640, 640, 3), dtype=np.uint8)
    expected_res[:, :] = (255, 0, 0)
    expected_res = cv.drawContours(
        expected_res,
        [contours + np.array([300, 0])],
        -1,
        (0, 0, 255),
        -1,
    )
    expected_res = cv.line(
        expected_res, [600, 5], [600, 600], (0, 150, 150), 5, cv.LINE_8
    )
    assert np.allclose(res, expected_res)

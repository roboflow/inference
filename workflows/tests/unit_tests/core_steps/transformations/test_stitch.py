import cv2 as cv
import numpy as np
import pytest
from roboflow_workflows.core_steps.transformations.stitch_images.v1 import (
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

    # The anchor stays pixel-exact, with the moving image extending about 300
    # pixels to the right. Homography estimation can round the outer edge by one.
    assert res.shape[0] == 640
    assert abs(res.shape[1] - 940) <= 1
    np.testing.assert_array_equal(res[:, :640], img2)
    extension = res[1:-1, 640:-1]
    np.testing.assert_array_equal(
        extension, np.broadcast_to([255, 0, 0], extension.shape)
    )


@pytest.mark.parametrize("offset", [(12, 7), (12, -7), (-12, 7), (-12, -7), (0, 0)])
def test_stitch_canvas_contains_both_images(
    offset: tuple[int, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    """An exact homography isolates the real warp and paste from estimation noise."""
    height, width = 96, 108
    image = np.random.default_rng(14).integers(
        12, 240, size=(height, width, 3), dtype=np.uint8
    )
    shift_x, shift_y = offset
    homography = np.array([[1.0, 0.0, shift_x], [0.0, 1.0, shift_y], [0.0, 0.0, 1.0]])
    monkeypatch.setattr(cv, "findHomography", lambda **_: (homography, None))

    stitched = stitch_images(
        image1=image,
        image2=image.copy(),
        count_of_best_matches_per_query_descriptor=2,
        max_allowed_reprojection_error=3,
    )

    assert stitched.shape == (height + abs(shift_y), width + abs(shift_x), 3)
    anchor_x, anchor_y = max(0, -shift_x), max(0, -shift_y)
    np.testing.assert_array_equal(
        stitched[anchor_y : anchor_y + height, anchor_x : anchor_x + width], image
    )
    source_x = width - 1 if shift_x >= 0 else 0
    source_y = height - 1 if shift_y >= 0 else 0
    moved_x = anchor_x + shift_x + source_x
    moved_y = anchor_y + shift_y + source_y
    np.testing.assert_array_equal(stitched[moved_y, moved_x], image[source_y, source_x])

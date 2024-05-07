import numpy as np

from inference.enterprise.workflows.core_steps.transformations.absolute_static_crop import (
    take_static_crop,
)


def test_take_absolute_static_crop() -> None:
    # given
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[50:70, 45:55] = 30  # painted the crop into (30, 30, 30)

    # when
    result = take_static_crop(
        image=image,
        x_center=50,
        y_center=60,
        width=10,
        height=20,
        origin_size={"height": 100, "width": 100},
        image_parent="parent_id",
    )

    # then
    assert (
        result["value"] == (np.ones((20, 10, 3), dtype=np.uint8) * 30)
    ).all(), "Crop must have the exact size and color"
    assert (
        result["parent_id"] == "parent_id"
    ), "Parent must be set at crop step identifier"
    assert result["origin_coordinates"] == {
        "left_top_x": 45,
        "left_top_y": 50,
        "origin_image_size": {"height": 100, "width": 100},
    }, "Origin coordinates of crop and image size metadata must be preserved through the operation"

import numpy as np

from inference.enterprise.workflows.core_steps.transformations.relative_static_crop import (
    take_static_crop,
)


def test_take_relative_static_crop() -> None:
    # given
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[50:70, 45:55] = 30  # painted the crop into (30, 30, 30)

    # when
    result = take_static_crop(
        image=image,
        x_center=0.5,
        y_center=0.6,
        width=0.1,
        height=0.2,
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

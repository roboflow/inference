import numpy as np
import pytest
import supervision as sv
from pydantic import ValidationError

from inference.core.workflows.core_steps.classical_cv.pixel_color_count.v1 import (
    ColorPixelCountManifest,
    PixelationCountBlockV1,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)


@pytest.mark.parametrize("images_field_alias", ["images", "image"])
def test_pixelation_validation_when_valid_manifest_is_given(
    images_field_alias: str,
) -> None:
    # given
    data = {
        "type": "roboflow_core/pixel_color_count@v1",  # Correct type
        "name": "pixelation1",
        "predictions": "$steps.od_model.predictions",
        images_field_alias: "$inputs.image",
        "pixel_size": 10,
        "target_color": [255, 0, 0],  # Add target_color field
    }

    # when
    result = ColorPixelCountManifest.model_validate(data)

    # then
    assert result == ColorPixelCountManifest(
        type="roboflow_core/pixel_color_count@v1",
        name="pixelation1",
        images="$inputs.image",
        predictions="$steps.od_model.predictions",
        pixel_size=10,
        target_color=[255, 0, 0],
    )


def test_pixelation_validation_when_invalid_image_is_given() -> None:
    # given
    data = {
        "type": "roboflow_core/pixel_color_count@v1",  # Correct type
        "name": "pixelation1",
        "images": "invalid",
        "predictions": "$steps.od_model.predictions",
        "pixel_size": 10,
        "target_color": [255, 0, 0],  # Add target_color field
    }

    # when
    with pytest.raises(ValidationError):
        _ = ColorPixelCountManifest.model_validate(data)


def test_pixelation_block() -> None:
    # given
    block = PixelationCountBlockV1()
    image = np.zeros((1000, 1000, 3), dtype=np.uint8)
    image[0:100, 0:100] = (0, 0, 245)
    image[0:10, 0:10] = (0, 0, 255)

    # when
    output = block.run(
        image=WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="some"),
            numpy_image=image,
        ),
        target_color=(255, 0, 0),
        tolerance=10,
    )

    assert output is not None
    assert output["color_pixel_count"] == 100 * 100, (
        "Expected 100*100 square to be matched, as 100 pixels match dominant color, "
        "and remaining are within margin"
    )

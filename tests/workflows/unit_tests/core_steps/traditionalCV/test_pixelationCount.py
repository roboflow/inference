import numpy as np
import pytest
from pydantic import ValidationError
import supervision as sv

from inference.core.workflows.core_steps.traditional.pixelationCount.v1 import (
    ColorPixelCountManifest,
    PixelationCountBlockV1,
)

from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)

@pytest.mark.parametrize("images_field_alias", ["images", "image"])
def test_pixelation_validation_when_valid_manifest_is_given(images_field_alias: str) -> None:
    # given
    data = {
        "type": "ColorPixelCount",  # Correct type
        "name": "pixelation1",
        "predictions": "$steps.od_model.predictions",
        images_field_alias: "$inputs.image",
        "pixel_size": 10,
        "target_color": [255, 0, 0]  # Add target_color field
    }

    # when
    result = ColorPixelCountManifest.model_validate(data)

    # then
    assert result == ColorPixelCountManifest(
        type="ColorPixelCount",
        name="pixelation1",
        images="$inputs.image",
        predictions="$steps.od_model.predictions",
        pixel_size=10,
        target_color=[255, 0, 0]
    )


def test_pixelation_validation_when_invalid_image_is_given() -> None:
    # given
    data = {
        "type": "ColorPixelCount",  # Correct type
        "name": "pixelation1",
        "images": "invalid",
        "predictions": "$steps.od_model.predictions",
        "pixel_size": 10,
        "target_color": [255, 0, 0]  # Add target_color field
    }

    # when
    with pytest.raises(ValidationError):
        _ = ColorPixelCountManifest.model_validate(data)


@pytest.mark.parametrize("target_color, tolerance", [
    ((255, 0, 0), 10),  # Red color with tolerance 10
    ((0, 255, 0), 15),  # Green color with tolerance 15
    ((0, 0, 255), 20),  # Blue color with tolerance 20
])
def test_pixelation_block(target_color, tolerance) -> None:
    # given
    block = PixelationCountBlockV1()

    output = block.run(
        image=WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="some"),
            numpy_image=np.zeros((1000, 1000, 3), dtype=np.uint8),
        ),
        target_color=target_color,  # Parametrized target_color
        tolerance=tolerance  # Parametrized tolerance
    )

    assert output is not None
    assert "image" in output
    assert hasattr(output.get("image"), "numpy_image")
    assert output['color_pixel_count'] >=0
    assert isinstance(output['color_pixel_count'], (int))
    

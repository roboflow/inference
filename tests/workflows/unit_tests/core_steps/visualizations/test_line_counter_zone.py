import cv2 as cv
import numpy as np
import pytest
from pydantic import ValidationError

from inference.core.workflows.core_steps.visualizations.line_zone.v1 import (
    LineCounterZoneVisualizationBlockV1,
    LineCounterZoneVisualizationManifest,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)


@pytest.mark.parametrize("type_alias", ["roboflow_core/line_counter_visualization@v1"])
@pytest.mark.parametrize("images_field_alias", ["images", "image"])
def test_line_counter_zone_validation_when_valid_manifest_is_given(
    type_alias: str,
    images_field_alias: str,
) -> None:
    # given
    data = {
        "type": type_alias,
        "name": "line_counter_zone_1",
        "zone": "$inputs.zone",
        images_field_alias: "$inputs.image",
        "color": "#FFFFFF",
        "opacity": 0.5,
        "thickness": 3,
        "text_thickness": 1,
        "text_scale": 2.0,
        "count_in": 7,
        "count_out": 1,
    }

    # when
    result = LineCounterZoneVisualizationManifest.model_validate(data)

    # then
    assert result == LineCounterZoneVisualizationManifest(
        type=type_alias,
        name="line_counter_zone_1",
        images="$inputs.image",
        zone="$inputs.zone",
        color="#FFFFFF",
        opacity=0.5,
        thickness=3,
        text_thickness=1,
        text_scale=2.0,
        count_in=7,
        count_out=1,
    )


def test_line_counter_zone_validation_when_invalid_image_is_given() -> None:
    # given
    data = {
        "type": "roboflow_core/line_counter_visualization@v1",
        "name": "line_counter_zone_1",
        "zone": "$inputs.zone",
        "images": "invalid",
        "color": "#FFFFFF",
        "opacity": 0.5,
        "thickness": 3,
        "text_thickness": 1,
        "text_scale": 2.0,
        "count_in": 7,
        "count_out": 1,
    }

    # when
    with pytest.raises(ValidationError):
        _ = LineCounterZoneVisualizationManifest.model_validate(data)


def test_line_counter_zone_visualization_block() -> None:
    # given
    block = LineCounterZoneVisualizationBlockV1()

    start_image = np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)
    output = block.run(
        image=WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="some"),
            numpy_image=start_image,
        ),
        zone=[(10, 10), (100, 100)],
        copy_image=True,
        color="#FF0000",
        opacity=1,
        thickness=3,
        text_thickness=1,
        text_scale=1.0,
        count_in=7,
        count_out=1,
    )

    assert isinstance(output, dict)
    assert "image" in output
    assert hasattr(output["image"], "numpy_image")

    # dimensions of output match input
    assert output.get("image").numpy_image.shape == (1000, 1000, 3)
    # check if the image is modified
    assert not np.array_equal(output.get("image").numpy_image, start_image)

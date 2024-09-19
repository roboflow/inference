import cv2 as cv
import numpy as np
import pytest
from pydantic import ValidationError

from inference.core.workflows.core_steps.visualizations.polygon_zone.v1 import (
    PolygonZoneVisualizationBlockV1,
    PolygonZoneVisualizationManifest,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)


@pytest.mark.parametrize("type_alias", ["roboflow_core/polygon_zone_visualization@v1"])
@pytest.mark.parametrize("images_field_alias", ["images", "image"])
def test_polygon_zone_validation_when_valid_manifest_is_given(
    type_alias: str,
    images_field_alias: str,
) -> None:
    # given
    data = {
        "type": type_alias,
        "name": "polygon_zone_1",
        "zone": "$inputs.zone",
        images_field_alias: "$inputs.image",
        "color": "#FFFFFF",
        "opacity": 0.5,
    }

    # when
    result = PolygonZoneVisualizationManifest.model_validate(data)

    # then
    assert result == PolygonZoneVisualizationManifest(
        type=type_alias,
        name="polygon_zone_1",
        images="$inputs.image",
        zone="$inputs.zone",
        color="#FFFFFF",
        opacity=0.5,
    )


def test_polygon_zone_validation_when_invalid_image_is_given() -> None:
    # given
    data = {
        "type": "roboflow_core/polygon_zone_visualization@v1",
        "name": "polygon_zone_1",
        "images": "invalid",
        "color": "#FFFFFF",
        "opacity": 0.5,
    }

    # when
    with pytest.raises(ValidationError):
        _ = PolygonZoneVisualizationManifest.model_validate(data)


def test_polygon_zone_visualization_block() -> None:
    # given
    block = PolygonZoneVisualizationBlockV1()

    start_image = np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)
    output = block.run(
        image=WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="some"),
            numpy_image=start_image,
        ),
        zone=[(10, 10), (100, 100), (100, 10), (50, 0)],
        copy_image=True,
        color="#FF0000",
        opacity=0.5,
    )

    assert isinstance(output, dict)
    assert "image" in output
    assert hasattr(output["image"], "numpy_image")

    # dimensions of output match input
    assert output.get("image").numpy_image.shape == (1000, 1000, 3)
    # check if the image is modified
    assert not np.array_equal(output.get("image").numpy_image, start_image)

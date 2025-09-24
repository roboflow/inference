import numpy as np
import pytest
from pydantic import ValidationError

from inference.core.workflows.core_steps.classical_cv.dominant_color.v1 import (
    DominantColorBlockV1,
    DominantColorManifest,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)


@pytest.mark.parametrize("images_field_alias", ["images", "image"])
def test_dominant_color_validation_when_valid_manifest_is_given(
    images_field_alias: str,
) -> None:
    # given
    data = {
        "type": "roboflow_core/dominant_color@v1",
        "name": "dominant_color1",
        "images": "$inputs.image",
    }

    # when
    result = DominantColorManifest.model_validate(data)

    # then
    assert result == DominantColorManifest(
        type="roboflow_core/dominant_color@v1",
        name="dominant_color1",
        images="$inputs.image",
        color_clusters=4,
        max_iterations=100,
    ), "Expected the manifest to be validated successfully with default values"


def test_dominant_color_validation_when_invalid_image_is_given() -> None:
    # given
    data = {
        "type": "DominantColor",
        "name": "dominant_color1",
        "images": "invalid",
    }

    # when
    with pytest.raises(ValidationError):
        _ = DominantColorManifest.model_validate(data)


def test_dominant_color_block() -> None:
    # given
    block = DominantColorBlockV1()

    # generate a red image to test
    red_image = np.zeros((1000, 1000, 3), dtype=np.uint8)
    red_image[:, :, 2] = 255

    output = block.run(
        image=WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="some"),
            numpy_image=red_image,
        ),
        color_clusters=4,
        max_iterations=100,
        target_size=100,
    )

    assert output is not None, "Expected an output but got None"
    assert isinstance(output["rgb_color"], tuple), " Expected rgb_color to be a tuple"
    assert len(output["rgb_color"]) == 3, " Expected rgb_color to have 3 elements"
    assert all(
        0 <= color <= 255 for color in output["rgb_color"]
    ), " Expected all elements in rgb_color to be between 0 and 255"
    assert output["rgb_color"] == (
        255,
        0,
        0,
    ), " Expected rgb_color to be [255, 0, 0], aka a red image"

import numpy as np
import pytest
import supervision as sv
from pydantic import ValidationError

from inference.core.workflows.core_steps.utilities.dominant_color.v1 import (
    DominantColorManifest,
    DominantColorBlockV1,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)


@pytest.mark.parametrize(
    "type_alias", ["roboflow_core/dominant_color@v1", "DominantColor"]
)
@pytest.mark.parametrize("images_field_alias", ["images", "image"])
def test_dominant_color_validation_when_valid_manifest_is_given(
    type_alias: str, images_field_alias: str
) -> None:
    # given
    data = {
        "type": type_alias,
        "name": "dominant_color1",
        "images": "$inputs.image",
    }

    # when
    result = DominantColorManifest.model_validate(data)

    # then
    assert result == DominantColorManifest(
        type=type_alias,
        name="dominant_color1",
        images="$inputs.image",
    )


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
        copy_image=True,
    )

    assert output is not None
    assert isinstance(output['rgb_color'], list)
    assert len(output['rgb_color']) == 3
    assert all(0 <= color <= 255 for color in output['rgb_color'])
    assert output['rgb_color'] == [255, 0, 0]
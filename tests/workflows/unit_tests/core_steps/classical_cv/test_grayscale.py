import numpy as np
import pytest
from pydantic import ValidationError

from inference.core.workflows.core_steps.classical_cv.convert_grayscale.v1 import (
    ConvertGrayscaleBlockV1,
    ConvertGrayscaleManifest,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)


@pytest.mark.parametrize("images_field_alias", ["images", "image"])
def test_convert_grayscale_validation_when_valid_manifest_is_given(
    images_field_alias: str,
) -> None:
    # given
    data = {
        "type": "roboflow_core/convert_grayscale@v1",
        "name": "grayscale1",
        images_field_alias: "$inputs.image",
    }

    # when
    result = ConvertGrayscaleManifest.model_validate(data)

    # then
    assert result == ConvertGrayscaleManifest(
        type="roboflow_core/convert_grayscale@v1",
        name="grayscale1",
        image="$inputs.image",
    )


def test_convert_grayscale_validation_when_invalid_image_is_given() -> None:
    # given
    data = {
        "type": "roboflow_core/convert_grayscale@v1",
        "name": "grayscale1",
        "image": "invalid",
    }

    # when
    with pytest.raises(ValidationError):
        _ = ConvertGrayscaleManifest.model_validate(data)


def test_convert_grayscale_block() -> None:
    # given
    block = ConvertGrayscaleBlockV1()

    start_image = np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)

    output = block.run(
        image=WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="some"),
            numpy_image=start_image,
        ),
    )

    assert output is not None
    assert "image" in output
    assert hasattr(output.get("image"), "numpy_image")

    # dimensions of output must be 1 dimensional
    assert output.get("image").numpy_image.shape == (1000, 1000)
    # check if the image is modified
    assert not np.array_equal(output.get("image").numpy_image, start_image)

import numpy as np
import pytest
from pydantic import ValidationError

from inference.core.workflows.core_steps.classical_cv.image_blur.v1 import (
    ImageBlurBlockV1,
    ImageBlurManifest,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)


@pytest.mark.parametrize("images_field_alias", ["images", "image"])
def test_image_blur_validation_when_valid_manifest_is_given(
    images_field_alias: str,
) -> None:
    # given
    data = {
        "type": "roboflow_core/image_blur@v1",
        "name": "blur1",
        images_field_alias: "$inputs.image",
        "blur_type": "gaussian",
        "kernel_size": 5,
    }

    # when
    result = ImageBlurManifest.model_validate(data)
    print(result)

    # then
    assert result == ImageBlurManifest(
        type="roboflow_core/image_blur@v1",
        name="blur1",
        image="$inputs.image",
        blur_type="gaussian",
        kernel_size=5,
    )


def test_image_blur_validation_when_invalid_image_is_given() -> None:
    # given
    data = {
        "type": "roboflow_core/image_blur@v1",
        "name": "image_blur1",
        "image": "invalid",
        "blur_type": "gaussian",
        "kernel_size": 5,
    }

    # when
    with pytest.raises(ValidationError):
        _ = ImageBlurManifest.model_validate(data)


def test_image_blur_block() -> None:
    # given
    block = ImageBlurBlockV1()

    start_image = np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)

    output = block.run(
        image=WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="some"),
            numpy_image=start_image,
        ),
        blur_type="gaussian",
        kernel_size=5,
    )

    assert output is not None
    assert "image" in output
    assert hasattr(output.get("image"), "numpy_image")

    # dimensions of output match input
    assert output.get("image").numpy_image.shape == (1000, 1000, 3)
    # check if the image is modified
    assert not np.array_equal(output.get("image").numpy_image, start_image)

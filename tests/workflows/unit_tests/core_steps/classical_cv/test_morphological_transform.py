import numpy as np
import pytest
from pydantic import ValidationError

from inference.core.workflows.core_steps.classical_cv.contrast_equalization.v1 import (
    ContrastEqualizationBlockV1,
    ContrastEqualizationManifest,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)


@pytest.mark.parametrize("images_field_alias", ["images", "image"])
def test_contrast_equalization_validation_when_valid_manifest_is_given(
    images_field_alias: str,
) -> None:
    # given
    data = {
        "type": "roboflow_core/contrast_equalization@v1",
        "name": "contrast1",
        images_field_alias: "$inputs.image",
        "equalization_type": "Contrast Stretching",
    }

    # when
    result = ContrastEqualizationManifest.model_validate(data)
    print(result)

    # then
    assert result == ContrastEqualizationManifest(
        type="roboflow_core/contrast_equalization@v1",
        name="contrast1",
        image="$inputs.image",
        equalization_type="Contrast Stretching",
    )


def test_contrast_equalization_validation_when_invalid_image_is_given() -> None:
    # given
    data = {
        "type": "roboflow_core/contrast_equalization@v1",
        "name": "contrast1",
        "image": "invalid",
        "equalization_type": "Contrast Stretching",
    }

    # when
    with pytest.raises(ValidationError):
        _ = ContrastEqualizationManifest.model_validate(data)


def test_contrast_equalization_block() -> None:
    # given
    block = ContrastEqualizationBlockV1()

    start_image = np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)

    output = block.run(
        image=WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="some"),
            numpy_image=start_image,
        ),
        equalization_type="Contrast Stretching",
    )

    assert output is not None
    assert "image" in output
    assert hasattr(output.get("image"), "numpy_image")

    # dimensions of output match input
    assert output.get("image").numpy_image.shape == (1000, 1000, 3)
    # check if the image is modified
    assert not np.array_equal(output.get("image").numpy_image, start_image)

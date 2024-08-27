import numpy as np
import pytest
from pydantic import ValidationError

from inference.core.workflows.core_steps.classical_cv.threshold.v1 import (
    ImageThresholdBlockV1,
    ImageThresholdManifest,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)


@pytest.mark.parametrize("images_field_alias", ["images", "image"])
def test_threshold_validation_when_valid_manifest_is_given(
    images_field_alias: str,
) -> None:
    # given
    data = {
        "type": "roboflow_core/threshold@v1",
        "name": "threshold1",
        images_field_alias: "$inputs.image",
        "threshold_type": "binary",
        "thresh_value": 210,
        "max_value": 255,
    }

    # when
    result = ImageThresholdManifest.model_validate(data)

    # then
    assert result == ImageThresholdManifest(
        type="roboflow_core/threshold@v1",
        name="threshold1",
        image="$inputs.image",
        threshold_type="binary",
        thresh_value=210,
        max_value=255,
    )


def test_threshold_validation_when_invalid_image_is_given() -> None:
    # given
    data = {
        "type": "roboflow_core/threshold@v1",
        "name": "threshold1",
        "image": "invalid",
        "threshold_type": "binary",
        "thresh_value": 210,
        "max_value": 255,
    }

    # when
    with pytest.raises(ValidationError):
        _ = ImageThresholdManifest.model_validate(data)


def test_threshold_block(dogs_image: np.ndarray) -> None:
    # given
    block = ImageThresholdBlockV1()

    output = block.run(
        image=WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="some"),
            numpy_image=dogs_image,
        ),
        threshold_type="binary",
        thresh_value=210,
        max_value=255,
    )

    assert output is not None
    assert "image" in output
    assert hasattr(output.get("image"), "numpy_image")

    # dimensions of output must be 3 dimensional
    assert output.get("image").numpy_image.shape == dogs_image.shape
    # check if the image is modified
    assert not np.array_equal(output.get("image").numpy_image, dogs_image)

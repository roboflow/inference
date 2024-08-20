import numpy as np
import pytest
from pydantic import ValidationError

from inference.core.workflows.core_steps.classical_cv.sift.v1 import (
    SIFTBlockV1,
    SIFTDetectionManifest,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)


@pytest.mark.parametrize("images_field_alias", ["images", "image"])
def test_sift_validation_when_valid_manifest_is_given(images_field_alias: str) -> None:
    # given
    data = {
        "type": "roboflow_core/sift@v1",
        "name": "sift1",
        images_field_alias: "$inputs.image",
    }

    # when
    result = SIFTDetectionManifest.model_validate(data)

    # then
    assert result == SIFTDetectionManifest(
        type="roboflow_core/sift@v1",
        name="sift1",
        image="$inputs.image",
    )


def test_sift_validation_when_invalid_image_is_given() -> None:
    # given
    data = {
        "type": "roboflow_core/sift@v1",  # Correct type
        "name": "sift1",
        "image": "invalid",
    }

    # when
    with pytest.raises(ValidationError):
        _ = SIFTDetectionManifest.model_validate(data)


def test_sift_block(dogs_image: np.ndarray) -> None:
    # given
    block = SIFTBlockV1()

    # when
    output = block.run(
        image=WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="some"),
            numpy_image=dogs_image,
        )
    )

    # then
    assert "image" in output
    assert isinstance(output["image"], WorkflowImageData)
    assert output.get("image").numpy_image.shape == (427, 640, 3)
    assert "keypoints" in output
    assert isinstance(output["keypoints"], list)
    assert "descriptors" in output
    assert isinstance(output["descriptors"], np.ndarray)

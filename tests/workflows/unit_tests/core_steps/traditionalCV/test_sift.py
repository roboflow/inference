import numpy as np
import pytest
from pydantic import ValidationError

from inference.core.workflows.core_steps.traditional.sift.v1 import (
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
        "type": "SIFT",  # Correct type
        "name": "sift1",
        images_field_alias: "$inputs.image",
    }

    # when
    result = SIFTDetectionManifest.model_validate(data)

    # then
    assert result == SIFTDetectionManifest(
        type="SIFT",
        name="sift1",
        image="$inputs.image",
    )


def test_sift_validation_when_invalid_image_is_given() -> None:
    # given
    data = {
        "type": "SIFT",  # Correct type
        "name": "sift1",
        "image": "invalid",
    }

    # when
    with pytest.raises(ValidationError):
        _ = SIFTDetectionManifest.model_validate(data)


@pytest.mark.asyncio
async def test_sift_block() -> None:
    # given
    block = SIFTBlockV1()

    output = block.run(
        image=WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="some"),
            numpy_image=np.zeros((1000, 1000, 3), dtype=np.uint8),
        )
    )

    assert output is not None
    assert "image" in output
    assert hasattr(output.get("image"), "numpy_image")

    # dimensions of output match input
    assert output.get("image").numpy_image.shape == (1000, 1000, 3)
    # check if the image is modified
    # check if keypoints and descriptors are present
    assert "keypoints" in output
    assert isinstance(output["keypoints"], list)
    assert "descriptors" in output
    if output["descriptors"] is not None:
        assert isinstance(output["descriptors"], np.ndarray)
    else:
        assert output["descriptors"] is None

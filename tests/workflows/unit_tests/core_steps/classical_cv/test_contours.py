import numpy as np
import pytest
from pydantic import ValidationError

from inference.core.workflows.core_steps.classical_cv.contours.v1 import (
    ImageContoursDetectionManifest,
    ImageContoursDetectionBlockV1,
)

from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)

@pytest.mark.parametrize("images_field_alias", ["images", "image"])
def test_contours_validation_when_valid_manifest_is_given(images_field_alias: str) -> None:
    # given
    data = {
      "type": "roboflow_core/contours_detection@v1",
      "name": "image_contours",
      images_field_alias: "$inputs.image",
      "raw_image": "$inputs.image",
    }

    # when
    result = ImageContoursDetectionManifest.model_validate(data)

    # then
    assert result == ImageContoursDetectionManifest(
        type="roboflow_core/contours_detection@v1",
        name="image_contours",
        image="$inputs.image",
        raw_image="$inputs.image",
        line_thickness=3,
    )

def test_contours_validation_when_invalid_image_is_given() -> None:
    # given
    data = {
      "type": "roboflow_core/contours_detection@v1",
      "name": "image_contours",
      "image": "invalid",
      "raw_image": "$inputs.image",
      "line_thickness": 3,
    }

    # when
    with pytest.raises(ValidationError):
        _ = ImageContoursDetectionManifest.model_validate(data)


def test_contours_block() -> None:
    # given
    block = ImageContoursDetectionBlockV1()

    start_image = np.random.randint(0, 255, (1000, 1000, 1), dtype=np.uint8)

    output = block.run(
        image=WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="some"),
            numpy_image=start_image,
        ),
        raw_image=WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="some"),
            numpy_image=start_image,
        ),
        line_thickness=3,
    )

    assert output is not None
    assert "image" in output
    assert hasattr(output.get("image"), "numpy_image")

    # dimensions of output must be 3 dimensional
    assert output.get("image").numpy_image.shape == (1000, 1000, 3)
    # check if the image is modified
    assert not np.array_equal(output.get("image").numpy_image, start_image)
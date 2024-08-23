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
    }

    # when
    result = ImageContoursDetectionManifest.model_validate(data)

    # then
    assert result == ImageContoursDetectionManifest(
        type="roboflow_core/contours_detection@v1",
        name="image_contours",
        image="$inputs.image",
    )

def test_contours_validation_when_invalid_image_is_given() -> None:
    # given
    data = {
      "type": "roboflow_core/contours_detection@v1",
      "name": "image_contours",
      "image": "invalid",
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
    )

    assert output is not None
    assert "number_contours" in output
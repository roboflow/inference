import numpy as np
import pytest
from pydantic import ValidationError

from inference.core.workflows.core_steps.classical_cv.contours.v1 import (
    ImageContoursDetectionBlockV1,
    ImageContoursDetectionManifest,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)


@pytest.mark.parametrize("images_field_alias", ["images", "image"])
def test_contours_validation_when_valid_manifest_is_given(
    images_field_alias: str,
) -> None:
    # given
    data = {
        "type": "roboflow_core/contours_detection@v1",
        "name": "image_contours",
        images_field_alias: "$inputs.image",
        "line_thickness": 3,
    }

    # when
    result = ImageContoursDetectionManifest.model_validate(data)

    # then
    assert result == ImageContoursDetectionManifest(
        type="roboflow_core/contours_detection@v1",
        name="image_contours",
        image="$inputs.image",
        line_thickness=3,
    )


def test_contours_validation_when_invalid_image_is_given() -> None:
    # given
    data = {
        "type": "roboflow_core/contours_detection@v1",
        "name": "image_contours",
        "image": "invalid",
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
        line_thickness=3,
    )

    assert output is not None
    assert "contours" in output
    assert "hierarchy" in output
    assert "number_contours" in output

    assert output["hierarchy"].size > 0
    assert output["number_contours"] > 0

import numpy as np
import pytest
from pydantic import ValidationError

from inference.core.workflows.core_steps.classical_cv.camera_focus.v1 import (
    CameraFocusBlockV1,
    CameraFocusManifest,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)


@pytest.mark.parametrize("images_field_alias", ["images", "image"])
def test_camera_focus_validation_when_valid_manifest_is_given(
    images_field_alias: str,
) -> None:
    # given
    data = {
        "type": "roboflow_core/camera_focus@v1",
        "name": "camera_focus",
        images_field_alias: "$inputs.image",
    }

    # when
    result = CameraFocusManifest.model_validate(data)

    # then
    assert result == CameraFocusManifest(
        type="roboflow_core/camera_focus@v1",
        name="camera_focus",
        image="$inputs.image",
    )


def test_camera_focus_validation_when_invalid_image_is_given() -> None:
    # given
    data = {
        "type": "roboflow_core/camera_focus@v1",
        "name": "image_contours",
        "image": "invalid",
    }

    # when
    with pytest.raises(ValidationError):
        _ = CameraFocusManifest.model_validate(data)


def test_camera_focus_block(dogs_image: np.ndarray) -> None:
    # given
    block = CameraFocusBlockV1()

    start_image = dogs_image

    output = block.run(
        image=WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="some"),
            numpy_image=start_image,
        ),
    )

    assert output is not None
    assert "focus_measure" in output
    assert output["focus_measure"] >= 0

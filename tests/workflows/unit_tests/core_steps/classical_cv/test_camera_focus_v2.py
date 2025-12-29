import numpy as np
import pytest
import supervision as sv
from pydantic import ValidationError

from inference.core.workflows.core_steps.classical_cv.camera_focus.v2 import (
    CameraFocusBlockV2,
    CameraFocusManifest,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)


@pytest.mark.parametrize("images_field_alias", ["images", "image"])
def test_camera_focus_v2_validation_when_valid_manifest_is_given(
    images_field_alias: str,
) -> None:
    data = {
        "type": "roboflow_core/camera_focus@v2",
        "name": "camera_focus",
        images_field_alias: "$inputs.image",
    }

    result = CameraFocusManifest.model_validate(data)

    assert result == CameraFocusManifest(
        type="roboflow_core/camera_focus@v2",
        name="camera_focus",
        image="$inputs.image",
    )


def test_camera_focus_v2_validation_when_invalid_image_is_given() -> None:
    data = {
        "type": "roboflow_core/camera_focus@v2",
        "name": "image_contours",
        "image": "invalid",
    }

    with pytest.raises(ValidationError):
        _ = CameraFocusManifest.model_validate(data)


def test_camera_focus_v2_block(dogs_image: np.ndarray) -> None:
    block = CameraFocusBlockV2()

    start_image = dogs_image

    output = block.run(
        image=WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="some"),
            numpy_image=start_image,
        ),
        underexposed_threshold_percent=3.0,
        overexposed_threshold_percent=97.0,
        show_zebra_warnings=True,
        grid_overlay="3x3",
        show_hud=True,
        show_focus_peaking=True,
        show_center_marker=True,
        detections=None,
    )

    assert output is not None
    assert "focus_measure" in output
    assert output["focus_measure"] >= 0
    assert "bbox_focus_measures" in output
    assert output["bbox_focus_measures"] == []


def test_camera_focus_v2_block_with_detections(dogs_image: np.ndarray) -> None:
    block = CameraFocusBlockV2()

    start_image = dogs_image

    detections = sv.Detections(
        xyxy=np.array([
            [10, 10, 100, 100],
            [150, 150, 300, 300],
        ]),
    )

    output = block.run(
        image=WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="some"),
            numpy_image=start_image,
        ),
        underexposed_threshold_percent=3.0,
        overexposed_threshold_percent=97.0,
        show_zebra_warnings=True,
        grid_overlay="3x3",
        show_hud=True,
        show_focus_peaking=True,
        show_center_marker=True,
        detections=detections,
    )

    assert output is not None
    assert "focus_measure" in output
    assert output["focus_measure"] >= 0
    assert "bbox_focus_measures" in output
    assert len(output["bbox_focus_measures"]) == 2
    assert all(fm >= 0 for fm in output["bbox_focus_measures"])

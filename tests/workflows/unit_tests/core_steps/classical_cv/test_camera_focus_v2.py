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


def test_camera_focus_v2_block_returns_same_image_when_all_visualizations_disabled(
    dogs_image: np.ndarray,
) -> None:
    block = CameraFocusBlockV2()

    input_image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="some"),
        numpy_image=dogs_image,
    )

    output = block.run(
        image=input_image,
        underexposed_threshold_percent=3.0,
        overexposed_threshold_percent=97.0,
        show_zebra_warnings=False,
        grid_overlay="None",
        show_hud=False,
        show_focus_peaking=False,
        show_center_marker=False,
        detections=None,
    )

    assert output is not None
    assert output["image"] is input_image
    assert "focus_measure" in output
    assert output["focus_measure"] >= 0
    assert output["bbox_focus_measures"] == []


def test_camera_focus_v2_block_with_grayscale_image() -> None:
    block = CameraFocusBlockV2()
    gray_image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)

    output = block.run(
        image=WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="some"),
            numpy_image=gray_image,
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

    assert output["focus_measure"] >= 0
    assert output["image"].numpy_image.shape == (100, 100, 3)


def test_camera_focus_v2_block_with_small_image() -> None:
    block = CameraFocusBlockV2()
    small_image = np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)

    output = block.run(
        image=WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="some"),
            numpy_image=small_image,
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

    assert output["focus_measure"] >= 0
    assert output["bbox_focus_measures"] == []


def test_camera_focus_v2_block_with_out_of_bounds_detections() -> None:
    block = CameraFocusBlockV2()
    image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

    detections = sv.Detections(
        xyxy=np.array([
            [-50, -50, 50, 50],
            [80, 80, 200, 200],
        ]),
    )

    output = block.run(
        image=WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="some"),
            numpy_image=image,
        ),
        underexposed_threshold_percent=3.0,
        overexposed_threshold_percent=97.0,
        show_zebra_warnings=False,
        grid_overlay="None",
        show_hud=False,
        show_focus_peaking=False,
        show_center_marker=False,
        detections=detections,
    )

    assert output["focus_measure"] >= 0
    assert len(output["bbox_focus_measures"]) == 2
    assert all(fm >= 0 for fm in output["bbox_focus_measures"] if fm is not None)


def test_camera_focus_v2_block_with_completely_out_of_bounds_detections() -> None:
    block = CameraFocusBlockV2()
    image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

    detections = sv.Detections(
        xyxy=np.array([
            [200, 200, 300, 300],
            [-100, -100, -50, -50],
            [10, 10, 50, 50],
        ]),
    )

    output = block.run(
        image=WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="some"),
            numpy_image=image,
        ),
        underexposed_threshold_percent=3.0,
        overexposed_threshold_percent=97.0,
        show_zebra_warnings=False,
        grid_overlay="None",
        show_hud=False,
        show_focus_peaking=False,
        show_center_marker=False,
        detections=detections,
    )

    assert output["focus_measure"] >= 0
    assert len(output["bbox_focus_measures"]) == 3
    assert output["bbox_focus_measures"][0] is None
    assert output["bbox_focus_measures"][1] is None
    assert output["bbox_focus_measures"][2] is not None
    assert output["bbox_focus_measures"][2] >= 0

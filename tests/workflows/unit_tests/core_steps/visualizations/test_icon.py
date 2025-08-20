import numpy as np
import pytest
import supervision as sv
from pydantic import ValidationError

from inference.core.workflows.core_steps.visualizations.icon.v1 import (
    IconManifest,
    IconVisualizationBlockV1,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)


@pytest.mark.parametrize(
    "type_alias", ["roboflow_core/icon_visualization@v1", "IconVisualization"]
)
@pytest.mark.parametrize("images_field_alias", ["images", "image"])
def test_icon_validation_when_valid_manifest_is_given_for_dynamic_mode(
    type_alias: str, images_field_alias: str
) -> None:
    # given
    data = {
        "type": type_alias,
        "name": "icon1",
        images_field_alias: "$inputs.image",
        "mode": "dynamic",
        "icon": "$inputs.icon_image",
        "predictions": "$steps.od_model.predictions",
        "icon_width": 64,
        "icon_height": 64,
        "position": "TOP_CENTER",
    }

    # when
    result = IconManifest.model_validate(data)

    # then
    assert result == IconManifest(
        type=type_alias,
        name="icon1",
        images="$inputs.image",
        mode="dynamic",
        icon="$inputs.icon_image",
        predictions="$steps.od_model.predictions",
        icon_width=64,
        icon_height=64,
        position="TOP_CENTER",
    )


def test_icon_validation_when_valid_manifest_is_given_for_static_mode() -> None:
    # given
    data = {
        "type": "IconVisualization",
        "name": "icon1",
        "image": "$inputs.image",
        "mode": "static",
        "icon": "$inputs.icon_image",
        "icon_width": 100,
        "icon_height": 100,
        "x_position": -150,
        "y_position": -150,
    }

    # when
    result = IconManifest.model_validate(data)

    # then
    assert result == IconManifest(
        type="IconVisualization",
        name="icon1",
        images="$inputs.image",
        mode="static",
        icon="$inputs.icon_image",
        icon_width=100,
        icon_height=100,
        x_position=-150,
        y_position=-150,
    )


def test_icon_validation_when_dynamic_mode_missing_predictions() -> None:
    # given
    data = {
        "type": "IconVisualization",
        "name": "icon1",
        "image": "$inputs.image",
        "mode": "dynamic",
        "icon": "$inputs.icon_image",
        "icon_width": 64,
        "icon_height": 64,
        # Missing predictions
    }

    # when
    with pytest.raises(ValidationError):
        _ = IconManifest.model_validate(data)


def test_icon_validation_when_dynamic_mode_with_default_position() -> None:
    # given
    data = {
        "type": "IconVisualization",
        "name": "icon1",
        "image": "$inputs.image",
        "mode": "dynamic",
        "icon": "$inputs.icon_image",
        "predictions": "$steps.od_model.predictions",
        "icon_width": 64,
        "icon_height": 64,
        # Position should use default value
    }

    # when
    result = IconManifest.model_validate(data)
    
    # then
    assert result.position == "TOP_CENTER"  # Check default value is used


def test_icon_validation_when_invalid_image_is_given() -> None:
    # given
    data = {
        "type": "IconVisualization",
        "name": "icon1",
        "images": "invalid",
        "mode": "dynamic",
        "icon": "$inputs.icon_image",
        "predictions": "$steps.od_model.predictions",
        "icon_width": 64,
        "icon_height": 64,
        "position": "TOP_CENTER",
    }

    # when
    with pytest.raises(ValidationError):
        _ = IconManifest.model_validate(data)


def test_icon_visualization_block_static_mode():
    # given
    block = IconVisualizationBlockV1()
    
    # Create test images
    test_image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="test"),
        numpy_image=np.zeros((1000, 1000, 3), dtype=np.uint8),
    )
    
    # Create test icon (red square)
    test_icon_np = np.zeros((32, 32, 3), dtype=np.uint8)
    test_icon_np[:, :, 2] = 255  # Make it red
    test_icon = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="icon"),
        numpy_image=test_icon_np,
    )

    output = block.run(
        image=test_image,
        copy_image=True,
        mode="static",
        icon=test_icon,
        predictions=None,
        icon_width=32,
        icon_height=32,
        position=None,
        x_position=100,
        y_position=100,
    )

    assert output is not None
    assert "image" in output
    assert hasattr(output.get("image"), "numpy_image")

    # dimensions of output match input
    assert output.get("image").numpy_image.shape == (1000, 1000, 3)
    # check if the image is modified
    assert not np.array_equal(
        output.get("image").numpy_image, np.zeros((1000, 1000, 3), dtype=np.uint8)
    )


def test_icon_visualization_block_dynamic_mode():
    # given
    block = IconVisualizationBlockV1()
    
    # Create test images
    test_image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="test"),
        numpy_image=np.zeros((1000, 1000, 3), dtype=np.uint8),
    )
    
    # Create test icon (blue square)
    test_icon_np = np.zeros((32, 32, 3), dtype=np.uint8)
    test_icon_np[:, :, 0] = 255  # Make it blue
    test_icon = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="icon"),
        numpy_image=test_icon_np,
    )

    output = block.run(
        image=test_image,
        copy_image=True,
        mode="dynamic",
        icon=test_icon,
        predictions=sv.Detections(
            xyxy=np.array(
                [[0, 0, 20, 20], [80, 80, 120, 120], [450, 450, 550, 550]],
                dtype=np.float64,
            ),
            class_id=np.array([1, 1, 1]),
        ),
        icon_width=32,
        icon_height=32,
        position="TOP_CENTER",
        x_position=None,
        y_position=None,
    )

    assert output is not None
    assert "image" in output
    assert hasattr(output.get("image"), "numpy_image")

    # dimensions of output match input
    assert output.get("image").numpy_image.shape == (1000, 1000, 3)
    # check if the image is modified
    assert not np.array_equal(
        output.get("image").numpy_image, np.zeros((1000, 1000, 3), dtype=np.uint8)
    )


def test_icon_visualization_block_static_mode_negative_positioning():
    # given
    block = IconVisualizationBlockV1()
    
    # Create test images
    test_image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="test"),
        numpy_image=np.zeros((1000, 1000, 3), dtype=np.uint8),
    )
    
    # Create test icon (green square)
    test_icon_np = np.zeros((50, 50, 3), dtype=np.uint8)
    test_icon_np[:, :, 1] = 255  # Make it green
    test_icon = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="icon"),
        numpy_image=test_icon_np,
    )

    output = block.run(
        image=test_image,
        copy_image=True,
        mode="static",
        icon=test_icon,
        predictions=None,
        icon_width=50,
        icon_height=50,
        position=None,
        x_position=-100,  # 100px from right edge
        y_position=-100,  # 100px from bottom edge
    )

    assert output is not None
    assert "image" in output
    assert hasattr(output.get("image"), "numpy_image")

    # dimensions of output match input
    assert output.get("image").numpy_image.shape == (1000, 1000, 3)
    # check if the image is modified (icon should be placed in bottom-right area)
    assert not np.array_equal(
        output.get("image").numpy_image, np.zeros((1000, 1000, 3), dtype=np.uint8)
    )


def test_icon_validation_when_static_mode_with_defaults() -> None:
    # given
    data = {
        "type": "IconVisualization",
        "name": "icon1",
        "image": "$inputs.image",
        "mode": "static",
        "icon": "$inputs.icon_image",
        # Should use default x_position=10, y_position=10
    }

    # when
    result = IconManifest.model_validate(data)
    
    # then
    assert result.x_position == 10
    assert result.y_position == 10

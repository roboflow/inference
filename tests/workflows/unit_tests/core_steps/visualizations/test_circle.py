import numpy as np
import pytest
import supervision as sv
from pydantic import ValidationError

from inference.core.workflows.core_steps.visualizations.circle.v1 import (
    CircleManifest,
    CircleVisualizationBlockV1,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)


@pytest.mark.parametrize(
    "type_alias", ["roboflow_core/circle_visualization@v1", "CircleVisualization"]
)
@pytest.mark.parametrize("images_field_alias", ["images", "image"])
def test_circle_validation_when_valid_manifest_is_given(
    type_alias: str,
    images_field_alias: str,
) -> None:
    # given
    data = {
        "type": type_alias,
        "name": "circle1",
        "predictions": "$steps.od_model.predictions",
        images_field_alias: "$inputs.image",
        "thickness": 10,
    }

    # when
    result = CircleManifest.model_validate(data)

    # then
    assert result == CircleManifest(
        type=type_alias,
        name="circle1",
        images="$inputs.image",
        predictions="$steps.od_model.predictions",
        thickness=10,
    )


def test_circle_validation_when_invalid_image_is_given() -> None:
    # given
    data = {
        "type": "CircleVisualization",
        "name": "circle1",
        "images": "invalid",
        "predictions": "$steps.od_model.predictions",
        "thickness": 10,
    }

    # when
    with pytest.raises(ValidationError):
        _ = CircleManifest.model_validate(data)


def test_circle_visualization_block() -> None:
    # given
    block = CircleVisualizationBlockV1()

    output = block.run(
        image=WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="some"),
            numpy_image=np.zeros((1000, 1000, 3), dtype=np.uint8),
        ),
        predictions=sv.Detections(
            xyxy=np.array(
                [[0, 0, 20, 20], [80, 80, 120, 120], [450, 450, 550, 550]],
                dtype=np.float64,
            ),
            class_id=np.array([1, 1, 1]),
        ),
        copy_image=True,
        color_palette="DEFAULT",
        palette_size=10,
        custom_colors=None,
        color_axis="CLASS",
        thickness=10,
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

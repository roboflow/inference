import numpy as np
import pytest
import supervision as sv
from pydantic import ValidationError

from inference.core.workflows.core_steps.visualizations.polygon.v1 import (
    PolygonManifest,
    PolygonVisualizationBlockV1,
)
from inference.core.workflows.core_steps.visualizations.polygon.v2 import (
    PolygonVisualizationBlockV2,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)


@pytest.mark.parametrize(
    "type_alias", ["roboflow_core/polygon_visualization@v1", "PolygonVisualization"]
)
@pytest.mark.parametrize("images_field_alias", ["images", "image"])
def test_polygon_validation_when_valid_manifest_is_given(
    type_alias: str,
    images_field_alias: str,
) -> None:
    # given
    data = {
        "type": type_alias,
        "name": "polygon1",
        "predictions": "$steps.od_model.predictions",
        images_field_alias: "$inputs.image",
        "thickness": 2,
    }

    # when
    result = PolygonManifest.model_validate(data)

    # then
    assert result == PolygonManifest(
        type=type_alias,
        name="polygon1",
        images="$inputs.image",
        predictions="$steps.od_model.predictions",
        thickness=2,
    )


def test_polygon_validation_when_invalid_image_is_given() -> None:
    # given
    data = {
        "type": "PolygonVisualization",
        "name": "polygon1",
        "images": "invalid",
        "predictions": "$steps.od_model.predictions",
        "thickness": 2,
    }

    # when
    with pytest.raises(ValidationError):
        _ = PolygonManifest.model_validate(data)


def test_polygon_visualization_block_v1() -> None:
    # given
    block = PolygonVisualizationBlockV1()

    mask = np.zeros((3, 1000, 1000), dtype=np.bool_)
    mask[0, 0:20, 0:20] = True
    mask[1, 80:120, 80:120] = True
    mask[2, 450:550, 450:550] = True

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
            mask=mask,
            class_id=np.array([1, 1, 1]),
        ),
        copy_image=True,
        color_palette="tab10",
        palette_size=10,
        custom_colors=["#FF0000", "#00FF00", "#0000FF"],
        color_axis="CLASS",
        thickness=2,
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


def test_polygon_visualization_block_v2() -> None:
    # given
    block = PolygonVisualizationBlockV2()

    mask = np.zeros((3, 1000, 1000), dtype=np.bool_)
    mask[0, 0:20, 0:20] = True
    mask[1, 80:120, 80:120] = True
    mask[2, 450:550, 450:550] = True

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
            mask=mask,
            class_id=np.array([1, 1, 1]),
        ),
        copy_image=True,
        color_palette="tab10",
        palette_size=10,
        custom_colors=["#FF0000", "#00FF00", "#0000FF"],
        color_axis="CLASS",
        thickness=2,
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

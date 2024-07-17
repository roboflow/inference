import numpy as np
import pytest
import supervision as sv
from pydantic import ValidationError

from inference.core.workflows.core_steps.visualizations.bounding_box import (
    BoundingBoxManifest,
    BoundingBoxVisualizationBlock,
)

from inference.core.workflows.entities.base import (
    WorkflowImageData,
)


@pytest.mark.parametrize("images_field_alias", ["images", "image"])
def test_bounding_box_validation_when_valid_manifest_is_given(images_field_alias: str) -> None:
    # given
    data = {
      "type": "BoundingBoxVisualization",
      "name": "square1",
      "predictions": "$steps.od_model.predictions",
      images_field_alias: "$inputs.image",
      "thickness": 1,
      "roundness": 0
    }

    # when
    result = BoundingBoxManifest.model_validate(data)

    # then
    assert result == BoundingBoxManifest(
        type="BoundingBoxVisualization",
        name="square1",
        images="$inputs.image",
        predictions="$steps.od_model.predictions",
        thickness=1,
        roundness=0
    )

def test_bounding_box_validation_when_invalid_image_is_given() -> None:
    # given
    data = {
        "type": "BoundingBoxVisualization",
        "name": "square1",
        "images": "invalid",
        "predictions": "$steps.od_model.predictions",
        "thickness": 1,
        "roundness": 0
    }

    # when
    with pytest.raises(ValidationError):
        _ = BoundingBoxManifest.model_validate(data)

async def test_bounding_box_visualization_block() -> None:
    # given
    block = BoundingBoxVisualizationBlock()

    output = await block.run(
        image=WorkflowImageData(
            parent_metadata=None,
            numpy_image=np.zeros((1000, 1000, 3), dtype=np.uint8),
        ),
        predictions=sv.Detections(
            xyxy=np.array(
                [[0, 0, 20, 20], [80, 80, 120, 120], [450, 450, 550, 550]], dtype=np.float64
            ),
            class_id=np.array([1, 1, 1]),
        ),
        copy_image=True,
        color_palette="DEFAULT",
        palette_size=10,
        custom_colors=None,
        color_axis="CLASS",
        thickness=1,
        roundness=0,
    )

    assert output is not None
    assert hasattr(output, "image")
    assert hasattr(output.image, "numpy_image")
    
    # dimensions of output match input
    assert output.image.numpy_image.shape == (1000, 1000, 3)
    # check if the image is modified
    assert not np.array_equal(output.image.numpy_image, np.zeros((1000, 1000, 3), dtype=np.uint8))
    

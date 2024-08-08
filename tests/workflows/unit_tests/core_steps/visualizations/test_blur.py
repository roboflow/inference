import numpy as np
import pytest
import supervision as sv
from pydantic import ValidationError

from inference.core.workflows.core_steps.visualizations.blur.v1 import (
    BlurManifest,
    BlurVisualizationBlockV1,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)


@pytest.mark.parametrize(
    "type_alias", ["roboflow_core/blur_visualization@v1", "BlurVisualization"]
)
@pytest.mark.parametrize("images_field_alias", ["images", "image"])
def test_blur_validation_when_valid_manifest_is_given(
    type_alias: str, images_field_alias: str
) -> None:
    # given
    data = {
        "type": type_alias,
        "name": "blur1",
        "predictions": "$steps.od_model.predictions",
        images_field_alias: "$inputs.image",
        "kernel_size": 5,
    }

    # when
    result = BlurManifest.model_validate(data)

    # then
    assert result == BlurManifest(
        type=type_alias,
        name="blur1",
        images="$inputs.image",
        predictions="$steps.od_model.predictions",
        kernel_size=5,
    )


def test_blur_validation_when_invalid_image_is_given() -> None:
    # given
    data = {
        "type": "BlurVisualization",
        "name": "blur1",
        "images": "invalid",
        "predictions": "$steps.od_model.predictions",
        "kernel_size": 5,
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlurManifest.model_validate(data)


def test_blur_visualization_block() -> None:
    # given
    block = BlurVisualizationBlockV1()

    start_image = np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)
    output = block.run(
        image=WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="some"),
            numpy_image=start_image,
        ),
        predictions=sv.Detections(
            xyxy=np.array(
                [[0, 0, 20, 20], [80, 80, 120, 120], [450, 450, 550, 550]],
                dtype=np.float64,
            ),
            class_id=np.array([1, 1, 1]),
        ),
        copy_image=True,
        kernel_size=5,
    )

    assert output is not None
    assert "image" in output
    assert hasattr(output.get("image"), "numpy_image")

    # dimensions of output match input
    assert output.get("image").numpy_image.shape == (1000, 1000, 3)
    # check if the image is modified
    assert not np.array_equal(output.get("image").numpy_image, start_image)

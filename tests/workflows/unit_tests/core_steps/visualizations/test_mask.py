import numpy as np
import pytest
import supervision as sv
from pydantic import ValidationError

from inference.core.workflows.core_steps.visualizations.mask.v1 import (
    MaskManifest,
    MaskVisualizationBlockV1,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)


@pytest.mark.parametrize(
    "type_alias", ["roboflow_core/mask_visualization@v1", "MaskVisualization"]
)
@pytest.mark.parametrize("images_field_alias", ["images", "image"])
def test_mask_validation_when_valid_manifest_is_given(
    type_alias: str, images_field_alias: str
) -> None:
    # given
    data = {
        "type": type_alias,
        "name": "mask1",
        "predictions": "$steps.od_model.predictions",
        images_field_alias: "$inputs.image",
        "opacity": 0.5,
    }

    # when
    result = MaskManifest.model_validate(data)

    # then
    assert result == MaskManifest(
        type=type_alias,
        name="mask1",
        images="$inputs.image",
        predictions="$steps.od_model.predictions",
        opacity=0.5,
    )


def test_mask_validation_when_invalid_image_is_given() -> None:
    # given
    data = {
        "type": "MaskVisualization",
        "name": "mask1",
        "images": "invalid",
        "predictions": "$steps.od_model.predictions",
        "opacity": 0.5,
    }

    # when
    with pytest.raises(ValidationError):
        _ = MaskManifest.model_validate(data)


def test_mask_visualization_block() -> None:
    # given
    block = MaskVisualizationBlockV1()

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
        color_palette="viridis",
        palette_size=10,
        custom_colors=["#000000", "#FFFFFF"],
        color_axis="CLASS",
        opacity=0.5,
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

import numpy as np
import pytest
import supervision as sv
from pydantic import ValidationError

from inference.core.workflows.core_steps.visualizations.halo.v1 import (
    HaloManifest,
    HaloVisualizationBlockV1,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)


@pytest.mark.parametrize(
    "type_alias", ["roboflow_core/halo_visualization@v1", "HaloVisualization"]
)
@pytest.mark.parametrize("images_field_alias", ["images", "image"])
def test_halo_validation_when_valid_manifest_is_given(
    type_alias: str, images_field_alias: str
) -> None:
    # given
    data = {
        "type": type_alias,
        "name": "halo1",
        "predictions": "$steps.od_model.predictions",
        images_field_alias: "$inputs.image",
        "opacity": 0.8,
        "kernel_size": 40,
    }

    # when
    result = HaloManifest.model_validate(data)

    # then
    assert result == HaloManifest(
        type=type_alias,
        name="halo1",
        images="$inputs.image",
        predictions="$steps.od_model.predictions",
        opacity=0.8,
        kernel_size=40,
    )


def test_halo_validation_when_invalid_image_is_given() -> None:
    # given
    data = {
        "type": "HaloVisualization",
        "name": "halo1",
        "images": "invalid",
        "predictions": "$steps.od_model.predictions",
        "opacity": 0.8,
        "kernel_size": 40,
    }

    # when
    with pytest.raises(ValidationError):
        _ = HaloManifest.model_validate(data)


def test_halo_visualization_block() -> None:
    # given
    block = HaloVisualizationBlockV1()

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
        color_palette="DEFAULT",
        palette_size=10,
        custom_colors=[],
        color_axis="CLASS",
        opacity=0.8,
        kernel_size=40,
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

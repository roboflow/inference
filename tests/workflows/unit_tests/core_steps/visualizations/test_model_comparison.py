import numpy as np
import pytest
import supervision as sv
from pydantic import ValidationError

from inference.core.workflows.core_steps.visualizations.model_comparison.v1 import (
    ModelComparisonManifest,
    ModelComparisonVisualizationBlockV1,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)


@pytest.mark.parametrize("images_field_alias", ["images", "image"])
def test_model_comparison_validation_when_valid_manifest_is_given(
    images_field_alias: str,
) -> None:
    # given
    data = {
        "type": "roboflow_core/model_comparison_visualization@v1",
        "name": "comparison1",
        "predictions_a": "$steps.od_model.predictions",
        "predictions_b": "$steps.od_model.predictions",
        images_field_alias: "$inputs.image",
    }

    # when
    result = ModelComparisonManifest.model_validate(data)

    print(result)

    # then
    assert result == ModelComparisonManifest(
        type="roboflow_core/model_comparison_visualization@v1",
        name="comparison1",
        images="$inputs.image",
        predictions_a="$steps.od_model.predictions",
        predictions_b="$steps.od_model.predictions",
    )


def test_model_comparison_validation_when_invalid_image_is_given() -> None:
    # given
    data = {
        "type": "roboflow_core/model_comparison_visualization@v1",
        "name": "comparison1",
        "images": "invalid",
        "predictions": "$steps.od_model.predictions",
    }

    # when
    with pytest.raises(ValidationError):
        _ = ModelComparisonManifest.model_validate(data)


def test_halo_visualization_block() -> None:
    # given
    block = ModelComparisonVisualizationBlockV1()

    mask_a = np.zeros((3, 1000, 1000), dtype=np.bool_)
    mask_a[0, 0:20, 0:20] = True
    mask_a[1, 80:120, 80:120] = True
    mask_a[2, 450:550, 450:550] = True

    mask_b = np.zeros((3, 1000, 1000), dtype=np.bool_)
    mask_b[0, 10:20, 10:20] = True
    mask_b[1, 80:120, 80:120] = True
    mask_b[2, 450:550, 450:550] = True

    output = block.run(
        image=WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="some"),
            numpy_image=np.zeros((1000, 1000, 3), dtype=np.uint8),
        ),
        predictions_a=sv.Detections(
            xyxy=np.array(
                [[0, 0, 20, 20], [80, 80, 120, 120], [450, 450, 550, 550]],
                dtype=np.float64,
            ),
            mask=mask_a,
            class_id=np.array([1, 1, 1]),
        ),
        color_a="RED",
        predictions_b=sv.Detections(
            xyxy=np.array(
                [[10, 10, 20, 20], [80, 80, 120, 120], [450, 450, 550, 550]],
                dtype=np.float64,
            ),
            mask=mask_b,
            class_id=np.array([1, 1, 1]),
        ),
        color_b="GREEN",
        background_color="BLACK",
        opacity=0.7,
        copy_image=True,
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

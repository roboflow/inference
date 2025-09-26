import numpy as np
import pytest
from pydantic import ValidationError

from inference.core.workflows.core_steps.classical_cv.morphological_transformation.v1 import (
    MorphologicalTransformationBlockV1,
    MorphologicalTransformationManifest,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)


@pytest.mark.parametrize("images_field_alias", ["images", "image"])
def test_morphological_transformation_validation_when_valid_manifest_is_given(
    images_field_alias: str,
) -> None:
    # given
    data = {
        "type": "roboflow_core/morphological_transformation@v1",
        "name": "morph1",
        images_field_alias: "$inputs.image",
        "operation": "Opening",
    }

    # when
    result = MorphologicalTransformationManifest.model_validate(data)
    print(result)

    # then
    assert result == MorphologicalTransformationManifest(
        type="roboflow_core/morphological_transformation@v1",
        name="morph1",
        image="$inputs.image",
        operation="Opening",
    )


def test_morphological_transformation_validation_when_invalid_image_is_given() -> None:
    # given
    data = {
        "type": "roboflow_core/morphological_transformation@v1",
        "name": "morph1",
        "image": "invalid",
        "operation": "Closing",
    }

    # when
    with pytest.raises(ValidationError):
        _ = MorphologicalTransformationManifest.model_validate(data)


def test_morphological_transformation_block() -> None:
    # given
    block = MorphologicalTransformationBlockV1()

    start_image = np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)

    output = block.run(
        image=WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="some"),
            numpy_image=start_image,
        ),
        operation="Erosion",
    )

    assert output is not None
    assert "image" in output
    assert hasattr(output.get("image"), "numpy_image")

    # dimensions of output match input
    assert output.get("image").numpy_image.shape == (1000, 1000, 1)
    # check if the image is modified
    assert not np.array_equal(output.get("image").numpy_image, start_image)

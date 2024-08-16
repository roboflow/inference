import numpy as np
import pytest
from pydantic import ValidationError

from inference.core.workflows.core_steps.classical_cv.template_matching.v1 import (
    TemplateMatchingBlockV1,
    TemplateMatchingManifest,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)


@pytest.mark.parametrize("images_field_alias", ["images", "image"])
def test_template_matching_validation_when_valid_manifest_is_given(
    images_field_alias: str,
) -> None:
    # given
    data = {
        "type": "roboflow_core/template_matching@v1",
        "name": "template_matching1",
        images_field_alias: "$inputs.image",
        "template": "$inputs.template",
        "matching_threshold": 0.8,
        "apply_nms": False,
        "nms_threshold": 0.6,
    }

    # when
    result = TemplateMatchingManifest.model_validate(data)

    # then
    assert result == TemplateMatchingManifest(
        type="roboflow_core/template_matching@v1",
        name="template_matching1",
        image="$inputs.image",
        template="$inputs.template",
        matching_threshold=0.8,
        apply_nms=False,
        nms_threshold=0.6,
    )


def test_template_matching_validation_when_invalid_image_is_given() -> None:
    # given
    data = {
        "type": "roboflow_core/template_matching@v1",
        "name": "template_matching1",
        "image": "invalid",
        "template": "$inputs.template",
        "threshold": 0.8,
    }

    # when
    with pytest.raises(ValidationError):
        _ = TemplateMatchingManifest.model_validate(data)


def test_template_matching_block(dogs_image: np.ndarray) -> None:
    # given
    block = TemplateMatchingBlockV1()
    template = dogs_image[220:280, 310:410]  # dog's head

    # when
    output = block.run(
        image=WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="some"),
            numpy_image=dogs_image,
        ),
        template=WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="some"),
            numpy_image=template,
        ),
        matching_threshold=0.8,
        apply_nms=True,
        nms_threshold=0.5,
    )

    # then
    assert np.allclose(
        output["predictions"].xyxy, np.array([312, 222, 412, 282])
    ), "Expected to find single template match"
    assert output["predictions"].class_id.tolist() == [0], "Expected fixed class id 0"
    assert output["predictions"]["class_name"].tolist() == [
        "template_match"
    ], "Expected fixed class name"
    assert np.allclose(
        output["predictions"].confidence, np.array([1.0])
    ), "Expected fixed confidence"
    assert output["number_of_matches"] == 1, "Expected one match to be reported"

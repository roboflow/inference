import numpy as np
import pytest
from pydantic import ValidationError

from inference.core.workflows.core_steps.traditional.templateMatching.v1 import (
    TemplateMatchingManifest,
    TemplateMatchingBlockV1,
)

from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)

@pytest.mark.parametrize("images_field_alias", ["images", "image"])
def test_template_matching_validation_when_valid_manifest_is_given(images_field_alias: str) -> None:
    # given
    data = {
        "type": "TemplateMatching",  # Correct type
        "name": "template_matching1",
        images_field_alias: "$inputs.image",
        "template": "$inputs.template",
        "threshold": 0.8,
    }

    # when
    result = TemplateMatchingManifest.model_validate(data)

    # then
    assert result == TemplateMatchingManifest(
        type="TemplateMatching",
        name="template_matching1",
        image="$inputs.image",
        template="$inputs.template",
        threshold=0.8,
    )


def test_template_matching_validation_when_invalid_image_is_given() -> None:
    # given
    data = {
        "type": "TemplateMatching",  # Correct type
        "name": "template_matching1",
        "image": "invalid",
        "template": "$inputs.template",
        "threshold": 0.8,
    }

    # when
    with pytest.raises(ValidationError):
        _ = TemplateMatchingManifest.model_validate(data)


@pytest.mark.asyncio
async def test_template_matching_block() -> None:
    # given
    block = TemplateMatchingBlockV1()

    output = block.run(
        image=WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="some"),
            numpy_image=np.zeros((1000, 1000, 3), dtype=np.uint8),
        ),
        template=WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="some"),
            numpy_image=np.zeros((100, 100, 3), dtype=np.uint8),
        ),
        threshold=0.8
    )

    assert output is not None
    assert "image" in output
    assert hasattr(output.get("image"), "numpy_image")
    
    # dimensions of output match input
    assert output.get("image").numpy_image.shape == (1000, 1000, 3)
    # check if the image is modified
    assert not np.array_equal(output.get("image").numpy_image, np.zeros((1000, 1000, 3), dtype=np.uint8))
    # check if num_matches is present and is an integer
    assert "num_matches" in output
    assert isinstance(output["num_matches"], int)
    assert output["num_matches"] >= 0
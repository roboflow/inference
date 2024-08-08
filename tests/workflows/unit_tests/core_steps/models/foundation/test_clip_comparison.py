import pytest
from pydantic import ValidationError

from inference.core.workflows.core_steps.models.foundation.clip_comparison.v1 import (
    BlockManifest,
)


@pytest.mark.parametrize(
    "type_alias", ["roboflow_core/clip_comparison@v1", "ClipComparison"]
)
@pytest.mark.parametrize("images_field_alias", ["images", "image"])
@pytest.mark.parametrize("texts_field_alias", ["texts", "text"])
def test_manifest_parsing_when_data_is_valid(
    type_alias: str,
    images_field_alias: str,
    texts_field_alias: str,
) -> None:
    # given
    data = {
        "type": type_alias,
        "name": "some",
        images_field_alias: "$inputs.some",
        texts_field_alias: "$inputs.classes",
    }

    # when
    result = BlockManifest.model_validate(data)

    # then
    assert result == BlockManifest(
        type=type_alias,
        name="some",
        images="$inputs.some",
        texts="$inputs.classes",
    )


def test_manifest_parsing_when_image_is_invalid() -> None:
    # given
    data = {
        "type": "ClipComparison",
        "name": "some",
        "images": "invalid",
        "texts": "$inputs.classes",
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(data)


def test_manifest_parsing_when_text_is_invalid() -> None:
    # given
    data = {
        "type": "ClipComparison",
        "name": "some",
        "images": "$inputs.some",
        "texts": "invalid",
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(data)

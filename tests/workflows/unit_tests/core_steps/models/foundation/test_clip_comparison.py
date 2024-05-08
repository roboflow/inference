import pytest
from pydantic import ValidationError

from inference.enterprise.workflows.core_steps.models.foundation.clip_comparison import (
    BlockManifest,
)


@pytest.mark.parametrize("images_field_alias", ["images", "image"])
@pytest.mark.parametrize("texts_field_alias", ["texts", "text"])
def test_manifest_parsing_when_data_is_valid(
    images_field_alias: str,
    texts_field_alias: str,
) -> None:
    # given
    data = {
        "type": "ClipComparison",
        "name": "some",
        images_field_alias: "$inputs.some",
        texts_field_alias: "$inputs.classes",
    }

    # when
    result = BlockManifest.model_validate(data)

    # then
    assert result == BlockManifest(
        type="ClipComparison",
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

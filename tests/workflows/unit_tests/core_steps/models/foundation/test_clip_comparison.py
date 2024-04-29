import pytest
from pydantic import ValidationError

from inference.enterprise.workflows.core_steps.models.foundation.clip_comparison import (
    BlockManifest,
)


def test_manifest_parsing_when_data_is_valid() -> None:
    # given
    data = {
        "type": "ClipComparison",
        "name": "some",
        "image": "$inputs.some",
        "text": "$inputs.classes",
    }

    # when
    result = BlockManifest.model_validate(data)

    # then
    assert result == BlockManifest(
        type="ClipComparison",
        name="some",
        image="$inputs.some",
        text="$inputs.classes",
    )


def test_manifest_parsing_when_image_is_invalid() -> None:
    # given
    data = {
        "type": "ClipComparison",
        "name": "some",
        "image": "invalid",
        "text": "$inputs.classes",
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(data)


def test_manifest_parsing_when_text_is_invalid() -> None:
    # given
    data = {
        "type": "ClipComparison",
        "name": "some",
        "image": "$inputs.some",
        "text": "invalid",
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(data)

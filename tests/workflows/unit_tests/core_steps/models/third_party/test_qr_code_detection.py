import pytest
from pydantic import ValidationError

from inference.enterprise.workflows.core_steps.models.third_party.qr_code_detection import (
    BlockManifest,
)


@pytest.mark.parametrize("type_alias", ["QRCodeDetector", "QRCodeDetection"])
def test_manifest_parsing_when_data_is_valid(type_alias: str) -> None:
    # given
    data = {
        "type": type_alias,
        "name": "some",
        "image": "$inputs.image",
    }

    # when
    result = BlockManifest.model_validate(data)

    # then
    assert result == BlockManifest(
        type=type_alias,
        name="some",
        image="$inputs.image",
    )


@pytest.mark.parametrize("type_alias", ["QRCodeDetector", "QRCodeDetection"])
def test_manifest_parsing_when_image_is_invalid_valid(type_alias: str) -> None:
    # given
    data = {
        "type": type_alias,
        "name": "some",
        "image": "invalid",
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(data)

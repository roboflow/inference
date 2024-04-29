from typing import Any

import pytest
from pydantic import ValidationError

from inference.enterprise.workflows.core_steps.transformations.detection_offset import (
    BlockManifest,
)


def test_manifest_parsing_when_valid_data_provided() -> None:
    # given
    data = {
        "type": "DetectionOffset",
        "name": "some",
        "predictions": "$steps.some.predictions",
        "offset_x": "$inputs.offset_x",
        "offset_y": 40,
        "image_metadata": "$steps.some.image",
        "prediction_type": "$steps.some.prediction_type",
    }

    # when
    result = BlockManifest.model_validate(data)

    # then
    assert result == BlockManifest(
        type="DetectionOffset",
        name="some",
        predictions="$steps.some.predictions",
        offset_x="$inputs.offset_x",
        offset_y=40,
        image_metadata="$steps.some.image",
        prediction_type="$steps.some.prediction_type",
    )


@pytest.mark.parametrize(
    "field_name, value",
    [
        ("predictions", "invalid"),
        ("offset_x", -1),
        ("offset_y", -1),
        ("image_metadata", "invalid"),
        ("prediction_type", "invalid"),
    ],
)
def test_manifest_parsing_when_invalid_data_provided(
    field_name: str,
    value: Any,
) -> None:
    # given
    data = {
        "type": "DetectionOffset",
        "name": "some",
        "predictions": "$steps.some.predictions",
        "offset_x": "$inputs.offset_x",
        "offset_y": 40,
        "image_metadata": "$steps.some.image",
        "prediction_type": "$steps.some.prediction_type",
    }
    data[field_name] = value

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(data)

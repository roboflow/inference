from typing import Any

import pytest
from pydantic import ValidationError

from inference.enterprise.workflows.core_steps.transformations.detection_offset import (
    BlockManifest,
)


@pytest.mark.parametrize("offset_width_alias", ["offset_width", "offset_x"])
@pytest.mark.parametrize("offset_height_alias", ["offset_height", "offset_y"])
def test_manifest_parsing_when_valid_data_provided(
    offset_width_alias: str,
    offset_height_alias: str,
) -> None:
    # given
    data = {
        "type": "DetectionOffset",
        "name": "some",
        "predictions": "$steps.some.predictions",
        offset_width_alias: "$inputs.offset_x",
        offset_height_alias: 40,
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
        offset_width="$inputs.offset_x",
        offset_height=40,
        image_metadata="$steps.some.image",
        prediction_type="$steps.some.prediction_type",
    )


@pytest.mark.parametrize(
    "field_name, value",
    [
        ("predictions", "invalid"),
        ("offset_width", -1),
        ("offset_height", -1),
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
        "offset_width": "$inputs.offset_x",
        "offset_height": 40,
        "image_metadata": "$steps.some.image",
        "prediction_type": "$steps.some.prediction_type",
    }
    data[field_name] = value

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(data)

from typing import Any

import pytest
from pydantic import ValidationError

from inference.core.workflows.core_steps.models.roboflow.multi_label_classification.v3 import (
    BlockManifest,
)


@pytest.mark.parametrize("images_field_alias", ["images", "image"])
def test_multi_label_classification_model_validation_when_minimalistic_config_is_provided(
    images_field_alias: str,
) -> None:
    # given
    data = {
        "type": "roboflow_core/roboflow_multi_label_classification_model@v3",
        "name": "some",
        images_field_alias: "$inputs.image",
        "model_id": "some/1",
    }

    # when
    result = BlockManifest.model_validate(data)

    # then
    assert result == BlockManifest(
        type="roboflow_core/roboflow_multi_label_classification_model@v3",
        name="some",
        images="$inputs.image",
        model_id="some/1",
    )


@pytest.mark.parametrize("field", ["type", "name", "images", "model_id"])
def test_multi_label_classification_model_validation_when_required_field_is_not_given(
    field: str,
) -> None:
    # given
    data = {
        "type": "roboflow_core/roboflow_multi_label_classification_model@v3",
        "name": "some",
        "images": "$inputs.image",
        "model_id": "some/1",
    }
    del data[field]

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(data)


def test_multi_label_classification_model_validation_when_invalid_type_provided() -> (
    None
):
    # given
    data = {
        "type": "invalid",
        "name": "some",
        "images": "$inputs.image",
        "model_id": "some/1",
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(data)


def test_multi_label_classification_model_validation_when_model_id_has_invalid_type() -> (
    None
):
    # given
    data = {
        "type": "roboflow_core/roboflow_multi_label_classification_model@v3",
        "name": "some",
        "images": "$inputs.image",
        "model_id": None,
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(data)


def test_multi_label_classification_model_validation_when_active_learning_flag_has_invalid_type() -> (
    None
):
    # given
    data = {
        "type": "roboflow_core/roboflow_multi_label_classification_model@v3",
        "name": "some",
        "images": "$inputs.image",
        "model_id": "some/1",
        "disable_active_learning": "some",
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(data)


def test_multi_label_classification_model_validation_when_custom_mode_missing_custom_confidence() -> (
    None
):
    # given
    data = {
        "type": "roboflow_core/roboflow_multi_label_classification_model@v3",
        "name": "some",
        "images": "$inputs.image",
        "model_id": "some/1",
        "confidence_mode": "custom",
        "custom_confidence": None,
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(data)


@pytest.mark.parametrize("mode", ["best", "default", "custom"])
def test_multi_label_classification_model_accepts_all_confidence_modes(
    mode: str,
) -> None:
    # given
    data = {
        "type": "roboflow_core/roboflow_multi_label_classification_model@v3",
        "name": "some",
        "images": "$inputs.image",
        "model_id": "some/1",
        "confidence_mode": mode,
        "custom_confidence": 0.5 if mode == "custom" else None,
    }

    # when
    result = BlockManifest.model_validate(data)

    # then
    assert result.confidence_mode == mode


@pytest.mark.parametrize(
    "parameter, value",
    [
        ("confidence_mode", "invalid-mode"),
        ("custom_confidence", "some"),
        ("custom_confidence", 1.1),
        ("images", "some"),
        ("disable_active_learning", "some"),
    ],
)
def test_multi_label_classification_model_when_parameters_have_invalid_type(
    parameter: str,
    value: Any,
) -> None:
    # given
    data = {
        "type": "roboflow_core/roboflow_multi_label_classification_model@v3",
        "name": "some",
        "images": "$inputs.image",
        "model_id": "some/1",
        parameter: value,
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(data)

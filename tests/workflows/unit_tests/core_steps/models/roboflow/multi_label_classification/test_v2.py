import pytest
from pydantic import ValidationError

from inference.core.workflows.core_steps.models.roboflow.multi_label_classification.v2 import (
    BlockManifest,
)


@pytest.mark.parametrize("images_field_alias", ["images", "image"])
def test_multi_label_classification_model_validation_when_minimalistic_config_is_provided(
    images_field_alias: str,
) -> None:
    # given
    data = {
        "type": "roboflow_core/roboflow_multi_label_classification_model@v2",
        "name": "some",
        images_field_alias: "$inputs.image",
        "model_id": "some/1",
    }

    # when
    result = BlockManifest.model_validate(data)

    # then
    assert result == BlockManifest(
        type="roboflow_core/roboflow_multi_label_classification_model@v2",
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
        "type": "roboflow_core/roboflow_multi_label_classification_model@v2",
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
        "type": "roboflow_core/roboflow_multi_label_classification_model@v2",
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
        "type": "roboflow_core/roboflow_multi_label_classification_model@v2",
        "name": "some",
        "images": "$inputs.image",
        "model_id": "some/1",
        "disable_active_learning": "some",
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(data)

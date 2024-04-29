import pytest
from pydantic import ValidationError

from inference.enterprise.workflows.core_steps.models.roboflow.multi_class_classification import (
    BlockManifest,
)


@pytest.mark.parametrize(
    "type_alias", ["RoboflowClassificationModel", "ClassificationModel"]
)
def test_classification_model_validation_when_minimalistic_config_is_provided(
    type_alias: str,
) -> None:
    # given
    data = {
        "type": type_alias,
        "name": "some",
        "image": "$inputs.image",
        "model_id": "some/1",
    }

    # when
    result = BlockManifest.model_validate(data)

    # then
    assert result == BlockManifest(
        type=type_alias,
        name="some",
        image="$inputs.image",
        model_id="some/1",
    )


@pytest.mark.parametrize("field", ["type", "name", "image", "model_id"])
@pytest.mark.parametrize(
    "type_alias", ["RoboflowClassificationModel", "ClassificationModel"]
)
def test_classification_model_validation_when_required_field_is_not_given(
    field: str,
    type_alias: str,
) -> None:
    # given
    data = {
        "type": type_alias,
        "name": "some",
        "image": "$inputs.image",
        "model_id": "some/1",
    }
    del data[field]

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(data)


def test_classification_model_validation_when_invalid_type_provided() -> None:
    # given
    data = {
        "type": "invalid",
        "name": "some",
        "image": "$inputs.image",
        "model_id": "some/1",
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(data)


@pytest.mark.parametrize(
    "type_alias", ["RoboflowClassificationModel", "ClassificationModel"]
)
def test_classification_model_validation_when_model_id_has_invalid_type(
    type_alias: str,
) -> None:
    # given
    data = {
        "type": type_alias,
        "name": "some",
        "image": "$inputs.image",
        "model_id": None,
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(data)


@pytest.mark.parametrize(
    "type_alias", ["RoboflowClassificationModel", "ClassificationModel"]
)
def test_classification_model_validation_when_active_learning_flag_has_invalid_type(
    type_alias: str,
) -> None:
    # given
    data = {
        "type": type_alias,
        "name": "some",
        "image": "$inputs.image",
        "model_id": "some/1",
        "disable_active_learning": "some",
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(data)

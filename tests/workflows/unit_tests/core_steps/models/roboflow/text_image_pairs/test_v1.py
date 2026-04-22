import pytest
from pydantic import ValidationError

from inference.core.workflows.core_steps.models.roboflow.text_image_pairs.v1 import (
    BlockManifest,
)


@pytest.mark.parametrize("images_field_alias", ["images", "image"])
def test_text_image_pairs_model_validation_when_minimalistic_config_is_provided(
    images_field_alias: str,
) -> None:
    # given
    data = {
        "type": "roboflow_core/roboflow_text_image_pairs_model@v1",
        "name": "some",
        images_field_alias: "$inputs.image",
        "model_id": "some/1",
    }

    # when
    result = BlockManifest.validate(data)

    # then
    assert result == BlockManifest(
        type="roboflow_core/roboflow_text_image_pairs_model@v1",
        name="some",
        images="$inputs.image",
        model_id="some/1",
    )


def test_text_image_pairs_model_validation_accepts_prompt() -> None:
    # given
    data = {
        "type": "roboflow_core/roboflow_text_image_pairs_model@v1",
        "name": "some",
        "images": "$inputs.image",
        "model_id": "some/1",
        "prompt": "describe this image",
    }

    # when
    result = BlockManifest.validate(data)

    # then
    assert result.prompt == "describe this image"


@pytest.mark.parametrize("field", ["type", "name", "images", "model_id"])
def test_text_image_pairs_model_validation_when_required_field_is_not_given(
    field: str,
) -> None:
    # given
    data = {
        "type": "roboflow_core/roboflow_text_image_pairs_model@v1",
        "name": "some",
        "images": "$inputs.image",
        "model_id": "some/1",
    }
    del data[field]

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.validate(data)


def test_text_image_pairs_model_validation_when_invalid_type_provided() -> None:
    # given
    data = {
        "type": "invalid",
        "name": "some",
        "images": "$inputs.image",
        "model_id": "some/1",
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.validate(data)


def test_text_image_pairs_model_validation_when_model_id_has_invalid_type() -> None:
    # given
    data = {
        "type": "roboflow_core/roboflow_text_image_pairs_model@v1",
        "name": "some",
        "images": "$inputs.image",
        "model_id": None,
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.validate(data)


def test_text_image_pairs_model_describe_outputs_returns_raw_vlm_response() -> None:
    # when
    outputs = BlockManifest.describe_outputs()

    # then
    assert len(outputs) == 1
    assert outputs[0].name == "response"
    assert outputs[0].kind[0].name == "language_model_output"


def test_text_image_pairs_model_is_compatible_with_text_image_pairs_task() -> None:
    # when
    task_types = BlockManifest.get_compatible_task_types()

    # then
    assert task_types == ["text-image-pairs"]

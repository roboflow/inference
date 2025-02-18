from typing import Any

import pytest
from pydantic import ValidationError

from inference.core.workflows.core_steps.models.foundation.lmm.v1 import LMMConfig
from inference.core.workflows.core_steps.models.foundation.lmm_classifier.v1 import (
    BlockManifest,
)


@pytest.mark.parametrize(
    "type_alias", ["roboflow_core/lmm_for_classification@v1", "LMMForClassification"]
)
@pytest.mark.parametrize("images_field_alias", ["images", "image"])
def test_llm_for_classification_step_validation_when_valid_input_given(
    type_alias: str,
    images_field_alias: str,
) -> None:
    # given
    specification = {
        "type": type_alias,
        "name": "step_3",
        images_field_alias: "$steps.step_2.crops",
        "lmm_type": "$inputs.lmm_type",
        "classes": "$inputs.classification_classes",
        "remote_api_key": "$inputs.open_ai_key",
    }

    # when
    result = BlockManifest.model_validate(specification)

    # then
    assert result == BlockManifest(
        type=type_alias,
        name="step_3",
        images="$steps.step_2.crops",
        lmm_type="$inputs.lmm_type",
        classes="$inputs.classification_classes",
        lmm_config=LMMConfig(),
        remote_api_key="$inputs.open_ai_key",
    )


@pytest.mark.parametrize("value", [1, "some", [], True])
def test_llm_for_classification_step_validation_when_invalid_image_given(
    value: Any,
) -> None:
    # given
    specification = {
        "type": "LMMForClassification",
        "name": "step_3",
        "images": value,
        "lmm_type": "$inputs.lmm_type",
        "classes": "$inputs.classification_classes",
        "remote_api_key": "$inputs.open_ai_key",
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(specification)


@pytest.mark.parametrize("value", [1, "some", [], True])
def test_llm_for_classification_step_validation_when_invalid_image_given(
    value: Any,
) -> None:
    # given
    specification = {
        "type": "LMMForClassification",
        "name": "step_3",
        "images": value,
        "lmm_type": "$inputs.lmm_type",
        "classes": "$inputs.classification_classes",
        "remote_api_key": "$inputs.open_ai_key",
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(specification)


@pytest.mark.parametrize("value", ["$inputs.model", "gpt_4v"])
def test_llm_for_classification_step_validation_when_lmm_type_valid(
    value: Any,
) -> None:
    # given
    specification = {
        "type": "LMMForClassification",
        "name": "step_3",
        "images": "$steps.step_2.crops",
        "lmm_type": value,
        "classes": "$inputs.classification_classes",
        "remote_api_key": "$inputs.open_ai_key",
    }

    # when
    result = BlockManifest.model_validate(specification)

    assert result == BlockManifest(
        type="LMMForClassification",
        name="step_3",
        images="$steps.step_2.crops",
        lmm_type=value,
        classes="$inputs.classification_classes",
        lmm_config=LMMConfig(),
        remote_api_key="$inputs.open_ai_key",
    )


@pytest.mark.parametrize("value", ["some", None])
def test_llm_for_classification_step_validation_when_lmm_type_invalid(
    value: Any,
) -> None:
    # given
    specification = {
        "type": "LMMForClassification",
        "name": "step_3",
        "images": "$steps.step_2.crops",
        "lmm_type": value,
        "classes": "$inputs.classification_classes",
        "remote_api_key": "$inputs.open_ai_key",
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(specification)


@pytest.mark.parametrize("value", ["$inputs.classes", ["a"], ["a", "b"]])
def test_llm_for_classification_step_validation_when_classes_field_valid(
    value: Any,
) -> None:
    # given
    specification = {
        "type": "LMMForClassification",
        "name": "step_3",
        "images": "$steps.step_2.crops",
        "lmm_type": "gpt_4v",
        "classes": value,
        "remote_api_key": "$inputs.open_ai_key",
    }

    # when
    result = BlockManifest.model_validate(specification)

    assert result == BlockManifest(
        type="LMMForClassification",
        name="step_3",
        images="$steps.step_2.crops",
        lmm_type="gpt_4v",
        classes=value,
        lmm_config=LMMConfig(),
        remote_api_key="$inputs.open_ai_key",
    )


@pytest.mark.parametrize("value", ["some", None, [], [1, 2]])
def test_llm_for_classification_step_validation_when_lmm_type_invalid(
    value: Any,
) -> None:
    # given
    specification = {
        "type": "LMMForClassification",
        "name": "step_3",
        "images": "$steps.step_2.crops",
        "lmm_type": "gpt_4v",
        "classes": value,
        "remote_api_key": "$inputs.open_ai_key",
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(specification)


@pytest.mark.parametrize("value", ["$inputs.api_key", "some", None])
def test_llm_for_classification_step_validation_when_remote_api_key_field_valid(
    value: Any,
) -> None:
    # given
    specification = {
        "type": "LMMForClassification",
        "name": "step_3",
        "images": "$steps.step_2.crops",
        "lmm_type": "gpt_4v",
        "classes": ["a", "b"],
        "remote_api_key": value,
    }

    # when
    result = BlockManifest.model_validate(specification)

    assert result == BlockManifest(
        type="LMMForClassification",
        name="step_3",
        images="$steps.step_2.crops",
        lmm_type="gpt_4v",
        classes=["a", "b"],
        lmm_config=LMMConfig(),
        remote_api_key=value,
    )


@pytest.mark.parametrize("value", [[], 1])
def test_llm_for_classification_step_validation_when_lmm_type_invalid(
    value: Any,
) -> None:
    # given
    specification = {
        "type": "LMMForClassification",
        "name": "step_3",
        "images": "$steps.step_2.crops",
        "lmm_type": "gpt_4v",
        "classes": ["a", "b"],
        "remote_api_key": value,
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(specification)

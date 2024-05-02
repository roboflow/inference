from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from inference.enterprise.workflows.core_steps.models.foundation.yolo_world import (
    BlockManifest,
)


@pytest.mark.parametrize("field_alias", ["YoloWorldModel", "YoloWorld"])
def test_yolo_world_step_configuration_decoding_when_valid_config_is_given(
    field_alias: str,
) -> None:
    # given
    specification = {
        "type": field_alias,
        "name": "step_1",
        "image": "$inputs.image",
        "class_names": "$inputs.classes",
        "confidence": "$inputs.confidence",
        "version": "s",
    }

    # when
    result = BlockManifest.model_validate(specification)

    # then
    assert result == BlockManifest(
        type=field_alias,
        name="step_1",
        image="$inputs.image",
        class_names="$inputs.classes",
        version="s",
        confidence="$inputs.confidence",
    )


@pytest.mark.parametrize("field_alias", ["YoloWorldModel", "YoloWorld"])
def test_yolo_world_step_configuration_decoding_when_valid_config_is_given_for_old_alias(
    field_alias: str,
) -> None:
    # given
    specification = {
        "type": field_alias,
        "name": "step_1",
        "image": "$inputs.image",
        "class_names": "$inputs.classes",
        "confidence": "$inputs.confidence",
        "version": "s",
    }

    # when
    result = BlockManifest.model_validate(specification)

    # then
    assert result == BlockManifest(
        type=field_alias,
        name="step_1",
        image="$inputs.image",
        class_names="$inputs.classes",
        version="s",
        confidence="$inputs.confidence",
    )


@pytest.mark.parametrize("value", ["some", [], np.zeros((192, 168, 3))])
@pytest.mark.parametrize("field_alias", ["YoloWorldModel", "YoloWorld"])
def test_yolo_world_step_image_validation_when_invalid_image_given(
    value: Any, field_alias: str
) -> None:
    # given
    specification = {
        "type": field_alias,
        "name": "step_1",
        "image": value,
        "class_names": "$inputs.classes",
        "confidence": "$inputs.confidence",
        "version": "s",
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(specification)


@pytest.mark.parametrize("value", ["some", [1, 2], True, 3])
@pytest.mark.parametrize("field_alias", ["YoloWorldModel", "YoloWorld"])
def test_yolo_world_step_image_validation_when_invalid_class_names_given(
    value: Any,
    field_alias: str,
) -> None:
    # given
    specification = {
        "type": field_alias,
        "name": "step_1",
        "image": "$inputs.image",
        "class_names": value,
        "confidence": "$inputs.confidence",
        "version": "s",
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(specification)


@pytest.mark.parametrize("field_alias", ["YoloWorldModel", "YoloWorld"])
def test_yolo_world_step_image_validation_when_valid_class_names_given(
    field_alias: str,
) -> None:
    # given
    specification = {
        "type": field_alias,
        "name": "step_1",
        "image": "$inputs.image",
        "class_names": ["a", "b"],
        "confidence": "$inputs.confidence",
        "version": "s",
    }

    # when
    result = BlockManifest.model_validate(specification)

    # then
    assert result == BlockManifest(
        type=field_alias,
        name="step_1",
        image="$inputs.image",
        class_names=["a", "b"],
        version="s",
        confidence="$inputs.confidence",
    )


@pytest.mark.parametrize("value", ["some", [1, 2], True, 3])
@pytest.mark.parametrize("field_alias", ["YoloWorldModel", "YoloWorld"])
def test_yolo_world_step_image_validation_when_invalid_version_given(
    value: Any,
    field_alias: str,
) -> None:
    # given
    specification = {
        "type": field_alias,
        "name": "step_1",
        "image": "$inputs.image",
        "class_names": ["a", "b"],
        "confidence": "$inputs.confidence",
        "version": value,
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(specification)


@pytest.mark.parametrize("value", ["s", "m", "l", "x", "v2-s", "v2-m", "v2-l", "v2-x"])
@pytest.mark.parametrize("field_alias", ["YoloWorldModel", "YoloWorld"])
def test_yolo_world_step_image_validation_when_valid_version_given(
    value: Any, field_alias: str
) -> None:
    # given
    specification = {
        "type": field_alias,
        "name": "step_1",
        "image": "$inputs.image",
        "class_names": ["a", "b"],
        "confidence": "$inputs.confidence",
        "version": value,
    }

    # when
    result = BlockManifest.model_validate(specification)

    # then
    assert result == BlockManifest(
        type=field_alias,
        name="step_1",
        image="$inputs.image",
        class_names=["a", "b"],
        version=value,
        confidence="$inputs.confidence",
    )


@pytest.mark.parametrize("value", ["some", [1, 2], 3, 1.1, -0.1])
@pytest.mark.parametrize("field_alias", ["YoloWorldModel", "YoloWorld"])
def test_yolo_world_step_image_validation_when_invalid_confidence_given(
    value: Any,
    field_alias: str,
) -> None:
    # given
    specification = {
        "type": field_alias,
        "name": "step_1",
        "image": "$inputs.image",
        "class_names": ["a", "b"],
        "confidence": value,
        "version": "s",
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(specification)


@pytest.mark.parametrize("value", [None, 0.3, 1.0, 0.0])
@pytest.mark.parametrize("field_alias", ["YoloWorldModel", "YoloWorld"])
def test_yolo_world_step_image_validation_when_valid_confidence_given(
    value: Any,
    field_alias: str,
) -> None:
    # given
    specification = {
        "type": field_alias,
        "name": "step_1",
        "image": "$inputs.image",
        "class_names": ["a", "b"],
        "confidence": value,
        "version": "s",
    }

    # when
    result = BlockManifest.model_validate(specification)

    # then
    assert result == BlockManifest(
        type=field_alias,
        name="step_1",
        image="$inputs.image",
        class_names=["a", "b"],
        version="s",
        confidence=value,
    )

from typing import Any

import pytest
from pydantic import ValidationError

from inference.enterprise.workflows.core_steps.models.foundation.lmm import (
    BlockManifest,
    LMMConfig,
)


@pytest.mark.parametrize("images_field_alias", ["images", "image"])
def test_lmm_step_validation_when_input_is_valid(images_field_alias: str) -> None:
    # given
    specification = {
        "type": "LMM",
        "name": "step_1",
        images_field_alias: "$inputs.image",
        "lmm_type": "$inputs.lmm_type",
        "prompt": "$inputs.prompt",
        "json_output": "$inputs.expected_output",
        "remote_api_key": "$inputs.open_ai_key",
    }

    # when
    result = BlockManifest.model_validate(specification)

    # then
    assert result == BlockManifest(
        type="LMM",
        name="step_1",
        images="$inputs.image",
        prompt="$inputs.prompt",
        lmm_type="$inputs.lmm_type",
        lmm_config=LMMConfig(),
        remote_api_key="$inputs.open_ai_key",
        json_output="$inputs.expected_output",
    )


@pytest.mark.parametrize("value", [None, 1, "a", True])
def test_lmm_step_validation_when_image_is_invalid(
    value: Any,
) -> None:
    # given
    specification = {
        "type": "LMM",
        "name": "step_1",
        "images": value,
        "lmm_type": "$inputs.lmm_type",
        "prompt": "$inputs.prompt",
        "json_output": "$inputs.expected_output",
        "remote_api_key": "$inputs.open_ai_key",
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(specification)


def test_lmm_step_validation_when_prompt_is_given_directly() -> None:
    # given
    specification = {
        "type": "LMM",
        "name": "step_1",
        "images": "$inputs.image",
        "lmm_type": "$inputs.lmm_type",
        "prompt": "This is my prompt",
        "json_output": "$inputs.expected_output",
        "remote_api_key": "$inputs.open_ai_key",
    }

    # when
    result = BlockManifest.model_validate(specification)

    # then
    assert result == BlockManifest(
        type="LMM",
        name="step_1",
        images="$inputs.image",
        prompt="This is my prompt",
        lmm_type="$inputs.lmm_type",
        lmm_config=LMMConfig(),
        remote_api_key="$inputs.open_ai_key",
        json_output="$inputs.expected_output",
    )


@pytest.mark.parametrize("value", [None, []])
def test_lmm_step_validation_when_prompt_is_invalid(
    value: Any,
) -> None:
    # given
    specification = {
        "type": "LMM",
        "name": "step_1",
        "images": "$inputs.image",
        "lmm_type": "$inputs.lmm_type",
        "prompt": value,
        "json_output": "$inputs.expected_output",
        "remote_api_key": "$inputs.open_ai_key",
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(specification)


@pytest.mark.parametrize("value", ["$inputs.model", "gpt_4v", "cog_vlm"])
def test_lmm_step_validation_when_lmm_type_valid(
    value: Any,
) -> None:
    # given
    specification = {
        "type": "LMM",
        "name": "step_1",
        "images": "$inputs.image",
        "lmm_type": value,
        "prompt": "$inputs.prompt",
        "json_output": "$inputs.expected_output",
        "remote_api_key": "$inputs.open_ai_key",
    }

    # when
    result = BlockManifest.model_validate(specification)

    assert result == BlockManifest(
        type="LMM",
        name="step_1",
        images="$inputs.image",
        prompt="$inputs.prompt",
        lmm_type=value,
        lmm_config=LMMConfig(),
        remote_api_key="$inputs.open_ai_key",
        json_output="$inputs.expected_output",
    )


@pytest.mark.parametrize("value", ["some", None])
def test_lmm_step_validation_when_lmm_type_invalid(
    value: Any,
) -> None:
    # given
    specification = {
        "type": "LMM",
        "name": "step_1",
        "images": "$inputs.image",
        "lmm_type": value,
        "prompt": "$inputs.prompt",
        "json_output": "$inputs.expected_output",
        "remote_api_key": "$inputs.open_ai_key",
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(specification)


@pytest.mark.parametrize("value", ["$inputs.api_key", "my-api-key", None])
def test_lmm_step_validation_when_remote_api_key_valid(
    value: Any,
) -> None:
    # given
    specification = {
        "type": "LMM",
        "name": "step_1",
        "images": "$inputs.image",
        "lmm_type": "gpt_4v",
        "prompt": "$inputs.prompt",
        "json_output": "$inputs.expected_output",
        "remote_api_key": value,
    }

    # when
    result = BlockManifest.model_validate(specification)

    assert result == BlockManifest(
        type="LMM",
        name="step_1",
        images="$inputs.image",
        prompt="$inputs.prompt",
        lmm_type="gpt_4v",
        lmm_config=LMMConfig(),
        remote_api_key=value,
        json_output="$inputs.expected_output",
    )


@pytest.mark.parametrize(
    "value", [None, "$inputs.some", {"my_field": "my_description"}]
)
def test_lmm_step_validation_when_json_output_valid(
    value: Any,
) -> None:
    # given
    specification = {
        "type": "LMM",
        "name": "step_1",
        "images": "$inputs.image",
        "lmm_type": "gpt_4v",
        "prompt": "$inputs.prompt",
        "json_output": value,
        "remote_api_key": "some",
    }

    # when
    result = BlockManifest.model_validate(specification)

    assert result == BlockManifest(
        type="LMM",
        name="step_1",
        images="$inputs.image",
        prompt="$inputs.prompt",
        lmm_type="gpt_4v",
        lmm_config=LMMConfig(),
        remote_api_key="some",
        json_output=value,
    )


@pytest.mark.parametrize(
    "value",
    [{"my_field": 3}, "some"],
)
def test_lmm_step_validation_when_json_output_invalid(
    value: Any,
) -> None:
    # given
    specification = {
        "type": "LMM",
        "name": "step_1",
        "images": "$inputs.image",
        "lmm_type": "gpt_4v",
        "prompt": "$inputs.prompt",
        "json_output": value,
        "remote_api_key": "some",
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(specification)

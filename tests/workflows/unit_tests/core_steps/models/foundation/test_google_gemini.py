from typing import Any
from unittest.mock import Mock, patch

import pytest
from pydantic import ValidationError

from inference.core.workflows.core_steps.models.foundation.google_gemini.v2 import (
    BlockManifest,
    execute_gemini_request,
    prepare_generation_config,
)


def test_gemini_step_validation_when_input_is_valid() -> None:
    # given
    specification = {
        "type": "roboflow_core/google_gemini@v2",
        "name": "step_1",
        "images": "$inputs.image",
        "task_type": "unconstrained",
        "prompt": "$inputs.prompt",
        "api_key": "$inputs.google_api_key",
    }

    # when
    result = BlockManifest.model_validate(specification)

    # then
    assert result.type == "roboflow_core/google_gemini@v2"
    assert result.name == "step_1"
    assert result.images == "$inputs.image"
    assert result.task_type == "unconstrained"
    assert result.prompt == "$inputs.prompt"
    assert result.api_key == "$inputs.google_api_key"


@pytest.mark.parametrize("value", [None, 1, "a", True])
def test_gemini_step_validation_when_image_is_invalid(value: Any) -> None:
    # given
    specification = {
        "type": "roboflow_core/google_gemini@v2",
        "name": "step_1",
        "images": value,
        "task_type": "unconstrained",
        "prompt": "$inputs.prompt",
        "api_key": "$inputs.google_api_key",
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(specification)


def test_gemini_step_validation_when_prompt_is_given_directly() -> None:
    # given
    specification = {
        "type": "roboflow_core/google_gemini@v2",
        "name": "step_1",
        "images": "$inputs.image",
        "task_type": "unconstrained",
        "prompt": "This is my prompt",
        "api_key": "$inputs.google_api_key",
    }

    # when
    result = BlockManifest.model_validate(specification)

    # then
    assert result.prompt == "This is my prompt"


@pytest.mark.parametrize(
    "model_version",
    ["gemini-3-pro-preview", "gemini-2.5-pro", "gemini-2.0-flash", "$inputs.model"],
)
def test_gemini_step_validation_when_model_version_valid(model_version: str) -> None:
    # given
    specification = {
        "type": "roboflow_core/google_gemini@v2",
        "name": "step_1",
        "images": "$inputs.image",
        "task_type": "caption",
        "api_key": "$inputs.google_api_key",
        "model_version": model_version,
    }

    # when
    result = BlockManifest.model_validate(specification)

    # then
    assert result.model_version == model_version


def test_gemini_step_validation_with_model_alias() -> None:
    # given
    specification = {
        "type": "roboflow_core/google_gemini@v2",
        "name": "step_1",
        "images": "$inputs.image",
        "task_type": "caption",
        "api_key": "$inputs.google_api_key",
        "model_version": "gemini-2.0-flash-exp",  # This is an alias
    }

    # when
    result = BlockManifest.model_validate(specification)

    # then
    # Should be converted to the canonical name
    assert result.model_version == "gemini-2.0-flash"


@pytest.mark.parametrize("thinking_level", ["low", "high", "$inputs.thinking"])
def test_gemini_step_validation_with_thinking_level(thinking_level: str) -> None:
    # given
    specification = {
        "type": "roboflow_core/google_gemini@v2",
        "name": "step_1",
        "images": "$inputs.image",
        "task_type": "caption",
        "api_key": "$inputs.google_api_key",
        "model_version": "gemini-3-pro-preview",
        "thinking_level": thinking_level,
    }

    # when
    result = BlockManifest.model_validate(specification)

    # then
    assert result.thinking_level == thinking_level


@pytest.mark.parametrize("value", ["invalid", 123])
def test_gemini_step_validation_when_thinking_level_invalid(value: Any) -> None:
    # given
    specification = {
        "type": "roboflow_core/google_gemini@v2",
        "name": "step_1",
        "images": "$inputs.image",
        "task_type": "caption",
        "api_key": "$inputs.google_api_key",
        "model_version": "gemini-3-pro-preview",
        "thinking_level": value,
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(specification)


def test_gemini_step_validation_with_temperature() -> None:
    # given
    specification = {
        "type": "roboflow_core/google_gemini@v2",
        "name": "step_1",
        "images": "$inputs.image",
        "task_type": "caption",
        "api_key": "$inputs.google_api_key",
        "model_version": "gemini-2.5-pro",
        "temperature": 0.7,
    }

    # when
    result = BlockManifest.model_validate(specification)

    # then
    assert result.temperature == 0.7


@pytest.mark.parametrize("value", [-0.1, 2.1, "invalid"])
def test_gemini_step_validation_when_temperature_invalid(value: Any) -> None:
    # given
    specification = {
        "type": "roboflow_core/google_gemini@v2",
        "name": "step_1",
        "images": "$inputs.image",
        "task_type": "caption",
        "api_key": "$inputs.google_api_key",
        "temperature": value,
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(specification)


def test_gemini_step_validation_with_max_tokens() -> None:
    # given
    specification = {
        "type": "roboflow_core/google_gemini@v2",
        "name": "step_1",
        "images": "$inputs.image",
        "task_type": "caption",
        "api_key": "$inputs.google_api_key",
        "max_tokens": 100,
    }

    # when
    result = BlockManifest.model_validate(specification)

    # then
    assert result.max_tokens == 100


def test_gemini_step_validation_without_required_prompt() -> None:
    # given - unconstrained requires prompt
    specification = {
        "type": "roboflow_core/google_gemini@v2",
        "name": "step_1",
        "images": "$inputs.image",
        "task_type": "unconstrained",
        "api_key": "$inputs.google_api_key",
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(specification)


def test_gemini_step_validation_without_required_classes() -> None:
    # given - classification requires classes
    specification = {
        "type": "roboflow_core/google_gemini@v2",
        "name": "step_1",
        "images": "$inputs.image",
        "task_type": "classification",
        "api_key": "$inputs.google_api_key",
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(specification)


def test_prepare_generation_config_with_max_tokens() -> None:
    # when
    result = prepare_generation_config(
        max_tokens=500,
        temperature=None,
        thinking_level=None,
        model_version="gemini-2.5-pro",
    )

    # then
    assert result["max_output_tokens"] == 500


def test_prepare_generation_config_without_max_tokens() -> None:
    # when
    result = prepare_generation_config(
        max_tokens=None,
        temperature=None,
        thinking_level=None,
        model_version="gemini-2.5-pro",
    )

    # then
    assert "max_output_tokens" not in result


def test_prepare_generation_config_with_temperature_for_older_models() -> None:
    # when
    result = prepare_generation_config(
        max_tokens=None,
        temperature=0.8,
        thinking_level=None,
        model_version="gemini-2.5-pro",
    )

    # then
    assert result["temperature"] == 0.8


def test_prepare_generation_config_with_temperature_for_newer_models() -> None:
    # when - temperature should be ignored for newer models
    result = prepare_generation_config(
        max_tokens=None,
        temperature=0.8,
        thinking_level=None,
        model_version="gemini-3-pro-preview",
    )

    # then
    assert "temperature" not in result


def test_prepare_generation_config_with_thinking_level_for_newer_models() -> None:
    # when
    result = prepare_generation_config(
        max_tokens=None,
        temperature=None,
        thinking_level="high",
        model_version="gemini-3-pro-preview",
    )

    # then
    assert result["thinking_config"] == {"thinking_level": "high"}


def test_prepare_generation_config_with_thinking_level_for_older_models() -> None:
    # when - thinking_level should be ignored for older models
    result = prepare_generation_config(
        max_tokens=None,
        temperature=None,
        thinking_level="high",
        model_version="gemini-2.5-pro",
    )

    # then
    assert "thinking_config" not in result


@patch(
    "inference.core.workflows.core_steps.models.foundation.google_gemini.v2.requests.post"
)
def test_execute_gemini_request_success(mock_post: Mock) -> None:
    # given
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "candidates": [
            {
                "content": {"parts": [{"text": "This is the generated response"}]},
                "finishReason": "STOP",
            }
        ]
    }
    mock_post.return_value = mock_response

    # when
    result = execute_gemini_request(
        prompt={"contents": {"parts": [{"text": "test"}]}},
        model_version="gemini-2.5-pro",
        google_api_key="test-key",
    )

    # then
    assert result == "This is the generated response"


@patch(
    "inference.core.workflows.core_steps.models.foundation.google_gemini.v2.requests.post"
)
def test_execute_gemini_request_max_tokens_error(mock_post: Mock) -> None:
    # given
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "candidates": [
            {
                "finishReason": "MAX_TOKENS",
            }
        ]
    }
    mock_post.return_value = mock_response

    # when/then
    with pytest.raises(ValueError) as exc_info:
        execute_gemini_request(
            prompt={"contents": {"parts": [{"text": "test"}]}},
            model_version="gemini-2.5-pro",
            google_api_key="test-key",
        )

    assert "max_tokens limit was reached" in str(exc_info.value)
    assert "increase the max_tokens parameter" in str(exc_info.value)


@patch(
    "inference.core.workflows.core_steps.models.foundation.google_gemini.v2.requests.post"
)
def test_execute_gemini_request_safety_error(mock_post: Mock) -> None:
    # given
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "candidates": [
            {
                "finishReason": "SAFETY",
            }
        ]
    }
    mock_post.return_value = mock_response

    # when/then
    with pytest.raises(ValueError) as exc_info:
        execute_gemini_request(
            prompt={"contents": {"parts": [{"text": "test"}]}},
            model_version="gemini-2.5-pro",
            google_api_key="test-key",
        )

    assert "SAFETY" in str(exc_info.value)


@patch(
    "inference.core.workflows.core_steps.models.foundation.google_gemini.v2.requests.post"
)
def test_execute_gemini_request_http_error(mock_post: Mock) -> None:
    # given
    mock_response = Mock()
    mock_response.status_code = 400
    mock_response.url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent?key=test-key"
    mock_response.raise_for_status.side_effect = Exception("HTTP Error")
    mock_response.json.return_value = {"error": "Invalid request"}
    mock_post.return_value = mock_response

    # when/then
    with pytest.raises(Exception) as exc_info:
        execute_gemini_request(
            prompt={"contents": {"parts": [{"text": "test"}]}},
            model_version="gemini-2.5-pro",
            google_api_key="test-key",
        )

    assert "HTTP Error" in str(exc_info.value)

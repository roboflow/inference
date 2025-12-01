from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest
from pydantic import ValidationError

from inference.core.workflows.core_steps.models.foundation.anthropic_claude.v2 import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    EXACT_MODEL_VERSIONS,
    MAX_OUTPUT_TOKENS,
    BlockManifest,
    execute_claude_request,
)


def test_claude_step_validation_when_input_is_valid() -> None:
    # given
    specification = {
        "type": "roboflow_core/anthropic_claude@v2",
        "name": "step_1",
        "images": "$inputs.image",
        "task_type": "unconstrained",
        "prompt": "$inputs.prompt",
        "api_key": "$inputs.anthropic_api_key",
    }

    # when
    result = BlockManifest.model_validate(specification)

    # then
    assert result.type == "roboflow_core/anthropic_claude@v2"
    assert result.name == "step_1"
    assert result.images == "$inputs.image"
    assert result.task_type == "unconstrained"
    assert result.prompt == "$inputs.prompt"
    assert result.api_key == "$inputs.anthropic_api_key"


@pytest.mark.parametrize("value", [None, 1, "a", True])
def test_claude_step_validation_when_image_is_invalid(value: Any) -> None:
    # given
    specification = {
        "type": "roboflow_core/anthropic_claude@v2",
        "name": "step_1",
        "images": value,
        "task_type": "unconstrained",
        "prompt": "$inputs.prompt",
        "api_key": "$inputs.anthropic_api_key",
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(specification)


def test_claude_step_validation_when_prompt_is_given_directly() -> None:
    # given
    specification = {
        "type": "roboflow_core/anthropic_claude@v2",
        "name": "step_1",
        "images": "$inputs.image",
        "task_type": "unconstrained",
        "prompt": "This is my prompt",
        "api_key": "$inputs.anthropic_api_key",
    }

    # when
    result = BlockManifest.model_validate(specification)

    # then
    assert result.prompt == "This is my prompt"


@pytest.mark.parametrize(
    "model_version",
    [
        "claude-sonnet-4-5",
        "claude-haiku-4-5",
        "claude-opus-4-5",
        "claude-sonnet-4",
        "claude-opus-4-1",
        "claude-opus-4",
        "$inputs.model",
    ],
)
def test_claude_step_validation_when_model_version_valid(model_version: str) -> None:
    # given
    specification = {
        "type": "roboflow_core/anthropic_claude@v2",
        "name": "step_1",
        "images": "$inputs.image",
        "task_type": "caption",
        "api_key": "$inputs.anthropic_api_key",
        "model_version": model_version,
    }

    # when
    result = BlockManifest.model_validate(specification)

    # then
    assert result.model_version == model_version


def test_claude_step_validation_with_extended_thinking() -> None:
    # given
    specification = {
        "type": "roboflow_core/anthropic_claude@v2",
        "name": "step_1",
        "images": "$inputs.image",
        "task_type": "caption",
        "api_key": "$inputs.anthropic_api_key",
        "extended_thinking": True,
    }

    # when
    result = BlockManifest.model_validate(specification)

    # then
    assert result.extended_thinking is True


def test_claude_step_validation_with_thinking_budget_tokens() -> None:
    # given
    specification = {
        "type": "roboflow_core/anthropic_claude@v2",
        "name": "step_1",
        "images": "$inputs.image",
        "task_type": "caption",
        "api_key": "$inputs.anthropic_api_key",
        "extended_thinking": True,
        "thinking_budget_tokens": 5000,
        "max_tokens": 10000,
    }

    # when
    result = BlockManifest.model_validate(specification)

    # then
    assert result.thinking_budget_tokens == 5000
    assert result.max_tokens == 10000


def test_claude_step_validation_thinking_budget_below_minimum() -> None:
    # given - thinking_budget_tokens must be >= 1024
    specification = {
        "type": "roboflow_core/anthropic_claude@v2",
        "name": "step_1",
        "images": "$inputs.image",
        "task_type": "caption",
        "api_key": "$inputs.anthropic_api_key",
        "extended_thinking": True,
        "thinking_budget_tokens": 500,
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(specification)


def test_claude_step_validation_thinking_budget_exceeds_max_tokens() -> None:
    # given - thinking_budget_tokens must be less than max_tokens
    specification = {
        "type": "roboflow_core/anthropic_claude@v2",
        "name": "step_1",
        "images": "$inputs.image",
        "task_type": "caption",
        "api_key": "$inputs.anthropic_api_key",
        "extended_thinking": True,
        "thinking_budget_tokens": 10000,
        "max_tokens": 5000,
    }

    # when
    with pytest.raises(ValidationError) as exc_info:
        _ = BlockManifest.model_validate(specification)

    assert "thinking_budget_tokens" in str(exc_info.value)
    assert "must be less than" in str(exc_info.value)


def test_claude_step_validation_temperature_with_extended_thinking() -> None:
    # given - temperature cannot be used with extended_thinking
    specification = {
        "type": "roboflow_core/anthropic_claude@v2",
        "name": "step_1",
        "images": "$inputs.image",
        "task_type": "caption",
        "api_key": "$inputs.anthropic_api_key",
        "extended_thinking": True,
        "temperature": 0.5,
    }

    # when
    with pytest.raises(ValidationError) as exc_info:
        _ = BlockManifest.model_validate(specification)

    assert "temperature" in str(exc_info.value)
    assert "extended_thinking" in str(exc_info.value)


def test_claude_step_validation_with_temperature() -> None:
    # given
    specification = {
        "type": "roboflow_core/anthropic_claude@v2",
        "name": "step_1",
        "images": "$inputs.image",
        "task_type": "caption",
        "api_key": "$inputs.anthropic_api_key",
        "temperature": 0.7,
    }

    # when
    result = BlockManifest.model_validate(specification)

    # then
    assert result.temperature == 0.7


@pytest.mark.parametrize("value", [-0.1, 1.1, "invalid"])
def test_claude_step_validation_when_temperature_invalid(value: Any) -> None:
    # given
    specification = {
        "type": "roboflow_core/anthropic_claude@v2",
        "name": "step_1",
        "images": "$inputs.image",
        "task_type": "caption",
        "api_key": "$inputs.anthropic_api_key",
        "temperature": value,
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(specification)


def test_claude_step_validation_with_max_tokens() -> None:
    # given
    specification = {
        "type": "roboflow_core/anthropic_claude@v2",
        "name": "step_1",
        "images": "$inputs.image",
        "task_type": "caption",
        "api_key": "$inputs.anthropic_api_key",
        "max_tokens": 1000,
    }

    # when
    result = BlockManifest.model_validate(specification)

    # then
    assert result.max_tokens == 1000


def test_claude_step_validation_without_required_prompt() -> None:
    # given - unconstrained requires prompt
    specification = {
        "type": "roboflow_core/anthropic_claude@v2",
        "name": "step_1",
        "images": "$inputs.image",
        "task_type": "unconstrained",
        "api_key": "$inputs.anthropic_api_key",
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(specification)


def test_claude_step_validation_without_required_classes() -> None:
    # given - classification requires classes
    specification = {
        "type": "roboflow_core/anthropic_claude@v2",
        "name": "step_1",
        "images": "$inputs.image",
        "task_type": "classification",
        "api_key": "$inputs.anthropic_api_key",
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(specification)


def test_claude_step_validation_without_required_output_structure() -> None:
    # given - structured-answering requires output_structure
    specification = {
        "type": "roboflow_core/anthropic_claude@v2",
        "name": "step_1",
        "images": "$inputs.image",
        "task_type": "structured-answering",
        "api_key": "$inputs.anthropic_api_key",
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(specification)


def test_claude_step_validation_with_classification_and_classes() -> None:
    # given
    specification = {
        "type": "roboflow_core/anthropic_claude@v2",
        "name": "step_1",
        "images": "$inputs.image",
        "task_type": "classification",
        "classes": ["cat", "dog"],
        "api_key": "$inputs.anthropic_api_key",
    }

    # when
    result = BlockManifest.model_validate(specification)

    # then
    assert result.task_type == "classification"
    assert result.classes == ["cat", "dog"]


def test_claude_step_validation_with_structured_answering() -> None:
    # given
    specification = {
        "type": "roboflow_core/anthropic_claude@v2",
        "name": "step_1",
        "images": "$inputs.image",
        "task_type": "structured-answering",
        "output_structure": {"name": "object name", "color": "object color"},
        "api_key": "$inputs.anthropic_api_key",
    }

    # when
    result = BlockManifest.model_validate(specification)

    # then
    assert result.task_type == "structured-answering"
    assert result.output_structure == {"name": "object name", "color": "object color"}


def test_max_output_tokens_mapping() -> None:
    # then - verify all models have max_output_tokens defined
    assert MAX_OUTPUT_TOKENS["claude-sonnet-4-5"] == 64000
    assert MAX_OUTPUT_TOKENS["claude-haiku-4-5"] == 64000
    assert MAX_OUTPUT_TOKENS["claude-opus-4-5"] == 64000
    assert MAX_OUTPUT_TOKENS["claude-sonnet-4"] == 64000
    assert MAX_OUTPUT_TOKENS["claude-opus-4-1"] == 32000
    assert MAX_OUTPUT_TOKENS["claude-opus-4"] == 32000
    assert DEFAULT_MAX_OUTPUT_TOKENS == 64000


def test_exact_model_versions_mapping() -> None:
    # then - verify all models have exact versions defined
    assert EXACT_MODEL_VERSIONS["claude-sonnet-4-5"] == "claude-sonnet-4-5-20250929"
    assert EXACT_MODEL_VERSIONS["claude-haiku-4-5"] == "claude-haiku-4-5-20251001"
    assert EXACT_MODEL_VERSIONS["claude-opus-4-5"] == "claude-opus-4-5-20251101"
    assert EXACT_MODEL_VERSIONS["claude-sonnet-4"] == "claude-sonnet-4-20250514"
    assert EXACT_MODEL_VERSIONS["claude-opus-4-1"] == "claude-opus-4-1-20250805"
    assert EXACT_MODEL_VERSIONS["claude-opus-4"] == "claude-opus-4-20250514"


@patch(
    "inference.core.workflows.core_steps.models.foundation.anthropic_claude.v2.anthropic.Anthropic"
)
def test_execute_claude_request_success(mock_anthropic_class: Mock) -> None:
    # given
    mock_client = MagicMock()
    mock_anthropic_class.return_value = mock_client

    mock_text_block = Mock()
    mock_text_block.type = "text"
    mock_text_block.text = "This is the generated response"

    mock_result = Mock()
    mock_result.stop_reason = "end_turn"
    mock_result.content = [mock_text_block]

    mock_stream = MagicMock()
    mock_stream.__enter__ = Mock(return_value=mock_stream)
    mock_stream.__exit__ = Mock(return_value=False)
    mock_stream.get_final_message.return_value = mock_result

    mock_client.messages.stream.return_value = mock_stream

    # when
    result = execute_claude_request(
        system_prompt="You are a helpful assistant",
        messages=[{"role": "user", "content": "Hello"}],
        model_version="claude-sonnet-4-5",
        max_tokens=1000,
        temperature=0.7,
        extended_thinking=None,
        thinking_budget_tokens=None,
        api_key="test-key",
    )

    # then
    assert result == "This is the generated response"
    mock_client.messages.stream.assert_called_once()


@patch(
    "inference.core.workflows.core_steps.models.foundation.anthropic_claude.v2.anthropic.Anthropic"
)
def test_execute_claude_request_with_extended_thinking(
    mock_anthropic_class: Mock,
) -> None:
    # given
    mock_client = MagicMock()
    mock_anthropic_class.return_value = mock_client

    mock_thinking_block = Mock()
    mock_thinking_block.type = "thinking"

    mock_text_block = Mock()
    mock_text_block.type = "text"
    mock_text_block.text = "Response after thinking"

    mock_result = Mock()
    mock_result.stop_reason = "end_turn"
    mock_result.content = [mock_thinking_block, mock_text_block]

    mock_stream = MagicMock()
    mock_stream.__enter__ = Mock(return_value=mock_stream)
    mock_stream.__exit__ = Mock(return_value=False)
    mock_stream.get_final_message.return_value = mock_result

    mock_client.messages.stream.return_value = mock_stream

    # when
    result = execute_claude_request(
        system_prompt=None,
        messages=[{"role": "user", "content": "Think about this"}],
        model_version="claude-sonnet-4-5",
        max_tokens=10000,
        temperature=None,
        extended_thinking=True,
        thinking_budget_tokens=5000,
        api_key="test-key",
    )

    # then
    assert result == "Response after thinking"
    call_kwargs = mock_client.messages.stream.call_args.kwargs
    assert "thinking" in call_kwargs
    assert call_kwargs["thinking"]["type"] == "enabled"
    assert call_kwargs["thinking"]["budget_tokens"] == 5000


@patch(
    "inference.core.workflows.core_steps.models.foundation.anthropic_claude.v2.anthropic.Anthropic"
)
def test_execute_claude_request_with_default_thinking_budget(
    mock_anthropic_class: Mock,
) -> None:
    # given
    mock_client = MagicMock()
    mock_anthropic_class.return_value = mock_client

    mock_text_block = Mock()
    mock_text_block.type = "text"
    mock_text_block.text = "Response"

    mock_result = Mock()
    mock_result.stop_reason = "end_turn"
    mock_result.content = [mock_text_block]

    mock_stream = MagicMock()
    mock_stream.__enter__ = Mock(return_value=mock_stream)
    mock_stream.__exit__ = Mock(return_value=False)
    mock_stream.get_final_message.return_value = mock_result

    mock_client.messages.stream.return_value = mock_stream

    # when - extended_thinking=True but thinking_budget_tokens=None
    execute_claude_request(
        system_prompt=None,
        messages=[{"role": "user", "content": "Think"}],
        model_version="claude-sonnet-4-5",
        max_tokens=None,
        temperature=None,
        extended_thinking=True,
        thinking_budget_tokens=None,
        api_key="test-key",
    )

    # then - should default to half of model's max output tokens (64000 // 2 = 32000)
    call_kwargs = mock_client.messages.stream.call_args.kwargs
    assert call_kwargs["thinking"]["budget_tokens"] == 32000


@patch(
    "inference.core.workflows.core_steps.models.foundation.anthropic_claude.v2.anthropic.Anthropic"
)
def test_execute_claude_request_with_default_max_tokens(
    mock_anthropic_class: Mock,
) -> None:
    # given
    mock_client = MagicMock()
    mock_anthropic_class.return_value = mock_client

    mock_text_block = Mock()
    mock_text_block.type = "text"
    mock_text_block.text = "Response"

    mock_result = Mock()
    mock_result.stop_reason = "end_turn"
    mock_result.content = [mock_text_block]

    mock_stream = MagicMock()
    mock_stream.__enter__ = Mock(return_value=mock_stream)
    mock_stream.__exit__ = Mock(return_value=False)
    mock_stream.get_final_message.return_value = mock_result

    mock_client.messages.stream.return_value = mock_stream

    # when - max_tokens=None should default to model's max
    execute_claude_request(
        system_prompt=None,
        messages=[{"role": "user", "content": "Hello"}],
        model_version="claude-sonnet-4-5",
        max_tokens=None,
        temperature=None,
        extended_thinking=None,
        thinking_budget_tokens=None,
        api_key="test-key",
    )

    # then - should default to model's max output tokens (64000)
    call_kwargs = mock_client.messages.stream.call_args.kwargs
    assert call_kwargs["max_tokens"] == 64000


@patch(
    "inference.core.workflows.core_steps.models.foundation.anthropic_claude.v2.anthropic.Anthropic"
)
def test_execute_claude_request_max_tokens_error(mock_anthropic_class: Mock) -> None:
    # given
    mock_client = MagicMock()
    mock_anthropic_class.return_value = mock_client

    mock_result = Mock()
    mock_result.stop_reason = "max_tokens"
    mock_result.content = []

    mock_stream = MagicMock()
    mock_stream.__enter__ = Mock(return_value=mock_stream)
    mock_stream.__exit__ = Mock(return_value=False)
    mock_stream.get_final_message.return_value = mock_result

    mock_client.messages.stream.return_value = mock_stream

    # when/then
    with pytest.raises(ValueError) as exc_info:
        execute_claude_request(
            system_prompt=None,
            messages=[{"role": "user", "content": "Hello"}],
            model_version="claude-sonnet-4-5",
            max_tokens=100,
            temperature=None,
            extended_thinking=None,
            thinking_budget_tokens=None,
            api_key="test-key",
        )

    assert "max_tokens limit was reached" in str(exc_info.value)


@patch(
    "inference.core.workflows.core_steps.models.foundation.anthropic_claude.v2.anthropic.Anthropic"
)
def test_execute_claude_request_unexpected_stop_reason(
    mock_anthropic_class: Mock,
) -> None:
    # given
    mock_client = MagicMock()
    mock_anthropic_class.return_value = mock_client

    mock_result = Mock()
    mock_result.stop_reason = "content_filter"
    mock_result.content = []

    mock_stream = MagicMock()
    mock_stream.__enter__ = Mock(return_value=mock_stream)
    mock_stream.__exit__ = Mock(return_value=False)
    mock_stream.get_final_message.return_value = mock_result

    mock_client.messages.stream.return_value = mock_stream

    # when/then
    with pytest.raises(ValueError) as exc_info:
        execute_claude_request(
            system_prompt=None,
            messages=[{"role": "user", "content": "Hello"}],
            model_version="claude-sonnet-4-5",
            max_tokens=1000,
            temperature=None,
            extended_thinking=None,
            thinking_budget_tokens=None,
            api_key="test-key",
        )

    assert "content_filter" in str(exc_info.value)


@patch(
    "inference.core.workflows.core_steps.models.foundation.anthropic_claude.v2.anthropic.Anthropic"
)
def test_execute_claude_request_no_text_content(mock_anthropic_class: Mock) -> None:
    # given
    mock_client = MagicMock()
    mock_anthropic_class.return_value = mock_client

    mock_result = Mock()
    mock_result.stop_reason = "end_turn"
    mock_result.content = []  # No content blocks

    mock_stream = MagicMock()
    mock_stream.__enter__ = Mock(return_value=mock_stream)
    mock_stream.__exit__ = Mock(return_value=False)
    mock_stream.get_final_message.return_value = mock_result

    mock_client.messages.stream.return_value = mock_stream

    # when/then
    with pytest.raises(ValueError) as exc_info:
        execute_claude_request(
            system_prompt=None,
            messages=[{"role": "user", "content": "Hello"}],
            model_version="claude-sonnet-4-5",
            max_tokens=1000,
            temperature=None,
            extended_thinking=None,
            thinking_budget_tokens=None,
            api_key="test-key",
        )

    assert "no text content" in str(exc_info.value)


@patch(
    "inference.core.workflows.core_steps.models.foundation.anthropic_claude.v2.anthropic.Anthropic"
)
def test_execute_claude_request_stop_sequence_is_valid(
    mock_anthropic_class: Mock,
) -> None:
    # given
    mock_client = MagicMock()
    mock_anthropic_class.return_value = mock_client

    mock_text_block = Mock()
    mock_text_block.type = "text"
    mock_text_block.text = "Response before stop sequence"

    mock_result = Mock()
    mock_result.stop_reason = "stop_sequence"
    mock_result.content = [mock_text_block]

    mock_stream = MagicMock()
    mock_stream.__enter__ = Mock(return_value=mock_stream)
    mock_stream.__exit__ = Mock(return_value=False)
    mock_stream.get_final_message.return_value = mock_result

    mock_client.messages.stream.return_value = mock_stream

    # when
    result = execute_claude_request(
        system_prompt=None,
        messages=[{"role": "user", "content": "Hello"}],
        model_version="claude-sonnet-4-5",
        max_tokens=1000,
        temperature=None,
        extended_thinking=None,
        thinking_budget_tokens=None,
        api_key="test-key",
    )

    # then - stop_sequence is a valid stop reason
    assert result == "Response before stop sequence"

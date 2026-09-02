from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest
from anthropic import NOT_GIVEN
from pydantic import ValidationError

from inference.core.workflows.core_steps.models.foundation.anthropic_claude.v1 import (
    EXACT_MODELS_VERSIONS_MAPPING as EXACT_MODEL_VERSIONS_V1,
)
from inference.core.workflows.core_steps.models.foundation.anthropic_claude.v1 import (
    BlockManifest as BlockManifestV1,
)
from inference.core.workflows.core_steps.models.foundation.anthropic_claude.v1 import (
    execute_claude_request as execute_claude_request_v1,
)
from inference.core.workflows.core_steps.models.foundation.anthropic_claude.v2 import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    EXACT_MODEL_VERSIONS,
    MAX_OUTPUT_TOKENS,
    BlockManifest,
    execute_claude_request,
)
from inference.core.workflows.core_steps.models.foundation.anthropic_claude.v3 import (
    EXACT_MODEL_VERSIONS as EXACT_MODEL_VERSIONS_V3,
)
from inference.core.workflows.core_steps.models.foundation.anthropic_claude.v3 import (
    MAX_OUTPUT_TOKENS as MAX_OUTPUT_TOKENS_V3,
)
from inference.core.workflows.core_steps.models.foundation.anthropic_claude.v3 import (
    BlockManifest as BlockManifestV3,
)
from inference.core.workflows.core_steps.models.foundation.anthropic_claude.v3 import (
    execute_claude_request as execute_claude_request_v3,
)
from inference.core.workflows.core_steps.models.foundation.anthropic_claude.v4 import (
    EXACT_MODEL_VERSIONS as EXACT_MODEL_VERSIONS_V4,
)
from inference.core.workflows.core_steps.models.foundation.anthropic_claude.v4 import (
    MAX_OUTPUT_TOKENS as MAX_OUTPUT_TOKENS_V4,
)
from inference.core.workflows.core_steps.models.foundation.anthropic_claude.v4 import (
    BlockManifest as BlockManifestV4,
)
from inference.core.workflows.core_steps.models.foundation.anthropic_claude.v4 import (
    execute_claude_request as execute_claude_request_v4,
)

# Claude 5-generation models that must be selectable in every block version,
# together with the wire id sent to Anthropic and the max output tokens the
# block falls back to when `max_tokens` is not set.
CLAUDE_5_GENERATION_MODELS = {
    "claude-fable-5-1": ("claude-fable-5-1", 128000),
    "claude-fable-5": ("claude-fable-5", 128000),
    "claude-opus-5": ("claude-opus-5", 128000),
    "claude-sonnet-5": ("claude-sonnet-5", 128000),
    "claude-opus-4-8": ("claude-opus-4-8", 128000),
}


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
        "claude-fable-5-1",
        "claude-fable-5",
        "claude-opus-5",
        "claude-sonnet-5",
        "claude-opus-4-8",
        "claude-opus-4-7",
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
    assert MAX_OUTPUT_TOKENS["claude-fable-5-1"] == 128000
    assert MAX_OUTPUT_TOKENS["claude-fable-5"] == 128000
    assert MAX_OUTPUT_TOKENS["claude-opus-5"] == 128000
    assert MAX_OUTPUT_TOKENS["claude-sonnet-5"] == 128000
    assert MAX_OUTPUT_TOKENS["claude-opus-4-8"] == 128000
    assert MAX_OUTPUT_TOKENS["claude-opus-4-7"] == 128000
    assert MAX_OUTPUT_TOKENS["claude-sonnet-4-5"] == 64000
    assert MAX_OUTPUT_TOKENS["claude-haiku-4-5"] == 64000
    assert MAX_OUTPUT_TOKENS["claude-opus-4-5"] == 64000
    assert MAX_OUTPUT_TOKENS["claude-sonnet-4"] == 64000
    assert MAX_OUTPUT_TOKENS["claude-opus-4-1"] == 32000
    assert MAX_OUTPUT_TOKENS["claude-opus-4"] == 32000
    assert DEFAULT_MAX_OUTPUT_TOKENS == 64000


def test_exact_model_versions_mapping() -> None:
    # then - verify all models have exact versions defined
    assert EXACT_MODEL_VERSIONS["claude-fable-5-1"] == "claude-fable-5-1"
    assert EXACT_MODEL_VERSIONS["claude-fable-5"] == "claude-fable-5"
    assert EXACT_MODEL_VERSIONS["claude-opus-5"] == "claude-opus-5"
    assert EXACT_MODEL_VERSIONS["claude-sonnet-5"] == "claude-sonnet-5"
    assert EXACT_MODEL_VERSIONS["claude-opus-4-8"] == "claude-opus-4-8"
    assert EXACT_MODEL_VERSIONS["claude-opus-4-7"] == "claude-opus-4-7"
    assert EXACT_MODEL_VERSIONS["claude-sonnet-4-5"] == "claude-sonnet-4-5-20250929"
    assert EXACT_MODEL_VERSIONS["claude-haiku-4-5"] == "claude-haiku-4-5-20251001"
    assert EXACT_MODEL_VERSIONS["claude-opus-4-5"] == "claude-opus-4-5-20251101"
    assert EXACT_MODEL_VERSIONS["claude-sonnet-4"] == "claude-sonnet-4-20250514"
    assert EXACT_MODEL_VERSIONS["claude-opus-4-1"] == "claude-opus-4-1-20250805"
    assert EXACT_MODEL_VERSIONS["claude-opus-4"] == "claude-opus-4-20250514"


def test_v3_claude_fable_model_metadata() -> None:
    # then
    assert MAX_OUTPUT_TOKENS_V3["claude-fable-5"] == 128000
    assert EXACT_MODEL_VERSIONS_V3["claude-fable-5"] == "claude-fable-5"


@pytest.mark.parametrize(
    "model_version, expected_exact_version, expected_max_output_tokens",
    [
        (model_version, exact_version, max_output_tokens)
        for model_version, (
            exact_version,
            max_output_tokens,
        ) in CLAUDE_5_GENERATION_MODELS.items()
    ],
)
def test_claude_5_generation_models_share_metadata_across_v2_v3_v4(
    model_version: str,
    expected_exact_version: str,
    expected_max_output_tokens: int,
) -> None:
    # then - every block version that owns a metadata table must agree on
    # the wire id and the output budget, so switching block versions never
    # silently changes which model is called or how much it may generate
    for exact_versions, max_output_tokens in [
        (EXACT_MODEL_VERSIONS, MAX_OUTPUT_TOKENS),
        (EXACT_MODEL_VERSIONS_V3, MAX_OUTPUT_TOKENS_V3),
        (EXACT_MODEL_VERSIONS_V4, MAX_OUTPUT_TOKENS_V4),
    ]:
        assert exact_versions[model_version] == expected_exact_version
        assert max_output_tokens[model_version] == expected_max_output_tokens
    assert EXACT_MODEL_VERSIONS_V1[model_version] == expected_exact_version


@pytest.mark.parametrize("model_version", list(CLAUDE_5_GENERATION_MODELS.keys()))
@pytest.mark.parametrize(
    "block_type, manifest_class",
    [
        ("roboflow_core/anthropic_claude@v1", BlockManifestV1),
        ("roboflow_core/anthropic_claude@v2", BlockManifest),
        ("roboflow_core/anthropic_claude@v3", BlockManifestV3),
        ("roboflow_core/anthropic_claude@v4", BlockManifestV4),
    ],
)
def test_claude_5_generation_models_accepted_by_every_block_version(
    model_version: str,
    block_type: str,
    manifest_class: type,
) -> None:
    # given
    specification = {
        "type": block_type,
        "name": "step_1",
        "images": "$inputs.image",
        "task_type": "unconstrained",
        "prompt": "This is my prompt",
        "api_key": "$inputs.anthropic_api_key",
        "model_version": model_version,
    }

    # when
    result = manifest_class.model_validate(specification)

    # then
    assert result.model_version == model_version


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


# --- temperature / thinking handling per model generation -------------------
#
# Claude Opus 4.7+, Sonnet 5, Opus 5 and the Fable line reject `temperature`
# and `thinking.type=enabled`; older models still take both. The request that
# leaves the block must differ accordingly on every code path (v1 direct, v2
# direct, v3/v4 direct and v3/v4 Roboflow proxy).

LEGACY_MODEL = "claude-sonnet-4-5"
NEW_GENERATION_MODEL = "claude-fable-5-1"


def _mock_streaming_client(mock_anthropic_class: Mock, text: str = "ok") -> MagicMock:
    mock_client = MagicMock()
    mock_anthropic_class.return_value = mock_client

    mock_text_block = Mock()
    mock_text_block.type = "text"
    mock_text_block.text = text

    mock_result = Mock()
    mock_result.stop_reason = "end_turn"
    mock_result.content = [mock_text_block]
    mock_result.usage = Mock(input_tokens=11, output_tokens=7)

    mock_stream = MagicMock()
    mock_stream.__enter__ = Mock(return_value=mock_stream)
    mock_stream.__exit__ = Mock(return_value=False)
    mock_stream.get_final_message.return_value = mock_result
    mock_client.messages.stream.return_value = mock_stream
    mock_client.messages.create.return_value = mock_result
    return mock_client


def _is_not_given(value: Any) -> bool:
    return value is NOT_GIVEN


@patch(
    "inference.core.workflows.core_steps.models.foundation.anthropic_claude.v1.anthropic.Anthropic"
)
def test_v1_forwards_temperature_for_legacy_model(mock_anthropic_class: Mock) -> None:
    # given
    mock_client = _mock_streaming_client(mock_anthropic_class)

    # when
    execute_claude_request_v1(
        system_prompt=None,
        messages=[{"role": "user", "content": "Hello"}],
        model_version=LEGACY_MODEL,
        max_tokens=100,
        temperature=0.4,
        api_key="test-key",
    )

    # then
    assert mock_client.messages.create.call_args.kwargs["temperature"] == 0.4


@patch(
    "inference.core.workflows.core_steps.models.foundation.anthropic_claude.v1.anthropic.Anthropic"
)
def test_v1_drops_temperature_for_new_generation_model(
    mock_anthropic_class: Mock,
) -> None:
    # given
    mock_client = _mock_streaming_client(mock_anthropic_class)

    # when
    execute_claude_request_v1(
        system_prompt=None,
        messages=[{"role": "user", "content": "Hello"}],
        model_version=NEW_GENERATION_MODEL,
        max_tokens=100,
        temperature=0.4,
        api_key="test-key",
    )

    # then
    call_kwargs = mock_client.messages.create.call_args.kwargs
    assert _is_not_given(call_kwargs["temperature"])
    assert call_kwargs["model"] == "claude-fable-5-1"


@patch(
    "inference.core.workflows.core_steps.models.foundation.anthropic_claude.v2.anthropic.Anthropic"
)
def test_v2_direct_request_translates_controls_for_new_generation_model(
    mock_anthropic_class: Mock,
) -> None:
    # given
    mock_client = _mock_streaming_client(mock_anthropic_class)

    # when - a budget and a temperature are configured but the model takes neither
    execute_claude_request(
        system_prompt=None,
        messages=[{"role": "user", "content": "Think"}],
        model_version=NEW_GENERATION_MODEL,
        max_tokens=None,
        temperature=0.4,
        extended_thinking=True,
        thinking_budget_tokens=5000,
        api_key="test-key",
    )

    # then
    call_kwargs = mock_client.messages.stream.call_args.kwargs
    assert call_kwargs["thinking"] == {"type": "adaptive"}
    assert _is_not_given(call_kwargs["temperature"])
    assert call_kwargs["max_tokens"] == 128000


@pytest.mark.parametrize(
    "execute_request, module",
    [
        (execute_claude_request_v3, "v3"),
        (execute_claude_request_v4, "v4"),
    ],
)
def test_direct_request_keeps_legacy_controls_for_legacy_model(
    execute_request: Any, module: str
) -> None:
    with patch(
        f"inference.core.workflows.core_steps.models.foundation.anthropic_claude.{module}.anthropic.Anthropic"
    ) as mock_anthropic_class:
        # given
        mock_client = _mock_streaming_client(mock_anthropic_class)

        # when - thinking off, temperature on
        execute_request(
            roboflow_api_key=None,
            anthropic_api_key="sk-ant-test",
            system_prompt=None,
            messages=[{"role": "user", "content": "Hello"}],
            model_version=LEGACY_MODEL,
            max_tokens=100,
            temperature=0.4,
            extended_thinking=None,
            thinking_budget_tokens=None,
        )
        no_thinking_kwargs = mock_client.messages.stream.call_args.kwargs

        # when - thinking on with an explicit budget
        execute_request(
            roboflow_api_key=None,
            anthropic_api_key="sk-ant-test",
            system_prompt=None,
            messages=[{"role": "user", "content": "Think"}],
            model_version=LEGACY_MODEL,
            max_tokens=10000,
            temperature=0.4,
            extended_thinking=True,
            thinking_budget_tokens=5000,
        )
        thinking_kwargs = mock_client.messages.stream.call_args.kwargs

    # then
    assert no_thinking_kwargs["temperature"] == 0.4
    assert "thinking" not in no_thinking_kwargs
    assert _is_not_given(thinking_kwargs["temperature"])
    assert thinking_kwargs["thinking"] == {"type": "enabled", "budget_tokens": 5000}


@pytest.mark.parametrize(
    "execute_request, module",
    [
        (execute_claude_request_v3, "v3"),
        (execute_claude_request_v4, "v4"),
    ],
)
def test_direct_request_translates_controls_for_new_generation_model(
    execute_request: Any, module: str
) -> None:
    with patch(
        f"inference.core.workflows.core_steps.models.foundation.anthropic_claude.{module}.anthropic.Anthropic"
    ) as mock_anthropic_class:
        # given
        mock_client = _mock_streaming_client(mock_anthropic_class)

        # when - temperature configured, thinking off
        execute_request(
            roboflow_api_key=None,
            anthropic_api_key="sk-ant-test",
            system_prompt=None,
            messages=[{"role": "user", "content": "Hello"}],
            model_version=NEW_GENERATION_MODEL,
            max_tokens=100,
            temperature=0.4,
            extended_thinking=None,
            thinking_budget_tokens=None,
        )
        no_thinking_kwargs = mock_client.messages.stream.call_args.kwargs

        # when - thinking on with a budget the model cannot take
        execute_request(
            roboflow_api_key=None,
            anthropic_api_key="sk-ant-test",
            system_prompt=None,
            messages=[{"role": "user", "content": "Think"}],
            model_version=NEW_GENERATION_MODEL,
            max_tokens=None,
            temperature=None,
            extended_thinking=True,
            thinking_budget_tokens=5000,
        )
        thinking_kwargs = mock_client.messages.stream.call_args.kwargs

    # then
    assert _is_not_given(no_thinking_kwargs["temperature"])
    assert "thinking" not in no_thinking_kwargs
    assert no_thinking_kwargs["model"] == "claude-fable-5-1"
    assert thinking_kwargs["thinking"] == {"type": "adaptive"}
    assert thinking_kwargs["max_tokens"] == 128000


PROXY_RESPONSE = {
    "stop_reason": "end_turn",
    "content": [{"type": "text", "text": "proxied"}],
    "usage": {"input_tokens": 3, "output_tokens": 2},
}


@pytest.mark.parametrize(
    "execute_request, module",
    [
        (execute_claude_request_v3, "v3"),
        (execute_claude_request_v4, "v4"),
    ],
)
def test_proxied_request_keeps_legacy_controls_for_legacy_model(
    execute_request: Any, module: str
) -> None:
    with patch(
        f"inference.core.workflows.core_steps.models.foundation.anthropic_claude.{module}.post_to_roboflow_api",
        return_value=PROXY_RESPONSE,
    ) as post_mock:
        # when
        execute_request(
            roboflow_api_key="rf-key",
            anthropic_api_key="rf_key:account",
            system_prompt="sys",
            messages=[{"role": "user", "content": "Hello"}],
            model_version=LEGACY_MODEL,
            max_tokens=100,
            temperature=0.4,
            extended_thinking=None,
            thinking_budget_tokens=None,
        )
        plain_payload = post_mock.call_args.kwargs["payload"]

        execute_request(
            roboflow_api_key="rf-key",
            anthropic_api_key="rf_key:account",
            system_prompt=None,
            messages=[{"role": "user", "content": "Think"}],
            model_version=LEGACY_MODEL,
            max_tokens=None,
            temperature=0.4,
            extended_thinking=True,
            thinking_budget_tokens=None,
        )
        thinking_payload = post_mock.call_args.kwargs["payload"]

    # then
    assert plain_payload["model"] == LEGACY_MODEL
    assert plain_payload["temperature"] == 0.4
    assert plain_payload["system"] == "sys"
    assert "thinking" not in plain_payload
    assert "temperature" not in thinking_payload
    assert thinking_payload["thinking"] == {"type": "enabled", "budget_tokens": 32000}
    assert thinking_payload["max_tokens"] == 64000


@pytest.mark.parametrize(
    "execute_request, module",
    [
        (execute_claude_request_v3, "v3"),
        (execute_claude_request_v4, "v4"),
    ],
)
def test_proxied_request_translates_controls_for_new_generation_model(
    execute_request: Any, module: str
) -> None:
    with patch(
        f"inference.core.workflows.core_steps.models.foundation.anthropic_claude.{module}.post_to_roboflow_api",
        return_value=PROXY_RESPONSE,
    ) as post_mock:
        # when
        execute_request(
            roboflow_api_key="rf-key",
            anthropic_api_key="rf_key:account",
            system_prompt=None,
            messages=[{"role": "user", "content": "Hello"}],
            model_version=NEW_GENERATION_MODEL,
            max_tokens=100,
            temperature=0.4,
            extended_thinking=None,
            thinking_budget_tokens=None,
        )
        plain_payload = post_mock.call_args.kwargs["payload"]

        execute_request(
            roboflow_api_key="rf-key",
            anthropic_api_key="rf_key:account",
            system_prompt=None,
            messages=[{"role": "user", "content": "Think"}],
            model_version=NEW_GENERATION_MODEL,
            max_tokens=None,
            temperature=None,
            extended_thinking=True,
            thinking_budget_tokens=5000,
        )
        thinking_payload = post_mock.call_args.kwargs["payload"]

    # then
    assert plain_payload["model"] == NEW_GENERATION_MODEL
    assert "temperature" not in plain_payload
    assert "thinking" not in plain_payload
    assert thinking_payload["thinking"] == {"type": "adaptive"}
    assert thinking_payload["max_tokens"] == 128000

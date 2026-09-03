"""Tests for the Anthropic Claude v5 block (v4 + reasoning_effort)."""

from typing import Any
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
from pydantic import ValidationError

from inference.core.workflows.core_steps.models.foundation.anthropic_claude.v5 import (
    AnthropicClaudeBlockV5,
    BlockManifest,
    execute_claude_request,
)
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    ImageParentMetadata,
    WorkflowImageData,
)


def _spec(**overrides: Any) -> dict:
    specification = {
        "type": "roboflow_core/anthropic_claude@v5",
        "name": "step_1",
        "images": "$inputs.image",
        "task_type": "unconstrained",
        "prompt": "This is my prompt",
        "api_key": "$inputs.anthropic_api_key",
    }
    specification.update(overrides)
    return specification


def test_v5_accepts_fable_with_medium_effort() -> None:
    result = BlockManifest.model_validate(
        _spec(model_version="claude-fable-5-1", reasoning_effort="medium")
    )

    assert result.reasoning_effort == "medium"


def test_v5_accepts_effort_alongside_extended_thinking_on_opus_4_5() -> None:
    result = BlockManifest.model_validate(
        _spec(
            model_version="claude-opus-4-5",
            reasoning_effort="low",
            extended_thinking=True,
            thinking_budget_tokens=2048,
            max_tokens=8000,
        )
    )

    assert result.reasoning_effort == "low"
    assert result.extended_thinking is True


def test_v5_rejects_effort_on_legacy_model() -> None:
    with pytest.raises(ValidationError, match="support"):
        BlockManifest.model_validate(
            _spec(model_version="claude-sonnet-4-5", reasoning_effort="high")
        )


def test_v5_rejects_xhigh_on_opus_4_6() -> None:
    with pytest.raises(ValidationError, match="support"):
        BlockManifest.model_validate(
            _spec(model_version="claude-opus-4-6", reasoning_effort="xhigh")
        )


def test_v5_rejects_none_effort() -> None:
    with pytest.raises(ValidationError):
        BlockManifest.model_validate(
            _spec(model_version="claude-fable-5-1", reasoning_effort="none")
        )


@pytest.mark.parametrize(
    "model_version, reasoning_effort",
    [
        ("claude-opus-4-6", "xhigh"),
        ("claude-sonnet-4-5", "high"),
        ("claude-fable-5-1", "turbo"),
    ],
)
@patch(
    "inference.core.workflows.core_steps.models.foundation.anthropic_claude.v5.post_to_roboflow_api"
)
@patch(
    "inference.core.workflows.core_steps.models.foundation.anthropic_claude.v5.anthropic.Anthropic"
)
def test_runtime_rejects_effort_the_model_cannot_take_before_any_request(
    mock_anthropic_class: Mock,
    post_mock: Mock,
    model_version: str,
    reasoning_effort: str,
) -> None:
    for anthropic_api_key in ("sk-ant-test", "rf_key:account"):
        with pytest.raises(ValueError, match="support"):
            execute_claude_request(
                roboflow_api_key="rf-key",
                anthropic_api_key=anthropic_api_key,
                system_prompt=None,
                messages=[{"role": "user", "content": "Hello"}],
                model_version=model_version,
                max_tokens=100,
                temperature=None,
                extended_thinking=None,
                thinking_budget_tokens=None,
                reasoning_effort=reasoning_effort,
            )

    assert mock_anthropic_class.call_count == 0
    assert post_mock.call_count == 0


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
    return mock_client


PROXY_RESPONSE = {
    "stop_reason": "end_turn",
    "content": [{"type": "text", "text": "proxied"}],
    "usage": {"input_tokens": 3, "output_tokens": 2},
}


@patch(
    "inference.core.workflows.core_steps.models.foundation.anthropic_claude.v5.anthropic.Anthropic"
)
def test_direct_request_sends_effort_without_thinking_for_fable(
    mock_anthropic_class: Mock,
) -> None:
    mock_client = _mock_streaming_client(mock_anthropic_class)

    execute_claude_request(
        roboflow_api_key=None,
        anthropic_api_key="sk-ant-test",
        system_prompt=None,
        messages=[{"role": "user", "content": "Hello"}],
        model_version="claude-fable-5-1",
        max_tokens=100,
        temperature=None,
        extended_thinking=None,
        thinking_budget_tokens=None,
        reasoning_effort="medium",
    )

    call_kwargs = mock_client.messages.stream.call_args.kwargs
    assert "thinking" not in call_kwargs
    assert call_kwargs["extra_body"] == {"output_config": {"effort": "medium"}}


@patch(
    "inference.core.workflows.core_steps.models.foundation.anthropic_claude.v5.anthropic.Anthropic"
)
def test_direct_request_sends_adaptive_thinking_and_effort_for_fable(
    mock_anthropic_class: Mock,
) -> None:
    mock_client = _mock_streaming_client(mock_anthropic_class)

    execute_claude_request(
        roboflow_api_key=None,
        anthropic_api_key="sk-ant-test",
        system_prompt=None,
        messages=[{"role": "user", "content": "Think"}],
        model_version="claude-fable-5-1",
        max_tokens=None,
        temperature=None,
        extended_thinking=True,
        thinking_budget_tokens=5000,
        reasoning_effort="high",
    )

    call_kwargs = mock_client.messages.stream.call_args.kwargs
    assert call_kwargs["thinking"] == {"type": "adaptive"}
    assert call_kwargs["extra_body"] == {"output_config": {"effort": "high"}}


@patch(
    "inference.core.workflows.core_steps.models.foundation.anthropic_claude.v5.anthropic.Anthropic"
)
def test_direct_request_keeps_legacy_thinking_without_output_config(
    mock_anthropic_class: Mock,
) -> None:
    mock_client = _mock_streaming_client(mock_anthropic_class)

    execute_claude_request(
        roboflow_api_key=None,
        anthropic_api_key="sk-ant-test",
        system_prompt=None,
        messages=[{"role": "user", "content": "Think"}],
        model_version="claude-sonnet-4-5",
        max_tokens=10000,
        temperature=None,
        extended_thinking=True,
        thinking_budget_tokens=5000,
        reasoning_effort=None,
    )

    call_kwargs = mock_client.messages.stream.call_args.kwargs
    assert call_kwargs["thinking"] == {"type": "enabled", "budget_tokens": 5000}
    assert "extra_body" not in call_kwargs


@patch(
    "inference.core.workflows.core_steps.models.foundation.anthropic_claude.v5.post_to_roboflow_api",
    return_value=PROXY_RESPONSE,
)
def test_proxied_request_sends_effort_without_thinking_for_fable(
    post_mock: Mock,
) -> None:
    execute_claude_request(
        roboflow_api_key="rf-key",
        anthropic_api_key="rf_key:account",
        system_prompt=None,
        messages=[{"role": "user", "content": "Hello"}],
        model_version="claude-fable-5-1",
        max_tokens=100,
        temperature=None,
        extended_thinking=None,
        thinking_budget_tokens=None,
        reasoning_effort="medium",
    )

    payload = post_mock.call_args.kwargs["payload"]
    assert "thinking" not in payload
    assert payload["output_config"] == {"effort": "medium"}


@patch(
    "inference.core.workflows.core_steps.models.foundation.anthropic_claude.v5.post_to_roboflow_api",
    return_value=PROXY_RESPONSE,
)
def test_proxied_request_sends_adaptive_thinking_and_effort_for_fable(
    post_mock: Mock,
) -> None:
    execute_claude_request(
        roboflow_api_key="rf-key",
        anthropic_api_key="rf_key:account",
        system_prompt=None,
        messages=[{"role": "user", "content": "Think"}],
        model_version="claude-fable-5-1",
        max_tokens=None,
        temperature=None,
        extended_thinking=True,
        thinking_budget_tokens=5000,
        reasoning_effort="low",
    )

    payload = post_mock.call_args.kwargs["payload"]
    assert payload["thinking"] == {"type": "adaptive"}
    assert payload["output_config"] == {"effort": "low"}


@patch(
    "inference.core.workflows.core_steps.models.foundation.anthropic_claude.v5.post_to_roboflow_api",
    return_value=PROXY_RESPONSE,
)
def test_proxied_request_keeps_legacy_thinking_without_output_config(
    post_mock: Mock,
) -> None:
    execute_claude_request(
        roboflow_api_key="rf-key",
        anthropic_api_key="rf_key:account",
        system_prompt=None,
        messages=[{"role": "user", "content": "Think"}],
        model_version="claude-sonnet-4-5",
        max_tokens=10000,
        temperature=None,
        extended_thinking=True,
        thinking_budget_tokens=5000,
        reasoning_effort=None,
    )

    payload = post_mock.call_args.kwargs["payload"]
    assert payload["thinking"] == {"type": "enabled", "budget_tokens": 5000}
    assert "output_config" not in payload


@patch(
    "inference.core.workflows.core_steps.models.foundation.anthropic_claude.v5.post_to_roboflow_api",
    return_value=PROXY_RESPONSE,
)
def test_block_run_threads_reasoning_effort_to_the_request(post_mock: Mock) -> None:
    block = AnthropicClaudeBlockV5(model_manager=MagicMock(), api_key="rf-key")
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="parent"),
        numpy_image=np.zeros((16, 16, 3), dtype=np.uint8),
    )

    result = block.run(
        images=Batch(content=[image], indices=[(0,)]),
        task_type="unconstrained",
        prompt="Describe the image",
        output_structure=None,
        classes=None,
        model_version="claude-fable-5-1",
        max_tokens=None,
        temperature=None,
        extended_thinking=None,
        thinking_budget_tokens=None,
        reasoning_effort="low",
        max_image_size=1024,
        max_concurrent_requests=None,
        api_key="rf_key:account",
    )

    payload = post_mock.call_args.kwargs["payload"]
    assert payload["model"] == "claude-fable-5-1"
    assert payload["output_config"] == {"effort": "low"}
    assert "thinking" not in payload
    assert result == [
        {
            "output": "proxied",
            "classes": None,
            "input_tokens": 3,
            "output_tokens": 2,
        }
    ]

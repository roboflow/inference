"""Tests for the Anthropic Claude v4 block (v3 + token-usage outputs).

The v1-v3 behavior suite lives in ``test_anthropic_claude.py``; this file
covers the v4 delta: ``input_tokens`` / ``output_tokens`` outputs on both
the proxied and direct execution paths.
"""

from unittest.mock import MagicMock, Mock, patch

from inference.core.workflows.core_steps.models.foundation.anthropic_claude.v4 import (
    BlockManifest,
    _execute_direct_claude_request,
    _execute_proxied_claude_request,
)

_CLAUDE_OK = {
    "stop_reason": "end_turn",
    "content": [{"type": "text", "text": "ok"}],
}


def test_manifest_declares_token_outputs():
    outputs = {output.name for output in BlockManifest.describe_outputs()}
    assert {"input_tokens", "output_tokens"} <= outputs


@patch(
    "inference.core.workflows.core_steps.models.foundation.anthropic_claude.v4.post_to_roboflow_api"
)
def test_proxied_request_returns_usage(mock_post: Mock) -> None:
    mock_post.return_value = {
        **_CLAUDE_OK,
        "usage": {"input_tokens": 30, "output_tokens": 9},
    }

    result = _execute_proxied_claude_request(
        roboflow_api_key="rf_api_key",
        anthropic_api_key="rf_key:account",
        system_prompt=None,
        messages=[],
        model_version="claude-sonnet-4-5",
        max_tokens=1000,
        temperature=None,
        extended_thinking=None,
        thinking_budget_tokens=None,
    )

    assert result == ("ok", 30, 9)


@patch(
    "inference.core.workflows.core_steps.models.foundation.anthropic_claude.v4.post_to_roboflow_api"
)
def test_proxied_request_usage_none_when_omitted(mock_post: Mock) -> None:
    mock_post.return_value = _CLAUDE_OK

    result = _execute_proxied_claude_request(
        roboflow_api_key="rf_api_key",
        anthropic_api_key="rf_key:account",
        system_prompt=None,
        messages=[],
        model_version="claude-sonnet-4-5",
        max_tokens=1000,
        temperature=None,
        extended_thinking=None,
        thinking_budget_tokens=None,
    )

    assert result == ("ok", None, None)


@patch(
    "inference.core.workflows.core_steps.models.foundation.anthropic_claude.v4.anthropic.Anthropic"
)
def test_direct_request_returns_usage(mock_anthropic_class: Mock) -> None:
    mock_client = MagicMock()
    mock_anthropic_class.return_value = mock_client

    mock_text_block = Mock()
    mock_text_block.type = "text"
    mock_text_block.text = "ok"

    mock_result = Mock()
    mock_result.stop_reason = "end_turn"
    mock_result.content = [mock_text_block]
    mock_result.usage = Mock(input_tokens=18, output_tokens=5)

    mock_stream = MagicMock()
    mock_stream.__enter__ = Mock(return_value=mock_stream)
    mock_stream.__exit__ = Mock(return_value=False)
    mock_stream.get_final_message.return_value = mock_result
    mock_client.messages.stream.return_value = mock_stream

    result = _execute_direct_claude_request(
        anthropic_api_key="sk-ant-test",
        system_prompt=None,
        messages=[],
        model_version="claude-sonnet-4-5",
        max_tokens=1000,
        temperature=None,
        extended_thinking=None,
        thinking_budget_tokens=None,
    )

    assert result == ("ok", 18, 5)

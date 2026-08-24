"""Tests for the OpenAI v6 block (v5 + token-usage outputs).

The v5 behavior suite lives in ``test_openai_v5.py``; this file covers the
v6 delta: ``input_tokens`` / ``output_tokens`` outputs on both the proxied
and direct execution paths.
"""

from unittest.mock import MagicMock, Mock, patch

from inference.core.workflows.core_steps.models.foundation.openai.v6 import (
    _execute_direct_openai_request,
    _execute_proxied_openai_request,
)

_OPENAI_OK = {
    "status": "completed",
    "output": [
        {
            "type": "message",
            "content": [{"type": "output_text", "text": "ok"}],
        }
    ],
}


@patch(
    "inference.core.workflows.core_steps.models.foundation.openai.v6.post_to_roboflow_api"
)
def test_proxied_request_returns_usage_and_none_when_omitted(mock_post: Mock) -> None:
    def call():
        return _execute_proxied_openai_request(
            roboflow_api_key="rf_api_key",
            openai_api_key="rf_key:account",
            instructions="test",
            input_content=[],
            model_version="gpt-5.1",
            reasoning_effort=None,
            max_tokens=None,
            temperature=None,
        )

    mock_post.return_value = {
        **_OPENAI_OK,
        "usage": {"input_tokens": 21, "output_tokens": 6},
    }
    assert call() == ("ok", 21, 6)

    mock_post.return_value = _OPENAI_OK
    assert call() == ("ok", None, None)


@patch(
    "inference.core.workflows.core_steps.models.foundation.openai.v6._get_openai_client"
)
def test_direct_request_returns_usage(mock_get_client: Mock) -> None:
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.status = "completed"
    mock_response.output_text = "response"
    mock_response.usage = MagicMock(input_tokens=14, output_tokens=3)
    mock_client.responses.create.return_value = mock_response
    mock_get_client.return_value = mock_client

    result = _execute_direct_openai_request(
        openai_api_key="sk-test",
        instructions="test",
        input_content=[{"role": "user", "content": []}],
        model_version="gpt-5.1",
        reasoning_effort=None,
        max_tokens=None,
        temperature=None,
    )

    assert result == ("response", 14, 3)

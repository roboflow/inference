"""Tests for the SpaceXAI v2 block (v1 + token-usage outputs).

The v1 behavior suite lives in ``test_spacexai.py``; this file covers the
v2 delta: ``input_tokens`` / ``output_tokens`` outputs on both the proxied
and direct execution paths.
"""

from unittest.mock import MagicMock, patch

import pytest

from inference.core.workflows.core_steps.models.foundation.spacexai.v2 import (
    _execute_direct_spacexai_request,
    _execute_proxied_spacexai_request,
)
from tests.workflows.unit_tests.prototypes.platform_client_double import (
    RecordingPlatformClient,
)

platform_client = RecordingPlatformClient()


@pytest.fixture(autouse=True)
def _reset_platform_client():
    platform_client.reset()


_XAI_OK = {
    "status": "completed",
    "output": [
        {
            "type": "message",
            "content": [{"type": "output_text", "text": "ok"}],
        }
    ],
}


def test_proxied_request_returns_usage_and_none_when_omitted() -> None:
    mock_post = platform_client.post_mock

    def call():
        return _execute_proxied_spacexai_request(
            roboflow_api_key="rf_abc",
            platform_client=platform_client,
            xai_api_key="rf_key:account",
            instructions=None,
            input_content=[],
            model_version="grok-4.6",
            reasoning_effort=None,
            max_tokens=None,
            temperature=None,
        )

    mock_post.return_value = {
        **_XAI_OK,
        "usage": {"input_tokens": 16, "output_tokens": 5},
    }
    assert call() == ("ok", 16, 5)

    mock_post.return_value = _XAI_OK
    assert call() == ("ok", None, None)


@patch("inference.core.workflows.core_steps.models.foundation.spacexai.v2.OpenAI")
def test_direct_request_returns_usage(mock_openai_cls: MagicMock) -> None:
    client = MagicMock()
    response = MagicMock()
    response.status = "completed"
    response.output_text = "ok"
    response.usage = MagicMock(input_tokens=12, output_tokens=4)
    client.responses.create.return_value = response
    mock_openai_cls.return_value = client

    result = _execute_direct_spacexai_request(
        xai_api_key="xai-secret",
        instructions=None,
        input_content=[],
        model_version="grok-4.6",
        reasoning_effort=None,
        max_tokens=None,
        temperature=None,
    )

    assert result == ("ok", 12, 4)

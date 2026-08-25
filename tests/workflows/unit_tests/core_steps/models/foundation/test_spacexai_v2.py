"""Tests for the SpaceXAI v2 block (v1 + token-usage outputs).

The v1 behavior suite lives in ``test_spacexai.py``; this file covers the
v2 delta: ``input_tokens`` / ``output_tokens`` outputs on both the proxied
and direct execution paths.
"""

from unittest.mock import MagicMock, patch

from inference.core.workflows.core_steps.models.foundation.spacexai.v2 import (
    BlockManifest,
    _execute_direct_spacexai_request,
    _execute_proxied_spacexai_request,
    execute_spacexai_request,
)

_XAI_OK = {
    "status": "completed",
    "output": [
        {
            "type": "message",
            "content": [{"type": "output_text", "text": "ok"}],
        }
    ],
}


def test_spacexai_v2_defaults_api_key_to_rf_key_account() -> None:
    result = BlockManifest.model_validate(
        {
            "type": "roboflow_core/spacexai@v2",
            "name": "step_1",
            "images": "$inputs.image",
            "task_type": "caption",
        }
    )

    assert result.api_key == "rf_key:account"


@patch(
    "inference.core.workflows.core_steps.models.foundation.spacexai.v2."
    "_execute_proxied_spacexai_request"
)
def test_execute_spacexai_request_routes_rf_key_to_proxy(
    proxied_mock: MagicMock,
) -> None:
    proxied_mock.return_value = ("proxied", 16, 5)
    result = execute_spacexai_request(
        roboflow_api_key="rf_abc",
        xai_api_key="rf_key:account",
        instructions=None,
        input_content=[{"role": "user", "content": []}],
        model_version="grok-4.6",
        reasoning_effort=None,
        max_tokens=None,
        temperature=None,
    )
    assert result == ("proxied", 16, 5)
    proxied_mock.assert_called_once()


@patch(
    "inference.core.workflows.core_steps.models.foundation.spacexai.v2.post_to_roboflow_api"
)
def test_proxied_request_returns_usage_and_none_when_omitted(
    mock_post: MagicMock,
) -> None:
    def call():
        return _execute_proxied_spacexai_request(
            roboflow_api_key="rf_abc",
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

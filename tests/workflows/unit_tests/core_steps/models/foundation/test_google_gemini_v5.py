"""Tests for the Google Gemini v5 block (v4 + token-usage outputs).

The v1-v4 behavior suite lives in ``test_google_gemini.py``; this file
covers the v5 delta: ``input_tokens`` / ``output_tokens`` outputs on both
the proxied and direct execution paths. ``output_tokens`` includes
Gemini's ``thoughtsTokenCount`` (billing parity).
"""

from unittest.mock import Mock, patch

from inference.core.workflows.core_steps.models.foundation.google_gemini.v5 import (
    BlockManifest,
    _execute_direct_gemini_request,
    _execute_proxied_gemini_request,
)

_GEMINI_OK = {
    "candidates": [
        {
            "content": {"parts": [{"text": "ok"}]},
            "finishReason": "STOP",
        }
    ]
}


def test_manifest_declares_token_outputs():
    outputs = {output.name for output in BlockManifest.describe_outputs()}
    assert {"input_tokens", "output_tokens"} <= outputs


@patch(
    "inference.core.workflows.core_steps.models.foundation.google_gemini.v5.post_to_roboflow_api"
)
def test_proxied_request_returns_usage_including_thoughts(mock_post: Mock) -> None:
    mock_post.return_value = {
        **_GEMINI_OK,
        "usageMetadata": {
            "promptTokenCount": 15,
            "candidatesTokenCount": 6,
            "thoughtsTokenCount": 4,
        },
    }

    result = _execute_proxied_gemini_request(
        roboflow_api_key="rf_api_key",
        google_api_key="rf_key:account",
        prompt={"contents": {"parts": [{"text": "test"}]}},
        model_version="gemini-2.5-pro",
    )

    assert result == ("ok", 15, 10)


@patch(
    "inference.core.workflows.core_steps.models.foundation.google_gemini.v5.post_to_roboflow_api"
)
def test_proxied_request_usage_none_when_omitted(mock_post: Mock) -> None:
    mock_post.return_value = _GEMINI_OK

    result = _execute_proxied_gemini_request(
        roboflow_api_key="rf_api_key",
        google_api_key="rf_key:account",
        prompt={"contents": {"parts": [{"text": "test"}]}},
        model_version="gemini-2.5-pro",
    )

    assert result == ("ok", None, None)


@patch(
    "inference.core.workflows.core_steps.models.foundation.google_gemini.v5.requests.post"
)
def test_direct_request_returns_usage(mock_post: Mock) -> None:
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        **_GEMINI_OK,
        "usageMetadata": {"promptTokenCount": 8, "candidatesTokenCount": 2},
    }
    mock_post.return_value = mock_response

    result = _execute_direct_gemini_request(
        google_api_key="user-google-key",
        prompt={"contents": {"parts": [{"text": "test"}]}},
        model_version="gemini-2.5-pro",
    )

    assert result == ("ok", 8, 2)

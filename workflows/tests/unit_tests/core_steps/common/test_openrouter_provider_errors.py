"""Provider diagnostics survive OpenRouter and Roboflow error envelopes."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from roboflow_workflows.core_steps.common.openrouter import (
    _build_proxy_error_handler,
    _execute_direct_openrouter_request,
    _execute_proxied_openrouter_request,
    _openrouter_error_message,
)
from roboflow_workflows.prototypes.platform_errors import (
    RoboflowAPIForbiddenError,
    RoboflowAPIUnsuccessfulRequestError,
)

DETAIL = "reasoning_effort none is not supported for this model, supported values: [high, low, medium]"
RAW = {
    "error": {"message": DETAIL, "type": "invalid_request_invalid_args", "code": 3051}
}


def _error(raw):
    return {"message": "Provider returned error", "metadata": {"raw": raw}}


@pytest.mark.parametrize("raw", [RAW, json.dumps(RAW), DETAIL])
@pytest.mark.parametrize("lane", ["direct", "proxy"])
def test_success_status_provider_error_retains_raw_message(raw, lane):
    error = _error(raw)
    kwargs = dict(
        model="mistralai/mistral-large-4-0",
        messages=[],
        max_tokens=None,
        temperature=None,
        privacy_level="deny",
    )
    client = MagicMock()
    with pytest.raises(RuntimeError, match="reasoning_effort none") as exc:
        if lane == "direct":
            client.chat.completions.create.return_value = SimpleNamespace(
                choices=[], error=error
            )
            with patch(
                "roboflow_workflows.core_steps.common.openrouter.OpenAI",
                return_value=client,
            ):
                _execute_direct_openrouter_request(api_key="test-key", **kwargs)
        else:
            client.post.return_value = {"choices": [], "error": error}
            _execute_proxied_openrouter_request(
                roboflow_api_key="test-key",
                platform_client=client,
                openrouter_api_key="rf_key:account",
                **kwargs,
            )
    assert "Provider returned error" in str(exc.value)
    assert DETAIL in str(exc.value)


@pytest.mark.parametrize("status", [400, 403, 502])
@pytest.mark.parametrize("wrapper", ["error", "details"])
def test_proxy_http_error_preserves_provider_message_and_status(status, wrapper):
    original = Exception("HTTP error")
    original.response = MagicMock()
    original.response.json.return_value = {wrapper: _error(json.dumps(RAW))}
    error_type = (
        RoboflowAPIForbiddenError
        if status == 403
        else RoboflowAPIUnsuccessfulRequestError
    )
    with pytest.raises(error_type, match="reasoning_effort none") as exc:
        _build_proxy_error_handler(status)(original)
    assert exc.value.status_code == status
    assert exc.value.__cause__ is original


@pytest.mark.parametrize(
    "data", [None, [], 7, {"metadata": []}, {"metadata": {"raw": []}}]
)
def test_malformed_error_uses_fallback(data):
    assert (
        _openrouter_error_message(data, fallback="original error") == "original error"
    )


def test_invalid_json_raw_is_preserved():
    assert (
        _openrouter_error_message(_error("upstream unavailable {"), fallback="fallback")
        == "Provider returned error: upstream unavailable {"
    )


def test_unrelated_metadata_is_not_included():
    error = _error(RAW)
    error["metadata"]["headers"] = {"Authorization": "not-for-output"}
    assert "not-for-output" not in _openrouter_error_message(error, fallback="fallback")


@pytest.mark.parametrize("retry", [False, True])
def test_direct_http_error_retains_provider_details_and_sdk_type(retry):
    from openai import BadRequestError

    response = MagicMock(status_code=400)
    provider_error = BadRequestError(
        "Provider returned error", response=response, body=_error(json.dumps(RAW))
    )
    client = MagicMock()
    if retry:
        first_error = BadRequestError(
            "reasoning is not supported", response=response, body={}
        )
        client.chat.completions.create.side_effect = [first_error, provider_error]
    else:
        client.chat.completions.create.side_effect = provider_error
    with patch(
        "roboflow_workflows.core_steps.common.openrouter.OpenAI", return_value=client
    ):
        with pytest.raises(BadRequestError, match="reasoning_effort none") as exc:
            _execute_direct_openrouter_request(
                api_key="test-key",
                model="mistralai/mistral-large-4-0",
                messages=[],
                max_tokens=None,
                temperature=None,
                privacy_level="deny",
                reasoning={"effort": "low"},
            )
    assert exc.value is provider_error
    assert exc.value.status_code == 400
    assert exc.value.body == _error(json.dumps(RAW))
    assert client.chat.completions.create.call_count == (2 if retry else 1)

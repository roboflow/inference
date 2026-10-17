import json

import aiohttp
import anthropic
import httpx
import openai
import pytest
import requests

from inference.core.interfaces.http.error_diagnostics import workflow_error_diagnostics


def proxy_error(body):
    response = requests.Response()
    response.status_code = 502
    response._content = json.dumps(body).encode()
    response._content_consumed = True
    return requests.HTTPError(response=response)


@pytest.mark.parametrize("value", [None, [], {}, 42, "sk-secret-value"])
def test_ignores_invalid_optional_fields(value):
    error = proxy_error(
        {
            "api_proxy_error_type": "upstream_error",
            "provider": "openai",
            "provider_request_id": value,
            "provider_error_code": value,
            "upstream_status_code": value,
            "failure_kind": value,
        }
    )
    assert workflow_error_diagnostics(error) == {
        "provider": "openai",
        "api_proxy_error_type": "upstream_error",
        "proxy_status_code": 502,
        "failure_kind": "http_error",
    }


@pytest.mark.parametrize(
    "body", [None, [], {}, {"provider": []}, {"api_proxy_error_type": {}}]
)
def test_ignores_unrecognized_proxy_responses(body):
    assert workflow_error_diagnostics(proxy_error(body)) is None


def test_does_not_read_streams_or_large_error_bodies():
    error = proxy_error(
        {
            "api_proxy_error_type": "upstream_error",
            "provider": "openai",
            "details": "x" * 16384,
        }
    )
    assert workflow_error_diagnostics(error) is None
    error.response._content_consumed = False
    assert workflow_error_diagnostics(error) is None


@pytest.mark.parametrize(
    "timeout_type,phase",
    [(requests.ReadTimeout, "read"), (requests.ConnectTimeout, "connect")],
)
def test_timeout_phase_is_observed_without_blame(timeout_type, phase):
    wrapped = RuntimeError("request failed")
    wrapped.__cause__ = timeout_type("https://secret.example?api_key=private")
    assert workflow_error_diagnostics(wrapped) == {
        "failure_kind": "timeout",
        "component": "http_client",
        "timeout_phase": phase,
    }


def test_direct_openai_sdk_error_does_not_assume_openai_provider():
    error = openai.RateLimitError(
        "private prompt sk-secret",
        response=httpx.Response(
            429,
            request=httpx.Request("POST", "https://example.test"),
            headers={"x-request-id": "req_456"},
        ),
        body={"code": "rate_limit_exceeded", "api_key": "sk-secret"},
    )
    assert workflow_error_diagnostics(error) == {
        "sdk": "openai",
        "provider_status_code": 429,
        "failure_kind": "http_error",
        "provider_error_code": "rate_limit_exceeded",
        "provider_request_id": "req_456",
    }


def test_direct_sdk_timeout_retains_classification():
    error = openai.APITimeoutError(
        request=httpx.Request("POST", "https://example.test")
    )
    assert workflow_error_diagnostics(error) == {
        "sdk": "openai",
        "failure_kind": "timeout",
    }


def test_cycle_and_unknown_error_are_ignored():
    error = ValueError("private prompt")
    error.__cause__ = error
    assert workflow_error_diagnostics(error) is None


@pytest.mark.parametrize(
    "error,kind",
    [
        (requests.ConnectionError("secret URL"), "network_error"),
        (aiohttp.ClientConnectionError("secret URL"), "network_error"),
        (aiohttp.ServerTimeoutError("secret URL"), "timeout"),
    ],
)
def test_sync_and_async_connection_failures(error, kind):
    assert workflow_error_diagnostics(error) == {
        "component": "http_client",
        "failure_kind": kind,
    }


def test_known_proxy_status_uses_actual_response_not_body_claim():
    error = proxy_error(
        {
            "api_proxy_error_type": "upstream_error",
            "provider": "openai",
            "proxy_status_code": 401,
            "upstream_status_code": 429,
        }
    )
    assert workflow_error_diagnostics(error)["proxy_status_code"] == 502


def test_direct_anthropic_sdk_retains_only_safe_fields():
    error = anthropic.RateLimitError(
        "private prompt sk-secret",
        response=httpx.Response(
            429,
            request=httpx.Request("POST", "https://example.test"),
            headers={"request-id": "req_anthropic"},
        ),
        body={
            "type": "rate_limit_error",
            "message": "private prompt",
            "api_key": "sk-secret",
        },
    )
    assert workflow_error_diagnostics(error) == {
        "sdk": "anthropic",
        "provider_status_code": 429,
        "failure_kind": "http_error",
        "provider_request_id": "req_anthropic",
        "provider_error_code": "rate_limit_error",
    }

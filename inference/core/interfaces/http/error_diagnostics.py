"""Allowlisted evidence from existing exception causes, without response bodies."""

import re
from typing import Dict, Optional, Union

import aiohttp
import requests

DiagnosticValue = Union[str, int]
PROVIDERS = {"anthropic", "openai", "openrouter", "gemini", "xai"}
FAILURE_KINDS = {"http_error", "timeout", "network_error", "invalid_response"}
ERROR_CODES = {
    "invalid_request_error",
    "authentication_error",
    "permission_error",
    "not_found_error",
    "rate_limit_error",
    "rate_limit_exceeded",
    "insufficient_quota",
    "overloaded_error",
    "api_error",
    "server_error",
    "context_length_exceeded",
    "invalid_api_key",
    "invalid_json_schema",
    "invalid_value",
    "invalid_type",
    "missing_required_parameter",
    "unsupported_parameter",
    "model_not_found",
    "INVALID_ARGUMENT",
    "UNAUTHENTICATED",
    "PERMISSION_DENIED",
    "NOT_FOUND",
    "RESOURCE_EXHAUSTED",
    "INTERNAL",
    "UNAVAILABLE",
    "DEADLINE_EXCEEDED",
}


def _one_of(value: object, *, choices: set) -> bool:
    return isinstance(value, str) and value in choices


def _request_id(value: object) -> Optional[str]:
    if not isinstance(value, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,128}", value):
        return None
    if re.match(r"sk-|AIza|Bearer", value, flags=re.IGNORECASE):
        return None
    return value


def _status(value: object) -> bool:
    return type(value) is int and 400 <= value <= 599


def _proxy_diagnostics(
    response: requests.Response,
) -> Optional[Dict[str, DiagnosticValue]]:
    # Only inspect buffered, small error envelopes. Never read a stream or copy
    # arbitrary response text into the public diagnostic fields.
    if (
        not _status(response.status_code)
        or not response._content_consumed
        or len(response.content) > 16384
    ):
        return None
    try:
        body = response.json()
    except ValueError:
        return None
    if not isinstance(body, dict):
        return None
    if not _one_of(
        body.get("api_proxy_error_type"),
        choices={"upstream_error", "proxy_internal_error"},
    ):
        return None
    if not _one_of(body.get("provider"), choices=PROVIDERS):
        return None
    result = {
        "provider": body["provider"],
        "api_proxy_error_type": body["api_proxy_error_type"],
        "proxy_status_code": response.status_code,
    }
    if _status(body.get("upstream_status_code")):
        result["upstream_status_code"] = body["upstream_status_code"]
    if _one_of(body.get("failure_kind"), choices=FAILURE_KINDS):
        result["failure_kind"] = body["failure_kind"]
    elif body["api_proxy_error_type"] == "upstream_error":
        result["failure_kind"] = "http_error"
    request_id = _request_id(body.get("provider_request_id"))
    if request_id:
        result["provider_request_id"] = request_id
    if _one_of(body.get("provider_error_code"), choices=ERROR_CODES):
        result["provider_error_code"] = body["provider_error_code"]
    return result


def _sdk_diagnostics(error: Exception) -> Optional[Dict[str, DiagnosticValue]]:
    module = type(error).__module__
    sdk = next(
        (
            name
            for prefix, name in (
                ("openai.", "openai"),
                ("anthropic.", "anthropic"),
                ("google.genai.", "gemini"),
            )
            if module == prefix[:-1] or module.startswith(prefix)
        ),
        None,
    )
    if sdk is None:
        return None
    # OpenAI-compatible SDKs also serve OpenRouter and xAI; the SDK alone
    # does not establish which provider received the request.
    result: Dict[str, DiagnosticValue] = {"sdk": sdk}
    status = getattr(error, "status_code", None)
    if status is None and sdk == "gemini":
        status = getattr(error, "code", None)
    if _status(status):
        result.update(provider_status_code=status, failure_kind="http_error")
    elif type(error).__name__ == "APITimeoutError":
        result["failure_kind"] = "timeout"
    elif type(error).__name__ == "APIConnectionError":
        result["failure_kind"] = "network_error"
    else:
        return None
    request_id = _request_id(getattr(error, "request_id", None))
    if request_id:
        result["provider_request_id"] = request_id
    code = (
        getattr(error, "status", None)
        if sdk == "gemini"
        else getattr(error, "code", None)
    )
    if not isinstance(code, str):
        body = getattr(error, "body", None)
        if isinstance(body, dict):
            details = body.get("error", body)
            if isinstance(details, dict):
                code = details.get("code") or details.get("type")
    if isinstance(code, str) and code in ERROR_CODES:
        result["provider_error_code"] = code
    return result


def workflow_error_diagnostics(
    error: Exception,
) -> Optional[Dict[str, DiagnosticValue]]:
    """Extract safe transport evidence from a workflow exception's explicit causes.

    Args:
        error: Workflow error wrapping a platform or provider SDK exception.

    Returns:
        Recognized status, provider, request identifier and failure category, or
        None when the cause does not carry known evidence. Existing messages and
        model outputs are not modified.
    """
    seen = set()
    current = error
    for _ in range(8):
        if not isinstance(current, Exception) or id(current) in seen:
            break
        seen.add(id(current))
        if (
            isinstance(current, requests.exceptions.HTTPError)
            and current.response is not None
        ):
            result = _proxy_diagnostics(current.response)
            if result:
                return result
        result = _sdk_diagnostics(current)
        if result:
            return result
        # These transport exceptions identify the API hop, not a vendor outage.
        if isinstance(current, requests.exceptions.Timeout):
            result = {"failure_kind": "timeout", "component": "http_client"}
            if isinstance(current, requests.exceptions.ReadTimeout):
                result["timeout_phase"] = "read"
            elif isinstance(current, requests.exceptions.ConnectTimeout):
                result["timeout_phase"] = "connect"
            return result
        if isinstance(current, aiohttp.ServerTimeoutError):
            return {"failure_kind": "timeout", "component": "http_client"}
        if isinstance(
            current,
            (requests.exceptions.ConnectionError, aiohttp.ClientConnectionError),
        ):
            return {"failure_kind": "network_error", "component": "http_client"}
        current = getattr(current, "inner_error", None) or current.__cause__
    return None

"""Tests for the shared OpenRouter base class and helpers."""

from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest
from openai import APIStatusError

from inference.core.exceptions import (
    RoboflowAPIConnectionError,
    RoboflowAPIUnsuccessfulRequestError,
)
from inference.core.workflows.core_steps.common.openrouter import (
    MODEL_NATIVE_QUANTIZATIONS,
    OpenRouterResult,
    OpenRouterWorkflowBlockBase,
    _execute_direct_openrouter_request,
    _execute_proxied_openrouter_request,
    _is_unsupported_reasoning_error,
    build_prompts_from_images,
    build_provider_routing,
    get_native_quantizations,
    validate_task_type_required_fields,
)

# Real error strings captured live from OpenRouter (2026-08-19):
# qwen/qwen3.8-max rejecting `reasoning: {"enabled": false}`:
MANDATORY_REASONING_ERROR = (
    "Reasoning is mandatory for this endpoint and cannot be disabled."
)
# qwen/qwen3.7-flash rejecting `reasoning: {"effort": "bogus"}`:
INVALID_REASONING_OPTION_ERROR = (
    "reasoning.effort: Invalid option: expected one of "
    '"max"|"xhigh"|"high"|"medium"|"low"|"minimal"|"none"'
)


def _openai_status_error(message: str, status_code: int) -> APIStatusError:
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    response = httpx.Response(status_code, request=request)
    return APIStatusError(message, response=response, body=None)


def _proxy_error(message: str, status_code: int) -> RoboflowAPIUnsuccessfulRequestError:
    """Mirror _build_proxy_error_handler: Roboflow exception with status attached."""
    error = RoboflowAPIUnsuccessfulRequestError(message)
    error.status_code = status_code
    return error


# ---------------------------------------------------------------------------
# build_provider_routing
# ---------------------------------------------------------------------------


def test_build_provider_routing_allow_returns_none():
    assert build_provider_routing("allow") is None


def test_build_provider_routing_deny_returns_data_collection_only():
    assert build_provider_routing("deny") == {"data_collection": "deny"}


def test_build_provider_routing_zdr_returns_zdr_and_deny():
    assert build_provider_routing("zdr") == {
        "data_collection": "deny",
        "zdr": True,
    }


def test_build_provider_routing_unknown_raises():
    with pytest.raises(ValueError, match="unknown privacy_level"):
        build_provider_routing("nope")


def test_build_provider_routing_merges_quantizations_with_privacy_filter():
    assert build_provider_routing("allow", quantizations=["bf16", "fp32"]) == {
        "quantizations": ["bf16", "fp32"]
    }
    assert build_provider_routing("deny", quantizations=["fp8", "bf16", "fp32"]) == {
        "data_collection": "deny",
        "quantizations": ["fp8", "bf16", "fp32"],
    }


# ---------------------------------------------------------------------------
# get_native_quantizations
# ---------------------------------------------------------------------------


def test_get_native_quantizations_returns_none_for_unregistered_model():
    # Deliberate: registering these `unknown`-precision SKUs would make them
    # unroutable.
    for slug in (
        "meta/muse-spark-1.1",
        "meta/muse-spark-1.2",
        "meta/muse-spark-1.3",
        "qwen/qwen3.8-max",
        "qwen/qwen3.8-flash",
        "qwen/qwen3.7-plus",
        "qwen/qwen3.5-flash-02-23",
        "z-ai/glm-5v-turbo",
        "deepseek/deepseek-v4-flash-vision-exp",
        "moonshotai/kimi-k2.6",
    ):
        assert get_native_quantizations(model=slug) is None, slug


def test_native_quantization_registry_never_allows_below_fp8():
    # Sub-FP8 labels would break the registry's native-or-higher guarantee.
    for slug, allowlist in MODEL_NATIVE_QUANTIZATIONS.items():
        assert not {"int4", "int8", "fp4", "fp6", "unknown"} & set(allowlist), slug


# ---------------------------------------------------------------------------
# OpenRouterWorkflowBlockBase.execute_openrouter_batch routing
# ---------------------------------------------------------------------------


class _FakeBlock(OpenRouterWorkflowBlockBase):
    """Concrete-enough subclass to instantiate the base (no manifest needed for routing tests)."""

    @classmethod
    def get_manifest(cls):
        return None  # not used in these tests

    def run(self, *args, **kwargs):
        raise NotImplementedError


def _stub_messages():
    return [[{"role": "user", "content": "hi"}], [{"role": "user", "content": "hello"}]]


@patch(
    "inference.core.workflows.core_steps.common.openrouter._execute_proxied_openrouter_request"
)
@patch(
    "inference.core.workflows.core_steps.common.openrouter._execute_direct_openrouter_request"
)
def test_execute_openrouter_batch_routes_to_proxy_for_managed_key(
    mock_direct, mock_proxied
):
    mock_proxied.side_effect = [
        OpenRouterResult(content="resp-1"),
        OpenRouterResult(content="resp-2"),
    ]
    block = _FakeBlock(model_manager=MagicMock(), api_key="rf-workspace-abc")

    out = block.execute_openrouter_batch(
        openrouter_api_key="rf_key:account",
        model="google/gemma-4-31b-it",
        prompts=_stub_messages(),
        max_tokens=100,
        temperature=0.5,
        privacy_level="deny",
        max_concurrent_requests=2,
    )

    assert out == ["resp-1", "resp-2"]
    assert mock_proxied.call_count == 2
    assert mock_direct.call_count == 0
    # Both calls share the same managed-key kwargs.
    for call in mock_proxied.call_args_list:
        kwargs = call.kwargs
        assert kwargs["roboflow_api_key"] == "rf-workspace-abc"
        assert kwargs["openrouter_api_key"] == "rf_key:account"
        assert kwargs["model"] == "google/gemma-4-31b-it"
        assert kwargs["privacy_level"] == "deny"
        assert kwargs["max_tokens"] == 100
        assert kwargs["temperature"] == 0.5


@patch(
    "inference.core.workflows.core_steps.common.openrouter._execute_proxied_openrouter_request"
)
@patch(
    "inference.core.workflows.core_steps.common.openrouter._execute_direct_openrouter_request"
)
def test_execute_openrouter_batch_routes_to_proxy_for_user_managed_key(
    mock_direct, mock_proxied
):
    """Both `rf_key:account` and `rf_key:user:<id>` route through the proxy.
    The platform decides whether to honor the user-stored variant."""
    mock_proxied.side_effect = [OpenRouterResult(content="resp")]
    block = _FakeBlock(model_manager=MagicMock(), api_key="ws-key")

    block.execute_openrouter_batch(
        openrouter_api_key="rf_key:user:abc-123",
        model="google/gemma-4-31b-it",
        prompts=[[{"role": "user", "content": "hi"}]],
        max_tokens=50,
        temperature=0.1,
        privacy_level="deny",
        max_concurrent_requests=1,
    )

    assert mock_proxied.call_count == 1
    assert mock_direct.call_count == 0
    assert mock_proxied.call_args.kwargs["openrouter_api_key"] == "rf_key:user:abc-123"


@patch(
    "inference.core.workflows.core_steps.common.openrouter._execute_proxied_openrouter_request"
)
@patch(
    "inference.core.workflows.core_steps.common.openrouter._execute_direct_openrouter_request"
)
def test_execute_openrouter_batch_routes_to_direct_for_user_key(
    mock_direct, mock_proxied
):
    mock_direct.side_effect = [OpenRouterResult(content="direct-resp")]
    block = _FakeBlock(model_manager=MagicMock(), api_key="ws-key")

    out = block.execute_openrouter_batch(
        openrouter_api_key="sk-or-v1-abcdef",
        model="google/gemma-4-31b-it",
        prompts=[[{"role": "user", "content": "hi"}]],
        max_tokens=50,
        temperature=0.7,
        privacy_level="zdr",
        max_concurrent_requests=1,
    )

    assert out == ["direct-resp"]
    assert mock_direct.call_count == 1
    assert mock_proxied.call_count == 0
    kwargs = mock_direct.call_args.kwargs
    assert kwargs["api_key"] == "sk-or-v1-abcdef"
    assert kwargs["privacy_level"] == "zdr"


@patch(
    "inference.core.workflows.core_steps.common.openrouter._execute_proxied_openrouter_request"
)
@patch(
    "inference.core.workflows.core_steps.common.openrouter._execute_direct_openrouter_request"
)
def test_execute_openrouter_batch_attaches_native_quantizations(
    mock_direct, mock_proxied
):
    """Registered models get the central allowlist on both routing paths."""
    mock_direct.side_effect = [OpenRouterResult(content="d")]
    mock_proxied.side_effect = [OpenRouterResult(content="p")]
    block = _FakeBlock(model_manager=MagicMock(), api_key="ws-key")
    common = dict(
        prompts=[[{"role": "user", "content": "hi"}]],
        max_tokens=50,
        temperature=0.1,
        privacy_level="deny",
        max_concurrent_requests=1,
    )

    block.execute_openrouter_batch(
        openrouter_api_key="sk-or-v1-abcdef",
        model="google/gemma-4-31b-it",
        **common,
    )
    block.execute_openrouter_batch(
        openrouter_api_key="rf_key:account",
        model="z-ai/glm-5.3-flash",
        **common,
    )

    assert mock_direct.call_args.kwargs["quantizations"] == ("bf16", "fp32")
    assert mock_proxied.call_args.kwargs["quantizations"] == ("fp8", "bf16", "fp32")


# ---------------------------------------------------------------------------
# _execute_proxied_openrouter_request: payload shape
# ---------------------------------------------------------------------------


@patch("inference.core.workflows.core_steps.common.openrouter.post_to_roboflow_api")
def test_proxied_request_sends_expected_payload_to_roboflow(mock_post):
    mock_post.return_value = {"choices": [{"message": {"content": "hello world"}}]}

    out = _execute_proxied_openrouter_request(
        roboflow_api_key="ws-key-xyz",
        openrouter_api_key="rf_key:account",
        model="moonshotai/kimi-k2.6",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=200,
        temperature=0.4,
        privacy_level="deny",
    )

    assert out == OpenRouterResult(content="hello world")
    mock_post.assert_called_once()
    call_kwargs = mock_post.call_args.kwargs
    assert call_kwargs["endpoint"] == "apiproxy/openrouter"
    assert call_kwargs["api_key"] == "ws-key-xyz"
    payload = call_kwargs["payload"]
    assert payload == {
        "openrouter_api_key": "rf_key:account",
        "model": "moonshotai/kimi-k2.6",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 200,
        "temperature": 0.4,
        "privacy_level": "deny",
    }
    # Block must NOT inject provider routing — that's the platform's job.
    assert "provider" not in payload


@patch("inference.core.workflows.core_steps.common.openrouter.post_to_roboflow_api")
def test_proxied_request_raises_when_choices_empty(mock_post):
    mock_post.return_value = {"choices": [], "error": {"message": "providers down"}}

    with pytest.raises(RuntimeError, match="providers down"):
        _execute_proxied_openrouter_request(
            roboflow_api_key="k",
            openrouter_api_key="rf_key:account",
            model="m",
            messages=[],
            max_tokens=1,
            temperature=0.0,
            privacy_level="deny",
        )


# ---------------------------------------------------------------------------
# OpenRouterResult: reasoning-trace and usage extraction
# ---------------------------------------------------------------------------


@patch("inference.core.workflows.core_steps.common.openrouter.post_to_roboflow_api")
def test_proxied_request_populates_reasoning_trace(mock_post):
    mock_post.return_value = {
        "choices": [{"message": {"content": "answer", "reasoning": "trace"}}]
    }

    out = _execute_proxied_openrouter_request(
        roboflow_api_key="k",
        openrouter_api_key="rf_key:account",
        model="qwen/qwen3.7-plus",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=10,
        temperature=None,
        privacy_level="deny",
    )

    assert out == OpenRouterResult(content="answer", reasoning_trace="trace")


@patch("inference.core.workflows.core_steps.common.openrouter.OpenAI")
def test_direct_request_returns_empty_trace_when_reasoning_missing(mock_openai_cls):
    client = MagicMock()
    response = MagicMock()
    choice = MagicMock()
    choice.message.content = "answer"
    # MagicMock auto-creates attributes; a non-str `reasoning` must map to ""
    # and non-int usage fields to None.
    response.choices = [choice]
    client.chat.completions.create.return_value = response
    mock_openai_cls.return_value = client

    out = _execute_direct_openrouter_request(
        api_key="sk-or-v1-test",
        model="qwen/qwen3.7-plus",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=10,
        temperature=None,
        privacy_level="deny",
    )

    assert out == OpenRouterResult(content="answer")


@patch("inference.core.workflows.core_steps.common.openrouter.post_to_roboflow_api")
def test_proxied_request_returns_usage_and_none_when_omitted(mock_post):
    def call():
        return _execute_proxied_openrouter_request(
            roboflow_api_key="k",
            openrouter_api_key="rf_key:account",
            model="qwen/qwen3.7-plus",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=10,
            temperature=None,
            privacy_level="deny",
        )

    mock_post.return_value = {
        "choices": [{"message": {"content": "answer"}}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7},
    }
    assert call() == OpenRouterResult(
        content="answer", input_tokens=11, output_tokens=7
    )

    mock_post.return_value = {"choices": [{"message": {"content": "answer"}}]}
    assert call() == OpenRouterResult(content="answer")


@patch("inference.core.workflows.core_steps.common.openrouter.OpenAI")
def test_direct_request_returns_trace_and_usage(mock_openai_cls):
    client = MagicMock()
    response = MagicMock()
    choice = MagicMock()
    choice.message.content = "answer"
    choice.message.reasoning = "trace"
    response.choices = [choice]
    response.usage = MagicMock(prompt_tokens=9, completion_tokens=4)
    client.chat.completions.create.return_value = response
    mock_openai_cls.return_value = client

    out = _execute_direct_openrouter_request(
        api_key="sk-or-v1-test",
        model="qwen/qwen3.7-plus",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=10,
        temperature=None,
        privacy_level="deny",
    )

    assert out == OpenRouterResult(
        content="answer", reasoning_trace="trace", input_tokens=9, output_tokens=4
    )


# ---------------------------------------------------------------------------
# _execute_direct_openrouter_request: provider injection
# ---------------------------------------------------------------------------


def _stub_openai_response(content: str) -> Any:
    """Build an object that mimics openai's ChatCompletion enough for the helper."""
    response = MagicMock()
    choice = MagicMock()
    choice.message.content = content
    response.choices = [choice]
    return response


@patch("inference.core.workflows.core_steps.common.openrouter.OpenAI")
def test_direct_request_injects_provider_data_collection_for_deny(mock_openai_cls):
    client = MagicMock()
    client.chat.completions.create.return_value = _stub_openai_response("ok")
    mock_openai_cls.return_value = client

    _execute_direct_openrouter_request(
        api_key="sk-or-v1-test",
        model="google/gemma-4-31b-it",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=10,
        temperature=0.1,
        privacy_level="deny",
    )

    mock_openai_cls.assert_called_once_with(
        base_url="https://openrouter.ai/api/v1",
        api_key="sk-or-v1-test",
    )
    create_kwargs = client.chat.completions.create.call_args.kwargs
    assert create_kwargs["extra_body"] == {"provider": {"data_collection": "deny"}}


@patch("inference.core.workflows.core_steps.common.openrouter.OpenAI")
def test_direct_request_injects_provider_zdr_when_zdr(mock_openai_cls):
    client = MagicMock()
    client.chat.completions.create.return_value = _stub_openai_response("ok")
    mock_openai_cls.return_value = client

    _execute_direct_openrouter_request(
        api_key="sk-or-v1-test",
        model="m",
        messages=[],
        max_tokens=1,
        temperature=0.0,
        privacy_level="zdr",
    )

    create_kwargs = client.chat.completions.create.call_args.kwargs
    assert create_kwargs["extra_body"] == {
        "provider": {"data_collection": "deny", "zdr": True},
    }


@patch("inference.core.workflows.core_steps.common.openrouter.OpenAI")
def test_direct_request_omits_provider_when_allow(mock_openai_cls):
    client = MagicMock()
    client.chat.completions.create.return_value = _stub_openai_response("ok")
    mock_openai_cls.return_value = client

    _execute_direct_openrouter_request(
        api_key="sk-or-v1-test",
        model="m",
        messages=[],
        max_tokens=1,
        temperature=0.0,
        privacy_level="allow",
    )

    create_kwargs = client.chat.completions.create.call_args.kwargs
    assert create_kwargs["extra_body"] == {}


@patch("inference.core.workflows.core_steps.common.openrouter.OpenAI")
def test_direct_request_injects_quantizations_into_provider(mock_openai_cls):
    client = MagicMock()
    client.chat.completions.create.return_value = _stub_openai_response("ok")
    mock_openai_cls.return_value = client

    _execute_direct_openrouter_request(
        api_key="sk-or-v1-test",
        model="google/gemma-4-31b-it",
        messages=[],
        max_tokens=1,
        temperature=0.0,
        privacy_level="deny",
        quantizations=["bf16", "fp32"],
    )

    create_kwargs = client.chat.completions.create.call_args.kwargs
    assert create_kwargs["extra_body"] == {
        "provider": {
            "data_collection": "deny",
            "quantizations": ["bf16", "fp32"],
        },
    }


@patch("inference.core.workflows.core_steps.common.openrouter.OpenAI")
def test_direct_request_sends_quantizations_even_when_privacy_allows(mock_openai_cls):
    """`allow` normally omits the provider object entirely; the quantization
    allowlist must still be delivered."""
    client = MagicMock()
    client.chat.completions.create.return_value = _stub_openai_response("ok")
    mock_openai_cls.return_value = client

    _execute_direct_openrouter_request(
        api_key="sk-or-v1-test",
        model="google/gemma-4-31b-it",
        messages=[],
        max_tokens=1,
        temperature=0.0,
        privacy_level="allow",
        quantizations=["bf16", "fp32"],
    )

    create_kwargs = client.chat.completions.create.call_args.kwargs
    assert create_kwargs["extra_body"] == {
        "provider": {"quantizations": ["bf16", "fp32"]},
    }


@patch("inference.core.workflows.core_steps.common.openrouter.post_to_roboflow_api")
def test_proxied_request_forwards_quantizations_in_payload(mock_post):
    mock_post.return_value = {"choices": [{"message": {"content": "ok"}}]}

    _execute_proxied_openrouter_request(
        roboflow_api_key="ws-key",
        openrouter_api_key="rf_key:account",
        model="z-ai/glm-5.3-flash",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=10,
        temperature=0.1,
        privacy_level="deny",
        quantizations=["fp8", "bf16", "fp32"],
    )

    payload = mock_post.call_args.kwargs["payload"]
    assert payload["quantizations"] == ["fp8", "bf16", "fp32"]


@patch("inference.core.workflows.core_steps.common.openrouter.OpenAI")
def test_direct_request_raises_when_choices_none(mock_openai_cls):
    response = MagicMock()
    response.choices = None
    response.error = {"message": "no providers available"}
    client = MagicMock()
    client.chat.completions.create.return_value = response
    mock_openai_cls.return_value = client

    with pytest.raises(RuntimeError, match="no providers available"):
        _execute_direct_openrouter_request(
            api_key="sk-or-v1",
            model="m",
            messages=[],
            max_tokens=1,
            temperature=0.0,
            privacy_level="deny",
        )


@patch("inference.core.workflows.core_steps.common.openrouter.OpenAI")
def test_direct_request_raises_when_message_content_is_none(mock_openai_cls):
    """OpenRouter can return a non-empty `choices` list with `message.content`
    None (e.g. when the model only emits tool calls or reasoning tokens)."""
    response = MagicMock()
    choice = MagicMock()
    choice.message.content = None
    response.choices = [choice]
    client = MagicMock()
    client.chat.completions.create.return_value = response
    mock_openai_cls.return_value = client

    with pytest.raises(RuntimeError, match="missing message.content"):
        _execute_direct_openrouter_request(
            api_key="sk-or-v1",
            model="m",
            messages=[],
            max_tokens=1,
            temperature=0.0,
            privacy_level="deny",
        )


# ---------------------------------------------------------------------------
# _is_unsupported_reasoning_error
# ---------------------------------------------------------------------------


def test_reasoning_error_matches_mandatory_reasoning_rejection_from_proxy():
    error = _proxy_error(MANDATORY_REASONING_ERROR, status_code=400)
    assert _is_unsupported_reasoning_error(error) is True


def test_reasoning_error_ignores_proxied_5xx_even_with_matching_message():
    # A transient upstream 502 relayed by the proxy carries the provider's
    # message verbatim; it must never trigger a duplicate billed request.
    error = _proxy_error(MANDATORY_REASONING_ERROR, status_code=502)
    assert _is_unsupported_reasoning_error(error) is False


def test_reasoning_error_ignores_proxy_exception_without_status():
    # Raised outside _PROXY_ERROR_HANDLERS (no status attached): not retryable.
    error = RoboflowAPIUnsuccessfulRequestError(MANDATORY_REASONING_ERROR)
    assert _is_unsupported_reasoning_error(error) is False


def test_reasoning_error_matches_invalid_option_rejection_from_direct_400():
    error = _openai_status_error(INVALID_REASONING_OPTION_ERROR, status_code=400)
    assert _is_unsupported_reasoning_error(error) is True


def test_reasoning_error_ignores_5xx_even_with_matching_message():
    error = _openai_status_error(MANDATORY_REASONING_ERROR, status_code=502)
    assert _is_unsupported_reasoning_error(error) is False


def test_reasoning_error_ignores_connection_errors_with_matching_message():
    # A transient failure whose text happens to mention reasoning must NOT
    # trigger a duplicate billed request.
    error = RoboflowAPIConnectionError(MANDATORY_REASONING_ERROR)
    assert _is_unsupported_reasoning_error(error) is False


def test_reasoning_error_ignores_unrelated_client_errors():
    error = _proxy_error("image exceeds maximum size", status_code=400)
    assert _is_unsupported_reasoning_error(error) is False


# ---------------------------------------------------------------------------
# reasoning / temperature payload shape
# ---------------------------------------------------------------------------


@patch("inference.core.workflows.core_steps.common.openrouter.post_to_roboflow_api")
def test_proxied_request_includes_reasoning_and_omits_none_temperature(mock_post):
    mock_post.return_value = {"choices": [{"message": {"content": "ok"}}]}

    _execute_proxied_openrouter_request(
        roboflow_api_key="ws-key",
        openrouter_api_key="rf_key:account",
        model="qwen/qwen3.7-flash",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=100,
        temperature=None,
        privacy_level="deny",
        reasoning={"enabled": False},
    )

    payload = mock_post.call_args.kwargs["payload"]
    assert payload["reasoning"] == {"enabled": False}
    assert "temperature" not in payload


@patch("inference.core.workflows.core_steps.common.openrouter.OpenAI")
def test_direct_request_includes_reasoning_and_omits_none_temperature(mock_openai_cls):
    client = MagicMock()
    client.chat.completions.create.return_value = _stub_openai_response("ok")
    mock_openai_cls.return_value = client

    _execute_direct_openrouter_request(
        api_key="sk-or-v1-test",
        model="qwen/qwen3.7-flash",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=100,
        temperature=None,
        privacy_level="deny",
        reasoning={"effort": "low"},
    )

    create_kwargs = client.chat.completions.create.call_args.kwargs
    assert create_kwargs["extra_body"]["reasoning"] == {"effort": "low"}
    assert "temperature" not in create_kwargs


# ---------------------------------------------------------------------------
# retry-without-reasoning fallback
# ---------------------------------------------------------------------------


@patch("inference.core.workflows.core_steps.common.openrouter.logger")
@patch("inference.core.workflows.core_steps.common.openrouter.post_to_roboflow_api")
def test_proxied_request_retries_without_reasoning_on_rejection(mock_post, mock_logger):
    mock_post.side_effect = [
        _proxy_error(MANDATORY_REASONING_ERROR, status_code=400),
        {"choices": [{"message": {"content": "ok"}}]},
    ]

    out = _execute_proxied_openrouter_request(
        roboflow_api_key="ws-key",
        openrouter_api_key="rf_key:account",
        model="qwen/qwen3.8-max",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=100,
        temperature=None,
        privacy_level="deny",
        reasoning={"enabled": False},
    )

    assert out == OpenRouterResult(content="ok")
    assert mock_post.call_count == 2
    assert "reasoning" in mock_post.call_args_list[0].kwargs["payload"]
    assert "reasoning" not in mock_post.call_args_list[1].kwargs["payload"]
    assert mock_logger.warning.call_count == 1


@patch("inference.core.workflows.core_steps.common.openrouter.logger")
@patch("inference.core.workflows.core_steps.common.openrouter.post_to_roboflow_api")
def test_proxied_retry_without_reasoning_keeps_quantizations(mock_post, mock_logger):
    """Dropping a rejected reasoning config must not drop the precision filter."""
    mock_post.side_effect = [
        _proxy_error(MANDATORY_REASONING_ERROR, status_code=400),
        {"choices": [{"message": {"content": "ok"}}]},
    ]

    _execute_proxied_openrouter_request(
        roboflow_api_key="ws-key",
        openrouter_api_key="rf_key:account",
        model="z-ai/glm-5.3-flash",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=100,
        temperature=None,
        privacy_level="deny",
        reasoning={"enabled": False},
        quantizations=["fp8", "bf16", "fp32"],
    )

    retry_payload = mock_post.call_args_list[1].kwargs["payload"]
    assert "reasoning" not in retry_payload
    assert retry_payload["quantizations"] == ["fp8", "bf16", "fp32"]


@patch("inference.core.workflows.core_steps.common.openrouter.post_to_roboflow_api")
def test_proxied_request_does_not_retry_on_relayed_502(mock_post):
    # Regression: the proxy relays upstream 5xx with the provider message
    # preserved; a reasoning-flavored 502 must not fire a duplicate request.
    mock_post.side_effect = _proxy_error(MANDATORY_REASONING_ERROR, status_code=502)

    with pytest.raises(RoboflowAPIUnsuccessfulRequestError):
        _execute_proxied_openrouter_request(
            roboflow_api_key="ws-key",
            openrouter_api_key="rf_key:account",
            model="qwen/qwen3.8-max",
            messages=[],
            max_tokens=1,
            temperature=None,
            privacy_level="deny",
            reasoning={"enabled": False},
        )

    assert mock_post.call_count == 1


@patch("inference.core.workflows.core_steps.common.openrouter.post_to_roboflow_api")
def test_proxied_request_does_not_retry_when_no_reasoning_sent(mock_post):
    mock_post.side_effect = _proxy_error(MANDATORY_REASONING_ERROR, status_code=400)

    with pytest.raises(RoboflowAPIUnsuccessfulRequestError):
        _execute_proxied_openrouter_request(
            roboflow_api_key="ws-key",
            openrouter_api_key="rf_key:account",
            model="qwen/qwen3.8-max",
            messages=[],
            max_tokens=1,
            temperature=None,
            privacy_level="deny",
        )

    assert mock_post.call_count == 1


@patch("inference.core.workflows.core_steps.common.openrouter.logger")
@patch("inference.core.workflows.core_steps.common.openrouter.OpenAI")
def test_direct_request_retries_without_reasoning_on_rejection(
    mock_openai_cls, mock_logger
):
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        _openai_status_error(INVALID_REASONING_OPTION_ERROR, status_code=400),
        _stub_openai_response("ok"),
    ]
    mock_openai_cls.return_value = client

    out = _execute_direct_openrouter_request(
        api_key="sk-or-v1-test",
        model="qwen/qwen3.7-flash",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=100,
        temperature=None,
        privacy_level="deny",
        reasoning={"effort": "low"},
    )

    assert out == OpenRouterResult(content="ok")
    assert client.chat.completions.create.call_count == 2
    first_extra = client.chat.completions.create.call_args_list[0].kwargs["extra_body"]
    second_extra = client.chat.completions.create.call_args_list[1].kwargs["extra_body"]
    assert "reasoning" in first_extra
    assert "reasoning" not in second_extra
    assert mock_logger.warning.call_count == 1


# ---------------------------------------------------------------------------
# batch-level reasoning forwarding
# ---------------------------------------------------------------------------


@patch(
    "inference.core.workflows.core_steps.common.openrouter._execute_proxied_openrouter_request"
)
def test_execute_openrouter_batch_forwards_reasoning_on_managed_key(mock_proxied):
    mock_proxied.side_effect = [OpenRouterResult(content="resp")]
    block = _FakeBlock(model_manager=MagicMock(), api_key="ws-key")

    block.execute_openrouter_batch(
        openrouter_api_key="rf_key:account",
        model="qwen/qwen3.7-flash",
        prompts=[[{"role": "user", "content": "hi"}]],
        max_tokens=50,
        temperature=None,
        privacy_level="deny",
        max_concurrent_requests=1,
        reasoning={"enabled": False},
    )

    assert mock_proxied.call_args.kwargs["reasoning"] == {"enabled": False}


# ---------------------------------------------------------------------------
# batch result shapes: legacy conversion vs. with_usage
# ---------------------------------------------------------------------------


@patch(
    "inference.core.workflows.core_steps.common.openrouter._execute_proxied_openrouter_request"
)
def test_execute_openrouter_batch_legacy_shapes_and_with_usage(mock_proxied):
    """Legacy method keeps its shipped shapes; with_usage exposes full results."""
    rich = OpenRouterResult(
        content="answer", reasoning_trace="trace", input_tokens=11, output_tokens=7
    )
    block = _FakeBlock(model_manager=MagicMock(), api_key="ws-key")
    kwargs = {
        "openrouter_api_key": "rf_key:account",
        "model": "qwen/qwen3.7-plus",
        "prompts": [[{"role": "user", "content": "hi"}]],
        "max_tokens": 50,
        "temperature": None,
        "privacy_level": "deny",
        "max_concurrent_requests": 1,
    }

    mock_proxied.return_value = rich
    assert block.execute_openrouter_batch(**kwargs) == ["answer"]
    assert block.execute_openrouter_batch(**kwargs, include_reasoning=True) == [
        ("answer", "trace")
    ]
    assert block.execute_openrouter_batch_with_usage(**kwargs) == [rich]


# ---------------------------------------------------------------------------
# validate_task_type_required_fields
# ---------------------------------------------------------------------------


def test_validate_task_type_unconstrained_requires_prompt():
    with pytest.raises(ValueError, match="`prompt`.*required"):
        validate_task_type_required_fields(
            task_type="unconstrained",
            prompt=None,
            classes=None,
            output_structure=None,
        )


def test_validate_task_type_classification_requires_classes():
    with pytest.raises(ValueError, match="`classes`.*required"):
        validate_task_type_required_fields(
            task_type="classification",
            prompt=None,
            classes=None,
            output_structure=None,
        )


def test_validate_task_type_structured_requires_output_structure():
    with pytest.raises(ValueError, match="`output_structure`.*required"):
        validate_task_type_required_fields(
            task_type="structured-answering",
            prompt=None,
            classes=None,
            output_structure=None,
        )


def test_validate_task_type_ocr_passes_with_no_extra_fields():
    # OCR doesn't need prompt/classes/output_structure — should not raise.
    validate_task_type_required_fields(
        task_type="ocr",
        prompt=None,
        classes=None,
        output_structure=None,
    )


@pytest.mark.parametrize(
    "effort, expected",
    [
        (None, None),
        ("none", {"enabled": False}),
        ("low", {"effort": "low"}),
        ("medium", {"effort": "medium"}),
        ("high", {"effort": "high"}),
        ("xhigh", {"effort": "xhigh"}),
    ],
)
def test_build_openrouter_reasoning_config(effort, expected):
    from inference.core.workflows.core_steps.common.reasoning import (
        build_openrouter_reasoning_config,
    )

    assert build_openrouter_reasoning_config(effort) == expected

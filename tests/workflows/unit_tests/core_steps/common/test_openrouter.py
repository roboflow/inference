"""Tests for the shared OpenRouter base class and helpers."""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from inference.core.workflows.core_steps.common.openrouter import (
    OpenRouterWorkflowBlockBase,
    build_provider_routing,
    build_prompts_from_images,
    validate_task_type_required_fields,
)
from inference.core.workflows.core_steps.common.openrouter import (
    _execute_direct_openrouter_request,
    _execute_proxied_openrouter_request,
)


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
    mock_proxied.side_effect = ["resp-1", "resp-2"]
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
    mock_proxied.side_effect = ["resp"]
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
    mock_direct.side_effect = ["direct-resp"]
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


# ---------------------------------------------------------------------------
# _execute_proxied_openrouter_request: payload shape
# ---------------------------------------------------------------------------


@patch("inference.core.workflows.core_steps.common.openrouter.post_to_roboflow_api")
def test_proxied_request_sends_expected_payload_to_roboflow(mock_post):
    mock_post.return_value = {
        "choices": [{"message": {"content": "hello world"}}]
    }

    out = _execute_proxied_openrouter_request(
        roboflow_api_key="ws-key-xyz",
        openrouter_api_key="rf_key:account",
        model="moonshotai/kimi-k2.6",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=200,
        temperature=0.4,
        privacy_level="deny",
    )

    assert out == "hello world"
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

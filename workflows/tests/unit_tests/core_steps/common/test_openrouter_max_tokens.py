"""How `max_tokens=None` reaches OpenRouter on each path.

Blocks that follow the provider-default convention (``max_tokens`` unset)
pass ``None`` through the shared executor. The direct OpenAI-SDK path must
omit the key. The Roboflow proxy requires it (absent -> its own 500-token
default, which truncated detection JSON live) and caps it at 16384, so the
proxied path sends that ceiling instead.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from roboflow_workflows.core_steps.common.openrouter import (
    PROXY_MAX_TOKENS_CEILING,
    _execute_direct_openrouter_request,
    _execute_proxied_openrouter_request,
)

from tests.unit_tests.prototypes.platform_client_double import RecordingPlatformClient

MESSAGES = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
PROXY_RESPONSE = {
    "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 3, "completion_tokens": 1},
}


def _sdk_response() -> SimpleNamespace:
    message = SimpleNamespace(content="ok", reasoning=None)
    choice = SimpleNamespace(message=message, finish_reason="stop")
    usage = SimpleNamespace(prompt_tokens=3, completion_tokens=1)
    return SimpleNamespace(choices=[choice], usage=usage)


@pytest.mark.parametrize(
    "max_tokens,expected_sent",
    [(None, PROXY_MAX_TOKENS_CEILING), (4096, 4096)],
)
def test_proxied_request_always_sends_an_integer_max_tokens(
    max_tokens, expected_sent
) -> None:
    client = RecordingPlatformClient(post_response=PROXY_RESPONSE)

    result = _execute_proxied_openrouter_request(
        roboflow_api_key="workspace-key",
        platform_client=client,
        openrouter_api_key="rf_key:account",
        model="mistralai/mistral-large-4-0",
        messages=MESSAGES,
        max_tokens=max_tokens,
        temperature=None,
        privacy_level="deny",
    )

    payload = client.posts[0]["payload"]
    assert result.content == "ok"
    assert payload["max_tokens"] == expected_sent
    assert 1 <= payload["max_tokens"] <= 16384
    assert "temperature" not in payload


@pytest.mark.parametrize("max_tokens", [None, 4096])
def test_direct_request_only_sends_max_tokens_when_set(max_tokens) -> None:
    client = MagicMock()
    client.chat.completions.create.return_value = _sdk_response()

    with patch(
        "roboflow_workflows.core_steps.common.openrouter.OpenAI", return_value=client
    ):
        result = _execute_direct_openrouter_request(
            api_key="sk-or-test",
            model="mistralai/mistral-large-4-0",
            messages=MESSAGES,
            max_tokens=max_tokens,
            temperature=None,
            privacy_level="deny",
        )

    kwargs = client.chat.completions.create.call_args.kwargs
    assert result.content == "ok"
    if max_tokens is None:
        assert "max_tokens" not in kwargs
    else:
        assert kwargs["max_tokens"] == max_tokens
    assert "temperature" not in kwargs

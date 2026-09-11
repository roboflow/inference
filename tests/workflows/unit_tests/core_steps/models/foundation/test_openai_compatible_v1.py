import json
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import numpy as np
import pytest
from pydantic import ValidationError

from inference.core.workflows.core_steps.models.foundation.openai_compatible import (
    v1 as openai_compatible,
)
from inference.core.workflows.core_steps.models.foundation.openai_compatible.v1 import (
    BlockManifest,
    OpenAICompatibleBlockV1,
    _build_messages,
    _build_prompt_content,
    _collect_image_data_urls,
    _encode_single_image,
    _is_image_list,
    _is_image_value,
    _resolve_parameters,
)
from inference.core.workflows.execution_engine.entities.base import WorkflowImageData


def test_manifest_validation_valid_input() -> None:
    specification = {
        "type": "roboflow_core/openai_compatible@v1",
        "name": "step_1",
        "base_url": "http://localhost:8000/v1",
        "model_name": "Qwen/Qwen2.5-VL-7B-Instruct",
        "prompt": "Describe what you see.",
    }
    result = BlockManifest.model_validate(specification)
    assert result.type == "roboflow_core/openai_compatible@v1"
    assert result.base_url == "http://localhost:8000/v1"
    assert result.model_name == "Qwen/Qwen2.5-VL-7B-Instruct"


def test_manifest_validation_with_selectors() -> None:
    specification = {
        "type": "roboflow_core/openai_compatible@v1",
        "name": "step_1",
        "base_url": "$inputs.base_url",
        "model_name": "$inputs.model_name",
        "prompt": "$inputs.prompt",
        "api_key": "$inputs.api_key",
    }
    result = BlockManifest.model_validate(specification)
    assert result.base_url == "$inputs.base_url"


def test_manifest_validation_missing_prompt() -> None:
    specification = {
        "type": "roboflow_core/openai_compatible@v1",
        "name": "step_1",
        "base_url": "http://localhost:8000/v1",
        "model_name": "test-model",
    }
    with pytest.raises(ValidationError):
        BlockManifest.model_validate(specification)


def test_manifest_validation_with_parameters() -> None:
    specification = {
        "type": "roboflow_core/openai_compatible@v1",
        "name": "step_1",
        "base_url": "http://localhost:8000/v1",
        "model_name": "test-model",
        "prompt": "Count {{ $parameters.obj_type }} objects",
        "prompt_parameters": {
            "obj_type": "$steps.classifier.class_name",
        },
    }
    result = BlockManifest.model_validate(specification)
    assert "obj_type" in result.prompt_parameters


def test_is_image_value_with_bytes() -> None:
    assert _is_image_value(b"\xff\xd8\xff") is True


def test_is_image_value_with_workflow_image() -> None:
    image = MagicMock(spec=WorkflowImageData)
    assert _is_image_value(image) is True


def test_is_image_value_with_string() -> None:
    assert _is_image_value("hello") is False


def test_is_image_list_with_bytes_list() -> None:
    assert _is_image_list([b"\xff\xd8\xff", b"\xff\xd8\xff"]) is True


def test_is_image_list_with_empty_list() -> None:
    assert _is_image_list([]) is False


def test_is_image_list_with_string_list() -> None:
    assert _is_image_list(["a", "b"]) is False


def test_encode_single_image_bytes() -> None:
    jpeg_bytes = b"\xff\xd8\xff\xe0"
    result = _encode_single_image(jpeg_bytes)
    assert result.startswith("data:image/jpeg;base64,")


def test_collect_image_data_urls_single_bytes() -> None:
    result = _collect_image_data_urls(b"\xff\xd8\xff")
    assert len(result) == 1
    assert result[0].startswith("data:image/jpeg;base64,")


def test_collect_image_data_urls_list_of_bytes() -> None:
    result = _collect_image_data_urls([b"\xff\xd8", b"\xff\xd9"])
    assert len(result) == 2


def test_collect_image_data_urls_non_image() -> None:
    result = _collect_image_data_urls("hello")
    assert result == []


def test_build_prompt_content_text_only() -> None:
    text, images = _build_prompt_content(
        prompt="Count {{ $parameters.obj_type }} in the scene",
        resolved_params={"obj_type": "cars"},
    )
    assert text == "Count cars in the scene"
    assert images == []


def test_build_prompt_content_image_param_removed_from_text() -> None:
    text, images = _build_prompt_content(
        prompt="Describe {{ $parameters.frame }}",
        resolved_params={"frame": b"\xff\xd8\xff"},
    )
    assert "frame" not in text
    assert len(images) == 1


def test_build_prompt_content_image_list() -> None:
    text, images = _build_prompt_content(
        prompt="What happens across these frames?",
        resolved_params={"frames": [b"\xff\xd8", b"\xff\xd9", b"\xff\xda"]},
    )
    assert len(images) == 3


def test_build_prompt_content_mixed_params() -> None:
    text, images = _build_prompt_content(
        prompt="Find {{ $parameters.object_type }} in the video",
        resolved_params={
            "object_type": "person",
            "frames": [b"\xff\xd8", b"\xff\xd9"],
        },
    )
    assert text == "Find person in the video"
    assert len(images) == 2


def test_build_messages_text_only() -> None:
    messages = _build_messages(
        system_prompt=None,
        text_prompt="Hello",
        image_parts=[],
    )
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == [{"type": "text", "text": "Hello"}]


def test_build_messages_with_system_prompt() -> None:
    messages = _build_messages(
        system_prompt="You are helpful.",
        text_prompt="Hello",
        image_parts=[],
    )
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "You are helpful."


def test_build_messages_with_images() -> None:
    messages = _build_messages(
        system_prompt=None,
        text_prompt="Describe",
        image_parts=["data:image/jpeg;base64,abc", "data:image/jpeg;base64,def"],
    )
    content = messages[0]["content"]
    assert len(content) == 3
    assert content[0]["type"] == "text"
    assert content[1]["type"] == "image_url"
    assert content[2]["type"] == "image_url"


def test_resolve_parameters_no_ops() -> None:
    result = _resolve_parameters(
        prompt_parameters={"a": "hello", "b": 42},
        prompt_parameters_operations={},
    )
    assert result == {"a": "hello", "b": 42}


@patch(
    "inference.core.workflows.core_steps.models.foundation.openai_compatible.v1._execute_request"
)
@patch.object(
    openai_compatible,
    "OPENAI_COMPATIBLE_ALLOWED_BASE_URLS",
    {"http://localhost:8000/v1"},
)
def test_block_run_success(mock_execute: MagicMock) -> None:
    mock_execute.return_value = "The model response"
    block = OpenAICompatibleBlockV1()
    result = block.run(
        base_url="http://localhost:8000/v1",
        model_name="test-model",
        api_key="test-key",
        system_prompt=None,
        prompt="Hello",
        prompt_parameters={},
        prompt_parameters_operations={},
        max_tokens=100,
        temperature=None,
        extra_body=None,
    )
    assert result["output"] == "The model response"
    assert result["error_status"] == ""


@patch(
    "inference.core.workflows.core_steps.models.foundation.openai_compatible.v1._execute_request"
)
@patch.object(
    openai_compatible,
    "OPENAI_COMPATIBLE_ALLOWED_BASE_URLS",
    {"http://localhost:8000/v1"},
)
def test_block_run_error(mock_execute: MagicMock) -> None:
    mock_execute.side_effect = RuntimeError("Connection refused")
    block = OpenAICompatibleBlockV1()
    result = block.run(
        base_url="http://localhost:8000/v1",
        model_name="test-model",
        api_key=None,
        system_prompt=None,
        prompt="Hello",
        prompt_parameters={},
        prompt_parameters_operations={},
        max_tokens=100,
        temperature=None,
        extra_body=None,
    )
    assert result["output"] == ""
    assert "Connection refused" in result["error_status"]


@patch(
    "inference.core.workflows.core_steps.models.foundation.openai_compatible.v1._execute_request"
)
@patch.object(
    openai_compatible,
    "OPENAI_COMPATIBLE_ALLOWED_BASE_URLS",
    {"http://localhost:8000/v1"},
)
def test_block_run_forwards_extra_body(mock_execute: MagicMock) -> None:
    mock_execute.return_value = "A"
    block = OpenAICompatibleBlockV1()
    extra_body = {
        "guided_choice": ["A", "B", "C", "D"],
        "chat_template_kwargs": {"enable_thinking": False},
    }
    block.run(
        base_url="http://localhost:8000/v1",
        model_name="test-model",
        api_key="test-key",
        system_prompt=None,
        prompt="Pick one.",
        prompt_parameters={},
        prompt_parameters_operations={},
        max_tokens=2,
        temperature=0.0,
        extra_body=extra_body,
    )
    assert mock_execute.call_args.kwargs["extra_body"] == extra_body


def test_manifest_validation_with_extra_body() -> None:
    specification = {
        "type": "roboflow_core/openai_compatible@v1",
        "name": "step_1",
        "base_url": "http://localhost:8000/v1",
        "model_name": "test-model",
        "prompt": "Pick one.",
        "extra_body": {
            "guided_choice": ["A", "B", "C"],
            "chat_template_kwargs": {"enable_thinking": False},
        },
    }
    result = BlockManifest.model_validate(specification)
    assert result.extra_body["guided_choice"] == ["A", "B", "C"]
    assert result.extra_body["chat_template_kwargs"] == {"enable_thinking": False}


def test_manifest_extra_body_defaults_to_none() -> None:
    specification = {
        "type": "roboflow_core/openai_compatible@v1",
        "name": "step_1",
        "base_url": "http://localhost:8000/v1",
        "model_name": "test-model",
        "prompt": "Hello.",
    }
    result = BlockManifest.model_validate(specification)
    assert result.extra_body is None


@patch(
    "inference.core.workflows.core_steps.models.foundation.openai_compatible.v1._execute_request"
)
@patch.object(
    openai_compatible,
    "OPENAI_COMPATIBLE_ALLOWED_BASE_URLS",
    {"http://localhost:8000/v1"},
)
def test_block_run_without_extra_body_kwarg_is_backwards_compatible(
    mock_execute: MagicMock,
) -> None:
    mock_execute.return_value = "ok"
    block = OpenAICompatibleBlockV1()
    result = block.run(
        base_url="http://localhost:8000/v1",
        model_name="test-model",
        api_key=None,
        system_prompt=None,
        prompt="Hello.",
        prompt_parameters={},
        prompt_parameters_operations={},
        max_tokens=10,
        temperature=None,
    )
    assert result["output"] == "ok"
    assert mock_execute.call_args.kwargs["extra_body"] is None


@patch(
    "inference.core.workflows.core_steps.models.foundation.openai_compatible.v1.OpenAI"
)
def test_execute_request_omits_extra_body_when_none(mock_openai: MagicMock) -> None:
    from inference.core.workflows.core_steps.models.foundation.openai_compatible.v1 import (
        _execute_request,
    )

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="hi"))]
    mock_client.chat.completions.create.return_value = mock_response

    _execute_request(
        client=mock_client,
        model_name="m",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=1,
        temperature=None,
        extra_body=None,
    )
    assert "extra_body" not in mock_client.chat.completions.create.call_args.kwargs


@patch(
    "inference.core.workflows.core_steps.models.foundation.openai_compatible.v1.OpenAI"
)
def test_execute_request_raises_on_none_content(mock_openai: MagicMock) -> None:
    import pytest

    from inference.core.workflows.core_steps.models.foundation.openai_compatible.v1 import (
        _execute_request,
    )

    mock_client = MagicMock()
    mock_response = MagicMock()
    choice = MagicMock(message=MagicMock(content=None), finish_reason="tool_calls")
    mock_response.choices = [choice]
    mock_client.chat.completions.create.return_value = mock_response

    with pytest.raises(RuntimeError, match="empty message content"):
        _execute_request(
            client=mock_client,
            model_name="m",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1,
            temperature=None,
            extra_body=None,
        )


@patch(
    "inference.core.workflows.core_steps.models.foundation.openai_compatible.v1.OpenAI"
)
def test_execute_request_forwards_empty_extra_body(mock_openai: MagicMock) -> None:
    from inference.core.workflows.core_steps.models.foundation.openai_compatible.v1 import (
        _execute_request,
    )

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="hi"))]
    mock_client.chat.completions.create.return_value = mock_response

    _execute_request(
        client=mock_client,
        model_name="m",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=1,
        temperature=None,
        extra_body={},
    )
    assert mock_client.chat.completions.create.call_args.kwargs["extra_body"] == {}


@pytest.mark.parametrize(
    "base_url",
    [
        "http://127.0.0.1/v1",
        "http://10.0.0.1/v1",
        "http://169.254.169.254/latest",
        "http://[::1]/v1",
        "https://attacker.example/v1",
        "https://approved.example.attacker.example/v1",
        "https://approved.example@attacker.example/v1",
        "https://approved.example:8443/v1",
        "https://approved.example/v1/other",
        "http://approved.example/v1",
    ],
)
def test_unapproved_endpoints_never_create_client(monkeypatch, base_url) -> None:
    monkeypatch.setattr(
        openai_compatible,
        "OPENAI_COMPATIBLE_ALLOWED_BASE_URLS",
        {"https://approved.example/v1"},
    )
    with patch.object(openai_compatible, "OpenAI") as create_client:
        with pytest.raises(ValueError, match="not approved"):
            OpenAICompatibleBlockV1()._get_client(base_url, "test-key")
    create_client.assert_not_called()


@pytest.mark.parametrize(
    "base_url",
    [
        "file:///tmp/v1",
        "ftp://approved.example/v1",
        "//approved.example/v1",
        "https://user:password@approved.example/v1",
        "https://approved.example/v1?path=other",
        "https://approved.example/v1#fragment",
    ],
)
def test_invalid_approved_url_never_creates_client(monkeypatch, base_url) -> None:
    monkeypatch.setattr(
        openai_compatible, "OPENAI_COMPATIBLE_ALLOWED_BASE_URLS", {base_url}
    )
    with patch.object(openai_compatible, "OpenAI") as create_client:
        with pytest.raises(ValueError, match=r"HTTP\(S\) URL"):
            OpenAICompatibleBlockV1()._get_client(base_url, "test-key")
    create_client.assert_not_called()


def test_empty_allowlist_blocks_run_before_network_access(monkeypatch) -> None:
    monkeypatch.setattr(openai_compatible, "OPENAI_COMPATIBLE_ALLOWED_BASE_URLS", set())
    with patch.object(openai_compatible, "OpenAI") as create_client:
        result = OpenAICompatibleBlockV1().run(
            base_url="http://localhost:8000/v1",
            model_name="test-model",
            api_key="test-key",
            system_prompt=None,
            prompt="Hello",
            prompt_parameters={},
            prompt_parameters_operations={},
            max_tokens=10,
            temperature=None,
        )
    assert result["output"] == ""
    assert "not approved" in result["error_status"]
    create_client.assert_not_called()


def test_cached_client_still_requires_approved_endpoint(monkeypatch) -> None:
    base_url = "http://localhost:8000/v1"
    allowed = {base_url}
    monkeypatch.setattr(
        openai_compatible, "OPENAI_COMPATIBLE_ALLOWED_BASE_URLS", allowed
    )
    block = OpenAICompatibleBlockV1()
    with block._get_client(base_url, "test-key") as client:
        assert block._get_client(base_url + "/", "test-key") is client
        allowed.clear()
        with pytest.raises(ValueError, match="not approved"):
            block._get_client(base_url, "test-key")


@pytest.mark.parametrize("allowed_url", ["*", "http://localhost:8000/v1"])
@pytest.mark.parametrize("status_code", [200, 301, 302, 303, 307, 308])
def test_approved_endpoint_uses_real_sdk_without_following_redirects(
    monkeypatch, status_code, allowed_url
) -> None:
    base_url = "http://localhost:8000/v1"
    monkeypatch.setattr(
        openai_compatible, "OPENAI_COMPATIBLE_ALLOWED_BASE_URLS", {allowed_url}
    )
    requests = []

    def respond(request):
        requests.append(request)
        if len(requests) == 1 and status_code != 200:
            return httpx.Response(
                status_code,
                headers={"Location": "http://169.254.169.254/redirect-target"},
            )
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "ok"}}]},
        )

    block = OpenAICompatibleBlockV1()
    with patch.object(httpx.HTTPTransport, "handle_request", side_effect=respond):
        with block._get_client(base_url, "synthetic-secret"):
            result = block.run(
                base_url=base_url + "/",
                model_name="test-model",
                api_key="synthetic-secret",
                system_prompt=None,
                prompt="Value: {{ $parameters.secret }}",
                prompt_parameters={"secret": "private-text", "image": b"image"},
                prompt_parameters_operations={},
                max_tokens=10,
                temperature=None,
            )
    assert len(requests) == 1
    assert str(requests[0].url) == base_url + "/chat/completions"
    assert requests[0].headers["Authorization"] == "Bearer synthetic-secret"
    content = json.loads(requests[0].content)["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "Value: private-text"}
    assert content[1]["image_url"]["url"] == "data:image/jpeg;base64,aW1hZ2U="
    if status_code == 200:
        assert result == {"output": "ok", "error_status": ""}
    else:
        assert result["output"] == ""
        assert result["error_status"]


@pytest.mark.parametrize("allowed", [{"*"}, {"*", "https://approved.example/v1"}])
@pytest.mark.parametrize(
    "base_url", ["http://127.0.0.1/v1", "https://other.example/v1"]
)
def test_wildcard_allows_any_destination(monkeypatch, allowed, base_url) -> None:
    monkeypatch.setattr(
        openai_compatible, "OPENAI_COMPATIBLE_ALLOWED_BASE_URLS", allowed
    )
    with OpenAICompatibleBlockV1()._get_client(base_url, "test-key") as client:
        assert str(client.base_url) == base_url + "/"

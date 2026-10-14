"""OpenRouter block: direct and proxied chat completions, retries, decoding.

Every request goes to an in-process fake transport or platform client.
"""

import json
from typing import Any, Dict

import pytest
import torch
from roboflow_workflows.v2.blocks.language.chat import ChatRequestError
from roboflow_workflows.v2.blocks.language.openrouter.v3 import OpenRouterBlock, OpenRouterProxyError, proxy_error_handlers
from roboflow_workflows.v2.data.batch import Batch
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.predictions import OBJECT_DETECTION_PREDICTION_KIND
from roboflow_workflows.v2.engine.declaration import spec_of

from tests.v2.support.language.fake_chat import (
    FakeChatServer,
    FakePlatformClient,
    chat_completion_body,
    error_body,
)

OPENROUTER_KEY = "sk-or-test-OPENROUTER-77aa"
ROBOFLOW_KEY = "rf-test-ROBOFLOW-0c3d"
FAKE_OPENROUTER_URL = "http://openrouter.test/api/v1"


def _image(width: int = 64, height: int = 48) -> ImageData:
    image = ImageData.from_tensor(torch.zeros((3, height, width), dtype=torch.uint8))

    return image


def _block(server: FakeChatServer, **resources: Any) -> OpenRouterBlock:
    block = OpenRouterBlock(
        openrouter_base_url=FAKE_OPENROUTER_URL,
        openrouter_http_client=server.http_client(),
        **resources,
    )

    return block


def _answering(content: str, **body: Any) -> FakeChatServer:
    server = FakeChatServer(
        lambda path, request: (200, chat_completion_body(content, **body))
    )

    return server


def _run(block: OpenRouterBlock, images, resolved=None, **raw: Any):
    defaults = {
        "images": "$inputs.images",
        "model_id": "qwen/qwen3.6-27b",
        "api_key": OPENROUTER_KEY,
        "prompt": "What is this?",
    }
    params = spec_of(OpenRouterBlock).validate_params({**defaults, **raw})
    arguments = {name: getattr(params, name) for name in type(params).model_fields}
    arguments.update(images=Batch(images), **(resolved or {}))
    results = block.run(**arguments)

    return results


def test_direct_request_routes_privacy_quantization_and_reasoning() -> None:
    server = _answering("A cat.", reasoning="It has whiskers.")

    results = _run(_block(server), [_image()], reasoning_effort="low")

    request = server.requests[0]
    assert request.url == f"{FAKE_OPENROUTER_URL}/chat/completions"
    assert request.headers["authorization"] == f"Bearer {OPENROUTER_KEY}"
    assert request.body["model"] == "qwen/qwen3.6-27b"
    assert request.body["max_tokens"] == 2048
    assert request.body["temperature"] == 0.1
    assert request.body["provider"] == {
        "data_collection": "deny",
        "quantizations": ["fp8", "bf16", "fp32"],
    }
    assert request.body["reasoning"] == {"effort": "low"}
    user = request.body["messages"][0]
    assert user["content"][0] == {"type": "text", "text": "What is this?"}
    assert user["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert results[0]["output"] == "A cat."
    assert results[0]["thinking"] == "It has whiskers."
    assert (results[0]["input_tokens"], results[0]["output_tokens"]) == (13, 5)
    assert results[0]["predictions"] is None


@pytest.mark.parametrize(
    ("privacy_level", "model_id", "provider"),
    [
        ("allow", "openai/gpt-4o-mini", None),
        ("zdr", "openai/gpt-4o-mini", {"data_collection": "deny", "zdr": True}),
    ],
)
def test_privacy_levels(privacy_level, model_id, provider) -> None:
    server = _answering("ok")

    _run(_block(server), [_image()], privacy_level=privacy_level, model_id=model_id)

    assert server.requests[0].body.get("provider") == provider


def test_reasoning_none_disables_reasoning() -> None:
    server = _answering("ok")

    _run(_block(server), [_image()], reasoning_effort="none")

    assert server.requests[0].body["reasoning"] == {"enabled": False}


def test_managed_key_posts_v1_payload_to_the_proxy() -> None:
    server = _answering("unused")
    platform = FakePlatformClient(
        lambda endpoint, payload: chat_completion_body("Proxied.", reasoning="r")
    )

    results = _run(
        _block(server, platform_client=platform, api_key=ROBOFLOW_KEY),
        [_image()],
        api_key="rf_key:account",
        reasoning_effort="high",
    )

    assert server.requests == []
    post = platform.posts[0]
    assert post["endpoint"] == "apiproxy/openrouter"
    assert post["api_key"] == ROBOFLOW_KEY
    payload = post["payload"]
    assert payload["openrouter_api_key"] == "rf_key:account"
    assert payload["privacy_level"] == "deny"
    assert payload["reasoning"] == {"effort": "high"}
    assert payload["quantizations"] == ["fp8", "bf16", "fp32"]
    assert sorted(post["http_errors_handlers"]) == [400, 401, 403, 413, 429, 502, 503]
    assert results[0]["output"] == "Proxied."
    assert results[0]["thinking"] == "r"


def test_rejected_reasoning_is_retried_once_without_it() -> None:
    def reply(path: str, body: Dict[str, Any]):
        if "reasoning" in body:
            return 400, error_body(
                "Reasoning is mandatory for this endpoint and cannot be disabled."
            )
        return 200, chat_completion_body("Retried.")

    server = FakeChatServer(reply)

    results = _run(_block(server), [_image()], reasoning_effort="none")

    assert [("reasoning" in request.body) for request in server.requests] == [
        True,
        False,
    ]
    assert results[0]["output"] == "Retried."


def test_other_client_errors_are_not_retried_and_are_scrubbed() -> None:
    server = FakeChatServer(
        lambda path, body: (
            401,
            error_body(f"No auth credentials for {OPENROUTER_KEY}"),
        )
    )

    with pytest.raises(ChatRequestError) as caught:
        _run(_block(server), [_image()], reasoning_effort="low")

    assert len(server.requests) == 1
    assert OPENROUTER_KEY not in str(caught.value)
    assert "<secret api_key>" in str(caught.value)


@pytest.mark.parametrize(("status", "attempts"), [(400, 2), (502, 1)])
def test_proxy_retry_only_for_client_rejections_of_reasoning(status, attempts) -> None:
    calls = []

    def reply(endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        calls.append(payload)
        if "reasoning" in payload:
            raise OpenRouterProxyError(
                "reasoning.effort: Invalid option", status_code=status
            )
        return chat_completion_body("ok")

    platform = FakePlatformClient(reply)
    block = _block(_answering("unused"), platform_client=platform, api_key=ROBOFLOW_KEY)

    if attempts == 1:
        with pytest.raises(ChatRequestError, match="Invalid option"):
            _run(block, [_image()], api_key="rf_key:account", reasoning_effort="max")
    else:
        _run(block, [_image()], api_key="rf_key:account", reasoning_effort="max")

    assert len(calls) == attempts


def test_proxy_error_handlers_keep_the_api_message_and_status() -> None:
    class Response:
        def json(self) -> Dict[str, Any]:
            return {"details": "credits exhausted"}

    class HTTPError(Exception):
        response = Response()

    with pytest.raises(OpenRouterProxyError) as caught:
        proxy_error_handlers()[429](HTTPError("429 Too Many Requests"))

    assert str(caught.value) == "credits exhausted"
    assert caught.value.status_code == 429


def test_format_detection_decodes_0_1000_boxes_into_image_pixels() -> None:
    answer = json.dumps([{"label": "car", "box_2d": [100, 250, 500, 750]}])
    server = _answering(answer)

    results = _run(
        _block(server),
        [_image(200, 100)],
        task_type="object-detection",
        classes=["car"],
        detection_format="xyxy_0_1000",
        prompt=None,
    )

    messages = server.requests[0].body["messages"]
    assert [message["role"] for message in messages] == ["user"]
    assert results[0]["predictions"].xyxy.tolist() == [[20.0, 25.0, 100.0, 75.0]]
    assert results[0]["classes"] == ["car"]


def test_absolute_detection_uses_the_full_size_upload() -> None:
    answer = json.dumps([{"label": "car", "box_2d": [10, 20, 30, 40]}])
    server = _answering(answer)

    results = _run(
        _block(server),
        [_image(3000, 100)],
        task_type="object-detection",
        classes=["car"],
        detection_format="xyxy_absolute",
    )

    text = server.requests[0].body["messages"][0]["content"][0]["text"]
    assert "3000x100 pixel image" in text
    assert results[0]["predictions"].xyxy.tolist() == [[10.0, 20.0, 30.0, 40.0]]


def test_legacy_detection_keeps_the_system_prompt() -> None:
    answer = json.dumps(
        {
            "detections": [
                {
                    "x_min": 0,
                    "y_min": 0,
                    "x_max": 0.5,
                    "y_max": 0.5,
                    "class_name": "car",
                    "confidence": 0.6,
                }
            ]
        }
    )
    server = _answering(answer)

    results = _run(
        _block(server),
        [_image(100, 100)],
        task_type="object-detection",
        classes=["car"],
    )

    system = server.requests[0].body["messages"][0]
    assert system["role"] == "system"
    assert system["content"].endswith("you are only allowed to return JSON document.")
    assert results[0]["predictions"].xyxy.tolist() == [[0.0, 0.0, 50.0, 50.0]]


def test_unknown_selected_detection_format_fails_before_sending() -> None:
    server = _answering("unused")

    with pytest.raises(ValueError):
        _run(
            _block(server),
            [_image()],
            task_type="object-detection",
            classes=["car"],
            detection_format="$inputs.format",
            resolved={"detection_format": "pixels_please"},
        )

    assert server.requests == []


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (
            {**chat_completion_body("x"), "choices": []},
            "provider failed in delivering response",
        ),
        (chat_completion_body(None, finish_reason="length"), "ran out of tokens"),
        (chat_completion_body(None, finish_reason="tool_calls"), "only tool calls"),
    ],
)
def test_missing_answers_raise_with_v1_messages(body, message) -> None:
    server = FakeChatServer(lambda path, request: (200, body))

    with pytest.raises(ChatRequestError, match=message):
        _run(_block(server), [_image()])


def test_proxy_missing_answer_names_the_proxy() -> None:
    platform = FakePlatformClient(
        lambda endpoint, payload: {"choices": [], "error": {"message": "busy"}}
    )
    block = _block(_answering("unused"), platform_client=platform, api_key=ROBOFLOW_KEY)

    with pytest.raises(ChatRequestError, match="via Roboflow proxy. Details: busy"):
        _run(block, [_image()], api_key="rf_key:account")


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (
            {"task_type": "classification", "prompt": None},
            "`classes` parameter required",
        ),
        ({"task_type": "unconstrained", "prompt": None}, "`prompt` parameter required"),
        ({"temperature": 2.5}, "temperature"),
        ({"privacy_level": "public"}, "privacy_level"),
        (
            {
                "task_type": "object-detection",
                "classes": ["a"],
                "detection_format": "pixels",
            },
            "detection_format",
        ),
        ({"task_type": "instance-segmentation", "classes": ["a"]}, "task_type"),
    ],
)
def test_invalid_configuration_is_rejected_at_validation(raw, message) -> None:
    defaults = {"images": "$inputs.images", "model_id": "m", "prompt": "x"}

    with pytest.raises(Exception, match=message):
        spec_of(OpenRouterBlock).validate_params({**defaults, **raw})


def test_outputs_narrow_predictions_and_include_thinking() -> None:
    spec = spec_of(OpenRouterBlock)
    params = spec.validate_params(
        {
            "images": "$inputs.i",
            "model_id": "m",
            "task_type": "object-detection",
            "classes": ["a"],
        }
    )

    outputs = spec.resolve_outputs(params)

    assert outputs["predictions"].kinds == (OBJECT_DETECTION_PREDICTION_KIND,)
    assert "thinking" in outputs
    resources = OpenRouterBlock.discover_dependent_resources(params)
    assert (resources[0].identifier, resources[0].details["provider"]) == (
        "m",
        "openrouter",
    )


def test_managed_key_without_a_platform_client_fails_without_sending() -> None:
    server = _answering("unused")

    with pytest.raises(ChatRequestError, match="provided no platform client"):
        _run(_block(server, api_key=ROBOFLOW_KEY), [_image()], api_key="rf_key:account")

    assert server.requests == []


@pytest.mark.parametrize("missing", [None])
def test_missing_openrouter_key_never_falls_back_to_environment(monkeypatch, missing):
    invented = "invented-environment-key-do-not-send"
    monkeypatch.setenv("OPENAI_API_KEY", invented)
    server = _answering("unused")
    platform = FakePlatformClient(
        lambda endpoint, payload: chat_completion_body("unused")
    )
    with pytest.raises(ValueError, match="OpenRouter API key is required") as caught:
        _run(
            _block(server, platform_client=platform),
            [_image()],
            resolved={"api_key": missing},
        )
    assert server.requests == []
    assert platform.posts == []
    assert invented not in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


@pytest.mark.parametrize("host_key", [None, ""])
def test_managed_openrouter_key_requires_host_key_before_proxy(host_key):
    server = _answering("unused")
    platform = FakePlatformClient(
        lambda endpoint, payload: chat_completion_body("unused")
    )
    with pytest.raises(ValueError, match="Roboflow API key is required") as caught:
        _run(
            _block(server, platform_client=platform, api_key=host_key),
            [_image()],
            api_key="rf_key:account",
        )
    assert server.requests == []
    assert platform.posts == []
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None

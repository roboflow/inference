"""Unit tests of ``roboflow_core/anthropic_claude@v5`` against in-process fakes.

``FakeAnthropicServer`` answers the SDK's streaming Messages request with
server-sent events through an ``httpx.MockTransport``; nothing leaves the
process. These tests check requests and wiring, not any real model.
"""

import base64
import json
import logging
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import httpx
import numpy as np
import pytest
import requests
import torch
from pydantic import ValidationError
from roboflow_workflows.v2.blocks.models.vision_language_models.anthropic_claude import (
    v5 as anthropic,
)
from roboflow_workflows.v2.blocks.models.vision_language_models.anthropic_claude.v5 import (
    CLAUDE_MODELS,
    DETECTION_MAX_PNG_PAYLOAD_BYTES,
    AnthropicClaudeBlock,
    claude_options,
    claude_request,
    claude_upload,
    downscaled_size_wh,
    normalize_model_id,
    read_claude_message,
)
from roboflow_workflows.v2.blocks.models.vision_language_models.chat import (
    ChatRequestError,
    encode_image,
)
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.secrets import Secret, SecretUnavailableError

from inference_models.models.base.classification import ClassificationPrediction
from inference_models.models.base.object_detection import Detections
from tests.v2.support.language.fake_chat import FakePlatformClient

ANTHROPIC_KEY = "sk-ant-test-CLAUDE-77aa"
ROBOFLOW_KEY = "rf-test-ROBOFLOW-1a2b"
Reply = Callable[[Dict[str, Any]], Tuple[int, Any]]
"""``request JSON -> (HTTP status, message dict or error dict)``."""


class FakeAnthropicServer:
    """Answers Anthropic Messages requests; records them in arrival order.

    A 200 reply is streamed as the SDK expects: ``message_start``, one
    ``content_block_*`` group per content block, ``message_delta`` and
    ``message_stop``. Other statuses return the JSON error body.
    """

    def __init__(self, reply: Reply, *, delay_s: float = 0.0) -> None:
        self._reply = reply
        self._delay_s = delay_s
        self._lock = threading.Lock()
        self.requests: List[Dict[str, Any]] = []
        self.in_flight = 0
        self.most_in_flight = 0

    def http_client(self) -> httpx.Client:
        client = httpx.Client(transport=httpx.MockTransport(self._handle))

        return client

    def _handle(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        with self._lock:
            self.requests.append(
                {
                    "url": str(request.url),
                    "headers": dict(request.headers),
                    "body": body,
                }
            )
            self.in_flight += 1
            self.most_in_flight = max(self.most_in_flight, self.in_flight)
        try:
            time.sleep(self._delay_s)
            status, payload = self._reply(body)
        finally:
            with self._lock:
                self.in_flight -= 1
        if status != 200:
            return httpx.Response(status, json=payload)

        response = httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=message_events(payload),
        )

        return response


def claude_message(
    text: Optional[str],
    *,
    thinking: Optional[str] = None,
    stop_reason: Optional[str] = "end_turn",
    input_tokens: int = 21,
    output_tokens: int = 9,
) -> Dict[str, Any]:
    """A Messages API result: optional thinking block, then optional text."""
    content: List[Dict[str, Any]] = []
    if thinking is not None:
        content.append({"type": "thinking", "thinking": thinking, "signature": "s"})
    if text is not None:
        content.append({"type": "text", "text": text})
    message = {
        "id": "msg_fake",
        "type": "message",
        "role": "assistant",
        "model": "fake",
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    }

    return message


def message_events(message: Dict[str, Any]) -> str:
    """Render a message as the server-sent events of a streamed response."""
    start = dict(message, content=[], stop_reason=None)
    start["usage"] = {
        "input_tokens": message["usage"]["input_tokens"],
        "output_tokens": 1,
    }
    events = [{"type": "message_start", "message": start}]
    for index, block in enumerate(message["content"]):
        if block["type"] == "text":
            empty = {"type": "text", "text": ""}
            delta = {"type": "text_delta", "text": block["text"]}
        else:
            empty = {"type": "thinking", "thinking": "", "signature": ""}
            delta = {"type": "thinking_delta", "thinking": block["thinking"]}
        events += [
            {"type": "content_block_start", "index": index, "content_block": empty},
            {"type": "content_block_delta", "index": index, "delta": delta},
            {"type": "content_block_stop", "index": index},
        ]
    events += [
        {
            "type": "message_delta",
            "delta": {"stop_reason": message["stop_reason"], "stop_sequence": None},
            "usage": {"output_tokens": message["usage"]["output_tokens"]},
        },
        {"type": "message_stop"},
    ]
    text = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)

    return text


def _image(width: int, height: int, *, seed: int = 0) -> ImageData:
    generator = torch.Generator().manual_seed(seed)
    pixels = torch.randint(
        0, 256, (3, height, width), dtype=torch.uint8, generator=generator
    )
    image = ImageData.from_tensor(pixels)

    return image


def _params(**fields: Any) -> AnthropicClaudeBlock.Params:
    params = AnthropicClaudeBlock.Params(images="$inputs.image", **fields)

    return params


def _run(
    block: AnthropicClaudeBlock,
    images: List[ImageData],
    *,
    overrides: Optional[Dict[str, Any]] = None,
    **fields: Any,
) -> List[Dict[str, Any]]:
    """Validate ``fields`` as Params, then run with run-time ``overrides``."""
    params = _params(**fields).model_dump(exclude={"images"})
    results = block.run(images=images, **{**params, **(overrides or {})})

    return results


def _direct_block(
    server: FakeAnthropicServer, **resources: Any
) -> AnthropicClaudeBlock:
    block = AnthropicClaudeBlock(
        anthropic_http_client=server.http_client(),
        anthropic_base_url="https://anthropic.test",
        **resources,
    )

    return block


def _chain_text(error: Optional[BaseException]) -> str:
    texts = []
    while error is not None:
        texts.append(f"{type(error).__name__}: {error} {error.args!r}")
        error = error.__cause__ or error.__context__

    return "\n".join(texts)


# Params


def test_params_default_to_v1_model_and_managed_key() -> None:
    params = _params(prompt="Describe.")

    assert params.model_version == "claude-sonnet-4-5"
    assert params.api_key == "rf_key:account"
    assert params.max_image_size == 1024
    assert params.task_type == "unconstrained"


@pytest.mark.parametrize(
    "fields, message",
    [
        ({"task_type": "unconstrained"}, "`prompt` parameter required"),
        ({"task_type": "object-detection"}, "`classes` parameter required"),
        ({"task_type": "structured-answering"}, "`output_structure` parameter"),
        (
            {"prompt": "x", "extended_thinking": True, "temperature": 0.5},
            "`temperature` cannot be used when `extended_thinking` is enabled",
        ),
        (
            {
                "prompt": "x",
                "extended_thinking": True,
                "thinking_budget_tokens": 2048,
                "max_tokens": 2048,
            },
            "`thinking_budget_tokens` (2048) must be less than `max_tokens` (2048)",
        ),
        ({"prompt": "x", "thinking_budget_tokens": 1000}, "greater than or equal"),
        ({"prompt": "x", "max_concurrent_requests": 0}, "greater than or equal"),
        ({"prompt": "x", "max_image_size": 0}, "greater than or equal"),
        ({"prompt": "x", "max_tokens": 0}, "greater than or equal"),
        ({"prompt": "x", "temperature": 1.5}, "less than or equal"),
        ({"prompt": "x", "model_version": "claude-unknown"}, "model_version"),
    ],
)
def test_params_reject_v1_invalid_configurations(fields, message) -> None:
    with pytest.raises(
        ValidationError, match=message.replace("(", r"\(").replace(")", r"\)")
    ):
        _params(**fields)


def test_params_accept_selectors_for_model_and_temperature() -> None:
    params = _params(
        prompt="x", model_version="$inputs.model", temperature="$inputs.temperature"
    )

    assert params.model_version == "$inputs.model"


def test_dependent_resource_names_the_exact_anthropic_model() -> None:
    by_label = AnthropicClaudeBlock.discover_dependent_resources(_params(prompt="x"))
    by_selector = AnthropicClaudeBlock.discover_dependent_resources(
        _params(prompt="x", model_version="$inputs.model")
    )

    assert by_label[0].identifier == "claude-sonnet-4-5-20250929"
    assert by_label[0].details == {
        "provider": "anthropic",
        "resource": "anthropic_http_client",
    }
    assert by_selector[0].identifier == "$inputs.model"


def test_outputs_narrow_predictions_to_the_task() -> None:
    detection = AnthropicClaudeBlock.describe_outputs(
        _params(task_type="object-detection", classes=["car"])
    )
    caption = AnthropicClaudeBlock.describe_outputs(_params(task_type="caption"))

    assert [kind.name for kind in detection["predictions"].kinds] == [
        "object_detection_prediction"
    ]
    assert len(caption["predictions"].kinds) == 2
    assert set(detection) == {
        "output",
        "classes",
        "input_tokens",
        "output_tokens",
        "predictions",
        "error_status",
        "inference_id",
    }


# Model capabilities


def test_normalize_model_id_strips_snapshot_and_latest_suffixes() -> None:
    assert normalize_model_id("Claude-Sonnet-4-5-20250929") == "claude-sonnet-4-5"
    assert normalize_model_id("claude-3-5-haiku-latest") == "claude-3-5-haiku"


def test_legacy_model_keeps_temperature_and_gets_a_half_limit_budget() -> None:
    plain = claude_options(
        "claude-sonnet-4-5",
        max_tokens=None,
        temperature=0.3,
        extended_thinking=None,
        thinking_budget_tokens=None,
    )
    thinking = claude_options(
        "claude-sonnet-4-5",
        max_tokens=None,
        temperature=None,
        extended_thinking=True,
        thinking_budget_tokens=None,
    )

    assert (plain.temperature, plain.thinking, plain.max_tokens) == (0.3, None, 64000)
    assert plain.exact_version == "claude-sonnet-4-5-20250929"
    assert thinking.thinking == {"type": "enabled", "budget_tokens": 32000}


def test_new_model_drops_temperature_and_thinks_adaptively(caplog) -> None:
    with caplog.at_level(logging.WARNING):
        options = claude_options(
            "claude-opus-5-5",
            max_tokens=None,
            temperature=0.3,
            extended_thinking=None,
            thinking_budget_tokens=None,
        )
        thinking = claude_options(
            "claude-opus-5-5",
            max_tokens=4096,
            temperature=None,
            extended_thinking=True,
            thinking_budget_tokens=2048,
        )

    assert options.temperature is None
    assert options.max_tokens == 128000
    assert thinking.thinking == {"type": "adaptive"}
    assert thinking.max_tokens == 4096
    assert "does not accept `temperature`" in caplog.text
    assert "only supports adaptive thinking" in caplog.text


def test_unknown_model_uses_v1_default_limit_and_its_own_id() -> None:
    options = claude_options(
        "claude-next",
        max_tokens=None,
        temperature=None,
        extended_thinking=None,
        thinking_budget_tokens=None,
    )

    assert (options.exact_version, options.max_tokens) == ("claude-next", 64000)


# Uploads and prompts


@pytest.mark.parametrize(
    "size_wh, expected",
    [
        ((800, 600), (800, 600)),
        ((2000, 1000), (1024, 512)),
        ((1000, 3000), (341, 1024)),
        ((1025, 1023), (1024, 1022)),
    ],
)
def test_downscaled_size_truncates_like_v1(size_wh, expected) -> None:
    assert downscaled_size_wh(size_wh, max_image_size=1024) == expected


def test_non_detection_upload_is_a_downscaled_jpeg() -> None:
    upload = claude_upload(_image(2000, 1000), task_type="ocr", max_image_size=1024)

    assert (upload.media_type, upload.size_wh) == ("image/jpeg", (1024, 512))


def test_detection_upload_is_a_png_at_the_size_claude_reads() -> None:
    flat = ImageData.from_tensor(torch.full((3, 1000, 3000), 90, dtype=torch.uint8))

    upload = claude_upload(flat, task_type="object-detection", max_image_size=64)
    pixels = cv2.imdecode(
        np.frombuffer(base64.b64decode(upload.base64), np.uint8), cv2.IMREAD_UNCHANGED
    )

    assert upload.media_type == "image/png"
    assert upload.size_wh == (pixels.shape[1], pixels.shape[0])
    assert max(upload.size_wh) <= 2576
    assert upload.size_wh[0] < 3000


def test_a_large_detection_png_falls_back_to_jpeg_at_the_same_size() -> None:
    image = _image(1900, 1900, seed=3)

    upload = claude_upload(image, task_type="object-detection", max_image_size=1024)

    png = encode_image(image, image_format="png", size_wh=upload.size_wh)
    assert len(base64.b64decode(png.base64)) > DETECTION_MAX_PNG_PAYLOAD_BYTES
    assert upload.media_type == "image/jpeg"
    pixels = cv2.imdecode(
        np.frombuffer(base64.b64decode(upload.base64), np.uint8), cv2.IMREAD_UNCHANGED
    )
    assert upload.size_wh == (pixels.shape[1], pixels.shape[0])


@pytest.mark.parametrize(
    "task_type, system_start, text",
    [
        ("unconstrained", None, "Count cars."),
        ("ocr", "You act as OCR model.", None),
        ("visual-question-answering", "You act as Visual", "Question: Count cars."),
        ("caption", "You act as image caption", None),
        ("classification", "You act as single-class", "List of all classes"),
        ("multi-label-classification", "You act as multi-label", "List of all classes"),
        ("structured-answering", "You are supposed to", "Specification of"),
    ],
)
def test_requests_put_the_image_first_with_v1_prompts(
    task_type, system_start, text
) -> None:
    upload = claude_upload(_image(8, 8), task_type=task_type, max_image_size=1024)

    request = claude_request(
        task_type,
        upload=upload,
        prompt="Count cars.",
        output_structure={"count": "cars"},
        classes=["car", "bus"],
    )
    content = request.messages[0]["content"]

    assert request.messages[0]["role"] == "user"
    assert content[0]["type"] == "image"
    assert content[0]["source"]["data"] == upload.base64
    if system_start is None:
        assert request.system is None
    else:
        assert request.system.startswith(system_start)
    if text is None:
        assert len(content) == 1
    else:
        assert content[1]["text"].startswith(text)


def test_detection_request_states_the_upload_size_without_a_system_prompt() -> None:
    upload = claude_upload(
        _image(64, 32), task_type="object-detection", max_image_size=8
    )

    request = claude_request(
        "object-detection",
        upload=upload,
        prompt=None,
        output_structure=None,
        classes=["car"],
    )

    assert request.system is None
    assert "64x32" in request.messages[0]["content"][1]["text"]
    assert request.upload_size_wh == (64, 32)


# Reading answers


def test_reading_skips_thinking_and_reports_usage() -> None:
    answer = read_claude_message(claude_message("Hi", thinking="hmm"), via_proxy=False)

    assert (answer.text, answer.input_tokens, answer.output_tokens) == ("Hi", 21, 9)


@pytest.mark.parametrize(
    "message, via_proxy, error",
    [
        (claude_message("x", stop_reason="max_tokens"), True, "max_tokens limit"),
        (claude_message("x", stop_reason="refusal"), False, "unexpected stop reason"),
        (claude_message("x", stop_reason=None), False, "unexpected stop reason"),
        (claude_message(None), True, "returned no content"),
        (claude_message(None, thinking="t"), False, "no text content"),
    ],
)
def test_reading_fails_with_v1_messages(message, via_proxy, error) -> None:
    with pytest.raises(ChatRequestError, match=error):
        read_claude_message(message, via_proxy=via_proxy)


def test_the_proxy_may_omit_the_stop_reason() -> None:
    answer = read_claude_message(claude_message("ok", stop_reason=None), via_proxy=True)

    assert answer.text == "ok"


# Direct requests


def test_direct_request_streams_with_the_exact_model_and_key() -> None:
    server = FakeAnthropicServer(lambda body: (200, claude_message("A cat.")))
    block = _direct_block(server)

    results = _run(
        block,
        [_image(16, 8)],
        task_type="caption",
        api_key=ANTHROPIC_KEY,
        temperature=0.4,
    )

    request = server.requests[0]
    assert request["url"] == "https://anthropic.test/v1/messages"
    assert request["headers"]["x-api-key"] == ANTHROPIC_KEY
    assert request["body"]["model"] == "claude-sonnet-4-5-20250929"
    assert request["body"]["stream"] is True
    assert request["body"]["max_tokens"] == 64000
    assert request["body"]["temperature"] == 0.4
    assert request["body"]["system"].endswith("Caption should be short.")
    assert "thinking" not in request["body"]
    assert results[0]["output"] == "A cat."
    assert (results[0]["input_tokens"], results[0]["output_tokens"]) == (21, 9)
    assert results[0]["predictions"] is None
    assert results[0]["error_status"] is False


def test_direct_request_sends_thinking_and_omits_unset_fields() -> None:
    server = FakeAnthropicServer(lambda body: (200, claude_message("x", thinking="t")))

    _run(
        _direct_block(server),
        [_image(8, 8)],
        prompt="Describe.",
        model_version="claude-opus-5-5",
        extended_thinking=True,
        api_key=ANTHROPIC_KEY,
    )

    body = server.requests[0]["body"]
    assert body["thinking"] == {"type": "adaptive"}
    assert "system" not in body
    assert "temperature" not in body
    assert body["max_tokens"] == 128000


def test_secret_key_is_revealed_only_on_the_wire() -> None:
    server = FakeAnthropicServer(lambda body: (200, claude_message("x")))

    _run(
        _direct_block(server),
        [_image(8, 8)],
        prompt="Describe.",
        overrides={"api_key": Secret(ANTHROPIC_KEY, name="anthropic")},
    )

    assert server.requests[0]["headers"]["x-api-key"] == ANTHROPIC_KEY


def test_detection_answer_becomes_detections_on_the_original_image() -> None:
    def reply(body):
        text = body["messages"][0]["content"][1]["text"]
        width, height = text.split(" pixel image")[0].rsplit(" ", 1)[1].split("x")
        answer = [{"box_2d": [0, 0, int(width) // 2, int(height) // 2], "label": "car"}]
        return 200, claude_message(json.dumps(answer))

    server = FakeAnthropicServer(reply)

    results = _run(
        _direct_block(server),
        [_image(4000, 2000), _image(200, 100)],
        task_type="object-detection",
        classes=["car"],
        api_key=ANTHROPIC_KEY,
    )

    assert isinstance(results[0]["predictions"], Detections)
    for result, (width, height) in zip(results, [(4000, 2000), (200, 100)]):
        x_max, y_max = result["predictions"].xyxy[0, 2:].tolist()
        assert abs(x_max - width / 2) <= width / 1000 + 1
        assert abs(y_max - height / 2) <= height / 1000 + 1
        assert result["error_status"] is False


def test_classification_answer_becomes_a_native_prediction() -> None:
    server = FakeAnthropicServer(
        lambda body: (200, claude_message('{"class_name": "bus", "confidence": 0.8}'))
    )

    results = _run(
        _direct_block(server),
        [_image(8, 8)],
        task_type="classification",
        classes=["car", "bus"],
        api_key=ANTHROPIC_KEY,
    )

    assert isinstance(results[0]["predictions"], ClassificationPrediction)
    assert results[0]["predictions"].class_id.tolist() == [1]


def test_unreadable_detection_answer_sets_error_status() -> None:
    server = FakeAnthropicServer(lambda body: (200, claude_message("no boxes")))

    results = _run(
        _direct_block(server),
        [_image(8, 8)],
        task_type="object-detection",
        classes=["car"],
        api_key=ANTHROPIC_KEY,
    )

    assert (results[0]["error_status"], results[0]["predictions"]) == (True, None)


# Managed key


def test_managed_key_posts_the_v1_proxy_payload_with_the_host_key() -> None:
    platform = FakePlatformClient(lambda endpoint, payload: claude_message("ok"))
    server = FakeAnthropicServer(lambda body: (200, claude_message("unused")))
    block = _direct_block(server, platform_client=platform, api_key=ROBOFLOW_KEY)

    results = _run(
        block,
        [_image(8, 8)],
        task_type="ocr",
        api_key="rf_key:user:42",
        model_version="claude-sonnet-4-6",
        temperature=0.2,
    )

    assert server.requests == []
    post = platform.posts[0]
    assert (post["endpoint"], post["api_key"]) == ("apiproxy/anthropic", ROBOFLOW_KEY)
    assert post["payload"]["model"] == "claude-sonnet-4-6"
    assert post["payload"]["anthropic_api_key"] == "rf_key:user:42"
    assert post["payload"]["max_tokens"] == 64000
    assert post["payload"]["temperature"] == 0.2
    assert post["payload"]["system"].startswith("You act as OCR model.")
    assert results[0]["output"] == "ok"


def test_managed_key_without_a_platform_client_sends_nothing() -> None:
    server = FakeAnthropicServer(lambda body: (200, claude_message("unused")))

    with pytest.raises(ChatRequestError, match="provided no platform client"):
        _run(
            _direct_block(server, api_key=ROBOFLOW_KEY),
            [_image(8, 8)],
            prompt="Describe.",
        )

    assert server.requests == []


def test_proxy_connection_errors_keep_the_v1_message() -> None:
    def fail(endpoint, payload):
        raise requests.exceptions.ConnectionError(f"refused for {ROBOFLOW_KEY}")

    block = AnthropicClaudeBlock(
        platform_client=FakePlatformClient(fail), api_key=ROBOFLOW_KEY
    )

    with pytest.raises(ChatRequestError) as caught:
        _run(block, [_image(8, 8)], prompt="Describe.")

    assert str(caught.value) == (
        "Failed to connect to Roboflow proxy: refused for <secret roboflow_api_key>"
    )


# Validation and failures before or during requests


def test_replayed_secret_fails_before_any_request() -> None:
    server = FakeAnthropicServer(lambda body: (200, claude_message("x")))

    with pytest.raises(SecretUnavailableError):
        _run(
            _direct_block(server),
            [_image(8, 8)],
            prompt="Describe.",
            overrides={"api_key": Secret.unavailable("anthropic")},
        )

    assert server.requests == []


def test_missing_key_fails_without_reading_the_environment(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-from-environment")
    server = FakeAnthropicServer(lambda body: (200, claude_message("x")))

    with pytest.raises(ChatRequestError, match="Anthropic API key is missing"):
        _run(
            _direct_block(server),
            [_image(8, 8)],
            prompt="Describe.",
            overrides={"api_key": None},
        )

    assert server.requests == []


@pytest.mark.parametrize(
    "override, message",
    [
        ({"max_concurrent_requests": 0}, "max_concurrent_requests must be at least 1"),
        ({"max_image_size": 0}, "max_image_size must be at least 1"),
        ({"classes": ["car", "car"]}, "classes repeat"),
    ],
)
def test_invalid_run_values_fail_before_any_request(override, message) -> None:
    server = FakeAnthropicServer(lambda body: (200, claude_message("[]")))

    with pytest.raises(ValueError, match=message):
        _run(
            _direct_block(server),
            [_image(8, 8)],
            task_type="object-detection",
            classes=["car"],
            api_key=ANTHROPIC_KEY,
            overrides=override,
        )

    assert server.requests == []


def test_failed_request_is_scrubbed_and_unchained() -> None:
    server = FakeAnthropicServer(
        lambda body: (
            400,
            {
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "message": f"bad {ANTHROPIC_KEY}",
                },
            },
        )
    )

    with pytest.raises(ChatRequestError) as caught:
        _run(_direct_block(server), [_image(8, 8)], prompt="x", api_key=ANTHROPIC_KEY)

    chain = _chain_text(caught.value)
    assert caught.value.__cause__ is None and caught.value.__context__ is None
    assert "Anthropic request failed" in chain
    assert "bad <secret api_key>" in chain
    assert ANTHROPIC_KEY not in chain


def test_answers_keep_image_order_within_the_concurrency_limit() -> None:
    def reply(body):
        size = len(body["messages"][0]["content"][0]["source"]["data"])
        return 200, claude_message(str(size))

    server = FakeAnthropicServer(reply, delay_s=0.02)
    images = [_image(8 + index, 8, seed=index) for index in range(6)]

    results = _run(
        _direct_block(server),
        images,
        prompt="x",
        api_key=ANTHROPIC_KEY,
        max_concurrent_requests=2,
    )

    sent = [
        str(len(request["body"]["messages"][0]["content"][0]["source"]["data"]))
        for request in server.requests
    ]
    assert sorted(result["output"] for result in results) == sorted(sent)
    expected = [
        str(
            len(
                claude_upload(
                    image, task_type="unconstrained", max_image_size=1024
                ).base64
            )
        )
        for image in images
    ]
    assert [result["output"] for result in results] == expected
    assert server.most_in_flight == 2


def test_model_table_matches_v1_labels_and_limits() -> None:
    assert list(CLAUDE_MODELS)[:3] == [
        "claude-opus-5-5",
        "claude-sonnet-5-5",
        "claude-fable-5-1",
    ]
    assert CLAUDE_MODELS["claude-opus-4-1"].max_output_tokens == 32000
    assert len(CLAUDE_MODELS) == 16


@pytest.mark.parametrize("host_key", [None, "", Secret("", name="empty-host")])
def test_managed_key_requires_nonempty_host_before_any_effect(
    host_key, monkeypatch
) -> None:
    managed_key = "rf_key:user:invented-managed-user"
    platform = FakePlatformClient(lambda endpoint, payload: claude_message("unused"))
    server = FakeAnthropicServer(lambda body: (200, claude_message("unused")))
    block = _direct_block(server, platform_client=platform, api_key=host_key)
    sdk_constructions = []
    real_sdk = anthropic.Anthropic

    def capture_sdk(**kwargs):
        sdk_constructions.append(kwargs)
        return real_sdk(**kwargs)

    monkeypatch.setattr(anthropic, "Anthropic", capture_sdk)
    with pytest.raises(ValueError, match="Roboflow API key is required") as caught:
        _run(block, [_image(8, 8)], api_key=managed_key, prompt="Describe.")

    assert "host must provide the 'api_key' resource" in str(caught.value)
    assert sdk_constructions == [] and server.requests == [] and platform.posts == []
    assert caught.value.__cause__ is None and caught.value.__context__ is None
    assert managed_key not in _chain_text(caught.value)
    assert ROBOFLOW_KEY not in _chain_text(caught.value)


def test_explicit_own_key_needs_no_host_key() -> None:
    server = FakeAnthropicServer(lambda body: (200, claude_message("Own key.")))
    platform = FakePlatformClient(lambda endpoint, payload: claude_message("unused"))
    block = _direct_block(server, platform_client=platform)

    results = _run(block, [_image(8, 8)], api_key=ANTHROPIC_KEY, prompt="Describe.")

    assert results[0]["output"] == "Own key."
    assert len(server.requests) == 1 and platform.posts == []
    assert server.requests[0]["headers"]["x-api-key"] == ANTHROPIC_KEY

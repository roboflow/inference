"""Unit tests of ``roboflow_core/google_gemini@v6`` against in-process fakes.

Gemini's REST API is a ``FakeChatServer`` behind an ``httpx.MockTransport``;
the Roboflow proxy is a ``FakePlatformClient``. Nothing leaves the process.
These tests check requests and wiring, not any real model.
"""

import json
import threading
import time
from typing import Any, Dict, List, Optional

import pytest
import requests
import torch
from pydantic import ValidationError
from roboflow_workflows.v2.blocks.models.vision_language_models.chat import (
    ChatRequestError,
    encode_image,
)
from roboflow_workflows.v2.blocks.models.vision_language_models.google_gemini.v6 import (
    GEMINI_MODELS,
    GoogleGeminiBlock,
    check_thinking_level,
    gemini_usage,
    generation_config,
    read_gemini_response,
)
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.secrets import Secret, SecretUnavailableError

from inference_models.models.base.classification import (
    MultiLabelClassificationPrediction,
)
from inference_models.models.base.object_detection import Detections
from tests.v2.support.language.fake_chat import FakeChatServer, FakePlatformClient

GOOGLE_KEY = "AIza-test-GEMINI-5c3e"
ROBOFLOW_KEY = "rf-test-ROBOFLOW-1a2b"
BASE_URL = "https://gemini.test/v1beta"


def gemini_body(
    text: Optional[str],
    *,
    finish_reason: Optional[str] = "STOP",
    usage: Optional[Dict[str, int]] = None,
    parts: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """A ``generateContent`` response with one candidate."""
    candidate: Dict[str, Any] = {
        "content": {"parts": parts if parts is not None else [{"text": text}]}
    }
    if finish_reason is not None:
        candidate["finishReason"] = finish_reason
    body = {
        "candidates": [candidate],
        "usageMetadata": usage
        or {"promptTokenCount": 30, "candidatesTokenCount": 4, "thoughtsTokenCount": 6},
    }

    return body


def _image(width: int, height: int, *, seed: int = 0) -> ImageData:
    generator = torch.Generator().manual_seed(seed)
    pixels = torch.randint(
        0, 256, (3, height, width), dtype=torch.uint8, generator=generator
    )
    image = ImageData.from_tensor(pixels)

    return image


def _params(**fields: Any) -> GoogleGeminiBlock.Params:
    params = GoogleGeminiBlock.Params(images="$inputs.image", **fields)

    return params


def _run(
    block: GoogleGeminiBlock,
    images: List[ImageData],
    *,
    overrides: Optional[Dict[str, Any]] = None,
    **fields: Any,
) -> List[Dict[str, Any]]:
    """Validate ``fields`` as Params, then run with run-time ``overrides``."""
    params = _params(**fields).model_dump(exclude={"images"})
    results = block.run(images=images, **{**params, **(overrides or {})})

    return results


def _direct_block(server: FakeChatServer, **resources: Any) -> GoogleGeminiBlock:
    block = GoogleGeminiBlock(
        gemini_http_client=server.http_client(), gemini_base_url=BASE_URL, **resources
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

    assert params.model_version == "gemini-3.1-pro-preview"
    assert params.api_key == "rf_key:account"
    assert params.google_code_execution is False


def test_retired_model_ids_are_renamed() -> None:
    params = _params(prompt="x", model_version="gemini-2.5-pro-preview-06-05")

    assert params.model_version == "gemini-2.5-pro"


@pytest.mark.parametrize(
    "fields, message",
    [
        ({"task_type": "visual-question-answering"}, "`prompt` parameter required"),
        ({"task_type": "classification"}, "`classes` parameter required"),
        ({"task_type": "structured-answering"}, "`output_structure` parameter"),
        (
            {
                "prompt": "x",
                "model_version": "gemini-3.1-pro-preview",
                "thinking_level": "minimal",
            },
            r"supports thinking_level values \['low', 'medium', 'high'\], got 'minimal'",
        ),
        (
            {
                "prompt": "x",
                "model_version": "gemini-2.5-flash",
                "thinking_level": "low",
            },
            "does not support configurable thinking_level",
        ),
        ({"prompt": "x", "temperature": 2.5}, "less than or equal"),
        ({"prompt": "x", "max_concurrent_requests": 0}, "greater than or equal"),
        ({"prompt": "x", "max_tokens": 0}, "greater than or equal"),
        ({"prompt": "x", "model_version": "gemini-unknown"}, "model_version"),
    ],
)
def test_params_reject_v1_invalid_configurations(fields, message) -> None:
    with pytest.raises(ValidationError, match=message):
        _params(**fields)


def test_params_accept_selectors() -> None:
    params = _params(
        prompt="x",
        model_version="$inputs.model",
        thinking_level="$inputs.level",
        temperature="$inputs.temperature",
        google_code_execution="$inputs.code",
    )

    assert params.temperature == "$inputs.temperature"


def test_dependent_resource_names_the_model() -> None:
    resources = GoogleGeminiBlock.discover_dependent_resources(_params(prompt="x"))

    assert resources[0].identifier == "gemini-3.1-pro-preview"
    assert resources[0].details == {
        "provider": "google",
        "resource": "gemini_http_client",
    }


def test_metadata_carries_v1_detection_description_and_levels() -> None:
    metadata = GoogleGeminiBlock.metadata

    assert metadata["task_metadata"]["object-detection"]["name"] == "Object Detection"
    assert metadata["model_version_metadata"]["gemini-3.8-flash"] == {
        "name": "Gemini 3.8 Flash",
        "reasoning_levels": ["low", "medium", "high"],
    }


# Generation config


def test_thinking_models_get_a_level_and_lose_temperature() -> None:
    config = generation_config(
        "gemini-3.5-flash", max_tokens=100, temperature=0.4, thinking_level="minimal"
    )

    assert config == {
        "response_mime_type": "text/plain",
        "candidate_count": 1,
        "max_output_tokens": 100,
        "thinking_config": {"thinking_level": "minimal"},
    }


def test_other_models_get_temperature_and_no_level() -> None:
    known = generation_config(
        "gemini-2.5-flash", max_tokens=None, temperature=0.4, thinking_level=None
    )
    unknown = generation_config(
        "gemini-next", max_tokens=None, temperature=0.4, thinking_level="high"
    )

    assert known["temperature"] == 0.4
    assert unknown["temperature"] == 0.4
    assert "thinking_config" not in unknown


def test_run_time_thinking_level_is_checked_before_any_request() -> None:
    server = FakeChatServer(lambda path, body: (200, gemini_body("x")))

    with pytest.raises(
        ValueError, match="does not support configurable thinking_level"
    ):
        _run(
            _direct_block(server),
            [_image(8, 8)],
            prompt="x",
            api_key=GOOGLE_KEY,
            overrides={"model_version": "gemini-2.5-pro", "thinking_level": "low"},
        )

    assert server.requests == []


def test_check_thinking_level_lets_selectors_and_unknown_models_pass() -> None:
    check_thinking_level("$inputs.model", "minimal")
    check_thinking_level("gemini-next", "minimal")
    check_thinking_level("gemini-3.1-pro-preview", None)


# Reading answers


def test_usage_adds_thinking_tokens_to_output() -> None:
    assert gemini_usage({"promptTokenCount": 3, "thoughtsTokenCount": 5}) == (3, 5)
    assert gemini_usage({"promptTokenCount": 3}) == (3, None)
    assert gemini_usage(None) == (None, None)


def test_reading_skips_code_parts() -> None:
    parts = [{"executableCode": {"code": "1+1"}}, {"text": "2"}]

    answer = read_gemini_response(gemini_body(None, parts=parts))

    assert (answer.text, answer.input_tokens, answer.output_tokens) == ("2", 30, 10)


@pytest.mark.parametrize(
    "body, error",
    [
        ({"candidates": []}, "returned no response candidates"),
        (gemini_body("x", finish_reason="MAX_TOKENS"), "max_tokens limit"),
        (gemini_body("x", finish_reason="SAFETY"), "finish reason: SAFETY"),
        (gemini_body(None, parts=[]), "Unable to parse Gemini API response"),
        (gemini_body(None, parts=[{"inlineData": {}}]), "Unable to parse"),
    ],
)
def test_reading_fails_with_v1_messages(body, error) -> None:
    with pytest.raises(ChatRequestError, match=error) as caught:
        read_gemini_response(body)

    assert caught.value.__cause__ is None


def test_missing_finish_reason_is_accepted() -> None:
    assert read_gemini_response(gemini_body("ok", finish_reason=None)).text == "ok"


# Direct requests


def test_direct_request_posts_v1_body_with_the_key_in_a_header() -> None:
    server = FakeChatServer(lambda path, body: (200, gemini_body("A dog.")))
    image = _image(40, 20)

    results = _run(
        _direct_block(server),
        [image],
        task_type="caption",
        model_version="gemini-2.5-flash",
        temperature=0.3,
        api_key=GOOGLE_KEY,
    )

    request = server.requests[0]
    assert request.url == f"{BASE_URL}/models/gemini-2.5-flash:generateContent"
    assert request.headers["x-goog-api-key"] == GOOGLE_KEY
    assert GOOGLE_KEY not in request.url
    assert request.body == {
        "systemInstruction": {
            "role": "system",
            "parts": [
                {
                    "text": "You act as image caption model. Your task is to provide "
                    "description of the image. Caption should be short."
                }
            ],
        },
        "contents": {
            "parts": [
                {
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": encode_image(image).base64,
                    }
                },
                {"text": "Caption the image"},
            ],
            "role": "user",
        },
        "generationConfig": {
            "response_mime_type": "text/plain",
            "candidate_count": 1,
            "temperature": 0.3,
        },
    }
    assert results[0]["output"] == "A dog."
    assert (results[0]["input_tokens"], results[0]["output_tokens"]) == (30, 10)
    assert results[0]["predictions"] is None


def test_code_execution_is_offered_only_to_supporting_models() -> None:
    server = FakeChatServer(lambda path, body: (200, gemini_body("x")))
    block = _direct_block(server)

    _run(
        block,
        [_image(8, 8)],
        prompt="x",
        google_code_execution=True,
        api_key=GOOGLE_KEY,
    )
    _run(
        block,
        [_image(8, 8)],
        prompt="x",
        model_version="gemini-2.5-pro",
        google_code_execution=True,
        api_key=GOOGLE_KEY,
    )

    assert server.requests[0].body["tools"] == [{"code_execution": {}}]
    assert "tools" not in server.requests[1].body


def test_detection_asks_for_box_2d_and_decodes_pixels() -> None:
    answer = json.dumps([{"box_2d": [0, 0, 500, 250], "label": "car"}])
    server = FakeChatServer(lambda path, body: (200, gemini_body(answer)))

    results = _run(
        _direct_block(server),
        [_image(400, 200), _image(800, 100)],
        task_type="object-detection",
        classes=["car", "bus"],
        api_key=GOOGLE_KEY,
    )

    config = server.requests[0].body["generationConfig"]
    assert config["response_mime_type"] == "application/json"
    assert config["response_schema"]["items"]["properties"]["label"]["enum"] == [
        "car",
        "bus",
    ]
    assert "systemInstruction" not in server.requests[0].body
    assert isinstance(results[0]["predictions"], Detections)
    assert results[0]["predictions"].xyxy.tolist() == [[0.0, 0.0, 100.0, 100.0]]
    assert results[1]["predictions"].xyxy.tolist() == [[0.0, 0.0, 200.0, 50.0]]


def test_multi_label_answer_becomes_a_native_prediction() -> None:
    answer = '{"predicted_classes": [{"class": "car", "confidence": 0.9}]}'
    server = FakeChatServer(lambda path, body: (200, gemini_body(answer)))

    results = _run(
        _direct_block(server),
        [_image(8, 8)],
        task_type="multi-label-classification",
        classes=["car", "bus"],
        api_key=GOOGLE_KEY,
    )

    assert isinstance(results[0]["predictions"], MultiLabelClassificationPrediction)
    assert server.requests[0].body["generationConfig"]["response_mime_type"] == (
        "application/json"
    )


def test_secret_key_is_revealed_only_on_the_wire() -> None:
    server = FakeChatServer(lambda path, body: (200, gemini_body("x")))

    _run(
        _direct_block(server),
        [_image(8, 8)],
        prompt="x",
        overrides={"api_key": Secret(GOOGLE_KEY, name="google")},
    )

    assert server.requests[0].headers["x-goog-api-key"] == GOOGLE_KEY


# Managed key


def test_managed_key_posts_the_v1_proxy_payload_with_the_host_key() -> None:
    platform = FakePlatformClient(lambda endpoint, payload: gemini_body("ok"))
    server = FakeChatServer(lambda path, body: (200, gemini_body("unused")))
    block = _direct_block(server, platform_client=platform, api_key=ROBOFLOW_KEY)

    results = _run(block, [_image(8, 8)], task_type="ocr", thinking_level="high")

    assert server.requests == []
    post = platform.posts[0]
    assert (post["endpoint"], post["api_key"]) == ("apiproxy/gemini", ROBOFLOW_KEY)
    assert post["payload"]["model"] == "gemini-3.1-pro-preview"
    assert post["payload"]["google_api_key"] == "rf_key:account"
    assert post["payload"]["contents"]["parts"][1] == {"text": "Read the text"}
    assert post["payload"]["generationConfig"]["thinking_config"] == {
        "thinking_level": "high"
    }
    assert results[0]["output"] == "ok"


def test_managed_key_without_a_platform_client_sends_nothing() -> None:
    server = FakeChatServer(lambda path, body: (200, gemini_body("unused")))

    with pytest.raises(ChatRequestError, match="provided no platform client"):
        _run(_direct_block(server, api_key=ROBOFLOW_KEY), [_image(8, 8)], prompt="x")

    assert server.requests == []


def test_proxy_connection_errors_keep_the_v1_message() -> None:
    def fail(endpoint, payload):
        raise requests.exceptions.ConnectionError(f"refused for {ROBOFLOW_KEY}")

    block = GoogleGeminiBlock(
        platform_client=FakePlatformClient(fail), api_key=ROBOFLOW_KEY
    )

    with pytest.raises(ChatRequestError) as caught:
        _run(block, [_image(8, 8)], prompt="x")

    assert str(caught.value) == (
        "Failed to connect to Roboflow proxy: refused for <secret roboflow_api_key>"
    )


# Validation and failures


def test_replayed_secret_fails_before_any_request() -> None:
    server = FakeChatServer(lambda path, body: (200, gemini_body("x")))

    with pytest.raises(SecretUnavailableError):
        _run(
            _direct_block(server),
            [_image(8, 8)],
            prompt="x",
            overrides={"api_key": Secret.unavailable("google")},
        )

    assert server.requests == []


def test_missing_key_fails_before_any_request() -> None:
    server = FakeChatServer(lambda path, body: (200, gemini_body("x")))

    with pytest.raises(ChatRequestError, match="Google API key is missing"):
        _run(
            _direct_block(server),
            [_image(8, 8)],
            prompt="x",
            overrides={"api_key": None},
        )

    assert server.requests == []


@pytest.mark.parametrize(
    "override, message",
    [
        ({"max_concurrent_requests": 0}, "max_concurrent_requests must be at least 1"),
        ({"classes": ["car", "car"]}, "classes repeat"),
    ],
)
def test_invalid_run_values_fail_before_any_request(override, message) -> None:
    server = FakeChatServer(lambda path, body: (200, gemini_body("[]")))

    with pytest.raises(ValueError, match=message):
        _run(
            _direct_block(server),
            [_image(8, 8)],
            task_type="object-detection",
            classes=["car"],
            api_key=GOOGLE_KEY,
            overrides=override,
        )

    assert server.requests == []


def test_http_error_is_scrubbed_and_unchained() -> None:
    server = FakeChatServer(
        lambda path, body: (403, {"error": {"message": f"bad key {GOOGLE_KEY}"}})
    )

    with pytest.raises(ChatRequestError) as caught:
        _run(_direct_block(server), [_image(8, 8)], prompt="x", api_key=GOOGLE_KEY)

    chain = _chain_text(caught.value)
    assert caught.value.__cause__ is None and caught.value.__context__ is None
    assert "Gemini request failed: Client error '403 Forbidden'" in chain
    assert GOOGLE_KEY not in chain


def test_a_failure_cancels_requests_not_yet_started() -> None:
    def reply(path, body):
        if body["contents"]["parts"][1]["text"] == "x" and not failed.is_set():
            failed.set()
            return 500, {"error": {"message": "boom"}}
        return 200, gemini_body("ok")

    failed = threading.Event()
    server = FakeChatServer(reply)

    with pytest.raises(ChatRequestError, match="500 Internal Server Error"):
        _run(
            _direct_block(server),
            [_image(8, 8, seed=index) for index in range(5)],
            prompt="x",
            api_key=GOOGLE_KEY,
            max_concurrent_requests=1,
        )

    assert len(server.requests) == 1


def test_answers_keep_image_order_within_the_concurrency_limit() -> None:
    lock = threading.Lock()
    state = {"in_flight": 0, "most": 0}

    def reply(path, body):
        with lock:
            state["in_flight"] += 1
            state["most"] = max(state["most"], state["in_flight"])
        time.sleep(0.02)
        with lock:
            state["in_flight"] -= 1
        data = body["contents"]["parts"][0]["inline_data"]["data"]
        return 200, gemini_body(str(len(data)))

    server = FakeChatServer(reply)
    images = [_image(8 + index, 8, seed=index) for index in range(6)]

    results = _run(
        _direct_block(server),
        images,
        prompt="x",
        api_key=GOOGLE_KEY,
        max_concurrent_requests=3,
    )

    assert [result["output"] for result in results] == [
        str(len(encode_image(image).base64)) for image in images
    ]
    assert state["most"] == 3


def test_model_table_matches_v1() -> None:
    assert list(GEMINI_MODELS)[0] == "gemini-3.8-flash"
    assert GEMINI_MODELS["gemini-3.6-flash"].thinking_levels == (
        "minimal",
        "low",
        "medium",
        "high",
    )
    assert not GEMINI_MODELS["gemini-2.5-flash-lite"].native_code_execution
    assert len(GEMINI_MODELS) == 11


@pytest.mark.parametrize("host_key", [None, "", Secret("", name="empty-host")])
def test_managed_key_requires_nonempty_host_before_any_effect(
    host_key, monkeypatch
) -> None:
    managed_key = "rf_key:user:invented-managed-user"
    platform = FakePlatformClient(lambda endpoint, payload: gemini_body("unused"))
    server = FakeChatServer(lambda path, body: (200, gemini_body("unused")))
    block = _direct_block(server, platform_client=platform, api_key=host_key)
    direct_calls = []
    real_direct = block._send_direct

    def capture_direct(*args, **kwargs):
        direct_calls.append((args, kwargs))
        return real_direct(*args, **kwargs)

    monkeypatch.setattr(block, "_send_direct", capture_direct)
    with pytest.raises(ValueError, match="Roboflow API key is required") as caught:
        _run(block, [_image(8, 8)], api_key=managed_key, prompt="Describe.")

    assert "host must provide the 'api_key' resource" in str(caught.value)
    assert direct_calls == [] and server.requests == [] and platform.posts == []
    assert caught.value.__cause__ is None and caught.value.__context__ is None
    assert managed_key not in _chain_text(caught.value)
    assert ROBOFLOW_KEY not in _chain_text(caught.value)


def test_explicit_own_key_needs_no_host_key() -> None:
    server = FakeChatServer(lambda path, body: (200, gemini_body("Own key.")))
    platform = FakePlatformClient(lambda endpoint, payload: gemini_body("unused"))
    block = _direct_block(server, platform_client=platform)

    results = _run(block, [_image(8, 8)], api_key=GOOGLE_KEY, prompt="Describe.")

    assert results[0]["output"] == "Own key."
    assert len(server.requests) == 1 and platform.posts == []
    assert server.requests[0].headers["x-goog-api-key"] == GOOGLE_KEY

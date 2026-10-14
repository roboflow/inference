"""Gemma and SpaceXAI blocks: requests, keys, failures, decoding.

Every request goes to an in-process fake transport or platform client; the
answers are fixed strings, not model output.
"""
import roboflow_workflows.v2.blocks.language._catalogue_compatible_vendors as _v2_family_0

import base64
import json
from typing import Any, Dict

import cv2
import numpy as np
import pytest
import torch
from roboflow_workflows.v2.blocks.language._shared import compatible_vendors as compatible_vendors
from roboflow_workflows.v2.blocks.language.chat import ChatRequestError
from roboflow_workflows.v2.blocks.language.google_gemma.v4 import GoogleGemmaBlock
from roboflow_workflows.v2.blocks.language.spacexai.v3 import SpaceXAIBlock
from roboflow_workflows.v2.blocks.language._shared.compatible_vendors import xai_task_request
from roboflow_workflows.v2.data.batch import Batch
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.predictions import (
    CLASSIFICATION_PREDICTION_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
)
from roboflow_workflows.v2.data.secrets import Secret, SecretUnavailableError
from roboflow_workflows.v2.engine.declaration import spec_of
from tests.v2.support.language.fake_chat import (
    FakeChatServer,
    FakePlatformClient,
    chat_completion_body,
    error_body,
    responses_body,
)

OPENROUTER_KEY = "sk-or-test-OPENROUTER-5e6f"
XAI_KEY = "xai-test-XAI-8b9c"
ROBOFLOW_KEY = "rf-test-ROBOFLOW-3d4e"
FAKE_OPENROUTER_URL = "http://openrouter.test/api/v1"
FAKE_XAI_URL = "http://xai.test/v1"
DETECTION_ANSWER_NORMALIZED = json.dumps(
    {
        "detections": [
            {
                "x_min": 0.25,
                "y_min": 0.5,
                "x_max": 0.75,
                "y_max": 1.0,
                "class_name": "car",
                "confidence": 0.9,
            }
        ]
    }
)


def _image(width: int = 64, height: int = 48, *, seed: int = 0) -> ImageData:
    generator = torch.Generator().manual_seed(seed)
    pixels = torch.randint(
        0, 256, (3, height, width), dtype=torch.uint8, generator=generator
    )
    image = ImageData.from_tensor(pixels)

    return image


def _run(block_class, block, images, resolved=None, **raw: Any):
    params = spec_of(block_class).validate_params({"images": "$inputs.images", **raw})
    arguments = {name: getattr(params, name) for name in type(params).model_fields}
    arguments.update(images=Batch(images), **(resolved or {}))
    results = block.run(**arguments)

    return results


def _chat_server(content: str = "A cat.", **body: Any) -> FakeChatServer:
    server = FakeChatServer(
        lambda path, request: (200, chat_completion_body(content, **body))
    )

    return server


def _responses_server(text: str = "A cat.", **body: Any) -> FakeChatServer:
    server = FakeChatServer(lambda path, request: (200, responses_body(text, **body)))

    return server


def _chain_text(error: BaseException) -> str:
    texts = []
    while error is not None:
        texts.append(f"{type(error).__name__}: {error}")
        error = error.__cause__ or error.__context__

    return "\n".join(texts)


def _gemma(server: FakeChatServer, **resources: Any) -> GoogleGemmaBlock:
    block = GoogleGemmaBlock(
        openrouter_base_url=FAKE_OPENROUTER_URL,
        openrouter_http_client=server.http_client(),
        **resources,
    )

    return block


def _xai(server: FakeChatServer, **resources: Any) -> SpaceXAIBlock:
    block = SpaceXAIBlock(
        xai_base_url=FAKE_XAI_URL, xai_http_client=server.http_client(), **resources
    )

    return block


# --- google_gemma@v4 -------------------------------------------------------


def test_gemma_direct_request_uses_the_slug_and_openrouter_routing() -> None:
    server = _chat_server("A cat.")

    results = _run(
        GoogleGemmaBlock,
        _gemma(server),
        [_image()],
        api_key=OPENROUTER_KEY,
        prompt="What is this?",
    )

    request = server.requests[0]
    assert request.url == f"{FAKE_OPENROUTER_URL}/chat/completions"
    assert request.headers["authorization"] == f"Bearer {OPENROUTER_KEY}"
    assert request.body["model"] == "google/gemma-4-31b-it"
    assert request.body["max_tokens"] == 2048
    assert request.body["temperature"] == 0.1
    assert request.body["provider"] == {
        "data_collection": "deny",
        "quantizations": ["bf16", "fp32"],
    }
    assert "reasoning" not in request.body
    assert results[0]["output"] == "A cat."
    assert results[0]["input_tokens"] == 13
    assert results[0]["output_tokens"] == 5
    assert "thinking" not in results[0]


def test_gemma_managed_key_goes_through_the_platform_proxy() -> None:
    server = _chat_server("unused")
    platform = FakePlatformClient(
        lambda endpoint, payload: chat_completion_body("Proxied.")
    )

    results = _run(
        GoogleGemmaBlock,
        _gemma(server, platform_client=platform, api_key=ROBOFLOW_KEY),
        [_image()],
        prompt="What is this?",
        model_version="Gemma 4 26B A4B - OpenRouter",
        reasoning_effort="none",
    )

    assert server.requests == []
    post = platform.posts[0]
    assert (post["endpoint"], post["api_key"]) == ("apiproxy/openrouter", ROBOFLOW_KEY)
    assert post["payload"]["openrouter_api_key"] == "rf_key:account"
    assert post["payload"]["model"] == "google/gemma-4-26b-a4b-it"
    assert post["payload"]["reasoning"] == {"enabled": False}
    assert results[0]["output"] == "Proxied."


def test_gemma_decodes_normalized_detections_into_image_pixels() -> None:
    server = _chat_server(DETECTION_ANSWER_NORMALIZED)

    results = _run(
        GoogleGemmaBlock,
        _gemma(server),
        [_image(64, 48)],
        api_key=OPENROUTER_KEY,
        task_type="object-detection",
        classes=["car"],
    )

    system = server.requests[0].body["messages"][0]["content"]
    assert '"x_min": 0.1' in system
    assert results[0]["error_status"] is False
    assert results[0]["predictions"].xyxy.tolist() == [[16.0, 24.0, 48.0, 48.0]]


def test_gemma_unknown_model_from_a_selector_fails_before_any_request() -> None:
    server = _chat_server()

    with pytest.raises(ValueError, match="Invalid model name: 'Gemma 9'"):
        _run(
            GoogleGemmaBlock,
            _gemma(server),
            [_image()],
            resolved={"model_version": "Gemma 9"},
            api_key=OPENROUTER_KEY,
            prompt="Hi",
            model_version="$inputs.model",
        )

    assert server.requests == []


def test_gemma_retries_once_without_a_rejected_reasoning_config() -> None:
    replies = iter(
        [
            (400, error_body("Reasoning is not supported for this model")),
            (200, chat_completion_body("Fine.")),
        ]
    )
    server = FakeChatServer(lambda path, request: next(replies))

    results = _run(
        GoogleGemmaBlock,
        _gemma(server),
        [_image()],
        api_key=OPENROUTER_KEY,
        prompt="Hi",
        reasoning_effort="low",
    )

    assert [request.body.get("reasoning") for request in server.requests] == [
        {"effort": "low"},
        None,
    ]
    assert results[0]["output"] == "Fine."


def test_gemma_outputs_and_resources_follow_the_configuration() -> None:
    spec = spec_of(GoogleGemmaBlock)
    detection = spec.validate_params(
        {"images": "$inputs.i", "task_type": "object-detection", "classes": ["a"]}
    )
    selected = spec.validate_params(
        {"images": "$inputs.i", "prompt": "Hi", "model_version": "$inputs.model"}
    )

    outputs = GoogleGemmaBlock.describe_outputs(detection)

    assert outputs["predictions"].kinds == (OBJECT_DETECTION_PREDICTION_KIND,)
    assert [
        resource.identifier
        for resource in GoogleGemmaBlock.discover_dependent_resources(detection)
    ] == ["google/gemma-4-31b-it"]
    assert [
        resource.identifier
        for resource in GoogleGemmaBlock.discover_dependent_resources(selected)
    ] == ["$inputs.model"]


def test_gemma_requires_the_field_its_task_needs() -> None:
    with pytest.raises(ValueError, match="`classes` parameter required"):
        spec_of(GoogleGemmaBlock).validate_params(
            {"images": "$inputs.i", "task_type": "classification"}
        )


# --- spacexai@v3 -----------------------------------------------------------


def test_xai_direct_request_is_a_responses_call_without_storage() -> None:
    server = _responses_server("Some text.")

    results = _run(
        SpaceXAIBlock,
        _xai(server),
        [_image()],
        api_key=XAI_KEY,
        task_type="ocr",
    )

    request = server.requests[0]
    assert request.url == f"{FAKE_XAI_URL}/responses"
    assert request.headers["authorization"] == f"Bearer {XAI_KEY}"
    assert request.body["model"] == "grok-4.6"
    assert request.body["store"] is False
    assert request.body["instructions"].startswith("You act as OCR model.")
    for absent in ("max_output_tokens", "temperature", "reasoning"):
        assert absent not in request.body
    image_part = request.body["input"][0]["content"][0]
    assert image_part["detail"] == "auto"
    assert image_part["image_url"].startswith("data:image/jpeg;base64,")
    assert results[0]["output"] == "Some text."
    assert (results[0]["input_tokens"], results[0]["output_tokens"]) == (11, 7)
    assert results[0]["predictions"] is None


def test_xai_detection_uploads_lossless_png_first_and_decodes_percent_boxes() -> None:
    image = _image(64, 48, seed=3)
    answer = json.dumps([{"label": "car", "box_2d": [25, 50, 75, 100]}])
    server = _responses_server(answer)

    results = _run(
        SpaceXAIBlock,
        _xai(server),
        [image],
        api_key=XAI_KEY,
        task_type="object-detection",
        classes=["car"],
        max_tokens=64,
        temperature=0.0,
        reasoning_effort="high",
    )

    body = server.requests[0].body
    image_part, text_part = body["input"][0]["content"]
    assert image_part["detail"] == "high"
    assert image_part["image_url"].startswith("data:image/png;base64,")
    data = base64.b64decode(image_part["image_url"].split(",", 1)[1])
    uploaded = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    expected = image.tensor_image.permute(1, 2, 0).numpy()[:, :, ::-1]
    assert np.array_equal(uploaded, expected)
    _, level_nine = cv2.imencode(
        ".png", np.ascontiguousarray(expected), [cv2.IMWRITE_PNG_COMPRESSION, 9]
    )
    assert data == level_nine.tobytes()
    assert "box_2d" in text_part["text"] and "car" in text_part["text"]
    assert "instructions" not in body
    assert (body["max_output_tokens"], body["temperature"]) == (64, 0.0)
    assert body["reasoning"] == {"effort": "high"}
    assert results[0]["predictions"].xyxy.tolist() == [[16.0, 24.0, 48.0, 48.0]]
    assert results[0]["error_status"] is False


def test_xai_classification_decodes_a_classification_prediction() -> None:
    server = _responses_server('{"class_name": "dog", "confidence": 0.8}')

    results = _run(
        SpaceXAIBlock,
        _xai(server),
        [_image()],
        api_key=XAI_KEY,
        task_type="classification",
        classes=["cat", "dog"],
    )

    assert results[0]["error_status"] is False
    assert results[0]["predictions"] is not None
    describe = SpaceXAIBlock.describe_outputs(
        spec_of(SpaceXAIBlock).validate_params(
            {"images": "$inputs.i", "task_type": "classification", "classes": ["a"]}
        )
    )
    assert describe["predictions"].kinds == (CLASSIFICATION_PREDICTION_KIND,)


def test_xai_managed_key_goes_through_the_platform_proxy() -> None:
    server = _responses_server("unused")
    platform = FakePlatformClient(lambda endpoint, payload: responses_body("Proxied."))

    results = _run(
        SpaceXAIBlock,
        _xai(server, platform_client=platform, api_key=ROBOFLOW_KEY),
        [_image()],
        task_type="caption",
    )

    assert server.requests == []
    post = platform.posts[0]
    assert (post["endpoint"], post["api_key"]) == ("apiproxy/xai", ROBOFLOW_KEY)
    assert post["http_errors_handlers"] is None
    assert post["payload"]["xai_api_key"] == "rf_key:account"
    assert post["payload"]["store"] is False
    assert results[0]["output"] == "Proxied."


def test_xai_managed_key_without_the_host_key_fails_before_any_request() -> None:
    server = _responses_server()
    platform = FakePlatformClient(lambda endpoint, payload: responses_body("x"))

    with pytest.raises(ValueError, match="Roboflow API key is required"):
        _run(
            SpaceXAIBlock,
            _xai(server, platform_client=platform),
            [_image()],
            task_type="ocr",
        )

    assert server.requests == [] and platform.posts == []


def test_xai_rejects_an_unsupported_reasoning_level() -> None:
    with pytest.raises(ValueError, match="grok-4.5 supports reasoning_effort"):
        spec_of(SpaceXAIBlock).validate_params(
            {
                "images": "$inputs.i",
                "task_type": "ocr",
                "model_version": "grok-4.5",
                "reasoning_effort": "xhigh",
            }
        )

    server = _responses_server()
    with pytest.raises(ValueError, match="grok-4.5 supports reasoning_effort"):
        _run(
            SpaceXAIBlock,
            _xai(server),
            [_image()],
            resolved={"model_version": "grok-4.5"},
            api_key=XAI_KEY,
            task_type="ocr",
            model_version="$inputs.model",
            reasoning_effort="xhigh",
        )
    assert server.requests == []


def test_xai_direct_retry_drops_a_rejected_reasoning_config() -> None:
    replies = iter(
        [
            (400, error_body("Argument not supported: reasoning")),
            (200, responses_body("Done.")),
        ]
    )
    server = FakeChatServer(lambda path, request: next(replies))

    results = _run(
        SpaceXAIBlock,
        _xai(server),
        [_image()],
        api_key=XAI_KEY,
        task_type="ocr",
        reasoning_effort="low",
    )

    assert [request.body.get("reasoning") for request in server.requests] == [
        {"effort": "low"},
        None,
    ]
    assert results[0]["output"] == "Done."


def test_xai_proxied_errors_are_not_retried() -> None:
    def reply(endpoint, payload):
        raise RuntimeError("reasoning is not supported")

    platform = FakePlatformClient(reply)

    with pytest.raises(ChatRequestError, match="xAI request failed"):
        _run(
            SpaceXAIBlock,
            _xai(_responses_server(), platform_client=platform, api_key=ROBOFLOW_KEY),
            [_image()],
            task_type="ocr",
            reasoning_effort="low",
        )

    assert len(platform.posts) == 1


@pytest.mark.parametrize(
    "body, message",
    [
        (
            {"status": "failed", "error": {"code": "server_error", "message": "boom"}},
            "xAI API request failed: server_error: boom",
        ),
        (
            {
                "status": "incomplete",
                "incomplete_details": {"reason": "max_output_tokens"},
            },
            "xAI API stopped generation because the max_tokens limit was reached",
        ),
        ({"status": "cancelled"}, "xAI API request was cancelled."),
        ({"output": []}, "xAI API returned no text content in response."),
    ],
)
def test_xai_unusable_answers_fail_with_v1_messages(body, message: str) -> None:
    server = FakeChatServer(
        lambda path, request: (200, {**responses_body("text"), **body})
    )

    with pytest.raises(ChatRequestError, match=message):
        _run(SpaceXAIBlock, _xai(server), [_image()], api_key=XAI_KEY, task_type="ocr")


def test_xai_failure_names_no_key_and_chains_nothing() -> None:
    server = FakeChatServer(
        lambda path, request: (403, error_body(f"forbidden {XAI_KEY}"))
    )

    with pytest.raises(ChatRequestError) as caught:
        _run(SpaceXAIBlock, _xai(server), [_image()], api_key=XAI_KEY, task_type="ocr")

    assert "forbidden <secret api_key>" in str(caught.value)
    assert caught.value.__cause__ is None and caught.value.__context__ is None
    assert XAI_KEY not in _chain_text(caught.value)


def test_xai_first_failure_skips_requests_not_yet_started() -> None:
    def reply(path: str, body: Dict[str, Any]):
        return 400, error_body("bad request")

    server = FakeChatServer(reply)
    images = [_image(seed=index) for index in range(5)]

    with pytest.raises(ChatRequestError):
        _run(
            SpaceXAIBlock,
            _xai(server),
            images,
            api_key=XAI_KEY,
            task_type="ocr",
            max_concurrent_requests=1,
        )

    assert (
        len(
            {
                request.body["input"][0]["content"][0]["image_url"]
                for request in server.requests
            }
        )
        == 1
    )


def test_xai_task_requests_follow_v1_part_order() -> None:
    image = _image()

    unconstrained = xai_task_request(
        "unconstrained", image=image, prompt="Hi", output_structure=None, classes=None
    )
    structured = xai_task_request(
        "structured-answering",
        image=image,
        prompt=None,
        output_structure={"count": "cars"},
        classes=None,
    )

    assert "instructions" not in unconstrained
    assert [part["type"] for part in unconstrained["input"][0]["content"]] == [
        "input_text",
        "input_image",
    ]
    assert structured["input"][0]["content"][0]["text"] == (
        'Specification of requirements regarding output fields: \n{\n    "count": "cars"\n}'
    )
    with pytest.raises(ValueError, match="not supported"):
        xai_task_request(
            "instance-segmentation",
            image=image,
            prompt=None,
            output_structure=None,
            classes=["a"],
        )


@pytest.mark.parametrize(
    "block_class, provider",  # all credentials below are invented
    [
        (SpaceXAIBlock, "xAI"),
    ],
)
def test_missing_resolved_provider_key_rejects_before_sdk_or_request(
    block_class, provider: str, monkeypatch
) -> None:
    ambient_key = "invented-ambient-openai-key-must-not-send"
    monkeypatch.setenv("OPENAI_API_KEY", ambient_key)
    server = _responses_server()
    block = _xai(server)
    constructions = []
    real_openai = compatible_vendors.OpenAI

    def capture_sdk(**kwargs):
        constructions.append(kwargs)
        return real_openai(**kwargs)

    monkeypatch.setattr(compatible_vendors, "OpenAI", capture_sdk)
    with pytest.raises(ValueError, match=f"requires an {provider} API key") as caught:
        _run(
            block_class,
            block,
            [_image()],
            api_key="$inputs.key",
            resolved={"api_key": None},
            prompt="Hi",
        )

    assert constructions == []
    assert server.requests == []
    assert caught.value.__cause__ is None and caught.value.__context__ is None
    assert ambient_key not in _chain_text(caught.value)


@pytest.mark.parametrize(
    "block_class, explicit_key",
    [
        (SpaceXAIBlock, XAI_KEY),
    ],
)
def test_explicit_provider_key_overrides_ambient_openai_key(
    block_class, explicit_key: str, monkeypatch
) -> None:
    ambient_key = "invented-ambient-openai-key-must-not-send"
    monkeypatch.setenv("OPENAI_API_KEY", ambient_key)
    server = _responses_server()
    block = _xai(server)
    _run(block_class, block, [_image()], api_key=explicit_key, prompt="Hi")

    assert len(server.requests) == 1
    request = server.requests[0]
    assert request.headers["authorization"] == f"Bearer {explicit_key}"
    assert ambient_key not in str(request.headers) + str(request.body)


def test_xai_managed_key_ignores_ambient_key_without_sdk(monkeypatch) -> None:
    ambient_key = "invented-ambient-openai-key-must-not-send"
    monkeypatch.setenv("OPENAI_API_KEY", ambient_key)
    server = _responses_server("unused")
    platform = FakePlatformClient(lambda endpoint, payload: responses_body("Proxied."))
    constructions = []

    def capture_sdk(**kwargs):
        constructions.append(kwargs)
        raise AssertionError("managed xAI key must not construct the SDK")

    monkeypatch.setattr(compatible_vendors, "OpenAI", capture_sdk)
    results = _run(
        SpaceXAIBlock,
        _xai(server, platform_client=platform, api_key=ROBOFLOW_KEY),
        [_image()],
        api_key="rf_key:user:invented-account",
        task_type="caption",
    )

    assert constructions == [] and server.requests == []
    assert len(platform.posts) == 1
    post = platform.posts[0]
    assert (post["endpoint"], post["api_key"]) == ("apiproxy/xai", ROBOFLOW_KEY)
    assert post["payload"]["xai_api_key"] == "rf_key:user:invented-account"
    assert post["payload"]["store"] is False
    assert ambient_key not in str(post)
    assert results[0]["output"] == "Proxied."


def test_only_active_compatible_vendor_blocks_are_discoverable() -> None:
    assert _v2_family_0.BLOCKS == (GoogleGemmaBlock, SpaceXAIBlock)
    assert all("qwen3_" not in block.type for block in _v2_family_0.BLOCKS)

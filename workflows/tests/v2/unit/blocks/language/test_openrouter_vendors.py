"""OpenRouter vendor blocks: Kimi, Meta and Z.ai requests and decoding.

Every request goes to an in-process fake transport or platform client; the
answers are fixed strings, not model output.
"""
import roboflow_workflows.v2.blocks.language._catalogue_openrouter_vendors as _v2_family_0

import base64
import re
import threading
import time
from typing import Any, Dict

import cv2
import numpy as np
import pytest
import torch
from roboflow_workflows.v2.blocks.language._shared import openrouter_vendors as openrouter_vendors
from roboflow_workflows.v2.blocks.language.chat import ChatRequestError
from roboflow_workflows.v2.blocks.language.kimi_openrouter.v3 import KimiOpenRouterBlock
from roboflow_workflows.v2.blocks.language.meta_vlm.v3 import MetaVlmBlock
from roboflow_workflows.v2.blocks.language._shared.openrouter_vendors import VendorModel, encode_capped_jpeg
from roboflow_workflows.v2.blocks.language.zai_vlm.v2 import ZaiVlmBlock
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
)

OPENROUTER_KEY = "sk-or-test-VENDOR-41be"
ROBOFLOW_KEY = "rf-test-ROBOFLOW-93c1"
FAKE_OPENROUTER_URL = "http://openrouter.test/api/v1"


def _image(width: int = 64, height: int = 48) -> ImageData:
    image = ImageData.from_tensor(torch.zeros((3, height, width), dtype=torch.uint8))

    return image


def _answering(content: Any, **body: Any) -> FakeChatServer:
    server = FakeChatServer(
        lambda path, request: (200, chat_completion_body(content, **body))
    )

    return server


def _block(block_class, server: FakeChatServer, **resources: Any):
    block = block_class(
        openrouter_base_url=FAKE_OPENROUTER_URL,
        openrouter_http_client=server.http_client(),
        **resources,
    )

    return block


def _run(block, images, resolved=None, **raw: Any):
    defaults = {"images": "$inputs.images", "api_key": OPENROUTER_KEY}
    params = spec_of(type(block)).validate_params({**defaults, **raw})
    arguments = {name: getattr(params, name) for name in type(params).model_fields}
    arguments.update(images=Batch(images), **(resolved or {}))
    results = block.run(**arguments)

    return results


def _uploaded_size(body: Dict[str, Any]) -> tuple:
    content = body["messages"][-1]["content"]
    url = next(part["image_url"]["url"] for part in content if "image_url" in part)
    data = base64.b64decode(url.split(",", 1)[1])
    height, width = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR).shape[
        :2
    ]

    return width, height


def test_kimi_sends_the_legacy_openrouter_request() -> None:
    server = _answering("A cat.")

    results = _run(_block(KimiOpenRouterBlock, server), [_image()], prompt="What?")

    body = server.requests[0].body
    assert server.requests[0].headers["authorization"] == f"Bearer {OPENROUTER_KEY}"
    assert body["model"] == "moonshotai/kimi-k2.6"
    assert (body["max_tokens"], body["temperature"]) == (500, 0.1)
    assert body["provider"] == {"data_collection": "deny"}
    assert "reasoning" not in body
    assert body["messages"][0]["content"][0] == {"type": "text", "text": "What?"}
    assert set(results[0]) == {
        "output",
        "classes",
        "predictions",
        "error_status",
        "inference_id",
    }
    assert results[0]["output"] == "A cat."


def test_kimi_decodes_normalized_detections_to_pixels() -> None:
    answer = (
        '{"detections": [{"x_min": 0.25, "y_min": 0.5, "x_max": 0.75, "y_max": 1.0,'
        ' "class_name": "car", "confidence": 0.9}]}'
    )
    server = _answering(answer)

    results = _run(
        _block(KimiOpenRouterBlock, server),
        [_image(200, 100)],
        task_type="object-detection",
        classes=["car"],
        model_version="Kimi K2.5 - OpenRouter",
    )

    assert server.requests[0].body["model"] == "moonshotai/kimi-k2.5"
    system = server.requests[0].body["messages"][0]
    assert system["role"] == "system"
    assert "object-detection model" in system["content"]
    detections = results[0]["predictions"]
    assert detections.xyxy.tolist() == [[50.0, 50.0, 150.0, 100.0]]
    assert results[0]["error_status"] is False


def test_meta_sends_one_image_first_user_message_with_low_reasoning() -> None:
    server = _answering("A street.", reasoning="Looked closely.")

    results = _run(_block(MetaVlmBlock, server), [_image()], prompt="Describe.")

    body = server.requests[0].body
    assert body["model"] == "meta/muse-spark-1.3"
    assert body["max_tokens"] == 2048
    assert "temperature" not in body
    assert body["reasoning"] == {"effort": "low"}
    assert [message["role"] for message in body["messages"]] == ["user"]
    content = body["messages"][0]["content"]
    assert content[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert content[1] == {"type": "text", "text": "Describe."}
    assert results[0]["thinking"] == "Looked closely."
    assert (results[0]["input_tokens"], results[0]["output_tokens"]) == (13, 5)


def test_meta_unset_max_tokens_means_2048_and_temperature_is_sent_when_set() -> None:
    server = _answering("ok")

    _run(
        _block(MetaVlmBlock, server),
        [_image()],
        prompt="p",
        max_tokens=None,
        temperature=0.3,
        model_version="Muse Glimmer",
    )

    body = server.requests[0].body
    assert (body["max_tokens"], body["temperature"]) == (2048, 0.3)
    assert body["provider"]["quantizations"] == ["bf16", "fp32"]


def test_meta_decodes_0_1000_named_boxes() -> None:
    server = _answering(
        '[{"label": "dog", "x_min": 0, "y_min": 500, "x_max": 500, "y_max": 1000}]'
    )

    results = _run(
        _block(MetaVlmBlock, server),
        [_image(400, 200)],
        task_type="object-detection",
        classes=["dog"],
    )

    text = server.requests[0].body["messages"][0]["content"][1]["text"]
    assert "0-1000" in text or "1000" in text
    assert results[0]["predictions"].xyxy.tolist() == [[0.0, 100.0, 200.0, 200.0]]


def test_meta_rejects_a_reasoning_level_the_model_lacks() -> None:
    with pytest.raises(Exception, match="Muse Glimmer supports reasoning_effort"):
        spec_of(MetaVlmBlock).validate_params(
            {
                "images": "$inputs.images",
                "prompt": "p",
                "model_version": "Muse Glimmer",
                "reasoning_effort": "minimal",
            }
        )


@pytest.mark.parametrize(
    ("model_version", "reasoning_effort", "reasoning"),
    [
        ("GLM 5V Turbo", "none", {"enabled": False}),
        ("GLM 5V Turbo", "high", {"effort": "high"}),
        ("GLM 5.3 Flash", "none", {"effort": "low"}),
        ("GLM 5.3 Flash", "medium", {"effort": "medium"}),
    ],
)
def test_zai_reasoning_rules(model_version, reasoning_effort, reasoning) -> None:
    server = _answering("ok")

    _run(
        _block(ZaiVlmBlock, server),
        [_image()],
        prompt="p",
        model_version=model_version,
        reasoning_effort=reasoning_effort,
    )

    assert server.requests[0].body["reasoning"] == reasoning


def test_zai_reads_the_shared_required_reasoning_fallback_at_call_time(
    monkeypatch,
) -> None:
    # given: the shared fallback changed after the block module was imported
    monkeypatch.setattr(openrouter_vendors, "ZAI_REQUIRED_REASONING_FALLBACK", "high")
    server = _answering("ok")
    block = _block(ZaiVlmBlock, server)

    # when
    for model_version in ("GLM 5.3 Flash", "GLM 5V Turbo"):
        _run(
            block,
            [_image()],
            prompt="p",
            model_version=model_version,
            reasoning_effort="none",
        )

    # then: only the model that requires reasoning uses the patched value
    reasoning = [request.body["reasoning"] for request in server.requests]
    assert reasoning == [{"effort": "high"}, {"enabled": False}]


def test_zai_flash_uses_its_own_detection_prompt_and_fp8_floor() -> None:
    server = _answering('[{"bbox_2d": [0, 0, 500, 500], "label": "car"}]')

    results = _run(
        _block(ZaiVlmBlock, server),
        [_image(100, 100)],
        task_type="object-detection",
        classes=["car", "bus"],
        model_version="GLM 5.3 Flash",
    )

    body = server.requests[0].body
    assert body["provider"]["quantizations"] == ["fp8", "bf16", "fp32"]
    text = body["messages"][0]["content"][1]["text"]
    assert text.endswith("Only use these labels: car, bus. Return a JSON array only.")
    assert results[0]["predictions"].xyxy.tolist() == [[0.0, 0.0, 50.0, 50.0]]


def test_a_rejected_reasoning_config_is_retried_once_without_it() -> None:
    def reply(path: str, body: Dict[str, Any]):
        if "reasoning" in body:
            return 400, error_body("reasoning is not supported for this model")
        return 200, chat_completion_body("Retried.")

    server = FakeChatServer(reply)

    results = _run(_block(MetaVlmBlock, server), [_image()], prompt="p")

    assert ["reasoning" in request.body for request in server.requests] == [
        True,
        False,
    ]
    assert results[0]["output"] == "Retried."


def test_managed_key_posts_through_the_host_platform_client() -> None:
    server = _answering("unused")
    platform = FakePlatformClient(
        lambda endpoint, payload: chat_completion_body("Proxied.")
    )
    block = _block(ZaiVlmBlock, server, platform_client=platform, api_key=ROBOFLOW_KEY)

    results = _run(block, [_image()], api_key="rf_key:account", prompt="p")

    assert server.requests == []
    post = platform.posts[0]
    assert (post["endpoint"], post["api_key"]) == ("apiproxy/openrouter", ROBOFLOW_KEY)
    assert post["payload"]["model"] == "z-ai/glm-5v-turbo"
    assert post["payload"]["reasoning"] == {"enabled": False}
    assert post["payload"]["openrouter_api_key"] == "rf_key:account"
    assert results[0]["output"] == "Proxied."


def test_managed_key_without_a_platform_client_sends_nothing() -> None:
    server = _answering("unused")

    with pytest.raises(ChatRequestError, match="provided no platform client"):
        _run(
            _block(KimiOpenRouterBlock, server, api_key=ROBOFLOW_KEY),
            [_image()],
            api_key="rf_key:account",
            prompt="p",
        )

    assert server.requests == []


def test_a_replayed_secret_fails_before_any_request() -> None:
    server = _answering("unused")

    with pytest.raises(SecretUnavailableError):
        _run(
            _block(MetaVlmBlock, server),
            [_image()],
            prompt="p",
            resolved={"api_key": Secret.unavailable("openrouter_key")},
        )

    assert server.requests == []


@pytest.mark.parametrize(
    "block_class", [KimiOpenRouterBlock, MetaVlmBlock, ZaiVlmBlock]
)
def test_a_key_resolved_to_none_never_sends_the_environment_key(
    block_class, monkeypatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-ENVIRONMENT-not-for-openrouter")
    server = _answering("unused")

    with pytest.raises(ValueError, match="`api_key` resolved to None"):
        _run(
            _block(block_class, server),
            [_image()],
            prompt="p",
            api_key="$inputs.key",
            resolved={"api_key": None},
        )

    assert server.requests == []


def test_a_failed_request_names_no_key_anywhere_in_the_chain() -> None:
    server = FakeChatServer(
        lambda path, body: (401, error_body(f"invalid key {OPENROUTER_KEY}"))
    )
    secret = Secret(OPENROUTER_KEY, name="openrouter_key")

    with pytest.raises(ChatRequestError) as caught:
        _run(
            _block(ZaiVlmBlock, server),
            [_image()],
            prompt="p",
            resolved={"api_key": secret},
        )

    assert OPENROUTER_KEY not in str(caught.value)
    assert "<secret openrouter_key>" in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ({"choices": [], "error": {"message": "overloaded"}}, "Details: overloaded"),
        (chat_completion_body(None, finish_reason="length"), "increasing `max_tokens`"),
    ],
)
def test_malformed_completions_fail_with_v1_messages(body, message) -> None:
    server = FakeChatServer(lambda path, request: (200, body))

    with pytest.raises(ChatRequestError, match=re.escape(message)):
        _run(_block(KimiOpenRouterBlock, server), [_image()], prompt="p")


def test_a_non_object_proxy_answer_fails_without_the_key() -> None:
    platform = FakePlatformClient(lambda endpoint, payload: ["not", "a", "completion"])
    block = MetaVlmBlock(platform_client=platform, api_key=ROBOFLOW_KEY)

    with pytest.raises(ChatRequestError) as caught:
        _run(block, [_image()], api_key="rf_key:account", prompt="p")

    assert ROBOFLOW_KEY not in str(caught.value)


def test_answers_keep_image_order_under_concurrency() -> None:
    widths = [40, 41, 42, 43, 44]
    lock = threading.Lock()
    arrivals = []

    def reply(path: str, body: Dict[str, Any]):
        width = _uploaded_size(body)[0]
        with lock:
            arrivals.append(width)
        time.sleep((44 - width) * 0.02)
        return 200, chat_completion_body(f"width {width}")

    server = FakeChatServer(reply)

    results = _run(
        _block(MetaVlmBlock, server),
        [_image(width, 30) for width in widths],
        prompt="p",
        max_concurrent_requests=5,
    )

    assert [result["output"] for result in results] == [
        f"width {width}" for width in widths
    ]
    assert sorted(arrivals) == widths


@pytest.mark.parametrize(
    ("block_class", "raw", "resolved", "message"),
    [
        (
            KimiOpenRouterBlock,
            {"prompt": "p"},
            {"model_version": "Kimi K9"},
            "Unknown Kimi model",
        ),
        (
            MetaVlmBlock,
            {"prompt": "p"},
            {"model_version": "Muse X"},
            "Unknown Muse variant",
        ),
        (
            MetaVlmBlock,
            {"prompt": "p"},
            {"model_version": "Muse Glimmer", "reasoning_effort": "minimal"},
            "supports reasoning_effort",
        ),
        (
            ZaiVlmBlock,
            {"task_type": "object-detection", "classes": "$inputs.classes"},
            {"classes": None},
            "`classes` parameter required",
        ),
        (
            KimiOpenRouterBlock,
            {"task_type": "classification", "classes": "$inputs.classes"},
            {"classes": ["car", "car"]},
            "classes repeat",
        ),
    ],
)
def test_resolved_values_are_checked_before_any_request(
    block_class, raw, resolved, message
) -> None:
    server = _answering("unused")

    with pytest.raises(ValueError, match=message):
        _run(_block(block_class, server), [_image()], resolved=resolved, **raw)

    assert server.requests == []


def test_an_oversized_upload_shrinks_by_tenths(monkeypatch) -> None:
    generator = torch.Generator().manual_seed(3)
    pixels = torch.randint(
        0, 256, (3, 120, 200), dtype=torch.uint8, generator=generator
    )
    image = ImageData.from_tensor(pixels)
    full = encode_capped_jpeg(image)
    monkeypatch.setattr(
        openrouter_vendors, "OPENROUTER_MAX_BASE64_BYTES", len(full.base64) // 2
    )

    upload = encode_capped_jpeg(image)

    assert full.size_wh == (200, 120)
    assert upload.size_wh[0] in (180, 162, 145, 130, 117, 105, 94)
    assert len(upload.base64) <= len(full.base64) // 2


def test_a_small_image_uploads_at_full_size() -> None:
    server = _answering("ok")

    _run(_block(ZaiVlmBlock, server), [_image(321, 123)], prompt="p")

    assert _uploaded_size(server.requests[0].body) == (321, 123)


def test_vendor_models_reject_absolute_box_formats() -> None:
    with pytest.raises(ValueError, match="needs the upload size"):
        VendorModel("x/y", box_format="xyxy_absolute")


def test_describe_outputs_follows_the_task() -> None:
    def kind(block_class, task_type, classes=None):
        params = spec_of(block_class).validate_params(
            {
                "images": "$inputs.images",
                "task_type": task_type,
                "classes": classes,
                "prompt": "p",
            }
        )
        outputs = block_class.describe_outputs(params)
        return outputs["predictions"].kinds

    assert kind(MetaVlmBlock, "object-detection", ["a"]) == (
        OBJECT_DETECTION_PREDICTION_KIND,
    )
    assert kind(KimiOpenRouterBlock, "classification", ["a"]) == (
        CLASSIFICATION_PREDICTION_KIND,
    )
    assert {"thinking", "input_tokens"} <= set(ZaiVlmBlock.outputs)
    assert "thinking" not in KimiOpenRouterBlock.outputs


@pytest.mark.parametrize(
    ("block_class", "model_version", "identifier"),
    [
        (KimiOpenRouterBlock, "Kimi K2.6 - OpenRouter", "moonshotai/kimi-k2.6"),
        (MetaVlmBlock, "Muse Glimmer", "meta/muse-glimmer-30b"),
        (ZaiVlmBlock, "$inputs.zai_model", "$inputs.zai_model"),
    ],
)
def test_dependent_resources_name_the_openrouter_slug(
    block_class, model_version, identifier
) -> None:
    params = spec_of(block_class).validate_params(
        {"images": "$inputs.images", "prompt": "p", "model_version": model_version}
    )

    resources = block_class.discover_dependent_resources(params)

    assert [(r.identifier, r.details["provider"]) for r in resources] == [
        (identifier, "openrouter")
    ]


def test_max_concurrent_requests_below_one_is_rejected() -> None:
    with pytest.raises(Exception, match="greater than or equal to 1"):
        spec_of(KimiOpenRouterBlock).validate_params(
            {"images": "$inputs.images", "prompt": "p", "max_concurrent_requests": 0}
        )


@pytest.mark.parametrize(
    "block_class", [KimiOpenRouterBlock, MetaVlmBlock, ZaiVlmBlock]
)
@pytest.mark.parametrize("provide_platform", [False, True])
def test_managed_key_without_host_fails_before_platform_effects(
    block_class, provide_platform: bool
) -> None:
    server = _answering("unused")
    platform = FakePlatformClient(
        lambda endpoint, payload: chat_completion_body("unused")
    )
    resources = {"platform_client": platform} if provide_platform else {}

    with pytest.raises(ValueError, match="Roboflow API key is required") as caught:
        _run(
            _block(block_class, server, **resources),
            [_image()],
            api_key="rf_key:account",
            prompt="p",
        )

    assert server.requests == [] and platform.posts == []
    assert caught.value.__cause__ is None and caught.value.__context__ is None
    assert ROBOFLOW_KEY not in str(caught.value)


def test_only_active_openrouter_vendor_blocks_are_discoverable() -> None:
    assert _v2_family_0.BLOCKS == (KimiOpenRouterBlock, MetaVlmBlock, ZaiVlmBlock)
    assert all("llama" not in block.type for block in _v2_family_0.BLOCKS)

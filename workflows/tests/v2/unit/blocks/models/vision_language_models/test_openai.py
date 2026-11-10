"""OpenAI (Responses API) and OpenAI-compatible (chat completions) blocks.

Every request goes to an in-process fake transport or platform client.
"""

import json
import re
from typing import Any, Dict

import pytest
import torch
from roboflow_workflows.v2.blocks.models.vision_language_models.chat import (
    ChatRequestError,
    encode_image,
)
from roboflow_workflows.v2.blocks.models.vision_language_models.open_ai.v7 import (
    OpenAIBlock,
)
from roboflow_workflows.v2.blocks.models.vision_language_models.openai_compatible.v1 import (
    OpenAICompatibleBlock,
)
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

OPENAI_KEY = "sk-test-OPENAI-5b1e"
ROBOFLOW_KEY = "rf-test-ROBOFLOW-0c3d"
FAKE_OPENAI_URL = "http://openai.test/v1"
FAKE_SERVER_URL = "http://llm.test/v1"


def _image(width: int = 64, height: int = 48) -> ImageData:
    image = ImageData.from_tensor(torch.zeros((3, height, width), dtype=torch.uint8))

    return image


def _arguments(block_class: type, raw: Dict[str, Any], **values: Any) -> Dict[str, Any]:
    params = spec_of(block_class).validate_params(raw)
    arguments = {name: getattr(params, name) for name in type(params).model_fields}
    arguments.update(values)

    return arguments


def _openai(server: FakeChatServer, **resources: Any) -> OpenAIBlock:
    block = OpenAIBlock(
        openai_base_url=FAKE_OPENAI_URL,
        openai_http_client=server.http_client(),
        **resources,
    )

    return block


def _answering(text: str) -> FakeChatServer:
    server = FakeChatServer(lambda path, body: (200, responses_body(text)))

    return server


def _run_openai(block: OpenAIBlock, images, resolved=None, **raw: Any):
    """Validate ``raw`` as written, then run with ``resolved`` selector values."""
    arguments = _arguments(
        OpenAIBlock,
        {"images": "$inputs.images", "api_key": OPENAI_KEY, **raw},
        images=Batch(images),
        **(resolved or {}),
    )
    results = block.run(**arguments)

    return results


def test_direct_request_uses_the_responses_api_with_the_step_options() -> None:
    server = _answering("Two cars.")
    block = _openai(server)

    results = _run_openai(
        block,
        [_image()],
        prompt="How many cars?",
        model_version="gpt-5.4",
        reasoning_effort="low",
        max_tokens=64,
        temperature=0.5,
    )

    request = server.requests[0]
    assert request.url == f"{FAKE_OPENAI_URL}/responses"
    assert request.headers["authorization"] == f"Bearer {OPENAI_KEY}"
    assert {key: request.body[key] for key in request.body if key != "input"} == {
        "model": "gpt-5.4",
        "max_output_tokens": 64,
        "temperature": 0.5,
        "reasoning": {"effort": "low"},
    }
    content = request.body["input"][0]["content"]
    assert content[0] == {"type": "input_text", "text": "How many cars?"}
    assert content[1]["image_url"].startswith("data:image/jpeg;base64,")
    assert content[1]["detail"] == "auto"
    assert results[0]["output"] == "Two cars."
    assert results[0]["input_tokens"] == 11
    assert results[0]["output_tokens"] == 7
    assert results[0]["predictions"] is None
    assert results[0]["error_status"] is False


def test_secret_key_is_revealed_only_for_the_request() -> None:
    server = _answering("ok")

    _run_openai(
        _openai(server),
        [_image()],
        prompt="hi",
        api_key="$inputs.key",
        resolved={"api_key": Secret(OPENAI_KEY, name="OPENAI_KEY")},
    )

    assert server.requests[0].headers["authorization"] == f"Bearer {OPENAI_KEY}"


def test_reasoning_is_not_sent_to_models_without_reasoning() -> None:
    server = _answering("ok")

    _run_openai(_openai(server), [_image()], prompt="hi", model_version="gpt-4o")

    assert "reasoning" not in server.requests[0].body


def test_managed_key_goes_through_the_platform_client_only() -> None:
    server = _answering("unused")
    platform = FakePlatformClient(lambda endpoint, payload: responses_body("Proxied."))
    block = _openai(server, platform_client=platform, api_key=ROBOFLOW_KEY)

    results = _run_openai(
        block, [_image()], prompt="hi", api_key="rf_key:account", max_tokens=32
    )

    assert server.requests == []
    post = platform.posts[0]
    assert post["endpoint"] == "apiproxy/openai/v2"
    assert post["api_key"] == ROBOFLOW_KEY
    assert post["payload"]["openai_api_key"] == "rf_key:account"
    assert post["payload"]["model"] == "gpt-5.1"
    assert post["payload"]["max_output_tokens"] == 32
    assert results[0]["output"] == "Proxied."


def test_managed_key_without_roboflow_key_fails_before_any_request() -> None:
    platform = FakePlatformClient(lambda endpoint, payload: responses_body("unused"))
    block = _openai(_answering("unused"), platform_client=platform)

    with pytest.raises(ValueError, match="Roboflow API key is required"):
        _run_openai(block, [_image()], prompt="hi", api_key="rf_key:user:42")

    assert platform.posts == []


def test_replayed_secret_fails_before_any_request() -> None:
    server = _answering("unused")

    with pytest.raises(SecretUnavailableError):
        _run_openai(
            _openai(server),
            [_image()],
            prompt="hi",
            api_key="$inputs.key",
            resolved={"api_key": Secret.unavailable("OPENAI_KEY")},
        )

    assert server.requests == []


def test_batch_results_follow_image_order_with_concurrent_requests() -> None:
    def reply(path: str, body: Dict[str, Any]):
        url = body["input"][0]["content"][1]["image_url"]
        return 200, responses_body(str(len(url)))

    server = FakeChatServer(reply)
    images = [_image(width) for width in (16, 640, 64, 320)]

    results = _run_openai(
        _openai(server), images, prompt="size?", max_concurrent_requests=3
    )

    expected = [str(len(encode_image(image).data_url)) for image in images]
    assert [result["output"] for result in results] == expected
    assert len({result["inference_id"] for result in results}) == 4


def test_absolute_detection_boxes_map_from_the_downscaled_upload() -> None:
    answer = json.dumps(
        {"detections": [{"label": "car", "box_2d": [100, 50, 300, 250]}]}
    )
    server = _answering(answer)

    results = _run_openai(
        _openai(server),
        [_image(4096, 1024)],
        task_type="object-detection",
        classes=["car", "person"],
        model_version="gpt-5.4",
    )

    request = server.requests[0]
    assert request.body["input"][0]["content"][0]["image_url"].startswith(
        "data:image/png;base64,"
    )
    assert "detail" not in request.body["input"][0]["content"][0]
    assert "2048x512 pixel image" in request.body["input"][0]["content"][1]["text"]
    assert request.body["text"]["format"]["name"] == "detections"
    predictions = results[0]["predictions"]
    assert predictions.xyxy.tolist() == [[200.0, 100.0, 600.0, 500.0]]
    assert predictions.class_id.tolist() == [0]
    assert results[0]["classes"] == ["car", "person"]


def test_normalized_detection_of_the_default_model() -> None:
    answer = json.dumps(
        {
            "detections": [
                {
                    "x_min": 0.25,
                    "y_min": 0.5,
                    "x_max": 0.5,
                    "y_max": 1.0,
                    "class_name": "person",
                    "confidence": 0.8,
                }
            ]
        }
    )
    server = _answering(answer)

    results = _run_openai(
        _openai(server),
        [_image(200, 100)],
        task_type="object-detection",
        classes=["car", "person"],
    )

    assert server.requests[0].body["model"] == "gpt-5.1"
    assert (
        server.requests[0]
        .body["instructions"]
        .startswith("You act as object-detection model.")
    )
    predictions = results[0]["predictions"]
    assert predictions.xyxy.tolist() == [[50.0, 50.0, 100.0, 100.0]]
    assert predictions.class_id.tolist() == [1]
    assert predictions.confidence.tolist() == pytest.approx([0.8])


def test_output_classes_constrain_labels_and_name_predictions() -> None:
    answer = json.dumps({"detections": [{"label": "CAR", "box_2d": [0, 0, 10, 10]}]})
    server = _answering(answer)

    results = _run_openai(
        _openai(server),
        [_image()],
        task_type="object-detection",
        classes=["a red car", "a person"],
        output_classes=["CAR", "PERSON"],
        model_version="gpt-5.4",
    )

    body = server.requests[0].body
    schema = body["text"]["format"]["schema"]
    label = schema["properties"]["detections"]["items"]["properties"]["label"]
    assert label["enum"] == ["CAR", "PERSON"]
    assert '"description": "a red car"' in body["input"][0]["content"][1]["text"]
    assert results[0]["classes"] == ["CAR", "PERSON"]
    assert results[0]["predictions"].class_id.tolist() == [0]


def test_classification_decodes_and_unreadable_answers_report_error_status() -> None:
    server = FakeChatServer(
        lambda path, body: (
            200,
            responses_body('```json\n{"class_name": "dog", "confidence": 0.9}\n```'),
        )
    )

    results = _run_openai(
        _openai(server), [_image()], task_type="classification", classes=["cat", "dog"]
    )

    assert results[0]["error_status"] is False
    assert results[0]["predictions"] is not None

    unreadable = _run_openai(
        _openai(_answering("I cannot tell.")),
        [_image()],
        task_type="classification",
        classes=["cat", "dog"],
    )

    assert unreadable[0]["error_status"] is True
    assert unreadable[0]["predictions"] is None
    assert unreadable[0]["output"] == "I cannot tell."


def test_segmentation_defaults_to_its_own_model_and_structured_output() -> None:
    answer = json.dumps(
        {"segmentations": [{"label": "car", "polygon": [0, 0, 20, 0, 20, 20, 0, 20]}]}
    )
    server = _answering(answer)

    results = _run_openai(
        _openai(server),
        [_image()],
        task_type="instance-segmentation",
        classes=["car"],
    )

    body = server.requests[0].body
    assert body["model"] == "gpt-6-astra"
    assert body["text"]["format"]["name"] == "segmentations"
    assert body["input"][0]["content"][0]["type"] == "input_image"
    assert results[0]["error_status"] is False
    assert len(results[0]["predictions"].xyxy) == 1


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (
            {"status": "failed", "error": {"code": "server_error", "message": "boom"}},
            "OpenAI API request failed: server_error: boom",
        ),
        (
            {
                "status": "incomplete",
                "incomplete_details": {"reason": "max_output_tokens"},
            },
            "max_tokens limit was reached",
        ),
        ({"status": "cancelled"}, "was cancelled"),
        ({"status": "completed", "output": []}, "no text content"),
    ],
)
def test_unsuccessful_responses_raise_with_v1_messages(body, message) -> None:
    platform = FakePlatformClient(lambda endpoint, payload: body)
    block = _openai(
        _answering("unused"), platform_client=platform, api_key=ROBOFLOW_KEY
    )

    with pytest.raises(ChatRequestError, match=message):
        _run_openai(block, [_image()], prompt="hi", api_key="rf_key:account")


def test_http_errors_are_scrubbed_of_the_key() -> None:
    server = FakeChatServer(
        lambda path, body: (
            400,
            error_body(f"Incorrect API key provided: {OPENAI_KEY}"),
        )
    )

    with pytest.raises(ChatRequestError) as caught:
        _run_openai(_openai(server), [_image()], prompt="hi")

    assert OPENAI_KEY not in str(caught.value)
    assert "<secret api_key>" in str(caught.value)
    assert caught.value.__cause__ is None


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ({"task_type": "unconstrained"}, "`prompt` parameter required"),
        ({"task_type": "object-detection"}, "`classes` parameter required"),
        (
            {"task_type": "structured-answering"},
            "`output_structure` parameter required",
        ),
        (
            {"task_type": "classification", "classes": ["a"], "output_classes": ["A"]},
            "only supported for object-detection",
        ),
        (
            {
                "task_type": "object-detection",
                "classes": ["a", "b"],
                "output_classes": ["A"],
            },
            "same length",
        ),
        (
            {"prompt": "x", "model_version": "gpt-4o", "reasoning_effort": "low"},
            "does not support configurable reasoning_effort",
        ),
        ({"prompt": "x", "model_version": "gpt-0"}, "model_version"),
        ({"prompt": "x", "max_concurrent_requests": 0}, "max_concurrent_requests"),
        ({"prompt": "x", "max_tokens": 8}, "max_tokens"),
    ],
)
def test_invalid_configuration_is_rejected_at_validation(raw, message) -> None:
    with pytest.raises(Exception, match=message):
        spec_of(OpenAIBlock).validate_params({"images": "$inputs.images", **raw})


def test_resolved_model_and_reasoning_are_checked_before_sending() -> None:
    server = _answering("unused")
    raw = {
        "images": "$inputs.images",
        "prompt": "hi",
        "api_key": OPENAI_KEY,
        "model_version": "$inputs.model",
        "reasoning_effort": "$inputs.effort",
    }
    arguments = _arguments(
        OpenAIBlock,
        raw,
        images=Batch([_image()]),
        model_version="gpt-5.4",
        reasoning_effort="minimal",
    )

    with pytest.raises(ValueError, match="supports reasoning_effort values"):
        _openai(server).run(**arguments)

    assert server.requests == []


def test_outputs_narrow_predictions_to_the_task_kind() -> None:
    spec = spec_of(OpenAIBlock)

    detection = spec.resolve_outputs(
        spec.validate_params(
            {"images": "$inputs.i", "task_type": "object-detection", "classes": ["a"]}
        )
    )
    classification = spec.resolve_outputs(
        spec.validate_params(
            {"images": "$inputs.i", "task_type": "classification", "classes": ["a"]}
        )
    )

    assert detection["predictions"].kinds == (OBJECT_DETECTION_PREDICTION_KIND,)
    assert classification["predictions"].kinds == (CLASSIFICATION_PREDICTION_KIND,)
    assert set(detection) == {
        "output",
        "classes",
        "input_tokens",
        "output_tokens",
        "predictions",
        "error_status",
        "inference_id",
    }
    assert [kind.name for kind in detection["output"].kinds] == [
        "string",
        "language_model_output",
    ]


def test_dependent_resource_reports_the_default_model() -> None:
    params = spec_of(OpenAIBlock).validate_params(
        {"images": "$inputs.i", "task_type": "instance-segmentation", "classes": ["a"]}
    )

    resources = OpenAIBlock.discover_dependent_resources(params)

    assert resources[0].identifier == "gpt-6-astra"
    assert resources[0].details["provider"] == "openai"


def _compatible(server: FakeChatServer, **resources: Any) -> OpenAICompatibleBlock:
    block = OpenAICompatibleBlock(
        allowed_base_urls=resources.pop("allowed_base_urls", ("*",)),
        openai_compatible_http_client=server.http_client(),
    )

    return block


def _run_compatible(block: OpenAICompatibleBlock, **raw: Any) -> Dict[str, Any]:
    images = raw.pop("resolved", {})
    defaults = {"base_url": FAKE_SERVER_URL, "model_name": "qwen", "prompt": "hi"}
    arguments = _arguments(OpenAICompatibleBlock, {**defaults, **raw})
    arguments["prompt_parameters"] = {**arguments["prompt_parameters"], **images}
    result = block.run(**arguments)

    return result


def test_compatible_sends_chat_completion_with_images_and_extra_body() -> None:
    server = FakeChatServer(lambda path, body: (200, chat_completion_body("Three.")))
    block = _compatible(server)

    result = _run_compatible(
        block,
        base_url=f"{FAKE_SERVER_URL}/",
        system_prompt="Be brief.",
        prompt="Count {{ $parameters.kind }} in {{ $parameters.photo }}",
        prompt_parameters={"kind": "cars"},
        resolved={"photo": _image()},
        temperature=0.0,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        api_key=OPENAI_KEY,
    )

    request = server.requests[0]
    assert request.url == f"{FAKE_SERVER_URL}/chat/completions"
    assert request.headers["authorization"] == f"Bearer {OPENAI_KEY}"
    assert request.body["model"] == "qwen"
    assert request.body["max_tokens"] == 500
    assert request.body["temperature"] == 0.0
    assert request.body["chat_template_kwargs"] == {"enable_thinking": False}
    system, user = request.body["messages"]
    assert system == {"role": "system", "content": "Be brief."}
    assert user["content"][0] == {"type": "text", "text": "Count cars in"}
    assert user["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert result == {"output": "Three.", "error_status": ""}


def test_compatible_applies_parameter_operations_and_sends_no_key_placeholder() -> None:
    server = FakeChatServer(lambda path, body: (200, chat_completion_body("ok")))

    _run_compatible(
        _compatible(server),
        prompt="{{ $parameters.name }}!",
        prompt_parameters={"name": "abc"},
        prompt_parameters_operations={"name": [{"type": "StringToUpperCase"}]},
    )

    request = server.requests[0]
    assert request.headers["authorization"] == "Bearer no-key"
    assert request.body["messages"][0]["content"][0]["text"] == "ABC!"


def test_compatible_operations_must_name_a_parameter() -> None:
    with pytest.raises(Exception, match="prompt_parameters|message_parameters|key"):
        spec_of(OpenAICompatibleBlock).validate_params(
            {
                "base_url": FAKE_SERVER_URL,
                "model_name": "m",
                "prompt": "x",
                "prompt_parameters_operations": {
                    "nope": [{"type": "StringToUpperCase"}]
                },
            }
        )


@pytest.mark.parametrize(
    ("allowed", "base_url", "message"),
    [
        (("http://other.test/v1",), FAKE_SERVER_URL, "not approved by the server"),
        ((), FAKE_SERVER_URL, "not approved by the server"),
        (("*",), "http://llm.test/v1?x=1", "without credentials, query strings"),
        (("*",), "ftp://llm.test/v1", "HTTP\\(S\\) URL"),
    ],
)
def test_compatible_refuses_unapproved_urls_without_sending(
    allowed, base_url, message
) -> None:
    server = FakeChatServer(lambda path, body: (200, chat_completion_body("unused")))

    result = _run_compatible(
        _compatible(server, allowed_base_urls=allowed), base_url=base_url
    )

    assert server.requests == []
    assert result["output"] == ""
    assert re.search(message, result["error_status"])


def test_compatible_allowlist_matches_ignoring_trailing_slash() -> None:
    server = FakeChatServer(lambda path, body: (200, chat_completion_body("ok")))
    block = _compatible(server, allowed_base_urls=(f"{FAKE_SERVER_URL}/",))

    result = _run_compatible(block)

    assert result == {"output": "ok", "error_status": ""}


@pytest.mark.parametrize(
    ("status", "body", "message"),
    [
        (400, error_body(f"bad key {OPENAI_KEY}"), "bad key <secret api_key>"),
        (200, {**chat_completion_body("x"), "choices": []}, "API returned no choices"),
        (
            200,
            chat_completion_body(None, finish_reason="length"),
            "finish_reason=length",
        ),
    ],
)
def test_compatible_failures_are_reported_in_error_status(
    status, body, message
) -> None:
    server = FakeChatServer(lambda path, request: (status, body))

    result = _run_compatible(_compatible(server), api_key=OPENAI_KEY)

    assert result["output"] == ""
    assert message in result["error_status"]
    assert OPENAI_KEY not in result["error_status"]


@pytest.mark.parametrize("missing", [None])
def test_missing_openai_key_never_falls_back_to_environment(monkeypatch, missing):
    invented = "invented-environment-key-do-not-send"
    monkeypatch.setenv("OPENAI_API_KEY", invented)
    server = _answering("unused")
    platform = FakePlatformClient(lambda endpoint, payload: responses_body("unused"))
    with pytest.raises(ValueError, match="OpenAI API key is required") as caught:
        _run_openai(
            _openai(server, platform_client=platform),
            [_image()],
            prompt="hi",
            resolved={"api_key": missing},
        )
    assert server.requests == []
    assert platform.posts == []
    assert invented not in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


@pytest.mark.parametrize("key", [None, "", "explicit-compatible-key"])
def test_compatible_anonymous_or_explicit_key_never_uses_environment(monkeypatch, key):
    monkeypatch.setenv("OPENAI_API_KEY", "invented-environment-key-do-not-send")
    server = FakeChatServer(lambda path, body: (200, chat_completion_body("ok")))
    arguments = _arguments(
        OpenAICompatibleBlock,
        {"base_url": FAKE_SERVER_URL, "model_name": "local", "prompt": "hi"},
        api_key=key,
    )
    result = _compatible(server).run(**arguments)
    assert result == {"output": "ok", "error_status": ""}
    assert server.requests[0].headers["authorization"] == f"Bearer {key or 'no-key'}"


def test_chat_completion_compatibility_exports_are_the_shared_helpers():
    from roboflow_workflows.v2.blocks.models.vision_language_models import (
        chat_completions,
    )
    from roboflow_workflows.v2.blocks.models.vision_language_models._shared import (
        openai as openai,
    )

    assert openai.chat_messages is chat_completions.chat_messages
    assert openai.chat_completion_usage is chat_completions.chat_completion_usage

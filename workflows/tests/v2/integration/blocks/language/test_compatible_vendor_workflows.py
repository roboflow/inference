"""Gemma, Qwen 3.5/3.6 and SpaceXAI blocks in compiled V2 workflows.

Nothing leaves the process: OpenRouter and xAI are ``httpx.MockTransport``
fakes injected as session resources, and the Roboflow platform client is a
recording fake. Answers are fixed strings. This checks wiring and
composition, not any real model::

    image ─▶ v2/google_gemma@v4 (detection, secret key) ─▶ v2/detection_offset@v1
    image ─▶ v2/spacexai@v3 (detection, managed key) ─▶ v2/detection_offset@v1
    image ─▶ v2/qwen3_6_openrouter@v1 (structured) ─▶ v2/json_parser@v1
    image ─▶ v2/qwen3_5_openrouter@v1 (classification) ─▶ v2/vlm_as_classifier@v2
"""

import json
from typing import Any, Dict, List

import pytest
import torch
from roboflow_workflows.v2.blocks.formatters.json import JsonParserBlock
from roboflow_workflows.v2.blocks.geometry.offset import DetectionOffsetBlock
from roboflow_workflows.v2.blocks.language.compatible_vendors import BLOCKS
from roboflow_workflows.v2.blocks.language.parsers import VLMAsClassifierBlock
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.language import LANGUAGE_MODEL_OUTPUT_KIND
from roboflow_workflows.v2.data.secrets import SECRET_KIND
from roboflow_workflows.v2.engine.catalogue import Catalogue
from roboflow_workflows.v2.engine.compilation import compile_workflow
from roboflow_workflows.v2.engine.introspection import discover_workload

from tests.v2.support.compilation.fixture_blocks import Gate
from tests.v2.support.language.fake_chat import (
    FakeChatServer,
    FakePlatformClient,
    chat_completion_body,
    error_body,
    responses_body,
)

OPENROUTER_KEY = "sk-or-test-OPENROUTER-71c2"
XAI_KEY = "xai-test-XAI-90de"
ROBOFLOW_KEY = "rf-test-ROBOFLOW-4f5a"
CATALOGUE = Catalogue(
    [*BLOCKS, DetectionOffsetBlock, JsonParserBlock, VLMAsClassifierBlock, Gate],
    kinds=[SECRET_KIND, LANGUAGE_MODEL_OUTPUT_KIND],
)
IMAGE_INPUT = {"type": "WorkflowBatchInput", "name": "image", "kind": ["image"]}
KEY_INPUT = {"type": "WorkflowParameter", "name": "key", "kind": ["secret"]}


def _image(width: int, height: int) -> ImageData:
    image = ImageData.from_tensor(torch.zeros((3, height, width), dtype=torch.uint8))

    return image


def _output(name: str, selector: str) -> Dict[str, str]:
    return {"type": "JsonField", "name": name, "selector": selector}


def _compile(steps: List[Dict[str, Any]], outputs: List[Dict[str, str]], inputs=None):
    definition = {
        "version": "2.0",
        "inputs": inputs or [IMAGE_INPUT, KEY_INPUT],
        "steps": steps,
        "outputs": outputs,
    }
    plan = compile_workflow(definition, catalogue=CATALOGUE)

    return plan


def _chain_text(error: BaseException) -> str:
    texts = []
    while error is not None:
        texts.append(f"{type(error).__name__}: {error}")
        error = error.__cause__ or error.__context__

    return "\n".join(texts)


def _offset(predictions: str) -> Dict[str, Any]:
    step = {
        "type": "v2/detection_offset@v1",
        "name": "offset",
        "predictions": predictions,
        "offset_width": 10,
        "offset_height": 10,
    }

    return step


def test_gemma_detections_from_a_secret_key_feed_a_geometry_step() -> None:
    plan = _compile(
        [
            {
                "type": "v2/google_gemma@v4",
                "name": "gemma",
                "images": "$inputs.image",
                "task_type": "object-detection",
                "classes": ["car"],
                "api_key": "$inputs.key",
            },
            _offset("$steps.gemma.predictions"),
        ],
        [
            _output("answer", "$steps.gemma.output"),
            _output("tokens", "$steps.gemma.output_tokens"),
            _output("boxes", "$steps.offset.predictions"),
        ],
    )
    answer = json.dumps(
        {
            "detections": [
                {
                    "x_min": 0.25,
                    "y_min": 0.25,
                    "x_max": 0.5,
                    "y_max": 0.5,
                    "class_name": "car",
                    "confidence": 0.9,
                }
            ]
        }
    )
    server = FakeChatServer(lambda path, body: (200, chat_completion_body(answer)))
    session = plan.create_session(
        resources={"openrouter_http_client": server.http_client()}
    )

    result = session.run(
        {"image": [_image(400, 200), _image(100, 40)], "key": OPENROUTER_KEY}
    )
    rows = result.rows()
    serialized = json.dumps(result.rows(serialize=True))

    assert [request.body["model"] for request in server.requests] == [
        "google/gemma-4-31b-it"
    ] * 2
    assert OPENROUTER_KEY not in serialized
    # Offset grows each box by 5 px per side.
    assert rows[0]["boxes"].xyxy.tolist() == [[95.0, 45.0, 205.0, 105.0]]
    assert rows[1]["boxes"].xyxy.tolist() == [[20.0, 5.0, 55.0, 25.0]]
    assert rows[0]["tokens"] == 5


def test_spacexai_managed_key_uses_the_host_platform_client() -> None:
    plan = _compile(
        [
            {
                "type": "v2/spacexai@v3",
                "name": "grok",
                "images": "$inputs.image",
                "task_type": "object-detection",
                "classes": ["car"],
                "model_version": "grok-4.7",
            },
            _offset("$steps.grok.predictions"),
        ],
        [_output("boxes", "$steps.offset.predictions")],
        inputs=[IMAGE_INPUT],
    )
    answer = json.dumps([{"label": "car", "box_2d": [10, 20, 30, 40]}])
    platform = FakePlatformClient(lambda endpoint, payload: responses_body(answer))
    server = FakeChatServer(lambda path, body: (200, responses_body("unused")))
    session = plan.create_session(
        resources={
            "platform_client": platform,
            "api_key": ROBOFLOW_KEY,
            "xai_http_client": server.http_client(),
        }
    )

    rows = session.run({"image": [_image(200, 100)]}).rows()

    assert server.requests == []
    assert [(post["endpoint"], post["api_key"]) for post in platform.posts] == [
        ("apiproxy/xai", ROBOFLOW_KEY)
    ]
    assert platform.posts[0]["payload"]["model"] == "grok-4.7"
    assert rows[0]["boxes"].xyxy.tolist() == [[15.0, 15.0, 65.0, 45.0]]


def test_qwen_structured_answer_feeds_the_json_parser() -> None:
    plan = _compile(
        [
            {
                "type": "v2/qwen3_6_openrouter@v1",
                "name": "qwen",
                "images": "$inputs.image",
                "task_type": "structured-answering",
                "output_structure": {"count": "number of cars"},
                "api_key": "$inputs.key",
            },
            {
                "type": "v2/json_parser@v1",
                "name": "parse",
                "raw_json": "$steps.qwen.output",
                "expected_fields": ["count"],
            },
        ],
        [_output("count", "$steps.parse.count")],
    )
    server = FakeChatServer(
        lambda path, body: (200, chat_completion_body('```json\n{"count": 3}\n```'))
    )
    session = plan.create_session(
        resources={"openrouter_http_client": server.http_client()}
    )

    rows = session.run({"image": [_image(32, 32)], "key": OPENROUTER_KEY}).rows()

    assert rows == [{"count": 3}]
    assert set(server.requests[0].body) == {
        "model",
        "messages",
        "max_tokens",
        "temperature",
    }


def test_qwen_classification_answer_feeds_the_vlm_classifier() -> None:
    plan = _compile(
        [
            {
                "type": "v2/qwen3_5_openrouter@v1",
                "name": "qwen",
                "images": "$inputs.image",
                "task_type": "classification",
                "classes": ["cat", "dog"],
                "api_key": "$inputs.key",
            },
            {
                "type": "v2/vlm_as_classifier@v2",
                "name": "classify",
                "image": "$inputs.image",
                "vlm_output": "$steps.qwen.output",
                "classes": "$steps.qwen.classes",
            },
        ],
        [
            _output("prediction", "$steps.classify.predictions"),
            _output("error", "$steps.classify.error_status"),
        ],
    )
    server = FakeChatServer(
        lambda path, body: (
            200,
            chat_completion_body('{"class_name": "dog", "confidence": 0.8}'),
        )
    )
    session = plan.create_session(
        resources={"openrouter_http_client": server.http_client()}
    )

    rows = session.run({"image": [_image(16, 16)], "key": OPENROUTER_KEY}).rows()

    assert rows[0]["error"] is False
    assert rows[0]["prediction"].class_id.tolist() == [1]


def test_a_gated_image_sends_no_request() -> None:
    plan = _compile(
        [
            {
                "type": Gate.type,
                "name": "keep",
                "value": "$inputs.keep",
                "next_steps": ["$steps.grok"],
            },
            {
                "type": "v2/spacexai@v3",
                "name": "grok",
                "images": "$inputs.image",
                "task_type": "caption",
                "api_key": "$inputs.key",
            },
        ],
        [_output("caption", "$steps.grok.output")],
        inputs=[IMAGE_INPUT, KEY_INPUT, {"type": "WorkflowBatchInput", "name": "keep"}],
    )
    server = FakeChatServer(lambda path, body: (200, responses_body("A box.")))
    session = plan.create_session(resources={"xai_http_client": server.http_client()})

    rows = session.run(
        {
            "image": [_image(8, 8), _image(9, 9), _image(10, 10)],
            "keep": [False, True, False],
            "key": XAI_KEY,
        }
    ).rows()

    assert len(server.requests) == 1
    assert [row["caption"] for row in rows] == [None, "A box.", None]


def test_a_failed_request_fails_the_step_without_the_key() -> None:
    plan = _compile(
        [
            {
                "type": "v2/spacexai@v3",
                "name": "grok",
                "images": "$inputs.image",
                "task_type": "ocr",
                "api_key": "$inputs.key",
            }
        ],
        [_output("text", "$steps.grok.output")],
    )
    server = FakeChatServer(lambda path, body: (401, error_body(f"bad key {XAI_KEY}")))
    session = plan.create_session(resources={"xai_http_client": server.http_client()})

    with pytest.raises(Exception) as caught:
        session.run({"image": [_image(8, 8)], "key": XAI_KEY})

    chain = _chain_text(caught.value)
    assert "bad key <secret input>" in chain
    assert XAI_KEY not in chain


@pytest.mark.parametrize(
    "step, message",
    [
        (
            {"type": "v2/qwen3_5_openrouter@v1", "prompt": "Hi"},
            "api_key",
        ),
        (
            {
                "type": "v2/spacexai@v3",
                "task_type": "ocr",
                "model_version": "grok-4.5",
                "reasoning_effort": "xhigh",
            },
            "grok-4.5 supports reasoning_effort",
        ),
        (
            {"type": "v2/google_gemma@v4", "task_type": "object-detection"},
            "`classes` parameter required",
        ),
    ],
)
def test_invalid_steps_fail_at_compilation(step: Dict[str, Any], message: str) -> None:
    with pytest.raises(Exception, match=message):
        _compile(
            [{"name": "vlm", "images": "$inputs.image", **step}],
            [_output("answer", "$steps.vlm.output")],
            inputs=[IMAGE_INPUT],
        )


def test_workload_introspection_reports_the_remote_models() -> None:
    plan = _compile(
        [
            {
                "type": "v2/google_gemma@v4",
                "name": "gemma",
                "images": "$inputs.image",
                "prompt": "Describe.",
            },
            {
                "type": "v2/qwen3_6_openrouter@v1",
                "name": "qwen",
                "images": "$inputs.image",
                "prompt": "Describe.",
                "api_key": "$inputs.key",
                "model_version": "$inputs.model",
            },
            {
                "type": "v2/spacexai@v3",
                "name": "grok",
                "images": "$inputs.image",
                "task_type": "ocr",
            },
        ],
        [
            _output("a", "$steps.gemma.output"),
            _output("b", "$steps.qwen.output"),
            _output("c", "$steps.grok.output"),
        ],
        inputs=[IMAGE_INPUT, KEY_INPUT, {"type": "WorkflowParameter", "name": "model"}],
    )

    workload = discover_workload(plan)

    text = json.dumps(workload, default=str)
    assert "google/gemma-4-31b-it" in text
    assert "$inputs.model" in text
    assert "grok-4.6" in text

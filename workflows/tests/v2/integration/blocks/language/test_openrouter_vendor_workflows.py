"""OpenRouter vendor blocks in compiled V2 workflows, against in-process fakes.

OpenRouter is an ``httpx.MockTransport`` fake injected as the
``openrouter_http_client`` session resource; the Roboflow platform client is
a recording fake. The answers are fixed strings. This checks wiring and
composition, not any real model::

    image ─▶ v2/meta_vlm@v3 (detection, secret key) ─▶ v2/detection_offset@v1
    image ─▶ v2/zai_vlm@v2 (structured) ─▶ v2/json_parser@v1
    image ─▶ v2/kimi_openrouter@v3 (native classification)
    image ─▶ v2/kimi_openrouter@v3 behind a gate: gated images send nothing
"""

import json
from typing import Any, Dict, List

import pytest
import torch
from roboflow_workflows.v2.blocks.formatters.json import JsonParserBlock
from roboflow_workflows.v2.blocks.geometry.offset import DetectionOffsetBlock
from roboflow_workflows.v2.blocks.language.openrouter_vendors import (
    KimiOpenRouterBlock,
    MetaVlmBlock,
    ZaiVlmBlock,
)
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
)

OPENROUTER_KEY = "sk-or-test-WORKFLOW-5e7d"
ROBOFLOW_KEY = "rf-test-ROBOFLOW-c0de"
CATALOGUE = Catalogue(
    [
        KimiOpenRouterBlock,
        MetaVlmBlock,
        ZaiVlmBlock,
        DetectionOffsetBlock,
        JsonParserBlock,
        Gate,
    ],
    kinds=[SECRET_KIND, LANGUAGE_MODEL_OUTPUT_KIND],
)
IMAGE_INPUT = {"type": "WorkflowBatchInput", "name": "image", "kind": ["image"]}
KEY_INPUT = {"type": "WorkflowParameter", "name": "openrouter_key", "kind": ["secret"]}


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


def _answering(content: str, **body: Any) -> FakeChatServer:
    server = FakeChatServer(
        lambda path, request: (200, chat_completion_body(content, **body))
    )

    return server


def _chain_text(error: BaseException) -> str:
    texts = []
    while error is not None:
        texts.append(f"{type(error).__name__}: {error}")
        error = error.__cause__ or error.__context__

    return "\n".join(texts)


def test_meta_detections_from_a_secret_keyed_request_feed_a_geometry_step() -> None:
    plan = _compile(
        [
            {
                "type": "v2/meta_vlm@v3",
                "name": "muse",
                "images": "$inputs.image",
                "task_type": "object-detection",
                "classes": ["car"],
                "api_key": "$inputs.openrouter_key",
            },
            {
                "type": "v2/detection_offset@v1",
                "name": "offset",
                "predictions": "$steps.muse.predictions",
                "offset_width": 10,
                "offset_height": 10,
            },
        ],
        [
            _output("answer", "$steps.muse.output"),
            _output("thinking", "$steps.muse.thinking"),
            _output("boxes", "$steps.offset.predictions"),
        ],
    )
    server = _answering(
        '[{"label": "car", "x_min": 0, "y_min": 0, "x_max": 500, "y_max": 500}]',
        reasoning="Found one car.",
    )
    session = plan.create_session(
        resources={"openrouter_http_client": server.http_client()}
    )

    result = session.run(
        {
            "image": [_image(4000, 2000), _image(200, 100)],
            "openrouter_key": OPENROUTER_KEY,
        }
    )
    rows = result.rows()
    serialized = json.dumps(result.rows(serialize=True))

    assert [request.headers["authorization"] for request in server.requests] == [
        f"Bearer {OPENROUTER_KEY}"
    ] * 2
    assert {request.body["model"] for request in server.requests} == {
        "meta/muse-spark-1.3"
    }
    assert OPENROUTER_KEY not in serialized
    # The offset step grows each box by 5 px per side, clipped to the image.
    assert rows[0]["boxes"].xyxy.tolist() == [[0.0, 0.0, 2005.0, 1005.0]]
    assert rows[1]["boxes"].xyxy.tolist() == [[0.0, 0.0, 105.0, 55.0]]
    assert rows[0]["thinking"] == "Found one car."


def test_zai_structured_answer_feeds_the_json_parser() -> None:
    plan = _compile(
        [
            {
                "type": "v2/zai_vlm@v2",
                "name": "glm",
                "images": "$inputs.image",
                "task_type": "structured-answering",
                "output_structure": {"count": "number of cars"},
                "api_key": "sk-or-literal",
            },
            {
                "type": "v2/json_parser@v1",
                "name": "parse",
                "raw_json": "$steps.glm.output",
                "expected_fields": ["count"],
            },
        ],
        [
            _output("count", "$steps.parse.count"),
            _output("tokens", "$steps.glm.output_tokens"),
        ],
        inputs=[IMAGE_INPUT],
    )
    server = _answering('```json\n{"count": 3}\n```')
    session = plan.create_session(
        resources={"openrouter_http_client": server.http_client()}
    )

    rows = session.run({"image": [_image(32, 32)]}).rows()

    assert rows == [{"count": 3, "tokens": 5}]
    assert server.requests[0].body["reasoning"] == {"enabled": False}


def test_kimi_classification_returns_a_native_prediction() -> None:
    plan = _compile(
        [
            {
                "type": "v2/kimi_openrouter@v3",
                "name": "classify",
                "images": "$inputs.image",
                "task_type": "classification",
                "classes": ["cat", "dog"],
                "api_key": "sk-or-literal",
            }
        ],
        [
            _output("prediction", "$steps.classify.predictions"),
            _output("error", "$steps.classify.error_status"),
        ],
        inputs=[IMAGE_INPUT],
    )
    server = _answering('{"class_name": "dog", "confidence": 0.8}')
    session = plan.create_session(
        resources={"openrouter_http_client": server.http_client()}
    )
    try:
        rows = session.run({"image": [_image(16, 16)]}).rows()
    finally:
        session.close()

    assert rows[0]["error"] is False
    prediction = rows[0]["prediction"]
    assert prediction.class_id.tolist() == [1]
    assert float(prediction.confidence[0, 1]) == pytest.approx(0.8)
    assert server.requests[0].body["model"] == "moonshotai/kimi-k2.6"


def test_a_gated_image_sends_no_request() -> None:
    plan = _compile(
        [
            {
                "type": Gate.type,
                "name": "keep",
                "value": "$inputs.keep",
                "next_steps": ["$steps.kimi"],
            },
            {
                "type": "v2/kimi_openrouter@v3",
                "name": "kimi",
                "images": "$inputs.image",
                "prompt": "Describe.",
                "api_key": "$inputs.openrouter_key",
            },
        ],
        [_output("answer", "$steps.kimi.output")],
        inputs=[IMAGE_INPUT, KEY_INPUT, {"type": "WorkflowBatchInput", "name": "keep"}],
    )
    server = _answering("An image.")
    session = plan.create_session(
        resources={"openrouter_http_client": server.http_client()}
    )

    rows = session.run(
        {
            "image": [_image(8, 8), _image(9, 9), _image(10, 10)],
            "keep": [True, False, True],
            "openrouter_key": OPENROUTER_KEY,
        }
    ).rows()

    assert len(server.requests) == 2
    assert [row["answer"] for row in rows] == ["An image.", None, "An image."]


def test_managed_key_uses_the_host_platform_client_and_roboflow_key() -> None:
    plan = _compile(
        [
            {
                "type": "v2/zai_vlm@v2",
                "name": "glm",
                "images": "$inputs.image",
                "task_type": "ocr",
                "model_version": "GLM 5.3 Flash",
            }
        ],
        [_output("answer", "$steps.glm.output")],
        inputs=[IMAGE_INPUT],
    )
    platform = FakePlatformClient(
        lambda endpoint, payload: chat_completion_body("STOP")
    )
    server = _answering("unused")
    session = plan.create_session(
        resources={
            "platform_client": platform,
            "api_key": ROBOFLOW_KEY,
            "openrouter_http_client": server.http_client(),
        }
    )

    rows = session.run({"image": [_image(16, 16)]}).rows()

    assert rows[0]["answer"] == "STOP"
    assert server.requests == []
    post = platform.posts[0]
    assert (post["endpoint"], post["api_key"]) == ("apiproxy/openrouter", ROBOFLOW_KEY)
    assert post["payload"]["reasoning"] == {"effort": "low"}
    assert post["payload"]["quantizations"] == ["fp8", "bf16", "fp32"]


def test_a_failed_request_fails_the_step_without_the_key() -> None:
    plan = _compile(
        [
            {
                "type": "v2/meta_vlm@v3",
                "name": "muse",
                "images": "$inputs.image",
                "prompt": "Describe.",
                "api_key": "$inputs.openrouter_key",
            }
        ],
        [_output("answer", "$steps.muse.output")],
    )
    server = FakeChatServer(
        lambda path, body: (401, error_body(f"bad key {OPENROUTER_KEY}"))
    )
    session = plan.create_session(
        resources={"openrouter_http_client": server.http_client()}
    )

    with pytest.raises(Exception) as caught:
        session.run({"image": [_image(8, 8)], "openrouter_key": OPENROUTER_KEY})

    chain = _chain_text(caught.value)
    assert "bad key <secret input>" in chain
    assert OPENROUTER_KEY not in chain


def test_an_unknown_model_label_from_an_input_sends_nothing() -> None:
    plan = _compile(
        [
            {
                "type": "v2/kimi_openrouter@v3",
                "name": "kimi",
                "images": "$inputs.image",
                "prompt": "Describe.",
                "model_version": "$inputs.model",
                "api_key": "sk-or-literal",
            }
        ],
        [_output("answer", "$steps.kimi.output")],
        inputs=[IMAGE_INPUT, {"type": "WorkflowParameter", "name": "model"}],
    )
    server = _answering("unused")
    session = plan.create_session(
        resources={"openrouter_http_client": server.http_client()}
    )

    with pytest.raises(Exception) as caught:
        session.run({"image": [_image(8, 8)], "model": "Kimi K9"})

    assert "Unknown Kimi model 'Kimi K9'" in _chain_text(caught.value)
    assert server.requests == []


def test_workload_introspection_reports_the_openrouter_models() -> None:
    plan = _compile(
        [
            {
                "type": "v2/meta_vlm@v3",
                "name": "muse",
                "images": "$inputs.image",
                "model_version": "Muse Glimmer",
                "prompt": "Describe.",
            },
            {
                "type": "v2/zai_vlm@v2",
                "name": "glm",
                "images": "$inputs.image",
                "model_version": "$inputs.model",
                "prompt": "Describe.",
            },
        ],
        [_output("a", "$steps.muse.output"), _output("b", "$steps.glm.output")],
        inputs=[IMAGE_INPUT, {"type": "WorkflowParameter", "name": "model"}],
    )

    workload = discover_workload(plan)

    text = json.dumps(workload, default=str)
    assert "meta/muse-glimmer-30b" in text
    assert "$inputs.model" in text

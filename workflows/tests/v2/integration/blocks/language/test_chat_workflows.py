"""Remote chat blocks in compiled V2 workflows, against in-process fakes.

Nothing leaves the process: OpenAI, OpenRouter and the OpenAI-compatible
server are ``httpx.MockTransport`` fakes injected as session resources, and
the Roboflow platform client is a recording fake. This checks wiring and
composition, not any real model::

    image ─▶ roboflow_core/open_ai@v7 (detection) ─▶ roboflow_core/detection_offset@v1 ─▶ outputs
    image ─▶ roboflow_core/openrouter@v3 (structured) ─▶ roboflow_core/json_parser@v1 ─▶ outputs
    image ─▶ roboflow_core/openai_compatible@v1 (image parameter) ─▶ outputs
"""

import json
from typing import Any, Dict, List

import pytest
import torch
from roboflow_workflows.v2.blocks.formatters.json_parser.v1 import JsonParserBlock
from roboflow_workflows.v2.blocks.geometry.detection_offset.v1 import DetectionOffsetBlock
from roboflow_workflows.v2.blocks.language.open_ai.v7 import OpenAIBlock
from roboflow_workflows.v2.blocks.language.openai_compatible.v1 import OpenAICompatibleBlock
from roboflow_workflows.v2.blocks.language.openrouter.v3 import OpenRouterBlock
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.language import LANGUAGE_MODEL_OUTPUT_KIND
from roboflow_workflows.v2.data.secrets import SECRET_KIND
from roboflow_workflows.v2.engine.catalogue import Catalogue
from roboflow_workflows.v2.engine.compilation import compile_workflow
from roboflow_workflows.v2.engine.errors import StepExecutionError
from roboflow_workflows.v2.engine.introspection import discover_workload

from tests.v2.support.compilation.fixture_blocks import Gate
from tests.v2.support.language.fake_chat import (
    FakeChatServer,
    FakePlatformClient,
    chat_completion_body,
    error_body,
    responses_body,
)

OPENAI_KEY = "sk-test-OPENAI-9d0f"
ROBOFLOW_KEY = "rf-test-ROBOFLOW-1a2b"
CATALOGUE = Catalogue(
    [
        OpenAIBlock,
        OpenAICompatibleBlock,
        OpenRouterBlock,
        DetectionOffsetBlock,
        JsonParserBlock,
        Gate,
    ],
    kinds=[SECRET_KIND, LANGUAGE_MODEL_OUTPUT_KIND],
)
IMAGE_INPUT = {"type": "WorkflowBatchInput", "name": "image", "kind": ["image"]}
KEY_INPUT = {"type": "WorkflowParameter", "name": "openai_key", "kind": ["secret"]}


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


def _detection_server() -> FakeChatServer:
    def reply(path: str, body: Dict[str, Any]):
        text = body["input"][0]["content"][1]["text"]
        width, height = text.split(" pixel image")[0].rsplit(" ", 1)[1].split("x")
        box = [0, 0, int(width) // 2, int(height) // 2]
        answer = json.dumps({"detections": [{"label": "car", "box_2d": box}]})
        return 200, responses_body(answer)

    server = FakeChatServer(reply)

    return server


def _chain_text(error: BaseException) -> str:
    texts = []
    while error is not None:
        texts.append(f"{type(error).__name__}: {error}")
        error = error.__cause__ or error.__context__

    return "\n".join(texts)


def test_detections_from_a_secret_keyed_request_feed_a_geometry_step() -> None:
    plan = _compile(
        [
            {
                "type": "roboflow_core/open_ai@v7",
                "name": "gpt",
                "images": "$inputs.image",
                "task_type": "object-detection",
                "classes": ["car"],
                "model_version": "gpt-5.4",
                "api_key": "$inputs.openai_key",
            },
            {
                "type": "roboflow_core/detection_offset@v1",
                "name": "offset",
                "predictions": "$steps.gpt.predictions",
                "offset_width": 10,
                "offset_height": 10,
            },
        ],
        [
            _output("answer", "$steps.gpt.output"),
            _output("tokens", "$steps.gpt.input_tokens"),
            _output("boxes", "$steps.offset.predictions"),
        ],
    )
    server = _detection_server()
    session = plan.create_session(
        resources={"openai_http_client": server.http_client()}
    )

    result = session.run(
        {"image": [_image(4096, 2048), _image(200, 100)], "openai_key": OPENAI_KEY}
    )
    rows = result.rows()
    serialized = json.dumps(result.rows(serialize=True))

    assert [request.headers["authorization"] for request in server.requests] == [
        f"Bearer {OPENAI_KEY}"
    ] * 2
    assert OPENAI_KEY not in serialized
    # The offset step grows each box by 5 px per side, clipped to the image.
    assert rows[0]["boxes"].xyxy.tolist() == [[0.0, 0.0, 2053.0, 1029.0]]
    assert rows[1]["boxes"].xyxy.tolist() == [[0.0, 0.0, 105.0, 55.0]]
    assert rows[0]["tokens"] == 11


def test_a_gated_image_sends_no_request() -> None:
    plan = _compile(
        [
            {
                "type": Gate.type,
                "name": "keep",
                "value": "$inputs.keep",
                "next_steps": ["$steps.gpt"],
            },
            {
                "type": "roboflow_core/open_ai@v7",
                "name": "gpt",
                "images": "$inputs.image",
                "prompt": "Describe.",
                "api_key": "$inputs.openai_key",
            },
        ],
        [_output("answer", "$steps.gpt.output")],
        inputs=[IMAGE_INPUT, KEY_INPUT, {"type": "WorkflowBatchInput", "name": "keep"}],
    )
    server = FakeChatServer(lambda path, body: (200, responses_body("An image.")))
    session = plan.create_session(
        resources={"openai_http_client": server.http_client()}
    )

    rows = session.run(
        {
            "image": [_image(8, 8), _image(9, 9), _image(10, 10)],
            "keep": [True, False, True],
            "openai_key": OPENAI_KEY,
        }
    ).rows()

    assert len(server.requests) == 2
    assert [row["answer"] for row in rows] == ["An image.", None, "An image."]


def test_managed_key_uses_the_host_platform_client_and_roboflow_key() -> None:
    plan = _compile(
        [
            {
                "type": "roboflow_core/open_ai@v7",
                "name": "gpt",
                "images": "$inputs.image",
                "task_type": "ocr",
            }
        ],
        [_output("answer", "$steps.gpt.output")],
        inputs=[IMAGE_INPUT],
    )
    platform = FakePlatformClient(lambda endpoint, payload: responses_body("STOP"))
    server = FakeChatServer(lambda path, body: (200, responses_body("unused")))
    session = plan.create_session(
        resources={
            "platform_client": platform,
            "api_key": ROBOFLOW_KEY,
            "openai_http_client": server.http_client(),
        }
    )

    rows = session.run({"image": [_image(16, 16)]}).rows()

    assert rows[0]["answer"] == "STOP"
    assert server.requests == []
    assert [(post["endpoint"], post["api_key"]) for post in platform.posts] == [
        ("apiproxy/openai/v2", ROBOFLOW_KEY)
    ]


def test_a_failed_request_fails_the_step_without_the_key() -> None:
    plan = _compile(
        [
            {
                "type": "roboflow_core/open_ai@v7",
                "name": "gpt",
                "images": "$inputs.image",
                "prompt": "Describe.",
                "api_key": "$inputs.openai_key",
            }
        ],
        [_output("answer", "$steps.gpt.output")],
    )
    server = FakeChatServer(
        lambda path, body: (400, error_body(f"bad key {OPENAI_KEY}"))
    )
    session = plan.create_session(
        resources={"openai_http_client": server.http_client()}
    )

    with pytest.raises(Exception) as caught:
        session.run({"image": [_image(8, 8)], "openai_key": OPENAI_KEY})

    chain = _chain_text(caught.value)
    assert "bad key <secret input>" in chain
    assert OPENAI_KEY not in chain


def test_openrouter_structured_answer_feeds_the_json_parser() -> None:
    plan = _compile(
        [
            {
                "type": "roboflow_core/openrouter@v3",
                "name": "router",
                "images": "$inputs.image",
                "model_id": "qwen/qwen3.6-27b",
                "task_type": "structured-answering",
                "output_structure": {"count": "number of cars"},
                "api_key": "sk-or-literal",
            },
            {
                "type": "roboflow_core/json_parser@v1",
                "name": "parse",
                "raw_json": "$steps.router.output",
                "expected_fields": ["count"],
            },
        ],
        [
            _output("count", "$steps.parse.count"),
            _output("thinking", "$steps.router.thinking"),
        ],
        inputs=[IMAGE_INPUT],
    )
    server = FakeChatServer(
        lambda path, body: (
            200,
            chat_completion_body('```json\n{"count": 3}\n```', reasoning="counted"),
        )
    )
    session = plan.create_session(
        resources={"openrouter_http_client": server.http_client()}
    )

    rows = session.run({"image": [_image(32, 32)]}).rows()

    assert rows == [{"count": 3, "thinking": "counted"}]
    assert server.requests[0].body["provider"]["data_collection"] == "deny"


def test_openai_compatible_sends_a_workflow_image_as_a_vision_part() -> None:
    plan = _compile(
        [
            {
                "type": "roboflow_core/openai_compatible@v1",
                "name": "local",
                "base_url": "http://llm.test/v1",
                "model_name": "qwen",
                "prompt": "Describe {{ $parameters.photo }}",
                "prompt_parameters": {"photo": "$inputs.image"},
            }
        ],
        [
            _output("answer", "$steps.local.output"),
            _output("error", "$steps.local.error_status"),
        ],
        inputs=[IMAGE_INPUT],
    )
    server = FakeChatServer(lambda path, body: (200, chat_completion_body("A box.")))
    session = plan.create_session(
        resources={
            "openai_compatible_http_client": server.http_client(),
            "allowed_base_urls": ("http://llm.test/v1",),
        }
    )

    rows = session.run({"image": [_image(8, 8), _image(9, 9)]}).rows()

    assert rows == [{"answer": "A box.", "error": ""}] * 2
    content = server.requests[0].body["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "Describe"}
    assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_workload_introspection_reports_the_remote_models() -> None:
    plan = _compile(
        [
            {
                "type": "roboflow_core/open_ai@v7",
                "name": "gpt",
                "images": "$inputs.image",
                "task_type": "instance-segmentation",
                "classes": ["car"],
            },
            {
                "type": "roboflow_core/openrouter@v3",
                "name": "router",
                "images": "$inputs.image",
                "model_id": "$inputs.model",
                "prompt": "Describe.",
            },
        ],
        [_output("a", "$steps.gpt.output"), _output("b", "$steps.router.output")],
        inputs=[IMAGE_INPUT, {"type": "WorkflowParameter", "name": "model"}],
    )

    workload = discover_workload(plan)

    text = json.dumps(workload, default=str)
    assert "gpt-6-astra" in text
    assert "$inputs.model" in text


@pytest.mark.parametrize(
    "block_type,resource,extra",
    [
        ("roboflow_core/open_ai@v7", "openai_http_client", {"images": "$inputs.image"}),
        (
            "roboflow_core/openrouter@v3",
            "openrouter_http_client",
            {"images": "$inputs.image", "model_id": "openai/gpt-4o-mini"},
        ),
        (
            "roboflow_core/openai_compatible@v1",
            "openai_compatible_http_client",
            {"base_url": "http://llm.test/v1", "model_name": "local"},
        ),
    ],
)
@pytest.mark.parametrize("temperature", [0.0, 0.5, 2.0, -0.1, 2.1])
def test_compiled_temperature_selectors_validate_resolved_values_before_requests(
    block_type, resource, extra, temperature
):
    plan = _compile(
        [
            {
                "type": block_type,
                "name": "chat",
                "prompt": "hi",
                "api_key": OPENAI_KEY,
                "temperature": "$inputs.temperature",
                **extra,
            }
        ],
        [_output("answer", "$steps.chat.output")],
        inputs=[
            IMAGE_INPUT,
            {"type": "WorkflowParameter", "name": "temperature", "kind": ["float"]},
        ],
    )
    body = (
        responses_body("ok")
        if block_type == "roboflow_core/open_ai@v7"
        else chat_completion_body("ok")
    )
    server = FakeChatServer(lambda path, request: (200, body))
    session = plan.create_session(resources={resource: server.http_client()})
    values = {"image": [_image(8, 8)], "temperature": temperature}
    if 0.0 <= temperature <= 2.0:
        assert session.run(values).rows()[0]["answer"] == "ok"
        assert server.requests[0].body["temperature"] == temperature
    else:
        with pytest.raises(StepExecutionError, match="temperature") as caught:
            session.run(values)
        assert isinstance(caught.value.__cause__, ValueError)
        assert server.requests == []

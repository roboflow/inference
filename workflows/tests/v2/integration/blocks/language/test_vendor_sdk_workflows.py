"""Claude and Gemini blocks in compiled V2 workflows, against in-process fakes.

Nothing leaves the process: Anthropic (streamed through the real SDK) and
Gemini are ``httpx.MockTransport`` fakes injected as session resources, and
the Roboflow platform client is a recording fake. This checks wiring and
composition, not any real model::

    image ─▶ roboflow_core/anthropic_claude@v5 (detection) ─▶ roboflow_core/detection_offset@v1
    image ─▶ roboflow_core/google_gemini@v6 (structured) ─▶ roboflow_core/json_parser@v1
    image ─▶ roboflow_core/google_gemini@v6 (managed key) ─▶ host platform client
"""

import json
from typing import Any, Dict, List

import pytest
import torch
from roboflow_workflows.v2.blocks.formatters.json_parser.v1 import JsonParserBlock
from roboflow_workflows.v2.blocks.geometry.detection_offset.v1 import DetectionOffsetBlock
from roboflow_workflows.v2.blocks.language.anthropic_claude.v5 import AnthropicClaudeBlock
from roboflow_workflows.v2.blocks.language.google_gemini.v6 import GoogleGeminiBlock
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.language import LANGUAGE_MODEL_OUTPUT_KIND
from roboflow_workflows.v2.data.secrets import SECRET_KIND, Secret
from roboflow_workflows.v2.engine.catalogue import Catalogue
from roboflow_workflows.v2.engine.compilation import compile_workflow
from roboflow_workflows.v2.engine.introspection import discover_workload

from inference_models.models.base.classification import ClassificationPrediction
from tests.v2.support.compilation.fixture_blocks import Gate
from tests.v2.support.language.fake_chat import FakeChatServer, FakePlatformClient
from tests.v2.unit.blocks.language.test_anthropic import (
    FakeAnthropicServer,
    claude_message,
)
from tests.v2.unit.blocks.language.test_gemini import gemini_body

ANTHROPIC_KEY = "sk-ant-test-CLAUDE-77aa"
GOOGLE_KEY = "AIza-test-GEMINI-5c3e"
ROBOFLOW_KEY = "rf-test-ROBOFLOW-1a2b"
CATALOGUE = Catalogue(
    [
        AnthropicClaudeBlock,
        GoogleGeminiBlock,
        DetectionOffsetBlock,
        JsonParserBlock,
        Gate,
    ],
    kinds=[SECRET_KIND, LANGUAGE_MODEL_OUTPUT_KIND],
)
IMAGE_INPUT = {"type": "WorkflowBatchInput", "name": "image", "kind": ["image"]}
KEY_INPUT = {"type": "WorkflowParameter", "name": "vendor_key", "kind": ["secret"]}


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
        texts.append(f"{type(error).__name__}: {error} {error.args!r}")
        error = error.__cause__ or error.__context__

    return "\n".join(texts)


def _claude_half_box(body: Dict[str, Any]):
    text = body["messages"][0]["content"][1]["text"]
    width, height = text.split(" pixel image")[0].rsplit(" ", 1)[1].split("x")
    answer = [{"box_2d": [0, 0, int(width) // 2, int(height) // 2], "label": "car"}]

    return 200, claude_message(json.dumps(answer))


def test_claude_detections_from_a_secret_key_feed_a_geometry_step() -> None:
    plan = _compile(
        [
            {
                "type": "roboflow_core/anthropic_claude@v5",
                "name": "claude",
                "images": "$inputs.image",
                "task_type": "object-detection",
                "classes": ["car"],
                "api_key": "$inputs.vendor_key",
            },
            {
                "type": "roboflow_core/detection_offset@v1",
                "name": "offset",
                "predictions": "$steps.claude.predictions",
                "offset_width": 10,
                "offset_height": 10,
            },
        ],
        [
            _output("answer", "$steps.claude.output"),
            _output("tokens", "$steps.claude.output_tokens"),
            _output("boxes", "$steps.offset.predictions"),
        ],
    )
    server = FakeAnthropicServer(_claude_half_box)
    session = plan.create_session(
        resources={"anthropic_http_client": server.http_client()}
    )

    result = session.run({"image": [_image(400, 200)], "vendor_key": ANTHROPIC_KEY})
    rows = result.rows()

    assert [request["headers"]["x-api-key"] for request in server.requests] == [
        ANTHROPIC_KEY
    ]
    assert server.requests[0]["body"]["model"] == "claude-sonnet-4-5-20250929"
    assert ANTHROPIC_KEY not in json.dumps(result.rows(serialize=True))
    # The 400x200 image fits Claude's limits, so it is uploaded as is; the
    # offset step grows the 200x100 box by 5 px per side.
    assert rows[0]["boxes"].xyxy.tolist() == [[0.0, 0.0, 205.0, 105.0]]
    assert rows[0]["tokens"] == 9


def test_gemini_structured_answer_feeds_the_json_parser() -> None:
    plan = _compile(
        [
            {
                "type": "roboflow_core/google_gemini@v6",
                "name": "gemini",
                "images": "$inputs.image",
                "task_type": "structured-answering",
                "output_structure": {"count": "number of cars"},
                "api_key": "$inputs.vendor_key",
            },
            {
                "type": "roboflow_core/json_parser@v1",
                "name": "parse",
                "raw_json": "$steps.gemini.output",
                "expected_fields": ["count"],
            },
        ],
        [_output("count", "$steps.parse.count")],
    )
    server = FakeChatServer(lambda path, body: (200, gemini_body('{"count": 3}')))
    session = plan.create_session(
        resources={"gemini_http_client": server.http_client()}
    )

    rows = session.run({"image": [_image(32, 32)], "vendor_key": GOOGLE_KEY}).rows()

    assert rows == [{"count": 3}]
    assert server.requests[0].headers["x-goog-api-key"] == GOOGLE_KEY
    assert (
        server.requests[0].path
        == "/v1beta/models/gemini-3.1-pro-preview:generateContent"
    )


def test_managed_gemini_key_uses_the_host_platform_client() -> None:
    plan = _compile(
        [
            {
                "type": "roboflow_core/google_gemini@v6",
                "name": "gemini",
                "images": "$inputs.image",
                "task_type": "classification",
                "classes": ["car", "bus"],
            }
        ],
        [_output("label", "$steps.gemini.predictions")],
        inputs=[IMAGE_INPUT],
    )
    platform = FakePlatformClient(
        lambda endpoint, payload: gemini_body(
            '{"class_name": "bus", "confidence": 0.7}'
        )
    )
    server = FakeChatServer(lambda path, body: (200, gemini_body("unused")))
    session = plan.create_session(
        resources={
            "platform_client": platform,
            "api_key": ROBOFLOW_KEY,
            "gemini_http_client": server.http_client(),
        }
    )

    rows = session.run({"image": [_image(16, 16), _image(8, 8)]}).rows()

    assert server.requests == []
    assert [(post["endpoint"], post["api_key"]) for post in platform.posts] == [
        ("apiproxy/gemini", ROBOFLOW_KEY)
    ] * 2
    assert all(isinstance(row["label"], ClassificationPrediction) for row in rows)


def test_a_gated_image_sends_no_request() -> None:
    plan = _compile(
        [
            {
                "type": Gate.type,
                "name": "keep",
                "value": "$inputs.keep",
                "next_steps": ["$steps.claude"],
            },
            {
                "type": "roboflow_core/anthropic_claude@v5",
                "name": "claude",
                "images": "$inputs.image",
                "prompt": "Describe.",
                "api_key": "$inputs.vendor_key",
            },
        ],
        [_output("answer", "$steps.claude.output")],
        inputs=[IMAGE_INPUT, KEY_INPUT, {"type": "WorkflowBatchInput", "name": "keep"}],
    )
    server = FakeAnthropicServer(lambda body: (200, claude_message("An image.")))
    session = plan.create_session(
        resources={"anthropic_http_client": server.http_client()}
    )

    rows = session.run(
        {
            "image": [_image(8, 8), _image(9, 9), _image(10, 10)],
            "keep": [True, False, True],
            "vendor_key": ANTHROPIC_KEY,
        }
    ).rows()

    assert len(server.requests) == 2
    assert [row["answer"] for row in rows] == ["An image.", None, "An image."]


@pytest.mark.parametrize("vendor", ["claude", "gemini"])
def test_a_failed_request_fails_the_step_without_the_key(vendor) -> None:
    key = ANTHROPIC_KEY if vendor == "claude" else GOOGLE_KEY
    block_type = (
        "roboflow_core/anthropic_claude@v5" if vendor == "claude" else "roboflow_core/google_gemini@v6"
    )
    plan = _compile(
        [
            {
                "type": block_type,
                "name": "vlm",
                "images": "$inputs.image",
                "prompt": "Describe.",
                "api_key": "$inputs.vendor_key",
            }
        ],
        [_output("answer", "$steps.vlm.output")],
    )
    error = {"type": "error", "error": {"type": "x", "message": f"bad key {key}"}}
    if vendor == "claude":
        server = FakeAnthropicServer(lambda body: (401, error))
        resources = {"anthropic_http_client": server.http_client()}
    else:
        server = FakeChatServer(lambda path, body: (401, error))
        resources = {"gemini_http_client": server.http_client()}
    session = plan.create_session(resources=resources)

    with pytest.raises(Exception) as caught:
        session.run({"image": [_image(8, 8)], "vendor_key": key})

    chain = _chain_text(caught.value)
    assert "request failed" in chain
    assert key not in chain


def test_workload_introspection_reports_both_vendors() -> None:
    plan = _compile(
        [
            {
                "type": "roboflow_core/anthropic_claude@v5",
                "name": "claude",
                "images": "$inputs.image",
                "prompt": "Describe.",
                "model_version": "claude-haiku-4-5",
            },
            {
                "type": "roboflow_core/google_gemini@v6",
                "name": "gemini",
                "images": "$inputs.image",
                "prompt": "Describe.",
                "model_version": "$inputs.model",
            },
        ],
        [_output("a", "$steps.claude.output"), _output("b", "$steps.gemini.output")],
        inputs=[IMAGE_INPUT, {"type": "WorkflowParameter", "name": "model"}],
    )

    text = json.dumps(discover_workload(plan), default=str)

    assert "claude-haiku-4-5-20251001" in text
    assert "$inputs.model" in text
    assert "gemini_http_client" in text


@pytest.mark.parametrize(
    "block_class, response, resource",
    [
        (AnthropicClaudeBlock, claude_message("unused"), "anthropic_http_client"),
        (GoogleGeminiBlock, gemini_body("unused"), "gemini_http_client"),
    ],
)
@pytest.mark.parametrize("host_key", [None, "", Secret("", name="empty-host")])
def test_managed_session_requires_nonempty_host_key(
    block_class, response, resource: str, host_key
) -> None:
    plan = _compile(
        [
            {
                "type": block_class.type,
                "name": "vendor",
                "images": "$inputs.image",
                "api_key": "$inputs.vendor_key",
                "prompt": "Describe.",
            }
        ],
        [_output("answer", "$steps.vendor.output")],
    )
    platform = FakePlatformClient(lambda endpoint, payload: response)
    server = FakeChatServer(lambda path, body: (200, response))
    session = plan.create_session(
        resources={
            resource: server.http_client(),
            "platform_client": platform,
            "api_key": host_key,
        }
    )
    managed_key = "rf_key:user:invented-managed-user"

    with pytest.raises(Exception, match="Roboflow API key is required") as caught:
        session.run({"image": [_image(8, 8)], "vendor_key": managed_key})

    chain = _chain_text(caught.value)
    assert "host must provide the 'api_key' resource" in chain
    assert managed_key not in chain and ROBOFLOW_KEY not in chain
    assert server.requests == [] and platform.posts == []


@pytest.mark.parametrize(
    "block_class, response, resource, endpoint, payload_key",
    [
        (
            AnthropicClaudeBlock,
            claude_message("Managed."),
            "anthropic_http_client",
            "apiproxy/anthropic",
            "anthropic_api_key",
        ),
        (
            GoogleGeminiBlock,
            gemini_body("Managed."),
            "gemini_http_client",
            "apiproxy/gemini",
            "google_api_key",
        ),
    ],
)
def test_managed_session_reveals_a_valid_host_secret(
    block_class, response, resource: str, endpoint: str, payload_key: str
) -> None:
    plan = _compile(
        [
            {
                "type": block_class.type,
                "name": "vendor",
                "images": "$inputs.image",
                "api_key": "$inputs.vendor_key",
                "prompt": "Describe.",
            }
        ],
        [_output("answer", "$steps.vendor.output")],
    )
    platform = FakePlatformClient(lambda endpoint, payload: response)
    server = FakeChatServer(lambda path, body: (200, response))
    session = plan.create_session(
        resources={
            resource: server.http_client(),
            "platform_client": platform,
            "api_key": Secret(ROBOFLOW_KEY, name="host"),
        }
    )
    managed_key = "rf_key:user:invented-managed-user"

    rows = session.run({"image": [_image(8, 8)], "vendor_key": managed_key}).rows()

    assert rows == [{"answer": "Managed."}]
    assert server.requests == [] and len(platform.posts) == 1
    post = platform.posts[0]
    assert (post["endpoint"], post["api_key"]) == (endpoint, ROBOFLOW_KEY)
    assert post["payload"][payload_key] == managed_key
    assert ROBOFLOW_KEY not in json.dumps(rows)

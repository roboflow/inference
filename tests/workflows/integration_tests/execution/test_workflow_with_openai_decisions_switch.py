"""Run Decisions Switch through server bindings, mocking only the HTTP transport."""

import base64
import json
import re
from typing import Any, Dict, List
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np
import pytest
import requests

from inference.core.env import ENABLE_TENSOR_DATA_REPRESENTATION
from inference.core.interfaces.roboflow_platform_client import (
    workflows_platform_bindings,
)
from inference.core.workflows.errors import StepExecutionError
from inference.core.workflows.execution_engine.core import ExecutionEngine

ROBOFLOW_API_KEY = "rf-test-switch-key"
PROXY_URL = re.compile(r".*/apiproxy/openai/decisions(\?.*)?$")
BLOCK_TYPE = "roboflow_core/open_ai_decisions_switch@v1"
QUESTION = "Which line should handle this item?"

# Tensor-origin inputs are supported by the tensor-mode deserializer only.
pytestmark = pytest.mark.parametrize(
    "image_as_workflow_input",
    (
        ["numpy-input", "tensor-input"]
        if ENABLE_TENSOR_DATA_REPRESENTATION
        else ["numpy-input"]
    ),
    indirect=True,
)


def _decision(choice: str, probability: Any) -> dict:
    return {
        "answers": [
            {
                "type": "choice",
                "name": "route",
                "choice": choice,
                "probabilities": [{"value": choice, "probability": probability}],
            }
        ]
    }


def _build_workflow(*, with_default: bool) -> dict:
    branches = ["on_damaged", "on_intact"]
    if with_default:
        branches.append("review")

    return {
        "version": "1.0",
        "inputs": [
            {"type": "WorkflowImage", "name": "image"},
            {"type": "WorkflowParameter", "name": "line"},
            {"type": "WorkflowParameter", "name": "min_confidence"},
        ],
        "steps": [
            {
                "type": "roboflow_core/property_definition@v1",
                "name": "width",
                "data": "$inputs.image",
                "operations": [
                    {"type": "ExtractImageProperty", "property_name": "width"}
                ],
            },
            {
                "type": BLOCK_TYPE,
                "name": "switch",
                "images": "$inputs.image",
                "question": QUESTION,
                "cases": {"damaged": "$steps.on_damaged", "intact": "$steps.on_intact"},
                "case_descriptions": {"damaged": "Visible damage"},
                "default_next_steps": ["$steps.review"] if with_default else [],
                "min_confidence": "$inputs.min_confidence",
                "context": {"width": "$steps.width.output", "line": "$inputs.line"},
                "image_detail": "low",
                "max_concurrent_requests": 2,
            },
            *(
                {
                    "type": "roboflow_core/property_definition@v1",
                    "name": name,
                    "data": "$inputs.image",
                    "operations": [
                        {"type": "ExtractImageProperty", "property_name": "width"}
                    ],
                }
                for name in branches
            ),
        ],
        "outputs": [
            {"type": "JsonField", "name": name, "selector": f"$steps.{name}.output"}
            for name in branches
        ],
    }


def _run_workflow(
    images: List[Any], *, with_default: bool = True, min_confidence: float = 0.9
) -> List[Dict[str, Any]]:
    engine = ExecutionEngine.init(
        workflow_definition=_build_workflow(with_default=with_default),
        init_parameters={
            **workflows_platform_bindings(),
            "workflows_core.api_key": ROBOFLOW_API_KEY,
        },
        max_concurrent_steps=1,
    )
    try:
        result = engine.run(
            runtime_parameters={
                "image": images,
                "line": "A",
                "min_confidence": min_confidence,
            },
            serialize_results=True,
        )
    finally:
        engine.close()

    return result


def _expected_row(branch: str, *, width: int, with_default: bool = True) -> dict:
    names = ["on_damaged", "on_intact"] + (["review"] if with_default else [])
    return {name: width if name == branch else None for name in names}


def test_switch_routes_each_image_and_sends_its_context(
    requests_mock, image_as_workflow_input
) -> None:
    dimensions = [(37, 83), (29, 61), (41, 97), (23, 53), (31, 71)]
    responses = {
        83: _decision("damaged", 0.95),
        61: _decision("intact", 0.99),
        97: _decision("damaged", 0.1),
        53: _decision("none of these", 0.99),
        71: {"answers": [{"type": "refusal", "name": "route"}]},
    }

    def respond(request, context):
        width = request.json()["input"][0]["content"][1]["data"]["width"]
        return responses[width]

    proxy = requests_mock.post(PROXY_URL, json=respond)
    images = [
        image_as_workflow_input(np.full((height, width, 3), (11, 71, 203), np.uint8))
        for height, width in dimensions
    ]

    result = _run_workflow(images)

    assert result == [
        _expected_row(branch, width=width)
        for (_, width), branch in zip(
            dimensions, ["on_damaged", "on_intact", "review", "review", "review"]
        )
    ]
    assert proxy.call_count == len(images)
    seen_widths = []
    for request in proxy.request_history:
        assert urlparse(request.url).path.endswith("/apiproxy/openai/decisions")
        assert parse_qs(urlparse(request.url).query)["api_key"] == [ROBOFLOW_API_KEY]
        body = request.json()
        assert body["openai_api_key"] == "rf_key:account"
        assert body["questions"] == [
            {
                "type": "choice",
                "name": "route",
                "instructions": QUESTION,
                "choices": [
                    {"value": "damaged", "description": "Visible damage"},
                    {"value": "intact"},
                    {
                        "value": "none of these",
                        "description": "None of the other options fits.",
                    },
                ],
            }
        ]
        image_part, context_part = body["input"][0]["content"]
        assert image_part["type"] == "input_image"
        assert image_part["detail"] == "low"
        width = context_part["data"]["width"]
        assert context_part == {
            "type": "input_json",
            "data": {"width": width, "line": "A"},
        }
        seen_widths.append(width)
        jpeg = base64.b64decode(image_part["image_url"].split(",", 1)[1])
        decoded = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        height = next(
            height for height, candidate_width in dimensions if candidate_width == width
        )
        assert decoded.shape == (height, width, 3)
        assert np.allclose(decoded.mean(axis=(0, 1)), (11, 71, 203), atol=3)
    assert sorted(seen_widths) == sorted(width for _, width in dimensions)


@pytest.mark.parametrize(
    "decision",
    [
        _decision("none of these", 0.99),
        _decision("damaged", 0.1),
        {"answers": [{"type": "refusal", "name": "route"}]},
    ],
    ids=["none-of-these", "below-threshold", "refusal"],
)
def test_unselected_image_terminates_without_a_default(
    requests_mock, image_as_workflow_input, decision: dict
) -> None:
    proxy = requests_mock.post(PROXY_URL, json=decision)
    image = image_as_workflow_input(np.zeros((37, 83, 3), dtype=np.uint8))

    result = _run_workflow([image], with_default=False)

    assert proxy.call_count == 1
    assert result == [_expected_row("review", width=83, with_default=False)]


@pytest.mark.parametrize("with_default", [True, False], ids=["default", "terminate"])
@pytest.mark.parametrize(
    "probability",
    [
        pytest.param(float("nan"), id="NaN"),
        pytest.param("NaN", id="string-NaN"),
        pytest.param(float("inf"), id="positive-infinity"),
        pytest.param(float("-inf"), id="negative-infinity"),
        pytest.param(1.1, id="above-one"),
        pytest.param(-0.1, id="below-zero"),
    ],
)
def test_invalid_probability_never_selects_a_case(
    requests_mock, image_as_workflow_input, probability: Any, with_default: bool
) -> None:
    # Use raw JSON text so non-standard NaN/Infinity reach the real response decoder.
    proxy = requests_mock.post(
        PROXY_URL,
        text=json.dumps(_decision("damaged", probability)),
        headers={"Content-Type": "application/json"},
    )
    image = image_as_workflow_input(np.zeros((37, 83, 3), dtype=np.uint8))

    result = _run_workflow([image], with_default=with_default, min_confidence=0.0)

    assert proxy.call_count == 1
    assert result == [_expected_row("review", width=83, with_default=with_default)]


@pytest.mark.parametrize(
    "probability, min_confidence, expected",
    [
        (0.95, 0.9, "on_damaged"),
        (0.89, 0.9, "review"),
        (0.9, 0.9, "on_damaged"),
        (0.0, 0.0, "on_damaged"),
        (1.0, 1.0, "on_damaged"),
    ],
)
def test_valid_probability_preserves_the_inclusive_threshold(
    requests_mock,
    image_as_workflow_input,
    probability: float,
    min_confidence: float,
    expected: str,
) -> None:
    proxy = requests_mock.post(PROXY_URL, json=_decision("damaged", probability))
    image = image_as_workflow_input(np.zeros((37, 83, 3), dtype=np.uint8))

    result = _run_workflow([image], min_confidence=min_confidence)

    assert proxy.call_count == 1
    assert result == [_expected_row(expected, width=83)]


@pytest.mark.parametrize(
    "failure",
    [
        {"status_code": 400, "json": {"error": "Invalid image"}},
        {"exc": requests.exceptions.ConnectionError},
        {"exc": requests.exceptions.ConnectTimeout},
    ],
    ids=["http-error", "connection-error", "timeout"],
)
def test_failed_proxy_request_fails_the_workflow(
    requests_mock, image_as_workflow_input, failure: dict
) -> None:
    proxy = requests_mock.post(PROXY_URL, **failure)
    image = image_as_workflow_input(np.zeros((37, 83, 3), dtype=np.uint8))

    with pytest.raises(StepExecutionError) as error:
        _run_workflow([image])

    assert proxy.call_count == 1
    assert error.value.block_type == BLOCK_TYPE
    assert ROBOFLOW_API_KEY not in str(error.value)

"""Offline runs of the OpenAI Decisions block through the Execution Engine.

Requests stop at the HTTP transport, `requests` for the Roboflow proxy and HTTPX2
for the OpenAI SDK, so the block's request and decoding code runs unchanged.
"""

import base64
import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple
from unittest import mock
from urllib.parse import parse_qs, urlparse

import cv2
import httpx2
import numpy as np
import pytest
import requests

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.interfaces.roboflow_platform_client import (
    workflows_platform_bindings,
)
from inference.core.workflows.errors import StepExecutionError
from inference.core.workflows.execution_engine.core import ExecutionEngine

ROBOFLOW_API_KEY = "rf-test-roboflow-key"
OPENAI_API_KEY = "sk-test-openai-key"
MANAGED_OPENAI_API_KEY = "rf_key:account"
PROXY_URL = re.compile(r".*/apiproxy/openai/decisions(\?.*)?$")

PROMPT = "Photos taken at a pet shelter."
CLASSES = ["cat", "dog", "bird"]
QUESTIONS = [
    {"type": "predicate", "name": "leash", "instructions": "The animal wears a leash."},
    {
        "type": "choice",
        "name": "place",
        "instructions": "Where was the photo taken?",
        "choices": [{"value": "indoors"}, {"value": "outdoors"}],
    },
    {
        "type": "score",
        "name": "calm",
        "instructions": "How calm does the animal look?",
        "levels": [{"label": "agitated"}, {"label": "calm"}],
    },
]
QUESTION_ANSWERS = [
    {"type": "predicate", "name": "leash", "probability": 0.1},
    {
        "type": "choice",
        "name": "place",
        "choice": "indoors",
        "probabilities": [
            {"value": "indoors", "probability": 0.9},
            {"value": "outdoors", "probability": 0.1},
        ],
        "confidence": 0.8,
    },
    {
        "type": "score",
        "name": "calm",
        "score": 0.75,
        "probabilities": [
            {"label": "agitated", "value": 0, "probability": 0.25},
            {"label": "calm", "value": 1, "probability": 0.75},
        ],
        "confidence": 0.5,
    },
]
USAGE = {
    "input_tokens": 562,
    "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
    "output_tokens": 0,
    "output_tokens_details": {"reasoning_tokens": 0},
    "total_tokens": 562,
}
OUTPUTS = [
    "predictions",
    "answers",
    "error_status",
    "inference_id",
    "input_tokens",
    "output_tokens",
]

Answer = Callable[[Dict[str, Any]], Tuple[int, Dict[str, Any]]]


@dataclass
class RecordedRequest:
    url: str
    headers: Any
    body: Dict[str, Any]


class FakeDecisionsAPI:
    """Answers Decisions requests with `answer` and records what was asked."""

    def __init__(self, transport: str, openai_api_key: str) -> None:
        self.transport = transport
        self.openai_api_key = openai_api_key
        self.answer: Optional[Answer] = None
        self.requests: List[RecordedRequest] = []

    def respond(self, request: RecordedRequest) -> Tuple[int, Dict[str, Any]]:
        self.requests.append(request)
        return self.answer(request.body)


@pytest.fixture(params=["proxy", "direct"])
def decisions_api(request, requests_mock) -> Generator[FakeDecisionsAPI, None, None]:
    if request.param == "proxy":
        api = FakeDecisionsAPI(transport="proxy", openai_api_key=MANAGED_OPENAI_API_KEY)

        def respond_to_proxy_request(proxy_request, context) -> Dict[str, Any]:
            context.status_code, payload = api.respond(
                RecordedRequest(
                    url=proxy_request.url,
                    headers=proxy_request.headers,
                    body=proxy_request.json(),
                )
            )
            return payload

        requests_mock.post(PROXY_URL, json=respond_to_proxy_request)
        yield api
        return

    api = FakeDecisionsAPI(transport="direct", openai_api_key=OPENAI_API_KEY)

    def respond_to_openai_request(openai_request) -> httpx2.Response:
        status_code, payload = api.respond(
            RecordedRequest(
                url=str(openai_request.url),
                headers=openai_request.headers,
                body=json.loads(openai_request.content),
            )
        )
        return httpx2.Response(status_code, json=payload)

    with mock.patch.object(
        httpx2.HTTPTransport, "handle_request", side_effect=respond_to_openai_request
    ):
        yield api


def _build_workflow(task_type: str) -> dict:
    task_input = "questions" if task_type == "questions" else "classes"
    return {
        "version": "1.0",
        "inputs": [
            {"type": "WorkflowImage", "name": "image"},
            {"type": "WorkflowParameter", "name": task_input},
            {"type": "WorkflowParameter", "name": "prompt"},
            {"type": "WorkflowParameter", "name": "openai_api_key"},
            {"type": "WorkflowParameter", "name": "image_detail"},
        ],
        "steps": [
            {
                "type": "roboflow_core/open_ai_decisions@v1",
                "name": "decisions",
                "images": "$inputs.image",
                "task_type": task_type,
                task_input: f"$inputs.{task_input}",
                "prompt": "$inputs.prompt",
                "api_key": "$inputs.openai_api_key",
                "image_detail": "$inputs.image_detail",
            }
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": output,
                "selector": f"$steps.decisions.{output}",
            }
            for output in OUTPUTS
        ],
    }


def _run_workflow(
    api: FakeDecisionsAPI,
    *,
    task_type: str,
    images: List[np.ndarray],
) -> List[Dict[str, Any]]:
    execution_engine = ExecutionEngine.init(
        workflow_definition=_build_workflow(task_type),
        init_parameters={
            **workflows_platform_bindings(),
            "workflows_core.api_key": ROBOFLOW_API_KEY,
        },
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )
    task_input = "questions" if task_type == "questions" else "classes"

    result = execution_engine.run(
        runtime_parameters={
            "image": images,
            task_input: QUESTIONS if task_type == "questions" else CLASSES,
            "prompt": PROMPT,
            "openai_api_key": api.openai_api_key,
            "image_detail": "low",
        },
        serialize_results=True,
    )

    assert json.loads(json.dumps(result)) == result, "Outputs must serialize to JSON"
    return result


def _image(value: int) -> np.ndarray:
    return np.full((48, 64, 3), value, dtype=np.uint8)


def _sent_image(body: Dict[str, Any]) -> np.ndarray:
    image_url = body["input"][0]["content"][0]["image_url"]
    jpeg = base64.b64decode(image_url.removeprefix("data:image/jpeg;base64,"))
    return cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)


def _assert_asked_about_the_image(
    api: FakeDecisionsAPI, request: RecordedRequest, questions: List[dict]
) -> None:
    assert request.body["model"] == "gpt-6-luna"
    assert request.body["questions"] == questions
    image_part, prompt_part = request.body["input"][0]["content"]
    assert image_part["type"] == "input_image"
    assert image_part["detail"] == "low"
    assert _sent_image(request.body).shape == (48, 64, 3)
    assert prompt_part == {"type": "input_text", "text": PROMPT}
    if api.transport == "proxy":
        assert parse_qs(urlparse(request.url).query)["api_key"] == [ROBOFLOW_API_KEY]
        assert request.body["openai_api_key"] == MANAGED_OPENAI_API_KEY
    else:
        assert urlparse(request.url).path == "/v1/decisions"
        assert request.headers["authorization"] == f"Bearer {OPENAI_API_KEY}"
        assert request.body["prompt_cache_options"] == {"mode": "explicit"}


def _choice_answer(choice: str, probabilities: Dict[str, float]) -> dict:
    return {
        "type": "choice",
        "name": "class",
        "choice": choice,
        "probabilities": [
            {"value": value, "probability": probability}
            for value, probability in probabilities.items()
        ],
        "confidence": 0.5,
    }


def _predicate_answer(name: str, probability: float) -> dict:
    return {"type": "predicate", "name": name, "probability": probability}


def test_classification_reports_every_class_probability(
    decisions_api: FakeDecisionsAPI,
) -> None:
    # Tensor-native predictions hold confidences as float32, so the probabilities
    # are exact in float32 to serialize the same under both tensor settings.
    answers = [_choice_answer("dog", {"cat": 0.125, "dog": 0.75, "bird": 0.125})]
    decisions_api.answer = lambda body: (200, {"answers": answers, "usage": USAGE})

    result = _run_workflow(
        decisions_api, task_type="classification", images=[_image(120)]
    )

    assert len(decisions_api.requests) == 1
    _assert_asked_about_the_image(
        decisions_api,
        decisions_api.requests[0],
        questions=[
            {
                "type": "choice",
                "name": "class",
                "instructions": "Which of these classes best describes the image?",
                "choices": [{"value": "cat"}, {"value": "dog"}, {"value": "bird"}],
            }
        ],
    )
    assert result == [
        {
            "predictions": {
                "image": {"width": 64, "height": 48},
                "predictions": [
                    {"class": "cat", "class_id": 0, "confidence": 0.125},
                    {"class": "dog", "class_id": 1, "confidence": 0.75},
                    {"class": "bird", "class_id": 2, "confidence": 0.125},
                ],
                "top": "dog",
                "confidence": 0.75,
                "inference_id": result[0]["inference_id"],
                "parent_id": "image.[0]",
            },
            "answers": answers,
            "error_status": False,
            "inference_id": result[0]["inference_id"],
            "input_tokens": 562,
            "output_tokens": 0,
        }
    ]


def test_multi_label_classification_reads_each_class_from_its_named_answer(
    decisions_api: FakeDecisionsAPI,
) -> None:
    answers = [
        _predicate_answer("class_2", 0.875),
        _predicate_answer("class_0", 0.5),
        _predicate_answer("class_1", 0.25),
    ]
    decisions_api.answer = lambda body: (200, {"answers": answers, "usage": USAGE})

    result = _run_workflow(
        decisions_api, task_type="multi-label-classification", images=[_image(120)]
    )

    assert len(decisions_api.requests) == 1
    _assert_asked_about_the_image(
        decisions_api,
        decisions_api.requests[0],
        questions=[
            {
                "type": "predicate",
                "name": f"class_{class_id}",
                "instructions": f"The image shows {class_name}.",
            }
            for class_id, class_name in enumerate(CLASSES)
        ],
    )
    assert result == [
        {
            "predictions": {
                "image": {"width": 64, "height": 48},
                "predictions": {
                    "cat": {"confidence": 0.5, "class_id": 0},
                    "dog": {"confidence": 0.25, "class_id": 1},
                    "bird": {"confidence": 0.875, "class_id": 2},
                },
                "predicted_classes": ["cat", "bird"],
                "inference_id": result[0]["inference_id"],
                "parent_id": "image.[0]",
            },
            "answers": answers,
            "error_status": False,
            "inference_id": result[0]["inference_id"],
            "input_tokens": 562,
            "output_tokens": 0,
        }
    ]


def test_questions_return_the_answers_without_predictions(
    decisions_api: FakeDecisionsAPI,
) -> None:
    decisions_api.answer = lambda body: (
        200,
        {"answers": QUESTION_ANSWERS, "usage": USAGE},
    )

    result = _run_workflow(decisions_api, task_type="questions", images=[_image(120)])

    assert len(decisions_api.requests) == 1
    _assert_asked_about_the_image(
        decisions_api, decisions_api.requests[0], questions=QUESTIONS
    )
    assert result == [
        {
            "predictions": None,
            "answers": QUESTION_ANSWERS,
            "error_status": False,
            "inference_id": result[0]["inference_id"],
            "input_tokens": 562,
            "output_tokens": 0,
        }
    ]


def test_a_refused_question_flags_only_the_image_it_was_about(
    decisions_api: FakeDecisionsAPI,
) -> None:
    refused = [
        QUESTION_ANSWERS[0],
        {"type": "refusal", "name": "place"},
        QUESTION_ANSWERS[2],
    ]
    decisions_api.answer = lambda body: (
        200,
        {
            "answers": refused if _sent_image(body).mean() > 127 else QUESTION_ANSWERS,
            "usage": USAGE,
        },
    )

    result = _run_workflow(
        decisions_api, task_type="questions", images=[_image(0), _image(255)]
    )

    assert len(decisions_api.requests) == 2
    assert [element["error_status"] for element in result] == [False, True]
    assert [element["answers"] for element in result] == [QUESTION_ANSWERS, refused]
    assert [element["predictions"] for element in result] == [None, None]
    assert [element["input_tokens"] for element in result] == [562, 562]


@pytest.mark.parametrize(
    "decision",
    [
        {"answers": [{"type": "refusal", "name": "class"}], "usage": USAGE},
        {"usage": USAGE},
        {"answers": [_choice_answer("fish", {"fish": 1.0})], "usage": USAGE},
        {
            "answers": [
                _choice_answer("cat", {"cat": 1.0}),
                _predicate_answer("leash", 1.0),
            ],
            "usage": USAGE,
        },
        {"answers": [_predicate_answer("class", 1.0)], "usage": USAGE},
    ],
    ids=["refusal", "no-answers", "unknown-class", "extra-answer", "wrong-type"],
)
def test_refused_or_undecodable_answers_flag_the_image(
    decisions_api: FakeDecisionsAPI,
    decision: dict,
) -> None:
    decisions_api.answer = lambda body: (200, decision)

    result = _run_workflow(
        decisions_api, task_type="classification", images=[_image(120)]
    )

    assert result == [
        {
            "predictions": None,
            "answers": decision.get("answers"),
            "error_status": True,
            "inference_id": result[0]["inference_id"],
            "input_tokens": 562,
            "output_tokens": 0,
        }
    ]


def test_a_failed_request_fails_the_run(decisions_api: FakeDecisionsAPI) -> None:
    decisions_api.answer = lambda body: (
        400,
        {"error": {"message": "Invalid image.", "type": "invalid_request_error"}},
    )

    with pytest.raises(StepExecutionError) as error:
        _run_workflow(decisions_api, task_type="classification", images=[_image(120)])

    assert len(decisions_api.requests) == 1
    assert error.value.block_type == "roboflow_core/open_ai_decisions@v1"
    assert ROBOFLOW_API_KEY not in str(error.value)
    assert OPENAI_API_KEY not in str(error.value)


@pytest.mark.parametrize(
    "failure",
    [requests.exceptions.ConnectionError, requests.exceptions.ConnectTimeout],
)
def test_an_unreachable_proxy_fails_the_run(requests_mock, failure: type) -> None:
    requests_mock.post(PROXY_URL, exc=failure)
    api = FakeDecisionsAPI(transport="proxy", openai_api_key=MANAGED_OPENAI_API_KEY)

    with pytest.raises(StepExecutionError) as error:
        _run_workflow(api, task_type="classification", images=[_image(120)])

    assert requests_mock.call_count == 1
    assert error.value.block_type == "roboflow_core/open_ai_decisions@v1"
    assert ROBOFLOW_API_KEY not in str(error.value)

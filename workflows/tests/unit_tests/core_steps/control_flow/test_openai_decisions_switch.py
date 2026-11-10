"""Tests for the OpenAI Decisions Switch block (``roboflow_core/open_ai_decisions_switch@v1``)."""

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import numpy as np
import pytest
import requests
import supervision as sv
from pydantic import ValidationError
from roboflow_workflows.core_steps.flow_control.openai_decisions_switch.v1 import (
    MAX_CASES,
    NONE_OPTION,
    NONE_OPTION_DESCRIPTION,
    BlockManifest,
    OpenAIDecisionsSwitchBlockV1,
)
from roboflow_workflows.execution_engine.core import ExecutionEngine
from roboflow_workflows.execution_engine.entities.base import (
    Batch,
    ImageParentMetadata,
    WorkflowImageData,
)
from roboflow_workflows.execution_engine.entities.workload import WorkOperation
from roboflow_workflows.execution_engine.v1.entities import FlowControl
from roboflow_workflows.prototypes.block import third_party_model

from tests.unit_tests.prototypes.platform_client_double import RecordingPlatformClient

MODULE = "roboflow_workflows.core_steps.flow_control.openai_decisions_switch.v1"
EXECUTE_REQUEST_SEAM = f"{MODULE}.execute_decision_request"

CASES = {"damaged": "$steps.on_damaged", "intact": "$steps.on_intact"}
DEFAULT_STEPS = ["$steps.review"]
ON_DAMAGED = FlowControl(mode="select_step", context=["$steps.on_damaged"])
ON_DEFAULT = FlowControl(mode="select_step", context=DEFAULT_STEPS)
TERMINATE = FlowControl(mode="terminate_branch")


def _build_image(height: int = 48, width: int = 64) -> WorkflowImageData:
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="parent"),
        numpy_image=np.zeros((height, width, 3), dtype=np.uint8),
    )


def _batch(*elements: Any) -> Batch:
    return Batch(content=list(elements), indices=[(i,) for i in range(len(elements))])


def _decision(
    choice: str, probability: float, confidence: Optional[float] = None
) -> Dict[str, Any]:
    others = [value for value in ("damaged", "intact", NONE_OPTION) if value != choice]
    rest = (1 - probability) / len(others)
    return {
        "answers": [
            {
                "type": "choice",
                "name": "route",
                "choice": choice,
                "confidence": probability if confidence is None else confidence,
                "probabilities": [
                    {"value": choice, "probability": probability},
                    *({"value": value, "probability": rest} for value in others),
                ],
            }
        ],
        "usage": {"input_tokens": 300, "output_tokens": 0},
    }


def _run_block(
    decision: Any = None,
    *,
    images: Optional[Batch] = None,
    default_next_steps: Optional[List[str]] = None,
    min_confidence: float = 0.9,
    case_descriptions: Optional[Dict[str, str]] = None,
    context: Optional[Dict[str, Any]] = None,
    context_operations: Optional[Dict[str, list]] = None,
    side_effect: Any = None,
) -> tuple:
    block = OpenAIDecisionsSwitchBlockV1(api_key="rf-key")
    with patch(
        EXECUTE_REQUEST_SEAM, return_value=decision, side_effect=side_effect
    ) as execute_request:
        result = block.run(
            images=images if images is not None else _batch(_build_image()),
            question="Which line should handle this item?",
            cases=CASES,
            case_descriptions=case_descriptions or {},
            default_next_steps=(
                DEFAULT_STEPS if default_next_steps is None else default_next_steps
            ),
            min_confidence=min_confidence,
            context=context or {},
            context_operations=context_operations or {},
            model_version="gpt-6-luna",
            image_detail="auto",
            max_concurrent_requests=2,
            api_key="sk-openai",
        )
    return result, execute_request


def _manifest(**overrides: Any) -> dict:
    return {
        "type": "roboflow_core/open_ai_decisions_switch@v1",
        "name": "switch",
        "images": "$inputs.image",
        "question": "Which line should handle this item?",
        "cases": CASES,
        **overrides,
    }


def test_manifest_parsing_applies_defaults() -> None:
    # when
    result = BlockManifest.model_validate(_manifest())

    # then
    assert result.min_confidence == 0.9
    assert result.default_next_steps == []
    assert result.case_descriptions == {}
    assert result.context == {}
    assert result.api_key == "rf_key:account"
    assert result.model_version == "gpt-6-luna"
    assert result.describe_outputs() == []


def test_manifest_parsing_accepts_selectors_and_a_single_case() -> None:
    # when
    result = BlockManifest.model_validate(
        _manifest(
            question="$inputs.question",
            cases={"damaged": "$steps.on_damaged"},
            min_confidence="$inputs.min_confidence",
            context={"label_text": "$steps.ocr.result", "line": "A", "belt": 2},
        )
    )

    # then
    assert result.question == "$inputs.question"
    assert result.min_confidence == "$inputs.min_confidence"
    assert result.context["belt"] == 2


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"cases": {}}, id="no cases"),
        pytest.param(
            {"cases": {f"c{i}": f"$steps.s{i}" for i in range(MAX_CASES + 1)}},
            id="too many cases",
        ),
        pytest.param({"cases": {" ": "$steps.a"}}, id="empty case"),
        pytest.param({"cases": {"None Of These": "$steps.a"}}, id="reserved case"),
        pytest.param(
            {"cases": {"a": "$steps.same", "b": "$steps.same"}}, id="repeated target"
        ),
        pytest.param(
            {"default_next_steps": ["$steps.on_damaged"]},
            id="target repeated in the default",
        ),
        pytest.param({"case_descriptions": {"other": "text"}}, id="unknown case"),
        pytest.param(
            {"context_operations": {"boxes": [{"type": "SequenceLength"}]}},
            id="operations for a missing value",
        ),
        pytest.param({"min_confidence": 1.5}, id="confidence out of range"),
    ],
)
def test_manifest_parsing_rejects_invalid_configuration(overrides: dict) -> None:
    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(_manifest(**overrides))


def test_manifest_declares_its_work_and_the_model_it_depends_on() -> None:
    # when
    manifest = BlockManifest.model_validate(_manifest())

    # then
    assert WorkOperation.FLOW_CONTROL in manifest.discover_work_operations()
    assert WorkOperation.MODEL_INFERENCE in manifest.discover_work_operations()
    assert manifest.discover_dependent_resources() == [
        third_party_model(provider="openai", model_id="gpt-6-luna")
    ]


@pytest.mark.parametrize(
    "decision, min_confidence, default_next_steps, expected",
    [
        pytest.param(
            _decision("damaged", 0.95), 0.9, DEFAULT_STEPS, ON_DAMAGED, id="chosen case"
        ),
        pytest.param(
            _decision("damaged", 0.9),
            0.9,
            DEFAULT_STEPS,
            ON_DAMAGED,
            id="at the minimum",
        ),
        pytest.param(
            _decision("damaged", 0.89), 0.9, DEFAULT_STEPS, ON_DEFAULT, id="below it"
        ),
        pytest.param(
            _decision("damaged", 0.4), 0.0, DEFAULT_STEPS, ON_DAMAGED, id="no minimum"
        ),
        pytest.param(
            _decision("damaged", 0.92, confidence=0.84),
            0.9,
            DEFAULT_STEPS,
            ON_DAMAGED,
            id="reads the probability, not the API's confidence",
        ),
        pytest.param(
            _decision(NONE_OPTION, 0.99),
            0.9,
            DEFAULT_STEPS,
            ON_DEFAULT,
            id="no case fits",
        ),
        pytest.param(
            {"answers": [{"type": "refusal", "name": "route"}]},
            0.9,
            DEFAULT_STEPS,
            ON_DEFAULT,
            id="refusal",
        ),
        pytest.param({"answers": []}, 0.9, DEFAULT_STEPS, ON_DEFAULT, id="no answers"),
        pytest.param("not a decision", 0.9, DEFAULT_STEPS, ON_DEFAULT, id="not a dict"),
        pytest.param(
            _decision("damaged", 0.5), 0.9, [], TERMINATE, id="below it, no default"
        ),
        pytest.param(
            _decision(NONE_OPTION, 0.99), 0.9, [], TERMINATE, id="no case, no default"
        ),
    ],
)
def test_run_routes_each_answer(
    decision: Any,
    min_confidence: float,
    default_next_steps: List[str],
    expected: FlowControl,
) -> None:
    # when
    result, _ = _run_block(
        decision,
        min_confidence=min_confidence,
        default_next_steps=default_next_steps,
    )

    # then
    assert result == [expected]


def test_run_routes_every_image_of_a_batch_on_its_own() -> None:
    # when
    result, execute_request = _run_block(
        side_effect=[_decision("damaged", 0.99), _decision("intact", 0.2)],
        images=_batch(_build_image(), _build_image()),
    )

    # then
    assert execute_request.call_count == 2
    assert result == [ON_DAMAGED, ON_DEFAULT]


def test_run_propagates_a_failed_request() -> None:
    # when
    with pytest.raises(RuntimeError, match="proxy down"):
        _ = _run_block(side_effect=RuntimeError("proxy down"))


def test_run_asks_one_choice_over_the_cases_and_the_none_option() -> None:
    # when
    _, execute_request = _run_block(
        _decision("damaged", 0.99),
        case_descriptions={"damaged": "The box is torn"},
    )

    # then
    assert execute_request.call_args.kwargs["questions"] == [
        {
            "type": "choice",
            "name": "route",
            "instructions": "Which line should handle this item?",
            "choices": [
                {"value": "damaged", "description": "The box is torn"},
                {"value": "intact"},
                {"value": NONE_OPTION, "description": NONE_OPTION_DESCRIPTION},
            ],
        }
    ]
    assert execute_request.call_args.kwargs["openai_api_key"] == "sk-openai"
    assert execute_request.call_args.kwargs["model_version"] == "gpt-6-luna"


def test_run_sends_only_the_image_when_there_is_no_context() -> None:
    # when
    _, execute_request = _run_block(_decision("damaged", 0.99))

    # then
    content = execute_request.call_args.kwargs["decision_input"][0]["content"]
    assert [part["type"] for part in content] == ["input_image"]
    assert content[0]["detail"] == "auto"


def test_run_sends_each_image_its_own_context_as_json() -> None:
    # given
    manifest = BlockManifest.model_validate(
        _manifest(
            context={"boxes": "$steps.model.predictions"},
            context_operations={"boxes": [{"type": "SequenceLength"}]},
        )
    )

    # when
    _, execute_request = _run_block(
        side_effect=[_decision("damaged", 0.99), _decision("intact", 0.99)],
        images=_batch(_build_image(), _build_image()),
        context={
            "label_text": _batch("FRAGILE", "OK"),
            "boxes": _batch([1, 2, 3], []),
            "line": "A",
            "seen_at": datetime(2026, 11, 3, tzinfo=timezone.utc),
        },
        context_operations=manifest.context_operations,
    )

    # then
    sent = [
        call.kwargs["decision_input"][0]["content"]
        for call in execute_request.call_args_list
    ]
    assert [part["type"] for part in sent[0]] == ["input_image", "input_json"]
    assert sent[0][1]["data"] == {
        "label_text": "FRAGILE",
        "boxes": 3,
        "line": "A",
        "seen_at": "2026-11-03T00:00:00+00:00",
    }
    assert sent[1][1]["data"] == {
        "label_text": "OK",
        "boxes": 0,
        "line": "A",
        "seen_at": "2026-11-03T00:00:00+00:00",
    }


def test_run_serializes_detections_in_the_context() -> None:
    # given
    detections = sv.Detections(
        xyxy=np.array([[10.0, 20.0, 110.0, 220.0]]),
        confidence=np.array([0.5]),
        class_id=np.array([3]),
        data={
            "class_name": np.array(["forklift"]),
            "detection_id": np.array(["first"]),
            "image_dimensions": np.array([[300, 400]]),
        },
    )

    # when
    _, execute_request = _run_block(
        _decision("damaged", 0.99), context={"boxes": _batch(detections)}
    )

    # then
    content = execute_request.call_args.kwargs["decision_input"][0]["content"]
    assert content[1]["data"]["boxes"] == {
        "image": {"width": 400, "height": 300},
        "predictions": [
            {
                "width": 100.0,
                "height": 200.0,
                "x": 60.0,
                "y": 120.0,
                "confidence": 0.5,
                "class_id": 3,
                "class": "forklift",
                "detection_id": "first",
            }
        ],
    }


@pytest.mark.parametrize(
    "context, expected_context",
    [
        pytest.param({"line": "A"}, {"line": "A"}, id="plain values"),
        pytest.param(
            {"depth": _batch(np.array([[0.25, 0.75]], dtype=np.float32))},
            {"depth": [[0.25, 0.75]]},
            id="numpy array from a batched output",
        ),
        pytest.param(
            {
                "count": _batch(np.int64(3)),
                "score": np.float32(0.5),
                "matches": np.bool_(True),
                "scalar_array": np.array(2),
            },
            {"count": 3, "score": 0.5, "matches": True, "scalar_array": 2},
            id="numpy scalars and zero-dimensional array",
        ),
        pytest.param(
            {
                "summary": _batch(
                    {
                        "counts": np.array([1, 2], dtype=np.int64),
                        "nested": [{"score": np.float32(0.5)}, (np.int64(3),)],
                    }
                )
            },
            {"summary": {"counts": [1, 2], "nested": [{"score": 0.5}, [3]]}},
            id="numpy values nested in containers",
        ),
    ],
)
def test_run_with_a_managed_key_posts_the_request_to_the_proxy(
    context: Dict[str, Any], expected_context: Dict[str, Any]
) -> None:
    # given
    platform_client = RecordingPlatformClient(post_response=_decision("damaged", 0.99))
    block = OpenAIDecisionsSwitchBlockV1(
        api_key="rf-key", platform_client=platform_client
    )

    # when
    result = block.run(
        images=_batch(_build_image()),
        question="Which line should handle this item?",
        cases=CASES,
        case_descriptions={},
        default_next_steps=DEFAULT_STEPS,
        min_confidence=0.9,
        context=context,
        context_operations={},
        model_version="gpt-6-luna",
        image_detail="low",
        max_concurrent_requests=None,
        api_key="rf_key:account",
    )

    # then
    assert result == [ON_DAMAGED]
    post = platform_client.posts[0]
    assert post["endpoint"] == "apiproxy/openai/decisions"
    assert post["api_key"] == "rf-key"
    assert post["payload"]["openai_api_key"] == "rf_key:account"
    assert post["payload"]["model"] == "gpt-6-luna"
    prepared_request = requests.Request(
        method="POST", url="https://example.invalid", json=post["payload"]
    ).prepare()
    payload = json.loads(prepared_request.body)
    content = payload["input"][0]["content"]
    assert content[0]["type"] == "input_image"
    assert content[0]["detail"] == "low"
    assert content[1] == {"type": "input_json", "data": expected_context}


ROUTING_WORKFLOW = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
        {"type": "WorkflowParameter", "name": "line"},
    ],
    "steps": [
        {
            "type": "roboflow_core/property_definition@v1",
            "name": "size",
            "data": "$inputs.image",
            "operations": [{"type": "ExtractImageProperty", "property_name": "size"}],
        },
        {
            "type": "roboflow_core/open_ai_decisions_switch@v1",
            "name": "switch",
            "images": "$inputs.image",
            "question": "Which line should handle this item?",
            "cases": {
                "damaged": "$steps.on_damaged",
                "intact": "$steps.on_intact",
            },
            "default_next_steps": ["$steps.review"],
            "context": {"pixels": "$steps.size.output", "line": "$inputs.line"},
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
            for name in ("on_damaged", "on_intact", "review")
        ),
    ],
    "outputs": [
        {"type": "JsonField", "name": name, "selector": f"$steps.{name}.output"}
        for name in ("on_damaged", "on_intact", "review")
    ],
}


def test_workflow_runs_only_the_branch_of_each_image() -> None:
    # given
    answers_by_pixels = {
        100: _decision("damaged", 0.99),
        400: _decision("intact", 0.97),
        900: _decision("damaged", 0.5),
        1600: _decision(NONE_OPTION, 0.99),
    }
    sent_context = []

    def answer(**kwargs: Any) -> Dict[str, Any]:
        context = kwargs["decision_input"][0]["content"][1]["data"]
        sent_context.append(context)
        return answers_by_pixels[context["pixels"]]

    execution_engine = ExecutionEngine.init(
        workflow_definition=ROUTING_WORKFLOW,
        init_parameters={
            "workflows_core.api_key": "rf-key",
            "workflows_core.platform_client": RecordingPlatformClient(),
        },
        max_concurrent_steps=1,
    )
    images = [np.zeros((side, side, 3), dtype=np.uint8) for side in (10, 20, 30, 40)]

    # when
    with patch(EXECUTE_REQUEST_SEAM, side_effect=answer):
        result = execution_engine.run(
            runtime_parameters={"image": images, "line": "A"},
        )

    # then
    assert [
        [name for name, value in row.items() if value is not None] for row in result
    ] == [["on_damaged"], ["on_intact"], ["review"], ["review"]]
    assert sorted(context["pixels"] for context in sent_context) == [
        100,
        400,
        900,
        1600,
    ]
    assert {context["line"] for context in sent_context} == {"A"}

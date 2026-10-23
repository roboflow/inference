"""Tests for the OpenAI Decisions block (``roboflow_core/open_ai_decisions@v1``)."""

from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from openai.types import Decision
from pydantic import ValidationError
from roboflow_workflows.core_steps.models.foundation.openai_decisions.v1 import (
    BlockManifest,
    OpenAIDecisionsBlockV1,
    build_multi_label_prediction,
    build_single_label_prediction,
)
from roboflow_workflows.execution_engine.entities.base import (
    Batch,
    ImageParentMetadata,
    WorkflowImageData,
)
from roboflow_workflows.prototypes.block import third_party_model

from tests.unit_tests.core_steps._vlm_prediction_readers import (
    classification_inference_id,
    classification_predicted_classes,
    classification_top_class,
    classification_top_confidence,
)
from tests.unit_tests.prototypes.platform_client_double import RecordingPlatformClient

MODULE = "roboflow_workflows.core_steps.models.foundation.openai_decisions.v1"
EXECUTE_REQUEST_SEAM = f"{MODULE}.execute_decision_request"
CLIENT_SEAM = f"{MODULE}._get_openai_client"

USAGE = {"input_tokens": 812, "output_tokens": 3}


def _build_image() -> WorkflowImageData:
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="parent"),
        numpy_image=np.zeros((48, 64, 3), dtype=np.uint8),
    )


def _choice_answer(choice: str, probabilities: Dict[str, float]) -> dict:
    return {
        "type": "choice",
        "name": "class",
        "choice": choice,
        "confidence": probabilities[choice],
        "probabilities": [
            {"value": value, "probability": probability}
            for value, probability in probabilities.items()
        ],
    }


def _predicate_answer(name: str, probability: float) -> dict:
    return {"type": "predicate", "name": name, "probability": probability}


def _run(
    block: OpenAIDecisionsBlockV1,
    *,
    task_type: str,
    classes: Optional[List[str]] = None,
    questions: Optional[List[Dict[str, Any]]] = None,
    prompt: Optional[str] = None,
    api_key: str = "sk-test",
) -> dict:
    results = block.run(
        images=Batch(content=[_build_image()], indices=[(0,)]),
        task_type=task_type,
        classes=classes,
        questions=questions,
        prompt=prompt,
        model_version="gpt-6-luna",
        image_detail="low",
        max_concurrent_requests=None,
        api_key=api_key,
    )
    assert len(results) == 1
    return results[0]


def _run_with_decision(
    decision: dict,
    *,
    task_type: str,
    classes: Optional[List[str]] = None,
    questions: Optional[List[Dict[str, Any]]] = None,
) -> dict:
    with patch(EXECUTE_REQUEST_SEAM, return_value=decision):
        return _run(
            OpenAIDecisionsBlockV1(api_key="rf-key"),
            task_type=task_type,
            classes=classes,
            questions=questions,
        )


@pytest.mark.parametrize(
    "task_type, fields, message",
    [
        ("classification", {}, "`classes` parameter required"),
        ("multi-label-classification", {}, "`classes` parameter required"),
        ("questions", {}, "`questions` parameter required"),
        ("classification", {"classes": ["cat"]}, "needs 2 to 255 classes"),
        ("multi-label-classification", {"classes": ["cat", "cat"]}, "repeat"),
        ("multi-label-classification", {"classes": ["cat", " "]}, "non-empty"),
        ("questions", {"questions": []}, "1 to 400 questions"),
        (
            "questions",
            {"questions": [{"type": "boolean", "instructions": "A cat?"}]},
            "needs a `type`",
        ),
    ],
)
def test_manifest_rejects_what_the_task_cannot_ask(
    task_type: str, fields: dict, message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        BlockManifest.model_validate(
            {
                "type": "roboflow_core/open_ai_decisions@v1",
                "name": "decisions",
                "images": "$inputs.image",
                "task_type": task_type,
                **fields,
            }
        )


def test_manifest_declares_the_openai_model_it_runs() -> None:
    manifest = BlockManifest.model_validate(
        {
            "type": "roboflow_core/open_ai_decisions@v1",
            "name": "decisions",
            "images": "$inputs.image",
            "classes": "$inputs.classes",
        }
    )

    assert manifest.discover_dependent_resources() == [
        third_party_model(provider="openai", model_id="gpt-6-luna"),
    ]


def test_classification_asks_one_choice_over_the_classes_through_the_proxy() -> None:
    platform_client = RecordingPlatformClient(
        post_response={
            "answers": [_choice_answer("dog", {"cat": 0.2, "dog": 0.8})],
            "usage": USAGE,
        }
    )
    block = OpenAIDecisionsBlockV1(api_key="rf-key", platform_client=platform_client)

    result = _run(
        block,
        task_type="classification",
        classes=["cat", "dog"],
        prompt="Pets at home.",
        api_key="rf_key:account",
    )

    [post] = platform_client.posts
    assert post["endpoint"] == "apiproxy/openai/decisions"
    assert post["api_key"] == "rf-key"
    payload = post["payload"]
    assert payload["model"] == "gpt-6-luna"
    assert payload["openai_api_key"] == "rf_key:account"
    assert payload["questions"] == [
        {
            "type": "choice",
            "name": "class",
            "instructions": "Which of these classes best describes the image?",
            "choices": [{"value": "cat"}, {"value": "dog"}],
        }
    ]
    [message] = payload["input"]
    image_part, text_part = message["content"]
    assert image_part["type"] == "input_image"
    assert image_part["detail"] == "low"
    assert image_part["image_url"].startswith("data:image/jpeg;base64,")
    assert text_part == {"type": "input_text", "text": "Pets at home."}
    assert (result["input_tokens"], result["output_tokens"]) == (812, 3)
    # The engine accepts exactly the declared outputs.
    assert set(result) == {output.name for output in BlockManifest.describe_outputs()}


@patch(CLIENT_SEAM)
def test_an_openai_key_asks_openai_with_caching_off(
    mock_get_client: MagicMock,
) -> None:
    questions = [
        {
            "type": "predicate",
            "name": "helmets",
            "instructions": "Every person in the image wears a helmet.",
        }
    ]
    answers = [_predicate_answer("helmets", 0.9)]
    mock_get_client.return_value.decisions.create.return_value = (
        Decision.model_validate(
            {
                "answers": answers,
                "model": "gpt-6-luna",
                "usage": {
                    "input_tokens": 812,
                    "input_tokens_details": {
                        "cached_tokens": 0,
                        "cache_write_tokens": 0,
                    },
                    "output_tokens": 0,
                    "output_tokens_details": {"reasoning_tokens": 0},
                    "total_tokens": 812,
                },
            }
        )
    )

    result = _run(
        OpenAIDecisionsBlockV1(api_key="rf-key"),
        task_type="questions",
        questions=questions,
    )

    mock_get_client.assert_called_once_with("sk-test")
    request = mock_get_client.return_value.decisions.create.call_args.kwargs
    assert request["model"] == "gpt-6-luna"
    assert request["questions"] == questions
    assert request["prompt_cache_options"] == {"mode": "explicit"}
    assert result["answers"] == answers
    assert result["predictions"] is None
    assert result["error_status"] is False
    assert (result["input_tokens"], result["output_tokens"]) == (812, 0)


def test_a_managed_key_needs_a_roboflow_api_key() -> None:
    with pytest.raises(ValueError, match="Roboflow API key is required"):
        _run(
            OpenAIDecisionsBlockV1(api_key=None),
            task_type="classification",
            classes=["cat", "dog"],
            api_key="rf_key:account",
        )


def test_run_decodes_the_chosen_class_with_its_probability() -> None:
    result = _run_with_decision(
        {"answers": [_choice_answer("dog", {"cat": 0.25, "dog": 0.75})]},
        task_type="classification",
        classes=["cat", "dog"],
    )

    assert result["error_status"] is False
    assert classification_top_class(result["predictions"]) == "dog"
    assert classification_top_confidence(result["predictions"]) == pytest.approx(0.75)
    assert classification_inference_id(result["predictions"]) == result["inference_id"]


def test_run_asks_one_predicate_per_class_and_keeps_the_likely_ones() -> None:
    decision = {
        "answers": [
            _predicate_answer("class_0", 0.5),
            _predicate_answer("class_1", 0.49),
            _predicate_answer("class_2", 0.97),
        ]
    }

    with patch(EXECUTE_REQUEST_SEAM, return_value=decision) as mock_execute:
        result = _run(
            OpenAIDecisionsBlockV1(api_key="rf-key"),
            task_type="multi-label-classification",
            classes=["cat", "dog", "bird"],
        )

    assert mock_execute.call_args.kwargs["questions"] == [
        {
            "type": "predicate",
            "name": f"class_{class_id}",
            "instructions": f"The image shows {class_name}.",
        }
        for class_id, class_name in enumerate(["cat", "dog", "bird"])
    ]
    assert result["error_status"] is False
    assert sorted(classification_predicted_classes(result["predictions"])) == [
        "bird",
        "cat",
    ]


@pytest.mark.parametrize(
    "decision",
    [
        {"answers": [{"type": "refusal", "name": "class"}]},
        {"answers": [_choice_answer("fish", {"cat": 0.1, "fish": 0.9})]},
        {"answers": [{"type": "choice", "name": "class", "choice": "cat"}]},
        {"answers": []},
        {"model": "gpt-6-luna"},
    ],
)
def test_run_flags_answers_it_cannot_decode(decision: dict) -> None:
    result = _run_with_decision(
        decision, task_type="classification", classes=["cat", "dog"]
    )

    assert result["error_status"] is True
    assert result["predictions"] is None
    assert result["answers"] == decision.get("answers")


def test_run_flags_a_refused_question() -> None:
    answers = [
        _predicate_answer("helmets", 0.9),
        {"type": "refusal", "name": "faces"},
    ]

    result = _run_with_decision(
        {"answers": answers},
        task_type="questions",
        questions=[
            {"type": "predicate", "name": "helmets", "instructions": "Helmets on."},
            {"type": "predicate", "name": "faces", "instructions": "Faces visible."},
        ],
    )

    assert result["error_status"] is True
    assert result["answers"] == answers


def test_single_label_prediction_gives_every_class_its_probability() -> None:
    prediction = build_single_label_prediction(
        _choice_answer("dog", {"cat": 0.25, "dog": 0.75}),
        image=_build_image(),
        classes=["cat", "dog", "bird"],
        inference_id="inference",
    )

    assert prediction == {
        "image": {"width": 64, "height": 48},
        "predictions": [
            {"class": "cat", "class_id": 0, "confidence": 0.25},
            {"class": "dog", "class_id": 1, "confidence": 0.75},
            {"class": "bird", "class_id": 2, "confidence": 0.0},
        ],
        "top": "dog",
        "confidence": 0.75,
        "inference_id": "inference",
        "parent_id": "parent",
    }


def test_multi_label_prediction_gives_every_class_its_statement_probability() -> None:
    prediction = build_multi_label_prediction(
        [_predicate_answer("class_0", 0.5), _predicate_answer("class_1", 0.49)],
        image=_build_image(),
        classes=["cat", "dog"],
        inference_id="inference",
    )

    assert prediction == {
        "image": {"width": 64, "height": 48},
        "predictions": {
            "cat": {"confidence": 0.5, "class_id": 0},
            "dog": {"confidence": 0.49, "class_id": 1},
        },
        "predicted_classes": ["cat"],
        "inference_id": "inference",
        "parent_id": "parent",
    }

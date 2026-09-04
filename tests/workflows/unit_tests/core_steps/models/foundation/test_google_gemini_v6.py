"""Tests for the Google Gemini v6 block (in-block VLM decoding).

The v1-v4 behavior suite lives in ``test_google_gemini.py`` and token usage
in ``test_google_gemini_v5.py``; this file covers the v6 delta:
``predictions`` / ``error_status`` / ``inference_id`` outputs decoded inside
the block from Gemini's native ``yxyx_0_1000`` boxes.
"""

import json
from typing import List, Optional
from unittest.mock import patch

import numpy as np
import pytest

from inference.core.workflows.core_steps.models.foundation.google_gemini.v6 import (
    DETECTION_BOX_FORMAT,
    BlockManifest,
    GoogleGeminiBlockV6,
    prepare_object_detection_prompt,
)
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    ImageParentMetadata,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    CLASSIFICATION_PREDICTION_KIND,
    INFERENCE_ID_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
)
from tests.workflows.unit_tests.core_steps._vlm_prediction_readers import (
    classification_inference_id,
    classification_top_class,
    classification_top_confidence,
    detection_boxes,
    detection_class_ids,
    detection_class_names,
    detection_count,
    detection_inference_ids,
    is_detection_prediction,
)

EXECUTE_REQUESTS_SEAM = (
    "inference.core.workflows.core_steps.models.foundation.google_gemini.v6"
    ".execute_gemini_requests"
)

# [y_min, x_min, y_max, x_max] normalized to 0-1000 - on an 800x400 image this
# is exactly [80, 100, 400, 300] pixels.
IMAGE_WIDTH = 800
IMAGE_HEIGHT = 400
DETECTION_OUTPUT = json.dumps([{"box_2d": [250, 100, 750, 500], "label": "cat"}])
EXPECTED_XYXY = [80.0, 100.0, 400.0, 300.0]


def _build_image(
    width: int = IMAGE_WIDTH, height: int = IMAGE_HEIGHT
) -> WorkflowImageData:
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="parent"),
        numpy_image=np.zeros((height, width, 3), dtype=np.uint8),
    )


def _run_block(
    task_type: str,
    raw_output: str,
    image: WorkflowImageData,
    classes: Optional[List[str]] = None,
    prompt: Optional[str] = None,
) -> dict:
    block = GoogleGeminiBlockV6(model_manager=None, api_key="rf-key")
    with patch(EXECUTE_REQUESTS_SEAM) as mock_execute:
        mock_execute.return_value = [(raw_output, 11, 3)]
        results = block.run(
            images=Batch(content=[image], indices=[(0,)]),
            task_type=task_type,
            prompt=prompt,
            output_structure=None,
            classes=classes,
            model_version="gemini-2.5-pro",
            max_tokens=None,
            temperature=None,
            thinking_level=None,
            google_code_execution=None,
            max_concurrent_requests=None,
            api_key="google-key",
        )
    assert len(results) == 1
    return results[0]


def test_manifest_parsing_for_new_block_type() -> None:
    # given
    raw_manifest = {
        "type": "roboflow_core/google_gemini@v6",
        "name": "gemini",
        "images": "$inputs.image",
        "task_type": "object-detection",
        "classes": ["cat", "dog"],
        "api_key": "$inputs.api_key",
    }

    # when
    result = BlockManifest.model_validate(raw_manifest)

    # then
    assert result.type == "roboflow_core/google_gemini@v6"
    assert result.task_type == "object-detection"
    assert result.classes == ["cat", "dog"]


def test_describe_outputs_declares_prediction_outputs() -> None:
    # when
    outputs = {output.name: output.kind for output in BlockManifest.describe_outputs()}

    # then
    assert outputs["predictions"] == [
        OBJECT_DETECTION_PREDICTION_KIND,
        CLASSIFICATION_PREDICTION_KIND,
    ]
    assert outputs["error_status"] == [BOOLEAN_KIND]
    assert outputs["inference_id"] == [INFERENCE_ID_KIND]
    assert "output" in outputs


@pytest.mark.parametrize(
    "task_type, expected_kind",
    [
        ("object-detection", [OBJECT_DETECTION_PREDICTION_KIND]),
        ("classification", [CLASSIFICATION_PREDICTION_KIND]),
        ("multi-label-classification", [CLASSIFICATION_PREDICTION_KIND]),
    ],
)
def test_get_actual_outputs_narrows_predictions(
    task_type: str, expected_kind: list
) -> None:
    # given
    manifest = BlockManifest.model_validate(
        {
            "type": "roboflow_core/google_gemini@v6",
            "name": "gemini",
            "images": "$inputs.image",
            "task_type": task_type,
            "classes": ["cat"],
        }
    )

    # when
    outputs = {output.name: output.kind for output in manifest.get_actual_outputs()}

    # then
    assert outputs["predictions"] == expected_kind
    assert "output" in outputs


def test_get_actual_outputs_keeps_union_for_unconstrained_task() -> None:
    # given
    manifest = BlockManifest.model_validate(
        {
            "type": "roboflow_core/google_gemini@v6",
            "name": "gemini",
            "images": "$inputs.image",
            "task_type": "unconstrained",
            "prompt": "what is on the image?",
        }
    )

    # when
    outputs = {output.name: output.kind for output in manifest.get_actual_outputs()}

    # then
    assert outputs["predictions"] == [
        OBJECT_DETECTION_PREDICTION_KIND,
        CLASSIFICATION_PREDICTION_KIND,
    ]
    assert [output.name for output in BlockManifest.describe_outputs()] == list(outputs)


def test_object_detection_prompt_keeps_pinned_wording() -> None:
    # when
    prompt = prepare_object_detection_prompt(
        base64_image="base64-image",
        classes=["cat", "dog"],
        model_version="gemini-2.5-pro",
        temperature=None,
        thinking_level=None,
        max_tokens=None,
    )

    # then
    assert DETECTION_BOX_FORMAT == "yxyx_0_1000"
    prompt_text = prompt["contents"]["parts"][1]["text"]
    assert prompt_text == (
        "Detect all objects in this image. "
        "Output a JSON list where each entry contains the 2D bounding box "
        'in the key "box_2d" and the text label in the key "label". '
        'The "box_2d" value must be [y_min, x_min, y_max, x_max]: integers '
        "between 0 and 1000, normalized to the image height and width. "
        "Return only the JSON list, with no extra text. "
        "Only use these labels: cat, dog"
    )
    assert prompt["generationConfig"]["response_schema"]["items"]["properties"][
        "label"
    ]["enum"] == ["cat", "dog"]


def test_run_decodes_object_detection_into_original_image_pixels() -> None:
    # given
    image = _build_image()

    # when
    result = _run_block(
        task_type="object-detection",
        raw_output=DETECTION_OUTPUT,
        image=image,
        classes=["cat", "dog"],
    )

    # then
    assert result["error_status"] is False
    assert result["output"] == DETECTION_OUTPUT
    predictions = result["predictions"]
    assert is_detection_prediction(predictions)
    assert detection_count(predictions) == 1
    assert detection_boxes(predictions)[0] == EXPECTED_XYXY
    assert detection_class_ids(predictions) == [0]
    assert detection_class_names(predictions) == ["cat"]
    assert detection_inference_ids(predictions) == [result["inference_id"]]


def test_run_decodes_classification_output() -> None:
    # given
    image = _build_image(width=128, height=64)
    raw_output = json.dumps({"class_name": "dog", "confidence": 0.7})

    # when
    result = _run_block(
        task_type="classification",
        raw_output=raw_output,
        image=image,
        classes=["cat", "dog"],
    )

    # then
    assert result["error_status"] is False
    assert classification_top_class(result["predictions"]) == "dog"
    assert classification_top_confidence(result["predictions"]) == pytest.approx(0.7)
    assert classification_inference_id(result["predictions"]) == result["inference_id"]


def test_run_returns_no_predictions_for_non_decoding_task() -> None:
    # given
    image = _build_image(width=128, height=64)

    # when
    result = _run_block(
        task_type="unconstrained",
        raw_output="some free-form answer",
        image=image,
        prompt="what is on the image?",
    )

    # then
    assert result["predictions"] is None
    assert result["error_status"] is False
    assert result["output"] == "some free-form answer"
    assert result["inference_id"]


def test_run_reports_error_status_for_unparsable_detection_output() -> None:
    # given
    image = _build_image()

    # when
    result = _run_block(
        task_type="object-detection",
        raw_output="I am sorry, I cannot help with that.",
        image=image,
        classes=["cat", "dog"],
    )

    # then
    assert result["error_status"] is True
    assert result["predictions"] is None


def test_manifest_accepts_gemini_3_8_flash() -> None:
    # given - v5 gained this model after v6 was cut; migrating workflows must
    # not lose it
    raw_manifest = {
        "type": "roboflow_core/google_gemini@v6",
        "name": "gemini",
        "images": "$inputs.image",
        "task_type": "object-detection",
        "classes": ["cat", "dog"],
        "api_key": "$inputs.api_key",
        "model_version": "gemini-3.8-flash",
    }

    # when
    result = BlockManifest.model_validate(raw_manifest)

    # then
    assert result.model_version == "gemini-3.8-flash"

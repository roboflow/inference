"""Tests for the Anthropic Claude v5 block (in-block VLM decoding).

The v1-v3 behavior suite lives in ``test_anthropic_claude.py``, token usage
in ``test_anthropic_claude_v4.py`` and the detection upload contract in
``test_anthropic_claude_v4_detection.py``. This file covers the v5 delta:
``predictions`` / ``error_status`` / ``inference_id`` outputs decoded inside
the block.
"""

import json
from typing import List, Optional
from unittest.mock import patch

import numpy as np
import pytest

from inference.core.workflows.core_steps.models.foundation.anthropic_claude.v5 import (
    AnthropicClaudeBlockV5,
    BlockManifest,
    detection_upload_dimensions,
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
    "inference.core.workflows.core_steps.models.foundation.anthropic_claude.v5"
    ".execute_claude_requests"
)

# A 3200x1600 image uploads at 2576x1288, so the box below maps onto exactly
# [800, 400, 1600, 1200] pixels of the original image.
DETECTION_IMAGE_WIDTH = 3200
DETECTION_IMAGE_HEIGHT = 1600
DETECTION_OUTPUT = json.dumps(
    [{"box_2d": [644, 322, 1288, 966], "label": "cat"}],
)
EXPECTED_XYXY = [800.0, 400.0, 1600.0, 1200.0]


def _build_image(width: int, height: int) -> WorkflowImageData:
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
    block = AnthropicClaudeBlockV5(model_manager=None, api_key="rf-key")
    with patch(EXECUTE_REQUESTS_SEAM) as mock_execute:
        mock_execute.return_value = [(raw_output, 11, 3)]
        results = block.run(
            images=Batch(content=[image], indices=[(0,)]),
            task_type=task_type,
            prompt=prompt,
            output_structure=None,
            classes=classes,
            model_version="claude-sonnet-4-5",
            max_tokens=None,
            temperature=None,
            extended_thinking=None,
            thinking_budget_tokens=None,
            max_image_size=1024,
            max_concurrent_requests=None,
            api_key="sk-ant-test",
        )
    assert len(results) == 1
    return results[0]


def test_manifest_parsing_for_new_block_type() -> None:
    # given
    raw_manifest = {
        "type": "roboflow_core/anthropic_claude@v5",
        "name": "claude",
        "images": "$inputs.image",
        "task_type": "object-detection",
        "classes": ["cat", "dog"],
        "api_key": "$inputs.api_key",
    }

    # when
    result = BlockManifest.model_validate(raw_manifest)

    # then
    assert result.type == "roboflow_core/anthropic_claude@v5"
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


def test_get_actual_outputs_narrows_predictions_for_object_detection() -> None:
    # given
    manifest = BlockManifest.model_validate(
        {
            "type": "roboflow_core/anthropic_claude@v5",
            "name": "claude",
            "images": "$inputs.image",
            "task_type": "object-detection",
            "classes": ["cat"],
        }
    )

    # when
    outputs = {output.name: output.kind for output in manifest.get_actual_outputs()}

    # then
    assert outputs["predictions"] == [OBJECT_DETECTION_PREDICTION_KIND]
    assert "output" in outputs


def test_get_actual_outputs_narrows_predictions_for_classification() -> None:
    # given
    manifest = BlockManifest.model_validate(
        {
            "type": "roboflow_core/anthropic_claude@v5",
            "name": "claude",
            "images": "$inputs.image",
            "task_type": "classification",
            "classes": ["cat"],
        }
    )

    # when
    outputs = {output.name: output.kind for output in manifest.get_actual_outputs()}

    # then
    assert outputs["predictions"] == [CLASSIFICATION_PREDICTION_KIND]


def test_get_actual_outputs_keeps_union_for_unconstrained_task() -> None:
    # given
    manifest = BlockManifest.model_validate(
        {
            "type": "roboflow_core/anthropic_claude@v5",
            "name": "claude",
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


def test_get_actual_outputs_names_match_describe_outputs() -> None:
    # given
    manifest = BlockManifest.model_validate(
        {
            "type": "roboflow_core/anthropic_claude@v5",
            "name": "claude",
            "images": "$inputs.image",
            "task_type": "object-detection",
            "classes": ["cat"],
        }
    )

    # when
    declared = [output.name for output in BlockManifest.describe_outputs()]
    actual = [output.name for output in manifest.get_actual_outputs()]

    # then
    assert declared == actual


def test_object_detection_prompt_keeps_pinned_wording() -> None:
    # when
    system_prompt, messages = prepare_object_detection_prompt(
        base64_image="base64-image",
        classes=["cat", "dog"],
        image_width=2212,
        image_height=1659,
    )

    # then
    assert system_prompt is None
    prompt_text = messages[0]["content"][1]["text"]
    assert prompt_text == (
        "Detect all objects in this image. "
        "Output a JSON list where each entry contains the 2D bounding box "
        'in the key "box_2d" and the text label in the key "label". '
        'The "box_2d" value must be [x_min, y_min, x_max, y_max]: the '
        "top-left and bottom-right corners in absolute pixel coordinates "
        "of the 2212x1659 pixel image. "
        "Return only the JSON list, with no extra text. "
        "Only use these labels: cat, dog"
    )


def test_detection_upload_dimensions_follow_anthropic_resize() -> None:
    # given
    image = _build_image(width=DETECTION_IMAGE_WIDTH, height=DETECTION_IMAGE_HEIGHT)

    # when
    detection_dimensions = detection_upload_dimensions(
        image=image, task_type="object-detection"
    )
    other_dimensions = detection_upload_dimensions(
        image=image, task_type="unconstrained"
    )

    # then
    assert detection_dimensions == (2576, 1288)
    assert other_dimensions == (None, None)


def test_run_decodes_object_detection_into_original_image_pixels() -> None:
    # given
    image = _build_image(width=DETECTION_IMAGE_WIDTH, height=DETECTION_IMAGE_HEIGHT)

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
    image = _build_image(width=DETECTION_IMAGE_WIDTH, height=DETECTION_IMAGE_HEIGHT)

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


# ---------------------------------------------------------------------------
# Model capabilities (mirrors the v4 coverage in test_anthropic_claude.py)
# ---------------------------------------------------------------------------

from typing import Any  # noqa: E402
from unittest.mock import MagicMock, Mock  # noqa: E402

from anthropic import NOT_GIVEN  # noqa: E402

from inference.core.workflows.core_steps.models.foundation.anthropic_claude.v5 import (  # noqa: E402
    EXACT_MODEL_VERSIONS,
    MAX_OUTPUT_TOKENS,
    execute_claude_request,
)

LEGACY_MODEL = "claude-sonnet-4-5"
NEW_GENERATION_MODEL = "claude-opus-4-7"
ANTHROPIC_CLIENT_SEAM = (
    "inference.core.workflows.core_steps.models.foundation.anthropic_claude.v5"
    ".anthropic.Anthropic"
)


def test_v5_claude_fable_5_1_model_metadata() -> None:
    specification = {
        "type": "roboflow_core/anthropic_claude@v5",
        "name": "step_1",
        "images": "$inputs.image",
        "task_type": "unconstrained",
        "prompt": "This is my prompt",
        "api_key": "$inputs.anthropic_api_key",
        "model_version": "claude-fable-5-1",
    }

    result = BlockManifest.model_validate(specification)

    assert result.model_version == "claude-fable-5-1"
    assert EXACT_MODEL_VERSIONS["claude-fable-5-1"] == "claude-fable-5-1"
    assert MAX_OUTPUT_TOKENS["claude-fable-5-1"] == 128000


def _mock_streaming_client(mock_anthropic_class: Mock, text: str = "ok") -> MagicMock:
    mock_client = MagicMock()
    mock_anthropic_class.return_value = mock_client

    mock_text_block = Mock()
    mock_text_block.type = "text"
    mock_text_block.text = text

    mock_result = Mock()
    mock_result.stop_reason = "end_turn"
    mock_result.content = [mock_text_block]
    mock_result.usage = Mock(input_tokens=11, output_tokens=7)

    mock_stream = MagicMock()
    mock_stream.__enter__ = Mock(return_value=mock_stream)
    mock_stream.__exit__ = Mock(return_value=False)
    mock_stream.get_final_message.return_value = mock_result
    mock_client.messages.stream.return_value = mock_stream
    return mock_client


def _direct_request(model_version: str, **overrides: Any) -> dict:
    kwargs = dict(
        roboflow_api_key=None,
        anthropic_api_key="sk-ant-test",
        system_prompt=None,
        messages=[{"role": "user", "content": "Hello"}],
        model_version=model_version,
        max_tokens=100,
        temperature=0.4,
        extended_thinking=None,
        thinking_budget_tokens=None,
    )
    kwargs.update(overrides)
    with patch(ANTHROPIC_CLIENT_SEAM) as mock_anthropic_class:
        mock_client = _mock_streaming_client(mock_anthropic_class)
        execute_claude_request(**kwargs)
        return mock_client.messages.stream.call_args.kwargs


def test_direct_request_keeps_legacy_controls_for_legacy_model() -> None:
    no_thinking_kwargs = _direct_request(LEGACY_MODEL)
    thinking_kwargs = _direct_request(
        LEGACY_MODEL,
        max_tokens=None,
        temperature=None,
        extended_thinking=True,
        thinking_budget_tokens=5000,
    )

    assert no_thinking_kwargs["temperature"] == 0.4
    assert "thinking" not in no_thinking_kwargs
    assert thinking_kwargs["thinking"] == {"type": "enabled", "budget_tokens": 5000}


def test_direct_request_translates_controls_for_new_generation_model() -> None:
    no_thinking_kwargs = _direct_request(NEW_GENERATION_MODEL)
    thinking_kwargs = _direct_request(
        NEW_GENERATION_MODEL,
        max_tokens=None,
        temperature=None,
        extended_thinking=True,
        thinking_budget_tokens=5000,
    )

    assert no_thinking_kwargs["temperature"] is NOT_GIVEN
    assert "thinking" not in no_thinking_kwargs
    assert no_thinking_kwargs["model"] == NEW_GENERATION_MODEL
    assert thinking_kwargs["thinking"] == {"type": "adaptive"}
    assert thinking_kwargs["max_tokens"] == MAX_OUTPUT_TOKENS[NEW_GENERATION_MODEL]

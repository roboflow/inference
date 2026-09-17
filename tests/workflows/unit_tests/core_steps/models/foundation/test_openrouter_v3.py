"""Tests for the OpenRouter v3 block: predictions are decoded inside the block.

The v2 behavior suite lives in ``test_openrouter_v2.py``; this file
covers the v3 delta - the ``predictions`` / ``error_status`` /
``inference_id`` outputs.
"""

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from pydantic import ValidationError

from inference.core.workflows.core_steps.common.openrouter import OpenRouterResult
from inference.core.workflows.core_steps.common.vlm_decoding import (
    DETECTION_BOX_FORMATS,
)
from inference.core.workflows.core_steps.models.foundation.openrouter.v3 import (
    DETECTION_FORMAT_METADATA,
    BlockManifest,
    OpenRouterBlockV3,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    CLASSIFICATION_PREDICTION_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
)
from tests.workflows.unit_tests.core_steps._vlm_prediction_readers import (
    classification_top_class,
    detection_boxes,
    detection_class_ids,
    is_detection_prediction,
)

IMAGE_WIDTH = 800
IMAGE_HEIGHT = 400
# 0.1/0.25/0.5/0.75 of an 800x400 image.
EXPECTED_XYXY = [[80.0, 100.0, 400.0, 300.0]]

DETECTION_OUTPUT = json.dumps(
    {
        "detections": [
            {
                "x_min": 0.1,
                "y_min": 0.25,
                "x_max": 0.5,
                "y_max": 0.75,
                "class_name": "cat",
                "confidence": 0.7,
            }
        ]
    }
)
CLASSIFICATION_OUTPUT = json.dumps({"class_name": "cat", "confidence": 0.9})

BASE_MANIFEST = {
    "type": "roboflow_core/openrouter@v3",
    "name": "step",
    "images": "$inputs.image",
    "model_id": "moonshotai/kimi-k2.6",
}


def _image() -> WorkflowImageData:
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="parent"),
        numpy_image=np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8),
    )


def _run(raw_output: str, task_type: str, **kwargs) -> dict:
    result, _ = _run_capturing_prompts(raw_output, task_type, **kwargs)
    return result


def _run_capturing_prompts(raw_output: str, task_type: str, **kwargs):
    """Run the block and also return the OpenRouter ``messages`` it sent."""
    with patch.object(
        OpenRouterBlockV3,
        "execute_openrouter_batch_with_usage",
        return_value=[OpenRouterResult(content=raw_output)],
    ) as execute:
        block = OpenRouterBlockV3(model_manager=MagicMock(), api_key="ws-key")
        results = block.run(
            images=[_image()],
            task_type=task_type,
            prompt="describe",
            output_structure=None,
            classes=["cat", "dog"],
            api_key="sk-or-v1-test",
            privacy_level="deny",
            model_id="moonshotai/kimi-k2.6",
            max_tokens=128,
            temperature=0.1,
            reasoning_effort=None,
            max_concurrent_requests=None,
            **kwargs,
        )
    assert len(results) == 1
    prompts = execute.call_args.kwargs["prompts"]
    assert len(prompts) == 1
    return results[0], prompts[0]


def _prompt_texts(messages: list) -> str:
    """Concatenate every text part of an OpenRouter ``messages`` array."""
    texts = []
    for message in messages:
        content = message["content"]
        if isinstance(content, str):
            texts.append(content)
            continue
        texts.extend(part["text"] for part in content if part["type"] == "text")
    return "\n".join(texts)


def test_manifest_parses_new_type() -> None:
    manifest = BlockManifest.model_validate(
        {**BASE_MANIFEST, "task_type": "object-detection", "classes": ["cat"]}
    )

    assert manifest.type == "roboflow_core/openrouter@v3"
    assert manifest.task_type == "object-detection"


def test_manifest_recommends_only_json_parser() -> None:
    recommended_parsers = BlockManifest.model_fields["task_type"].json_schema_extra[
        "recommended_parsers"
    ]

    assert recommended_parsers == {
        "structured-answering": "roboflow_core/json_parser@v1"
    }


def test_describe_outputs_declares_union_prediction_kind() -> None:
    outputs = {output.name: output for output in BlockManifest.describe_outputs()}

    assert outputs["predictions"].kind == [
        OBJECT_DETECTION_PREDICTION_KIND,
        CLASSIFICATION_PREDICTION_KIND,
    ]
    assert {"predictions", "error_status", "inference_id"}.issubset(outputs)


@pytest.mark.parametrize(
    "task_type, extra_fields, expected_kind",
    [
        (
            "object-detection",
            {"classes": ["cat"]},
            [OBJECT_DETECTION_PREDICTION_KIND],
        ),
        (
            "classification",
            {"classes": ["cat"]},
            [CLASSIFICATION_PREDICTION_KIND],
        ),
        (
            "unconstrained",
            {"prompt": "describe"},
            [OBJECT_DETECTION_PREDICTION_KIND, CLASSIFICATION_PREDICTION_KIND],
        ),
    ],
)
def test_get_actual_outputs_narrows_prediction_kind(
    task_type: str, extra_fields: dict, expected_kind: list
) -> None:
    manifest = BlockManifest.model_validate(
        {**BASE_MANIFEST, "task_type": task_type, **extra_fields}
    )

    outputs = {output.name: output for output in manifest.get_actual_outputs()}

    assert outputs["predictions"].kind == expected_kind
    assert {"predictions", "error_status", "inference_id"}.issubset(outputs)


def test_run_decodes_object_detection() -> None:
    result = _run(raw_output=DETECTION_OUTPUT, task_type="object-detection")

    assert result["error_status"] is False
    assert is_detection_prediction(result["predictions"])
    assert detection_boxes(result["predictions"]) == EXPECTED_XYXY
    assert detection_class_ids(result["predictions"]) == [0]
    assert result["inference_id"]


def test_run_returns_keys_matching_actual_outputs() -> None:
    manifest = BlockManifest.model_validate(
        {**BASE_MANIFEST, "task_type": "object-detection", "classes": ["cat"]}
    )

    result = _run(raw_output=DETECTION_OUTPUT, task_type="object-detection")

    assert set(result) == {output.name for output in manifest.get_actual_outputs()}


def test_run_decodes_classification() -> None:
    result = _run(raw_output=CLASSIFICATION_OUTPUT, task_type="classification")

    assert result["error_status"] is False
    assert classification_top_class(result["predictions"]) == "cat"


def test_run_does_not_decode_unconstrained_task() -> None:
    result = _run(raw_output="a free form answer", task_type="unconstrained")

    assert result["error_status"] is False
    assert result["predictions"] is None
    assert result["output"] == "a free form answer"


def test_run_flags_error_status_on_garbage_detection_output() -> None:
    result = _run(raw_output="this is not JSON", task_type="object-detection")

    assert result["error_status"] is True
    assert result["predictions"] is None


# --- detection_format -------------------------------------------------------
# Every payload below encodes the same 80..400 x 100..300 box of the 800x400
# test image in its own contract.
FORMAT_OUTPUTS = {
    "named_normalized": DETECTION_OUTPUT,
    "xyxy_0_1000": json.dumps([{"box_2d": [100, 250, 500, 750], "label": "cat"}]),
    "yxyx_0_1000": json.dumps([{"box_2d": [250, 100, 750, 500], "label": "cat"}]),
    "xyxy_absolute": json.dumps([{"box_2d": [80, 100, 400, 300], "label": "cat"}]),
    "xyxy_percent": json.dumps([{"box_2d": [10, 25, 50, 75], "label": "cat"}]),
    "named_0_1000": json.dumps(
        [{"label": "cat", "x_min": 100, "y_min": 250, "x_max": 500, "y_max": 750}]
    ),
}
# A phrase only that contract's prompt wording contains.
FORMAT_PROMPT_MARKERS = {
    "named_normalized": "All coordinates must be in range 0.0-1.0",
    "xyxy_0_1000": "[x_min, y_min, x_max, y_max]: the top-left and bottom-right "
    "corners as integers between 0 and 1000",
    "yxyx_0_1000": "[y_min, x_min, y_max, x_max]",
    "xyxy_absolute": f"of the {IMAGE_WIDTH}x{IMAGE_HEIGHT} pixel image",
    "xyxy_percent": "as percentages of image width and height",
    "named_0_1000": "object grounding expert",
}


def test_detection_format_covers_every_registered_box_format() -> None:
    assert set(FORMAT_OUTPUTS) == set(DETECTION_BOX_FORMATS)
    assert set(DETECTION_FORMAT_METADATA) == set(DETECTION_BOX_FORMATS)


def test_manifest_defaults_detection_format_to_legacy_contract() -> None:
    manifest = BlockManifest.model_validate(
        {**BASE_MANIFEST, "task_type": "object-detection", "classes": ["cat"]}
    )

    assert manifest.detection_format == "named_normalized"


@pytest.mark.parametrize("detection_format", sorted(DETECTION_BOX_FORMATS))
def test_manifest_accepts_every_registered_detection_format(
    detection_format: str,
) -> None:
    manifest = BlockManifest.model_validate(
        {
            **BASE_MANIFEST,
            "task_type": "object-detection",
            "classes": ["cat"],
            "detection_format": detection_format,
        }
    )

    assert manifest.detection_format == detection_format


def test_manifest_accepts_detection_format_selector() -> None:
    manifest = BlockManifest.model_validate(
        {
            **BASE_MANIFEST,
            "task_type": "object-detection",
            "classes": ["cat"],
            "detection_format": "$inputs.detection_format",
        }
    )

    assert manifest.detection_format == "$inputs.detection_format"


def test_manifest_rejects_unknown_detection_format() -> None:
    with pytest.raises(ValidationError):
        BlockManifest.model_validate(
            {
                **BASE_MANIFEST,
                "task_type": "object-detection",
                "classes": ["cat"],
                "detection_format": "polar_coordinates",
            }
        )


@pytest.mark.parametrize("detection_format", sorted(DETECTION_BOX_FORMATS))
def test_run_prompts_and_decodes_with_selected_detection_format(
    detection_format: str,
) -> None:
    result, messages = _run_capturing_prompts(
        raw_output=FORMAT_OUTPUTS[detection_format],
        task_type="object-detection",
        detection_format=detection_format,
    )

    assert FORMAT_PROMPT_MARKERS[detection_format] in _prompt_texts(messages)
    assert "cat, dog" in _prompt_texts(messages)
    assert result["error_status"] is False
    assert detection_boxes(result["predictions"]) == EXPECTED_XYXY
    assert detection_class_ids(result["predictions"]) == [0]


def test_run_without_detection_format_keeps_legacy_prompt_and_decoding() -> None:
    # given - a pre-existing workflow definition that never set the field
    result, messages = _run_capturing_prompts(
        raw_output=DETECTION_OUTPUT, task_type="object-detection"
    )

    assert messages[0]["role"] == "system"
    assert FORMAT_PROMPT_MARKERS["named_normalized"] in messages[0]["content"]
    assert detection_boxes(result["predictions"]) == EXPECTED_XYXY


def test_run_selected_format_rejects_answer_in_another_contract() -> None:
    result = _run(
        raw_output=FORMAT_OUTPUTS["named_0_1000"],
        task_type="object-detection",
        detection_format="xyxy_0_1000",
    )

    assert result["error_status"] is True
    assert result["predictions"] is None


def test_run_ignores_detection_format_for_non_detection_tasks() -> None:
    result, messages = _run_capturing_prompts(
        raw_output=CLASSIFICATION_OUTPUT,
        task_type="classification",
        detection_format="yxyx_0_1000",
    )

    assert classification_top_class(result["predictions"]) == "cat"
    assert "box_2d" not in _prompt_texts(messages)


def test_run_fails_on_unknown_detection_format_before_sending_requests() -> None:
    with patch.object(
        OpenRouterBlockV3, "execute_openrouter_batch_with_usage"
    ) as execute:
        block = OpenRouterBlockV3(model_manager=MagicMock(), api_key="ws-key")
        with pytest.raises(ValueError, match="Unknown detection box format"):
            block.run(
                images=[_image()],
                task_type="object-detection",
                prompt=None,
                output_structure=None,
                classes=["cat"],
                api_key="sk-or-v1-test",
                privacy_level="deny",
                model_id="moonshotai/kimi-k2.6",
                max_tokens=128,
                temperature=0.1,
                reasoning_effort=None,
                max_concurrent_requests=None,
                detection_format="polar_coordinates",
            )
    execute.assert_not_called()

"""Tests for the Gemma v4 block: predictions are decoded inside the block.

The v3 behavior suite lives in ``test_google_gemma_v3.py``; this file
covers the v4 delta - the ``predictions`` / ``error_status`` /
``inference_id`` outputs.
"""

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from inference.core.workflows.core_steps.common.openrouter import OpenRouterResult
from inference.core.workflows.core_steps.models.foundation.google_gemma.v4 import (
    BlockManifest,
    GoogleGemmaBlockV4,
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
    "type": "roboflow_core/google_gemma@v4",
    "name": "step",
    "images": "$inputs.image",
}


def _image() -> WorkflowImageData:
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="parent"),
        numpy_image=np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8),
    )


def _run(raw_output: str, task_type: str) -> dict:
    with patch.object(
        GoogleGemmaBlockV4,
        "execute_openrouter_batch_with_usage",
        return_value=[OpenRouterResult(content=raw_output)],
    ):
        block = GoogleGemmaBlockV4(model_manager=MagicMock(), api_key="ws-key")
        results = block.run(
            images=[_image()],
            task_type=task_type,
            prompt="describe",
            output_structure=None,
            classes=["cat", "dog"],
            api_key="sk-or-v1-test",
            privacy_level="deny",
            model_version="Gemma 4 31B - OpenRouter",
            max_tokens=128,
            temperature=0.1,
            reasoning_effort=None,
            max_concurrent_requests=None,
        )
    assert len(results) == 1
    return results[0]


def test_manifest_parses_new_type() -> None:
    manifest = BlockManifest.model_validate(
        {**BASE_MANIFEST, "task_type": "object-detection", "classes": ["cat"]}
    )

    assert manifest.type == "roboflow_core/google_gemma@v4"
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

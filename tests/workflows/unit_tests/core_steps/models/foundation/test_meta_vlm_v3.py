"""Tests for the Meta VLM v3 block (v2 + in-block prediction decoding).

The v2 behavior suite lives in ``test_meta_vlm_v2.py``; this file covers
the v3 delta: the ``predictions`` / ``error_status`` / ``inference_id``
outputs decoded inside the block from Muse's flat 0-1000 grounding format.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from inference.core.workflows.core_steps.common.openrouter import OpenRouterResult
from inference.core.workflows.core_steps.models.foundation.meta_vlm.v3 import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL_VERSION,
    DEFAULT_REASONING_EFFORT,
    BlockManifest,
    MetaVlmBlockV3,
)
from inference.core.workflows.execution_engine.entities.base import WorkflowImageData
from tests.workflows.unit_tests.core_steps._vlm_prediction_readers import (
    classification_top_class,
    classification_top_confidence,
    detection_boxes,
    detection_class_ids,
    detection_class_names,
    is_detection_prediction,
)

OPENROUTER_SEAM = (
    "inference.core.workflows.core_steps.models.foundation.meta_vlm.v3."
    "OpenRouterWorkflowBlockBase.execute_openrouter_batch_with_usage"
)

# Image is 200 wide x 100 high, so 0-1000 normalized named fields
# (100, 200, 500, 600) map onto pixels [20, 20, 100, 60].
DETECTION_OUTPUT = (
    '[{"label": "cat", "x_min": 100, "y_min": 200, "x_max": 500, "y_max": 600}]'
)
EXPECTED_XYXY = [[20.0, 20.0, 100.0, 60.0]]

CLASSIFICATION_OUTPUT = '{"class_name": "cat", "confidence": 0.9}'


def _stub_image() -> WorkflowImageData:
    return WorkflowImageData(
        parent_metadata=MagicMock(
            parent_id="root", workflow_root_ancestor_metadata=None
        ),
        numpy_image=np.zeros((100, 200, 3), dtype=np.uint8),
    )


def _base_run_kwargs(**overrides):
    kwargs = dict(
        images=[_stub_image()],
        model_version=DEFAULT_MODEL_VERSION,
        task_type="caption",
        prompt=None,
        output_structure=None,
        classes=None,
        reasoning_effort=DEFAULT_REASONING_EFFORT,
        api_key="rf_key:account",
        privacy_level="deny",
        max_tokens=DEFAULT_MAX_TOKENS,
        temperature=None,
        max_concurrent_requests=None,
    )
    kwargs.update(overrides)
    return kwargs


def _manifest(**overrides) -> BlockManifest:
    payload = {
        "type": "roboflow_core/meta_vlm@v3",
        "name": "muse",
        "images": "$inputs.image",
        "task_type": "unconstrained",
        "prompt": "describe",
    }
    payload.update(overrides)
    return BlockManifest.model_validate(payload)


def _kinds(outputs, name):
    return [k.name for o in outputs if o.name == name for k in o.kind]


def test_manifest_parses_new_type():
    manifest = _manifest()

    assert manifest.type == "roboflow_core/meta_vlm@v3"
    assert {o.name for o in BlockManifest.describe_outputs()} >= {
        "predictions",
        "error_status",
        "inference_id",
    }


def test_manifest_recommends_parser_only_for_structured_answering():
    recommended = BlockManifest.model_fields["task_type"].json_schema_extra[
        "recommended_parsers"
    ]

    assert recommended == {"structured-answering": "roboflow_core/json_parser@v1"}


def test_get_actual_outputs_narrows_predictions_kind_per_task():
    detection = _manifest(
        task_type="object-detection", prompt=None, classes=["cat"]
    ).get_actual_outputs()
    classification = _manifest(
        task_type="classification", prompt=None, classes=["cat"]
    ).get_actual_outputs()
    unconstrained = _manifest().get_actual_outputs()

    assert _kinds(detection, "predictions") == ["object_detection_prediction"]
    assert _kinds(classification, "predictions") == ["classification_prediction"]
    assert _kinds(unconstrained, "predictions") == [
        "object_detection_prediction",
        "classification_prediction",
    ]


@patch(OPENROUTER_SEAM)
def test_run_decodes_named_0_1000_detections(mock_or):
    mock_or.return_value = [
        OpenRouterResult(
            content=DETECTION_OUTPUT,
            reasoning_trace="trace",
            input_tokens=20,
            output_tokens=8,
        )
    ]
    block = MetaVlmBlockV3(model_manager=MagicMock(), api_key="rf_key")

    result = block.run(
        **_base_run_kwargs(task_type="object-detection", classes=["cat"])
    )

    predictions = result[0]["predictions"]
    assert is_detection_prediction(predictions)
    assert detection_boxes(predictions) == EXPECTED_XYXY
    assert detection_class_names(predictions) == ["cat"]
    assert detection_class_ids(predictions) == [0]
    assert result[0]["error_status"] is False
    assert result[0]["inference_id"]
    assert result[0]["thinking"] == "trace"


@patch(OPENROUTER_SEAM)
def test_run_decodes_classification(mock_or):
    mock_or.return_value = [
        OpenRouterResult(
            content=CLASSIFICATION_OUTPUT,
            reasoning_trace="",
            input_tokens=4,
            output_tokens=2,
        )
    ]
    block = MetaVlmBlockV3(model_manager=MagicMock(), api_key="rf_key")

    result = block.run(
        **_base_run_kwargs(task_type="classification", classes=["cat", "dog"])
    )

    assert classification_top_class(result[0]["predictions"]) == "cat"
    assert classification_top_confidence(result[0]["predictions"]) == pytest.approx(0.9)
    assert result[0]["error_status"] is False


@patch(OPENROUTER_SEAM)
def test_run_leaves_predictions_none_for_non_decoding_task(mock_or):
    mock_or.return_value = [
        OpenRouterResult(
            content="a cat on a mat",
            reasoning_trace="",
            input_tokens=4,
            output_tokens=2,
        )
    ]
    block = MetaVlmBlockV3(model_manager=MagicMock(), api_key="rf_key")

    result = block.run(**_base_run_kwargs(task_type="unconstrained", prompt="what?"))

    assert result[0]["predictions"] is None
    assert result[0]["error_status"] is False
    assert result[0]["inference_id"]


@patch(OPENROUTER_SEAM)
def test_run_reports_error_status_for_undecodable_detection_output(mock_or):
    mock_or.return_value = [
        OpenRouterResult(
            content="I am afraid I cannot help with that.",
            reasoning_trace="",
            input_tokens=4,
            output_tokens=2,
        )
    ]
    block = MetaVlmBlockV3(model_manager=MagicMock(), api_key="rf_key")

    result = block.run(
        **_base_run_kwargs(task_type="object-detection", classes=["cat"])
    )

    assert result[0]["predictions"] is None
    assert result[0]["error_status"] is True

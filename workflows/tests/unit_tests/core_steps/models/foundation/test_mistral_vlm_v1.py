"""Contract tests for the Mistral AI VLM v1 block.

Covers what would ship wrong boxes or dead calls if broken: manifest
defaults, the outbound OpenRouter request (slug, reasoning, token budget,
temperature, message layout), the 0-999 detection prompt actually sent, and
in-block decoding on the 0-999 grid.
"""

import itertools
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from pydantic import ValidationError
from roboflow_workflows.core_steps.common.openrouter import (
    SUPPORTED_TASK_TYPES_LIST,
    OpenRouterResult,
)
from roboflow_workflows.core_steps.models.foundation.mistral_vlm.v1 import (
    DEFAULT_MODEL_VERSION,
    DEFAULT_REASONING_EFFORT,
    REASONING_EFFORT_OPTIONS,
    BlockManifest,
    MistralVlmBlockV1,
)
from roboflow_workflows.execution_engine.entities.base import WorkflowImageData

from tests.unit_tests.core_steps._vlm_prediction_readers import (
    classification_top_class,
    classification_top_confidence,
    detection_boxes,
    detection_class_ids,
    detection_class_names,
    is_detection_prediction,
)

OPENROUTER_SEAM = (
    "roboflow_workflows.core_steps.models.foundation.mistral_vlm.v1."
    "OpenRouterWorkflowBlockBase.execute_openrouter_batch_with_usage"
)

# vlm-exam `_NORMALIZED_XYXY_PROMPT_TEMPLATE` with the grid changed from
# 0-1000 to Mistral's documented 0-999, copied here verbatim so an edit to
# the shared template fails this test.
EXPECTED_DETECTION_PROMPT = (
    "Detect all objects in this image. "
    "Output a JSON list where each entry contains the 2D bounding box "
    'in the key "box_2d" and the text label in the key "label". '
    'The "box_2d" value must be [x_min, y_min, x_max, y_max]: the '
    "top-left and bottom-right corners as integers between 0 and 999, "
    "normalized to the image width (x) and height (y). "
    "Return only the JSON list, with no extra text. "
    "Only use these labels: cat, dog"
)

# Image is 1998 wide x 999 high, so every 0-999 coordinate maps onto exactly
# 2x (x axis) / 1x (y axis) pixels. A 0-1000 decoder would land the far
# corner at 1996x998 instead of the full image.
IMAGE_WIDTH = 1998
IMAGE_HEIGHT = 999
DETECTION_OUTPUT = '[{"box_2d": [100, 250, 999, 999], "label": "cat"}]'
DETECTION_OUTPUT_BBOX_ALIAS = '[{"bbox_2d": [100, 250, 999, 999], "label": "cat"}]'
EXPECTED_XYXY = [[200.0, 250.0, 1998.0, 999.0]]

CLASSIFICATION_OUTPUT = '{"class_name": "cat", "confidence": 0.9}'

# Effort names OpenRouter accepts for other models; Mistral Large 4 maps all
# of them onto the same budget as `high`, so the block refuses them.
UNEXPOSED_REASONING_LEVELS = ["minimal", "low", "medium", "xhigh"]

# Minimum extra inputs each task needs to pass manifest validation and build
# a prompt; `output_structure` / `classes` double as the decode inputs.
TASK_INPUTS = {
    "unconstrained": {"prompt": "describe"},
    "visual-question-answering": {"prompt": "what is it?"},
    "structured-answering": {"output_structure": {"animal": "name"}},
    "classification": {"classes": ["cat", "dog"]},
    "multi-label-classification": {"classes": ["cat", "dog"]},
    "object-detection": {"classes": ["cat", "dog"]},
    "ocr": {},
    "caption": {},
    "detailed-caption": {},
}
DECODING_TASKS = {"object-detection", "classification", "multi-label-classification"}


def _stub_image() -> WorkflowImageData:
    return WorkflowImageData(
        parent_metadata=MagicMock(
            parent_id="root", workflow_root_ancestor_metadata=None
        ),
        numpy_image=np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8),
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
        max_tokens=None,
        temperature=None,
        max_concurrent_requests=None,
    )
    kwargs.update(overrides)
    return kwargs


def _manifest(**overrides) -> BlockManifest:
    payload = {
        "type": "roboflow_core/mistral_vlm@v1",
        "name": "mistral",
        "images": "$inputs.image",
        "task_type": "caption",
    }
    payload.update(overrides)
    return BlockManifest.model_validate(payload)


def _result(content: str) -> list:
    return [
        OpenRouterResult(
            content=content, reasoning_trace="", input_tokens=20, output_tokens=8
        )
    ]


def _kinds(outputs, name):
    return [k.name for o in outputs if o.name == name for k in o.kind]


def test_manifest_defaults():
    manifest = _manifest()

    assert manifest.model_version == "Mistral Large 4"
    assert manifest.reasoning_effort == "none"
    assert manifest.max_tokens is None
    assert manifest.temperature is None
    assert manifest.api_key == "rf_key:account"
    assert manifest.privacy_level == "deny"
    assert {o.name for o in BlockManifest.describe_outputs()} >= {
        "output",
        "thinking",
        "input_tokens",
        "output_tokens",
        "predictions",
        "error_status",
        "inference_id",
    }


@pytest.mark.parametrize("reasoning_effort", REASONING_EFFORT_OPTIONS)
def test_manifest_accepts_every_exposed_reasoning_level(reasoning_effort):
    assert _manifest(reasoning_effort=reasoning_effort).reasoning_effort == (
        reasoning_effort
    )


@pytest.mark.parametrize("reasoning_effort", UNEXPOSED_REASONING_LEVELS)
def test_manifest_rejects_reasoning_levels_mistral_does_not_expose(reasoning_effort):
    with pytest.raises(ValidationError):
        _manifest(reasoning_effort=reasoning_effort)


def test_manifest_accepts_selector_fed_model_version_without_level_check():
    manifest = _manifest(model_version="$inputs.model", reasoning_effort="high")

    assert manifest.model_version == "$inputs.model"
    assert manifest.reasoning_effort == "high"


def test_manifest_rejects_unknown_model_version():
    with pytest.raises(ValidationError):
        _manifest(model_version="Pixtral Large")


@pytest.mark.parametrize("max_tokens", [0, 1, -5])
def test_manifest_rejects_max_tokens_at_or_below_one(max_tokens):
    with pytest.raises(ValidationError):
        _manifest(max_tokens=max_tokens)


@pytest.mark.parametrize("max_tokens", [None, 2, 4096])
def test_manifest_accepts_unset_or_positive_max_tokens(max_tokens):
    assert _manifest(max_tokens=max_tokens).max_tokens == max_tokens


@pytest.mark.parametrize("temperature", [-0.1, 2.1])
def test_manifest_rejects_temperature_outside_range(temperature):
    with pytest.raises(ValidationError):
        _manifest(temperature=temperature)


def test_manifest_recommends_parser_only_for_structured_answering():
    recommended = BlockManifest.model_fields["task_type"].json_schema_extra[
        "recommended_parsers"
    ]

    assert recommended == {"structured-answering": "roboflow_core/json_parser@v1"}


def test_get_actual_outputs_narrows_predictions_kind_per_task():
    detection = _manifest(
        task_type="object-detection", classes=["cat"]
    ).get_actual_outputs()
    classification = _manifest(
        task_type="classification", classes=["cat"]
    ).get_actual_outputs()
    caption = _manifest().get_actual_outputs()

    assert _kinds(detection, "predictions") == ["object_detection_prediction"]
    assert _kinds(classification, "predictions") == ["classification_prediction"]
    assert _kinds(caption, "predictions") == [
        "object_detection_prediction",
        "classification_prediction",
    ]


@patch(OPENROUTER_SEAM)
def test_run_sends_vlm_exam_request_contract(mock_or):
    mock_or.return_value = _result(DETECTION_OUTPUT)
    block = MistralVlmBlockV1(model_manager=MagicMock(), api_key="rf_key")

    block.run(**_base_run_kwargs(task_type="object-detection", classes=["cat", "dog"]))

    kwargs = mock_or.call_args.kwargs
    assert kwargs["model"] == "mistralai/mistral-large-4-0"
    assert kwargs["reasoning"] == {"enabled": False}
    assert kwargs["max_tokens"] is None
    assert kwargs["temperature"] is None
    assert kwargs["privacy_level"] == "deny"
    messages = kwargs["prompts"][0]
    assert [message["role"] for message in messages] == ["user"]
    content = messages[0]["content"]
    assert content[0]["type"] == "image_url"
    assert content[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert content[1] == {"type": "text", "text": EXPECTED_DETECTION_PROMPT}


EXPECTED_REASONING_CONFIG = {"none": {"enabled": False}, "high": {"effort": "high"}}


@pytest.mark.parametrize(
    "reasoning_effort,max_tokens,temperature",
    list(itertools.product(REASONING_EFFORT_OPTIONS, [None, 4096], [None, 0.2])),
)
@patch(OPENROUTER_SEAM)
def test_run_forwards_every_generation_setting_combination(
    mock_or, reasoning_effort, max_tokens, temperature
):
    mock_or.return_value = _result("answer")
    block = MistralVlmBlockV1(model_manager=MagicMock(), api_key="rf_key")

    block.run(
        **_base_run_kwargs(
            reasoning_effort=reasoning_effort,
            max_tokens=max_tokens,
            temperature=temperature,
            api_key="sk-or-user-key",
            privacy_level="allow",
            max_concurrent_requests=3,
        )
    )

    kwargs = mock_or.call_args.kwargs
    assert kwargs["reasoning"] == EXPECTED_REASONING_CONFIG[reasoning_effort]
    assert kwargs["max_tokens"] == max_tokens
    assert kwargs["temperature"] == temperature
    assert kwargs["openrouter_api_key"] == "sk-or-user-key"
    assert kwargs["privacy_level"] == "allow"
    assert kwargs["max_concurrent_requests"] == 3


@pytest.mark.parametrize(
    "task_type,reasoning_effort",
    list(itertools.product(SUPPORTED_TASK_TYPES_LIST, REASONING_EFFORT_OPTIONS)),
)
@patch(OPENROUTER_SEAM)
def test_run_keeps_message_layout_and_decoding_per_task(
    mock_or, task_type, reasoning_effort
):
    mock_or.return_value = _result(
        {
            "object-detection": DETECTION_OUTPUT,
            "classification": CLASSIFICATION_OUTPUT,
            "multi-label-classification": (
                '{"predicted_classes": [{"class": "cat", "confidence": 0.8}]}'
            ),
        }.get(task_type, "free text answer")
    )
    block = MistralVlmBlockV1(model_manager=MagicMock(), api_key="rf_key")

    result = block.run(
        **_base_run_kwargs(
            task_type=task_type,
            reasoning_effort=reasoning_effort,
            **TASK_INPUTS[task_type],
        )
    )

    kwargs = mock_or.call_args.kwargs
    assert kwargs["reasoning"] == EXPECTED_REASONING_CONFIG[reasoning_effort]
    messages = kwargs["prompts"][0]
    assert [message["role"] for message in messages] == ["user"]
    assert [part["type"] for part in messages[0]["content"]] == ["image_url", "text"]
    assert result[0]["error_status"] is False
    if task_type in DECODING_TASKS:
        assert result[0]["predictions"] is not None
    else:
        assert result[0]["predictions"] is None


@pytest.mark.parametrize("task_type", SUPPORTED_TASK_TYPES_LIST)
def test_manifest_validates_each_task_with_its_required_inputs(task_type):
    manifest = _manifest(task_type=task_type, **TASK_INPUTS[task_type])

    assert manifest.task_type == task_type


@pytest.mark.parametrize("reasoning_effort", UNEXPOSED_REASONING_LEVELS)
def test_run_rejects_reasoning_levels_mistral_does_not_expose(reasoning_effort):
    block = MistralVlmBlockV1(model_manager=MagicMock(), api_key="rf_key")

    with pytest.raises(ValueError, match="supports reasoning_effort values"):
        block.run(**_base_run_kwargs(reasoning_effort=reasoning_effort))


@pytest.mark.parametrize("raw_output", [DETECTION_OUTPUT, DETECTION_OUTPUT_BBOX_ALIAS])
@patch(OPENROUTER_SEAM)
def test_run_decodes_detections_on_the_0_999_grid(mock_or, raw_output):
    mock_or.return_value = _result(raw_output)
    block = MistralVlmBlockV1(model_manager=MagicMock(), api_key="rf_key")

    result = block.run(
        **_base_run_kwargs(task_type="object-detection", classes=["cat", "dog"])
    )

    predictions = result[0]["predictions"]
    assert is_detection_prediction(predictions)
    assert detection_boxes(predictions) == EXPECTED_XYXY
    assert detection_class_names(predictions) == ["cat"]
    assert detection_class_ids(predictions) == [0]
    assert result[0]["error_status"] is False
    assert result[0]["inference_id"]
    assert result[0]["output"] == raw_output
    assert result[0]["classes"] == ["cat", "dog"]
    assert result[0]["input_tokens"] == 20
    assert result[0]["output_tokens"] == 8


@patch(OPENROUTER_SEAM)
def test_run_keeps_unknown_label_with_negative_class_id(mock_or):
    mock_or.return_value = _result(
        '[{"box_2d": [100, 250, 999, 999], "label": "giraffe"}]'
    )
    block = MistralVlmBlockV1(model_manager=MagicMock(), api_key="rf_key")

    result = block.run(
        **_base_run_kwargs(task_type="object-detection", classes=["cat"])
    )

    assert detection_class_names(result[0]["predictions"]) == ["giraffe"]
    assert detection_class_ids(result[0]["predictions"]) == [-1]
    assert result[0]["error_status"] is False


@patch(OPENROUTER_SEAM)
def test_run_decodes_classification(mock_or):
    mock_or.return_value = _result(CLASSIFICATION_OUTPUT)
    block = MistralVlmBlockV1(model_manager=MagicMock(), api_key="rf_key")

    result = block.run(
        **_base_run_kwargs(task_type="classification", classes=["cat", "dog"])
    )

    assert classification_top_class(result[0]["predictions"]) == "cat"
    assert classification_top_confidence(result[0]["predictions"]) == pytest.approx(0.9)
    assert result[0]["error_status"] is False


@patch(OPENROUTER_SEAM)
def test_run_leaves_predictions_none_for_non_decoding_task(mock_or):
    mock_or.return_value = _result("a cat on a mat")
    block = MistralVlmBlockV1(model_manager=MagicMock(), api_key="rf_key")

    result = block.run(**_base_run_kwargs(task_type="caption"))

    assert result[0]["predictions"] is None
    assert result[0]["error_status"] is False
    assert result[0]["output"] == "a cat on a mat"


@patch(OPENROUTER_SEAM)
def test_run_reports_error_status_for_undecodable_detection_output(mock_or):
    mock_or.return_value = _result("I am afraid I cannot help with that.")
    block = MistralVlmBlockV1(model_manager=MagicMock(), api_key="rf_key")

    result = block.run(
        **_base_run_kwargs(task_type="object-detection", classes=["cat"])
    )

    assert result[0]["predictions"] is None
    assert result[0]["error_status"] is True

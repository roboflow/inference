"""Tests for the OpenAI v7 block (in-block VLM decoding).

The v5/v6 behavior suites live in ``test_openai_v5.py`` and
``test_openai_v6.py``; this file covers the v7 delta: ``predictions`` /
``error_status`` / ``inference_id`` outputs decoded inside the block, for
each of the three per-model detection prompt styles.
"""

import json
from typing import List, Optional
from unittest.mock import patch

import numpy as np
import pytest
from roboflow_workflows.core_steps.models.foundation.openai.v7 import (
    INSTANCE_SEGMENTATION_DEFAULT_MODEL,
    STRUCTURED_INSTANCE_SEGMENTATION_OUTPUT_FORMAT,
    BlockManifest,
    OpenAIBlockV7,
    detection_upload_dimensions,
    encode_image_for_task,
    get_detection_box_format,
    prepare_instance_segmentation_prompt,
    prepare_object_detection_prompt,
)
from roboflow_workflows.execution_engine.entities.base import (
    Batch,
    ImageParentMetadata,
    WorkflowImageData,
)
from roboflow_workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    CLASSIFICATION_PREDICTION_KIND,
    INFERENCE_ID_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
)

from tests.unit_tests.core_steps._vlm_prediction_readers import (
    classification_inference_id,
    classification_top_class,
    classification_top_confidence,
    detection_boxes,
    detection_class_ids,
    detection_class_names,
    detection_count,
    detection_inference_ids,
    detection_masks,
    is_detection_prediction,
)

EXECUTE_REQUESTS_SEAM = (
    "roboflow_workflows.core_steps.models.foundation.openai.v7"
    ".execute_openai_requests"
)

# A 3200x1600 image uploads at 2048x1024 for the absolute formats, so both
# payloads below map onto exactly [800, 400, 1600, 1200] original pixels.
DETECTION_IMAGE_WIDTH = 3200
DETECTION_IMAGE_HEIGHT = 1600
PLAIN_ABSOLUTE_OUTPUT = json.dumps([{"box_2d": [512, 256, 1024, 768], "label": "cat"}])
STRUCTURED_ABSOLUTE_OUTPUT = json.dumps(
    {"detections": [{"box_2d": [512, 256, 1024, 768], "label": "cat"}]}
)
NORMALIZED_LEGACY_OUTPUT = json.dumps(
    {
        "detections": [
            {
                "x_min": 0.25,
                "y_min": 0.25,
                "x_max": 0.5,
                "y_max": 0.75,
                "class_name": "cat",
                "confidence": 0.7,
            }
        ]
    }
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
    model_version: str = "gpt-5.1",
    classes: Optional[List[str]] = None,
    prompt: Optional[str] = None,
) -> dict:
    block = OpenAIBlockV7(api_key="rf-key")
    with patch(EXECUTE_REQUESTS_SEAM) as mock_execute:
        mock_execute.return_value = [(raw_output, 11, 3)]
        results = block.run(
            images=Batch(content=[image], indices=[(0,)]),
            task_type=task_type,
            prompt=prompt,
            output_structure=None,
            classes=classes,
            model_version=model_version,
            reasoning_effort=None,
            image_detail="auto",
            max_tokens=None,
            temperature=None,
            max_concurrent_requests=None,
            api_key="sk-test",
        )
    assert len(results) == 1
    return results[0]


def test_manifest_parsing_for_new_block_type() -> None:
    # given
    raw_manifest = {
        "type": "roboflow_core/open_ai@v7",
        "name": "open_ai",
        "images": "$inputs.image",
        "task_type": "object-detection",
        "classes": ["cat", "dog"],
        "api_key": "$inputs.api_key",
    }

    # when
    result = BlockManifest.model_validate(raw_manifest)

    # then
    assert result.type == "roboflow_core/open_ai@v7"
    assert result.task_type == "object-detection"
    assert result.classes == ["cat", "dog"]


def test_describe_outputs_declares_prediction_outputs() -> None:
    # when
    outputs = {output.name: output.kind for output in BlockManifest.describe_outputs()}

    # then
    assert outputs["predictions"] == [
        OBJECT_DETECTION_PREDICTION_KIND,
        RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
        INSTANCE_SEGMENTATION_PREDICTION_KIND,
        CLASSIFICATION_PREDICTION_KIND,
    ]
    assert outputs["error_status"] == [BOOLEAN_KIND]
    assert outputs["inference_id"] == [INFERENCE_ID_KIND]
    assert "output" in outputs


@pytest.mark.parametrize(
    "task_type, expected_kind",
    [
        ("object-detection", [OBJECT_DETECTION_PREDICTION_KIND]),
        (
            "instance-segmentation",
            [
                RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
                INSTANCE_SEGMENTATION_PREDICTION_KIND,
            ],
        ),
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
            "type": "roboflow_core/open_ai@v7",
            "name": "open_ai",
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
            "type": "roboflow_core/open_ai@v7",
            "name": "open_ai",
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
        RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
        INSTANCE_SEGMENTATION_PREDICTION_KIND,
        CLASSIFICATION_PREDICTION_KIND,
    ]
    assert [output.name for output in BlockManifest.describe_outputs()] == list(outputs)


@pytest.mark.parametrize(
    "model_version, expected_format",
    [
        ("gpt-6-astra", "xyxy_absolute"),
        ("gpt-6-sol", "xyxy_absolute"),
        ("gpt-6-luna", "xyxy_absolute"),
        ("gpt-5.6-sol", "xyxy_absolute"),
        ("gpt-4o", "xyxy_absolute"),
        ("gpt-5.1", "named_normalized"),
        ("gpt-5-nano", "named_normalized"),
        ("some-future-model", "xyxy_absolute"),
    ],
)
def test_detection_box_format_follows_prompt_style(
    model_version: str, expected_format: str
) -> None:
    assert get_detection_box_format(model_version) == expected_format


def test_plain_absolute_prompt_keeps_pinned_wording() -> None:
    # when
    prompt = prepare_object_detection_prompt(
        base64_image="base64-image",
        classes=["cat", "dog"],
        image_width=2048,
        image_height=1024,
        model_version="gpt-4o",
    )

    # then
    prompt_text = prompt["input"][0]["content"][1]["text"]
    assert prompt_text == (
        "Detect all objects in this image. "
        "Output a JSON list where each entry contains the 2D bounding box "
        'in the key "box_2d" and the text label in the key "label". '
        'The "box_2d" value must be [x_min, y_min, x_max, y_max]: the '
        "top-left and bottom-right corners in absolute pixel coordinates "
        "of the 2048x1024 pixel image. "
        "Return only the JSON list, with no extra text. "
        "Only use these labels: cat, dog"
    )
    assert "text" not in prompt


@pytest.mark.parametrize("model_version", ["gpt-6-sol", "gpt-6-luna"])
def test_gpt6_sol_and_luna_reuse_astra_detection_contract(model_version: str) -> None:
    shared_kwargs = {
        "base64_image": "base64-image",
        "classes": ["cat", "dog"],
        "image_width": 2048,
        "image_height": 1024,
    }

    astra_prompt = prepare_object_detection_prompt(
        **shared_kwargs, model_version="gpt-6-astra"
    )
    model_prompt = prepare_object_detection_prompt(
        **shared_kwargs, model_version=model_version
    )

    assert model_prompt == astra_prompt


def test_structured_absolute_prompt_keeps_json_schema_wrapper() -> None:
    # when
    prompt = prepare_object_detection_prompt(
        base64_image="base64-image",
        classes=["cat", "dog"],
        image_width=2048,
        image_height=1024,
        model_version="gpt-5.6-sol",
    )

    # then
    prompt_text = prompt["input"][0]["content"][1]["text"]
    assert 'Output JSON with the key "detections"' in prompt_text
    assert "of the 2048x1024 pixel image" in prompt_text
    assert prompt["text"]["format"]["name"] == "detections"


def test_detection_upload_dimensions_only_apply_to_absolute_formats() -> None:
    # given
    image = _build_image(width=DETECTION_IMAGE_WIDTH, height=DETECTION_IMAGE_HEIGHT)

    # when / then
    assert detection_upload_dimensions(
        image=image, task_type="object-detection", box_format="xyxy_absolute"
    ) == (2048, 1024)
    assert detection_upload_dimensions(
        image=image, task_type="object-detection", box_format="named_normalized"
    ) == (None, None)
    assert detection_upload_dimensions(
        image=image, task_type="unconstrained", box_format="xyxy_absolute"
    ) == (None, None)
    # Segmentation uploads the original image, whatever the box format.
    assert detection_upload_dimensions(
        image=image, task_type="instance-segmentation", box_format="named_normalized"
    ) == (DETECTION_IMAGE_WIDTH, DETECTION_IMAGE_HEIGHT)


@pytest.mark.parametrize(
    "model_version, raw_output",
    [
        ("gpt-4o", PLAIN_ABSOLUTE_OUTPUT),
        ("gpt-5.6-sol", STRUCTURED_ABSOLUTE_OUTPUT),
        ("gpt-5.1", NORMALIZED_LEGACY_OUTPUT),
    ],
)
def test_run_decodes_object_detection_into_original_image_pixels(
    model_version: str, raw_output: str
) -> None:
    # given
    image = _build_image(width=DETECTION_IMAGE_WIDTH, height=DETECTION_IMAGE_HEIGHT)

    # when
    result = _run_block(
        task_type="object-detection",
        raw_output=raw_output,
        image=image,
        model_version=model_version,
        classes=["cat", "dog"],
    )

    # then
    assert result["error_status"] is False
    assert result["output"] == raw_output
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
        model_version="gpt-4o",
        classes=["cat", "dog"],
    )

    # then
    assert result["error_status"] is True
    assert result["predictions"] is None


# ---------------------------------------------------------------------------
# Instance segmentation
# ---------------------------------------------------------------------------

# Polygon vertices are absolute pixels of the ORIGINAL image (no downscale on
# this path), so the extent box is the polygon's own bounds.
SEGMENTATION_OUTPUT = json.dumps(
    {
        "segmentations": [
            {"label": "cat", "polygon": [800, 400, 1600, 400, 1600, 1200, 800, 1200]}
        ]
    }
)


def _segmentation_manifest(**overrides: object) -> BlockManifest:
    raw_manifest = {
        "type": "roboflow_core/open_ai@v7",
        "name": "open_ai",
        "images": "$inputs.image",
        "task_type": "instance-segmentation",
        "classes": ["cat", "dog"],
    }
    raw_manifest.update(overrides)
    return BlockManifest.model_validate(raw_manifest)


def test_segmentation_task_defaults_model_to_astra_when_unset() -> None:
    manifest = _segmentation_manifest()

    assert (
        manifest.model_version == INSTANCE_SEGMENTATION_DEFAULT_MODEL == "gpt-6-astra"
    )


@pytest.mark.parametrize("model_version", ["gpt-5.6-sol", "$inputs.openai_model"])
def test_segmentation_task_keeps_explicit_model(model_version: str) -> None:
    manifest = _segmentation_manifest(model_version=model_version)

    assert manifest.model_version == model_version


def test_other_tasks_keep_block_default_model() -> None:
    manifest = _segmentation_manifest(task_type="object-detection")

    assert manifest.model_version == "gpt-5.1"


def test_segmentation_task_requires_classes() -> None:
    with pytest.raises(ValueError):
        _segmentation_manifest(classes=None)


def test_instance_segmentation_prompt_enforces_polygon_schema() -> None:
    prompt = prepare_instance_segmentation_prompt(
        base64_image="abc",
        classes=["cat", "dog"],
        image_width=640,
        image_height=480,
        image_detail="high",
    )

    content = prompt["input"][0]["content"]
    assert content[0] == {
        "type": "input_image",
        "image_url": "data:image/jpeg;base64,abc",
        "detail": "high",
    }
    assert content[1]["type"] == "input_text"
    assert "640x480" in content[1]["text"]
    assert content[1]["text"].endswith("Only use these labels: cat, dog")
    assert prompt["text"] is STRUCTURED_INSTANCE_SEGMENTATION_OUTPUT_FORMAT
    schema = prompt["text"]["format"]["schema"]
    assert prompt["text"]["format"]["strict"] is True
    assert schema["required"] == ["segmentations"]
    entry = schema["properties"]["segmentations"]["items"]
    assert entry["required"] == ["label", "polygon"]
    assert entry["properties"]["polygon"] == {
        "type": "array",
        "items": {"type": "integer"},
    }
    assert "instructions" not in prompt


def test_encode_image_for_task_keeps_original_resolution_for_segmentation() -> None:
    image = np.zeros((DETECTION_IMAGE_HEIGHT, DETECTION_IMAGE_WIDTH, 3), dtype=np.uint8)

    _, width, height = encode_image_for_task(image, task_type="instance-segmentation")
    _, det_width, det_height = encode_image_for_task(
        image, task_type="object-detection"
    )

    assert (width, height) == (DETECTION_IMAGE_WIDTH, DETECTION_IMAGE_HEIGHT)
    assert (det_width, det_height) == (2048, 1024)


@pytest.mark.parametrize("model_version", ["gpt-6-astra", "gpt-5.1", "gpt-4o"])
def test_run_decodes_instance_segmentation_into_masks(model_version: str) -> None:
    # given
    image = _build_image(width=DETECTION_IMAGE_WIDTH, height=DETECTION_IMAGE_HEIGHT)

    # when
    result = _run_block(
        task_type="instance-segmentation",
        raw_output=SEGMENTATION_OUTPUT,
        image=image,
        model_version=model_version,
        classes=["cat", "dog"],
    )

    # then
    assert result["error_status"] is False
    assert result["output"] == SEGMENTATION_OUTPUT
    predictions = result["predictions"]
    assert is_detection_prediction(predictions)
    assert detection_count(predictions) == 1
    assert detection_boxes(predictions)[0] == EXPECTED_XYXY
    assert detection_class_ids(predictions) == [0]
    assert detection_class_names(predictions) == ["cat"]
    assert detection_inference_ids(predictions) == [result["inference_id"]]
    masks = detection_masks(predictions)
    assert len(masks) == 1
    assert masks[0].shape == (DETECTION_IMAGE_HEIGHT, DETECTION_IMAGE_WIDTH)
    assert masks[0][800, 1200]
    assert not masks[0][100, 100]


def test_run_reports_error_status_for_unparsable_segmentation_output() -> None:
    # given
    image = _build_image(width=DETECTION_IMAGE_WIDTH, height=DETECTION_IMAGE_HEIGHT)

    # when
    result = _run_block(
        task_type="instance-segmentation",
        raw_output="I am sorry, I cannot help with that.",
        image=image,
        model_version="gpt-6-astra",
        classes=["cat", "dog"],
    )

    # then
    assert result["error_status"] is True
    assert result["predictions"] is None

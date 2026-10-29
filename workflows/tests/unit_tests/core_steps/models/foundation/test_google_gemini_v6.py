"""Tests for the Google Gemini v6 block (in-block VLM decoding).

The v1-v4 behavior suite lives in ``test_google_gemini.py`` and token usage
in ``test_google_gemini_v5.py``; this file covers the v6 delta:
``predictions`` / ``error_status`` / ``inference_id`` outputs decoded inside
the block from Gemini's native ``yxyx_0_1000`` boxes.
"""

import base64
import json
from typing import List, Optional
from unittest.mock import patch

import cv2
import numpy as np
import pytest
from roboflow_workflows.core_steps.models.foundation.google_gemini.v6 import (
    DETECTION_BOX_FORMAT,
    BlockManifest,
    GoogleGeminiBlockV6,
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
    is_detection_prediction,
)

EXECUTE_REQUESTS_SEAM = (
    "roboflow_workflows.core_steps.models.foundation.google_gemini.v6"
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
    model_version: str = "gemini-2.5-pro",
) -> dict:
    block = GoogleGeminiBlockV6(api_key="rf-key")
    with patch(EXECUTE_REQUESTS_SEAM) as mock_execute:
        mock_execute.return_value = [(raw_output, 11, 3)]
        results = block.run(
            images=Batch(content=[image], indices=[(0,)]),
            task_type=task_type,
            prompt=prompt,
            output_structure=None,
            classes=classes,
            model_version=model_version,
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


@pytest.mark.parametrize("model", ["gemini-2.5-pro", "gemini-3.8-flash"])
def test_segmentation_rejects_other_gemini_models(model: str) -> None:
    with pytest.raises(ValueError, match="only for gemini-4-argon"):
        BlockManifest.model_validate(
            {
                "type": "roboflow_core/google_gemini@v6",
                "name": "segment",
                "images": "$inputs.image",
                "task_type": "instance-segmentation",
                "model_version": model,
                "classes": ["ring"],
            }
        )


def test_argon_segmentation_manifest_and_request() -> None:
    import base64
    import io

    from PIL import Image
    from roboflow_workflows.core_steps.models.foundation.google_gemini.v6 import (
        run_gemini_prompting,
    )

    manifest = BlockManifest.model_validate(
        {
            "type": "roboflow_core/google_gemini@v6",
            "name": "segment",
            "images": "$inputs.image",
            "task_type": "instance-segmentation",
            "model_version": "gemini-4-argon",
            "classes": ["ring"],
        }
    )
    outputs = {output.name: output.kind for output in manifest.get_actual_outputs()}
    assert RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND in outputs["predictions"]
    image = _build_image(width=2400, height=1200)
    with patch(EXECUTE_REQUESTS_SEAM, return_value=[]) as execute:
        run_gemini_prompting(
            roboflow_api_key=None,
            platform_client=None,
            images=[image.to_inference_format()],
            task_type="instance-segmentation",
            prompt=None,
            output_structure=None,
            classes=["ring"],
            google_api_key="key",
            model_version="gemini-4-argon",
            max_tokens=32768,
            temperature=0,
            thinking_level="low",
            google_code_execution=False,
            max_concurrent_requests=1,
        )
    payload = execute.call_args.kwargs["gemini_prompts"][0]
    parts = payload["contents"]["parts"]
    assert parts[0]["inline_data"]["mime_type"] == "image/png"
    encoded = base64.b64decode(parts[0]["inline_data"]["data"])
    assert Image.open(io.BytesIO(encoded)).size == (1600, 800)
    assert "[y,x]" in parts[1]["text"]
    component = payload["generationConfig"]["response_schema"]["items"]["properties"][
        "mask"
    ]["items"]
    assert set(component["required"]) == {"outer", "holes"}
    assert payload["generationConfig"]["temperature"] == 0


def test_argon_components_holes_and_yx_scaling() -> None:
    from pycocotools import mask as masks
    from roboflow_workflows.core_steps.common.vlm_decoding.argon_segmentation import (
        decode_argon_segmentations,
    )

    raw = [
        {
            "label": "ring",
            "box_2d": [0, 0, 1000, 1000],
            "mask": [
                {
                    "outer": [[100, 100], [100, 600], [800, 600], [800, 100]],
                    "holes": [[[300, 200], [300, 400], [600, 400], [600, 200]]],
                },
                {
                    "outer": [[400, 250], [400, 300], [500, 300], [500, 250]],
                    "holes": [],
                },
                {
                    "outer": [[100, 800], [100, 900], [200, 900], [200, 800]],
                    "holes": [],
                },
            ],
        }
    ]
    with patch(
        "roboflow_workflows.core_steps.common.vlm_decoding.argon_segmentation.to_tensor_native_predictions",
        side_effect=lambda **kw: kw["predictions"],
    ):
        error, result = decode_argon_segmentations(
            json.dumps(raw),
            image=_build_image(width=200, height=100),
            classes=["ring"],
            inference_id="test",
        )
    assert not error and len(result) == 1 and result.mask is None
    mask = masks.decode(result.data["rle_mask"][0])
    assert mask.shape == (100, 200)
    assert mask[15, 30] and mask[15, 170]
    assert not mask[35, 45] and mask[45, 55]
    assert not mask[15, 140]
    assert result.xyxy[0].tolist() == [20, 10, 181, 81]


@pytest.mark.parametrize(
    "raw",
    [
        "not json",
        '[{"label":"ring","mask":[]}]',
        '[{"label":"ring","mask":[{"outer":[[0,0],[0,1001],[500,500]],"holes":[]}]}]',
    ],
)
def test_argon_invalid_masks_are_errors(raw: str) -> None:
    from roboflow_workflows.core_steps.common.vlm_decoding.argon_segmentation import (
        decode_argon_segmentations,
    )

    error, result = decode_argon_segmentations(
        raw, image=_build_image(), classes=["ring"], inference_id="test"
    )
    assert error and result is None


def test_argon_segmentation_block_and_empty_output() -> None:
    raw = json.dumps(
        [
            {
                "label": "ring",
                "box_2d": [0, 0, 1000, 1000],
                "mask": [
                    {
                        "outer": [[100, 100], [100, 600], [800, 600], [800, 100]],
                        "holes": [],
                    }
                ],
            }
        ]
    )
    result = _run_block(
        task_type="instance-segmentation",
        raw_output=raw,
        image=_build_image(width=200, height=100),
        classes=["ring"],
        model_version="gemini-4-argon",
    )
    assert result["error_status"] is False
    assert detection_count(result["predictions"]) == 1
    np.testing.assert_allclose(
        detection_boxes(result["predictions"]), [[20, 10, 121, 81]]
    )
    empty = _run_block(
        task_type="instance-segmentation",
        raw_output="[]",
        image=_build_image(),
        classes=["ring"],
        model_version="gemini-4-argon",
    )
    assert empty["error_status"] is False
    assert detection_count(empty["predictions"]) == 0


def test_segmentation_runtime_rejects_resolved_other_model() -> None:
    with pytest.raises(ValueError, match="only for gemini-4-argon"):
        _run_block(
            task_type="instance-segmentation",
            raw_output="[]",
            image=_build_image(),
            classes=["ring"],
        )


def _reference_example(box_class: str = "cat") -> dict:
    image = np.full((100, 200, 3), 255, dtype=np.uint8)
    _, encoded = cv2.imencode(".png", image)
    return {
        "image": {
            "type": "base64",
            "value": base64.b64encode(encoded.tobytes()).decode("ascii"),
        },
        "boxes": [{"x": 50, "y": 50, "width": 40, "height": 20, "class": box_class}],
    }


def _run_detection_with_mock(reference_examples: Optional[list]) -> tuple:
    block = GoogleGeminiBlockV6(api_key="rf-key")
    with patch(EXECUTE_REQUESTS_SEAM) as mock_execute:
        mock_execute.return_value = [(DETECTION_OUTPUT, 11, 3)]
        results = block.run(
            images=Batch(content=[_build_image()], indices=[(0,)]),
            task_type="object-detection",
            prompt=None,
            output_structure=None,
            classes=["cat", "dog"],
            model_version="gemini-4-argon",
            max_tokens=16384,
            temperature=None,
            thinking_level="low",
            google_code_execution=None,
            max_concurrent_requests=None,
            api_key="rf_key:account",
            reference_examples=reference_examples,
        )
    parts = mock_execute.call_args[1]["gemini_prompts"][0]["contents"]["parts"]
    return results[0], parts


@pytest.mark.parametrize(
    "reference_examples", [[_reference_example()], "$inputs.reference_examples"]
)
def test_manifest_accepts_reference_examples(reference_examples) -> None:
    # when
    result = BlockManifest.model_validate(
        {
            "type": "roboflow_core/google_gemini@v6",
            "name": "gemini",
            "images": "$inputs.image",
            "task_type": "object-detection",
            "classes": ["cat", "dog"],
            "reference_examples": reference_examples,
        }
    )

    # then
    assert result.reference_examples == reference_examples


def test_manifest_rejects_reference_examples_for_other_tasks() -> None:
    with pytest.raises(ValueError, match="only supported for the object-detection"):
        BlockManifest.model_validate(
            {
                "type": "roboflow_core/google_gemini@v6",
                "name": "gemini",
                "images": "$inputs.image",
                "task_type": "classification",
                "classes": ["cat", "dog"],
                "reference_examples": [_reference_example()],
            }
        )


def test_manifest_rejects_reference_examples_with_unknown_labels() -> None:
    with pytest.raises(ValueError, match=r"does not have: \['bird'\]"):
        BlockManifest.model_validate(
            {
                "type": "roboflow_core/google_gemini@v6",
                "name": "gemini",
                "images": "$inputs.image",
                "task_type": "object-detection",
                "classes": ["cat", "dog"],
                "reference_examples": [_reference_example(box_class="bird")],
            }
        )


def test_run_sends_boxed_reference_images_before_the_target() -> None:
    # when
    result, parts = _run_detection_with_mock([_reference_example()])

    # then
    assert parts[0]["text"].startswith("Reference examples: other images")
    assert parts[1] == {"text": "Reference image 1:"}
    assert parts[2]["inline_data"]["mime_type"] == "image/jpeg"
    assert parts[3] == {"text": "Target image:"}
    assert parts[4]["inline_data"]["mime_type"] == "image/jpeg"
    assert parts[5]["text"].startswith("Detect all objects in the target image. ")
    assert parts[5]["text"].endswith(
        "The reference images are not part of the task: return detections for "
        "the target image only."
    )
    assert len(parts) == 6
    assert detection_boxes(result["predictions"])[0] == EXPECTED_XYXY


@pytest.mark.parametrize("reference_examples", [None, []])
def test_run_without_reference_examples_keeps_the_request(reference_examples) -> None:
    # when
    _, parts = _run_detection_with_mock(reference_examples)

    # then
    assert len(parts) == 2
    assert "inline_data" in parts[0]
    assert parts[1]["text"].startswith("Detect all objects in this image. ")


def test_run_rejects_resolved_reference_examples_for_other_tasks() -> None:
    with pytest.raises(ValueError, match="only supported for the object-detection"):
        block = GoogleGeminiBlockV6(api_key="rf-key")
        with patch(EXECUTE_REQUESTS_SEAM):
            block.run(
                images=Batch(content=[_build_image()], indices=[(0,)]),
                task_type="classification",
                prompt=None,
                output_structure=None,
                classes=["cat", "dog"],
                model_version="gemini-4-argon",
                max_tokens=None,
                temperature=None,
                thinking_level=None,
                google_code_execution=None,
                max_concurrent_requests=None,
                reference_examples=[_reference_example()],
            )

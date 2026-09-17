"""Parity of the shared VLM decoder's tensor-native carriers.

Under ``ENABLE_TENSOR_DATA_REPRESENTATION`` the kind serializers registered in
``core_steps/loader.py`` and the tensor consumers (visualization blocks read
``.image_metadata``) require the native ``inference_models`` carriers, so
``decode_vlm_output`` converts before returning (see
``vlm_decoding/tensor_native.py``). These tests pin that conversion against the
DEPRECATED tensor formatter blocks it replaces - the same raw string and image
go through ``VLMAsDetectorBlockV2`` / ``VLMAsClassifierBlockV2`` of the
``*_tensor.py`` modules, and the decoded carriers must agree field for field.

Runs only with the flag on: the conversion is a no-op otherwise, and the
serializers/consumers the parity protects are only swapped in that mode.
"""

import json

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("inference_models")

from inference.core.env import ENABLE_TENSOR_DATA_REPRESENTATION
from inference.core.workflows.core_steps.common.vlm_decoding import decode_vlm_output
from inference.core.workflows.core_steps.formatters.vlm_as_classifier.v2_tensor import (
    VLMAsClassifierBlockV2,
)
from inference.core.workflows.core_steps.formatters.vlm_as_detector.v2_tensor import (
    VLMAsDetectorBlockV2,
)
from inference.core.workflows.execution_engine.constants import (
    CLASS_NAME_KEY,
    CLASS_NAMES_KEY,
    CLASSIFICATION_STYLE_FORMATTER,
    CLASSIFICATION_STYLE_KEY,
    IMAGE_DIMENSIONS_KEY,
    INFERENCE_ID_KEY,
    PARENT_COORDINATES_KEY,
    PARENT_DIMENSIONS_KEY,
    PARENT_ID_KEY,
    PREDICTION_TYPE_KEY,
    ROOT_PARENT_COORDINATES_KEY,
    ROOT_PARENT_DIMENSIONS_KEY,
    ROOT_PARENT_ID_KEY,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)
from inference_models.models.base.classification import (
    ClassificationPrediction,
    MultiLabelClassificationPrediction,
)
from inference_models.models.base.object_detection import Detections

pytestmark = pytest.mark.skipif(
    not ENABLE_TENSOR_DATA_REPRESENTATION,
    reason="tensor-native carriers; runs only with "
    "ENABLE_TENSOR_DATA_REPRESENTATION=True",
)

IMAGE_WIDTH = 800
IMAGE_HEIGHT = 400
CLASSES = ["cat", "dog"]

# The OpenRouter/OpenAI normalized-named contract - what the deprecated
# formatter parses for `model_type="anthropic-claude"` and what the shared
# decoder parses as `named_normalized`.
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
SINGLE_LABEL_OUTPUT = json.dumps({"class_name": "dog", "confidence": 0.75})
MULTI_LABEL_OUTPUT = json.dumps(
    {
        "predicted_classes": [
            {"class": "dog", "confidence": 0.8},
            {"class": "cat", "confidence": 0.3},
        ]
    }
)


def _build_image() -> WorkflowImageData:
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="parent"),
        numpy_image=np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8),
    )


def _decoded_detections() -> Detections:
    error_status, predictions = decode_vlm_output(
        task_type="object-detection",
        raw_output=DETECTION_OUTPUT,
        image=_build_image(),
        classes=CLASSES,
        inference_id="inference-id",
        box_format="named_normalized",
    )
    assert error_status is False
    return predictions


def _formatter_detections() -> Detections:
    result = VLMAsDetectorBlockV2().run(
        image=_build_image(),
        vlm_output=DETECTION_OUTPUT,
        classes=CLASSES,
        model_type="anthropic-claude",
        task_type="object-detection",
    )
    assert result["error_status"] is False
    return result["predictions"]


def _decoded_classification(raw_output: str, task_type: str):
    error_status, predictions = decode_vlm_output(
        task_type=task_type,
        raw_output=raw_output,
        image=_build_image(),
        classes=CLASSES,
        inference_id="inference-id",
    )
    assert error_status is False
    return predictions


def _formatter_classification(raw_output: str):
    result = VLMAsClassifierBlockV2().run(
        image=_build_image(),
        vlm_output=raw_output,
        classes=CLASSES,
    )
    assert result["error_status"] is False
    return result["predictions"]


def test_object_detection_returns_the_native_detection_type() -> None:
    predictions = _decoded_detections()

    assert isinstance(predictions, Detections)
    assert isinstance(_formatter_detections(), Detections)


def test_object_detection_matches_the_deprecated_tensor_formatter() -> None:
    decoded = _decoded_detections()
    formatted = _formatter_detections()

    assert decoded.xyxy.cpu().tolist() == formatted.xyxy.cpu().tolist()
    assert decoded.class_id.cpu().tolist() == formatted.class_id.cpu().tolist()
    assert decoded.confidence.cpu().tolist() == formatted.confidence.cpu().tolist()
    assert decoded.image_metadata[CLASS_NAMES_KEY] == (
        formatted.image_metadata[CLASS_NAMES_KEY]
    )
    for key in (
        PREDICTION_TYPE_KEY,
        IMAGE_DIMENSIONS_KEY,
        PARENT_ID_KEY,
        PARENT_COORDINATES_KEY,
        PARENT_DIMENSIONS_KEY,
        ROOT_PARENT_ID_KEY,
        ROOT_PARENT_COORDINATES_KEY,
        ROOT_PARENT_DIMENSIONS_KEY,
    ):
        assert decoded.image_metadata[key] == formatted.image_metadata[key], key
    # The formatter mints its own inference id; the decoder is handed one.
    assert decoded.image_metadata[INFERENCE_ID_KEY] == "inference-id"
    assert formatted.image_metadata[INFERENCE_ID_KEY]
    assert [entry[CLASS_NAME_KEY] for entry in decoded.bboxes_metadata] == [
        entry[CLASS_NAME_KEY] for entry in formatted.bboxes_metadata
    ]


def test_object_detection_boxes_land_on_original_image_pixels() -> None:
    decoded = _decoded_detections()

    assert decoded.xyxy.cpu().tolist() == [[80.0, 100.0, 400.0, 300.0]]
    assert decoded.image_metadata[IMAGE_DIMENSIONS_KEY] == [IMAGE_HEIGHT, IMAGE_WIDTH]


def test_empty_object_detection_returns_native_detections() -> None:
    error_status, predictions = decode_vlm_output(
        task_type="object-detection",
        raw_output="[]",
        image=_build_image(),
        classes=CLASSES,
        inference_id="inference-id",
        box_format="named_normalized",
    )

    assert error_status is False
    assert isinstance(predictions, Detections)
    assert predictions.xyxy.shape == (0, 4)
    assert predictions.bboxes_metadata is None
    assert predictions.image_metadata[CLASS_NAMES_KEY] == {0: "cat", 1: "dog"}


def test_single_label_classification_matches_the_tensor_formatter() -> None:
    decoded = _decoded_classification(
        raw_output=SINGLE_LABEL_OUTPUT, task_type="classification"
    )
    formatted = _formatter_classification(raw_output=SINGLE_LABEL_OUTPUT)

    assert isinstance(decoded, ClassificationPrediction)
    assert isinstance(formatted, ClassificationPrediction)
    assert decoded.class_id.cpu().tolist() == formatted.class_id.cpu().tolist()
    assert decoded.confidence.cpu().tolist() == formatted.confidence.cpu().tolist()
    decoded_metadata = decoded.images_metadata[0]
    formatted_metadata = formatted.images_metadata[0]
    for key in (
        CLASS_NAMES_KEY,
        CLASSIFICATION_STYLE_KEY,
        PREDICTION_TYPE_KEY,
        IMAGE_DIMENSIONS_KEY,
        PARENT_ID_KEY,
    ):
        assert decoded_metadata[key] == formatted_metadata[key], key
    assert decoded_metadata[CLASSIFICATION_STYLE_KEY] == CLASSIFICATION_STYLE_FORMATTER
    assert decoded_metadata[INFERENCE_ID_KEY] == "inference-id"


def test_multi_label_classification_matches_the_tensor_formatter() -> None:
    decoded = _decoded_classification(
        raw_output=MULTI_LABEL_OUTPUT, task_type="multi-label-classification"
    )
    formatted = _formatter_classification(raw_output=MULTI_LABEL_OUTPUT)

    assert isinstance(decoded, MultiLabelClassificationPrediction)
    assert isinstance(formatted, MultiLabelClassificationPrediction)
    assert decoded.class_ids.cpu().tolist() == formatted.class_ids.cpu().tolist()
    assert decoded.confidence.cpu().tolist() == formatted.confidence.cpu().tolist()
    for key in (
        CLASS_NAMES_KEY,
        CLASSIFICATION_STYLE_KEY,
        PREDICTION_TYPE_KEY,
        IMAGE_DIMENSIONS_KEY,
        PARENT_ID_KEY,
    ):
        assert decoded.image_metadata[key] == formatted.image_metadata[key], key


def test_out_of_list_class_is_densified_like_the_tensor_formatter() -> None:
    # The numpy dict marks an unrequested class with `class_id == -1`; the
    # native carriers index the confidence vector BY class id, so the formatter
    # (and therefore the decoder) appends it at `len(classes)` instead.
    raw_output = json.dumps({"class_name": "bird", "confidence": 0.9})

    decoded = _decoded_classification(raw_output=raw_output, task_type="classification")
    formatted = _formatter_classification(raw_output=raw_output)

    assert decoded.class_id.cpu().tolist() == [2]
    assert decoded.class_id.cpu().tolist() == formatted.class_id.cpu().tolist()
    assert decoded.confidence.cpu().tolist() == formatted.confidence.cpu().tolist()
    assert decoded.images_metadata[0][CLASS_NAMES_KEY] == (
        formatted.images_metadata[0][CLASS_NAMES_KEY]
    )


def test_serialized_classification_is_identical_to_the_tensor_formatter() -> None:
    # The serialized `predictions` payload is what leaves the workflow, so pin
    # it: the "D4 / formatter shape" must survive the conversion.
    from inference.core.workflows.core_steps.common.serializers_tensor import (
        serialise_native_classification,
    )

    decoded = _decoded_classification(
        raw_output=SINGLE_LABEL_OUTPUT, task_type="classification"
    )
    formatted = _formatter_classification(raw_output=SINGLE_LABEL_OUTPUT)

    serialized_decoded = serialise_native_classification(decoded)
    serialized_formatted = serialise_native_classification(formatted)
    serialized_decoded.pop(INFERENCE_ID_KEY)
    serialized_formatted.pop(INFERENCE_ID_KEY)

    assert serialized_decoded == serialized_formatted
    assert list(serialized_decoded) == ["image", "predictions", "top", "confidence"] + [
        PARENT_ID_KEY
    ]


def test_empty_object_detection_carries_the_block_inference_id() -> None:
    error_status, predictions = decode_vlm_output(
        task_type="object-detection",
        raw_output="[]",
        image=_build_image(),
        classes=CLASSES,
        inference_id="inference-id",
        box_format="named_normalized",
    )

    assert error_status is False
    assert predictions.image_metadata[INFERENCE_ID_KEY] == "inference-id"


def test_duplicate_classes_report_error_status_instead_of_raising() -> None:
    # given - duplicate class names collapse to fewer dense ids than the
    # confidence vector expects; the native carrier cannot be built
    error_status, predictions = decode_vlm_output(
        task_type="classification",
        raw_output='{"class_name": "cat", "confidence": 0.9}',
        image=_build_image(),
        classes=["cat", "cat", "dog"],
        inference_id="inference-id",
    )

    assert error_status is True
    assert predictions is None

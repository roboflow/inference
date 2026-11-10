"""Classification, image and frame UQL operations, and API path handling."""

import base64
from datetime import datetime

import cv2
import numpy as np
import pytest
import torch
from roboflow_workflows.execution_engine.entities.base import VideoMetadata
from roboflow_workflows.v2.blocks.query import (
    QueryDefinitionError,
    QueryRuntimeError,
    compile_condition,
    compile_operations,
)
from roboflow_workflows.v2.data.frames import frame_of
from roboflow_workflows.v2.data.image import ImageData

from inference_models.models.base.classification import (
    ClassificationPrediction,
    MultiLabelClassificationPrediction,
)
from inference_models.models.base.object_detection import Detections

NAMES = {"class_names": {0: "cat", 1: "dog", 2: "bird"}}


def single_label(batch=1, metadata=NAMES):
    return ClassificationPrediction(
        class_id=torch.tensor([1] * batch),
        confidence=torch.tensor([[0.125, 0.75, 0.125]] * batch),
        images_metadata=None if metadata is None else [metadata] * batch,
    )


def multi_label(class_ids=(0, 2)):
    return MultiLabelClassificationPrediction(
        class_ids=torch.tensor(class_ids, dtype=torch.int64),
        confidence=torch.tensor([0.875, 0.25, 0.625]),
        image_metadata=NAMES,
    )


def frame_image(video_metadata=None) -> ImageData:
    pixels = torch.zeros((3, 40, 60), dtype=torch.uint8)
    pixels[0] = 255  # pure red in RGB
    return ImageData.from_tensor(
        pixels, image_id="frame", video_metadata=video_metadata
    )


def video(frame_number=31, fps=30.0):
    return VideoMetadata(
        video_identifier="camera-1",
        frame_number=frame_number,
        frame_timestamp=datetime(2026, 10, 12, 8, 30, 1),
        fps=fps,
    )


def _apply(operation, value):
    return compile_operations([operation]).apply(value)


def _classification(name):
    return {"type": "ClassificationPropertyExtract", "property_name": name}


@pytest.mark.parametrize(
    "name, expected",
    [
        ("top_class", "dog"),
        ("top_class_confidence", 0.75),
        ("top_class_confidence_single", 0.75),
        ("all_classes", ["cat", "dog", "bird"]),
        ("all_confidences", [0.125, 0.75, 0.125]),
    ],
)
def test_single_label_classification(name, expected) -> None:
    assert _apply(_classification(name), single_label()) == expected


@pytest.mark.parametrize(
    "class_ids, name, expected",
    [
        ((0, 2), "top_class", ["cat", "bird"]),
        ((0, 2), "top_class_confidence", [0.875, 0.625]),
        ((0, 2), "top_class_confidence_single", 0.875),
        ((), "top_class", []),
        ((), "top_class_confidence_single", 0.0),
        ((0, 2), "all_classes", ["cat", "dog", "bird"]),
        ((0, 2), "all_confidences", [0.875, 0.25, 0.625]),
    ],
)
def test_multi_label_classification(class_ids, name, expected) -> None:
    assert _apply(_classification(name), multi_label(class_ids)) == expected


@pytest.mark.parametrize(
    "value, detail",
    [
        (single_label(batch=2), "batch of 2"),
        (single_label(metadata=None), "class_names"),
        (single_label(metadata={"class_names": {0: "cat"}}), "class id 1"),
        ({"top": "dog"}, "expected ClassificationPrediction"),
    ],
)
def test_classification_failures(value, detail) -> None:
    # when
    with pytest.raises(QueryRuntimeError) as error:
        _apply(_classification("top_class"), value)

    # then
    assert detail in error.value.detail


@pytest.mark.parametrize(
    "name, expected",
    [("size", 2400), ("height", 40), ("width", 60), ("aspect_ratio", 1.5)],
)
def test_image_properties_read_the_shape(name, expected) -> None:
    assert (
        _apply({"type": "ExtractImageProperty", "property_name": name}, frame_image())
        == expected
    )


def test_jpeg_keeps_rgb_colour_order_and_quality() -> None:
    # when
    best = _apply(
        {"type": "ConvertImageToJPEG", "compression_level": 100}, frame_image()
    )
    small = _apply(
        {"type": "ConvertImageToJPEG", "compression_level": 5}, frame_image()
    )

    # then
    decoded_bgr = cv2.imdecode(np.frombuffer(best, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert best[:2] == b"\xff\xd8"
    assert decoded_bgr.shape == (40, 60, 3)
    assert decoded_bgr[20, 30].tolist() == pytest.approx([0, 0, 254], abs=2)
    assert len(small) < len(best)


def test_base64_is_the_default_quality_jpeg() -> None:
    # when
    text = _apply({"type": "ConvertImageToBase64"}, frame_image())
    jpeg = _apply({"type": "ConvertImageToJPEG"}, frame_image())

    # then
    assert base64.b64decode(text) == jpeg


def test_image_operations_reject_other_values() -> None:
    # when
    with pytest.raises(QueryRuntimeError) as error:
        _apply({"type": "ConvertImageToJPEG"}, np.zeros((4, 4, 3), dtype=np.uint8))

    # then
    assert "expected ImageData" in error.value.detail


@pytest.mark.parametrize(
    "name, expected",
    [
        ("frame_number", 31),
        ("frame_timestamp", datetime(2026, 10, 12, 8, 30, 1)),
        ("seconds_since_start", 1.0),
    ],
)
def test_frame_metadata_of_image_frame_and_prediction(name, expected) -> None:
    # given
    image = frame_image(video_metadata=video())
    prediction = Detections(
        xyxy=torch.zeros((0, 4)),
        class_id=torch.zeros(0, dtype=torch.int64),
        confidence=torch.zeros(0),
        image_metadata=image.prediction_metadata(),
    )
    operation = {"type": "ExtractFrameMetadata", "property_name": name}

    # when
    values = [
        _apply(operation, value) for value in (image, frame_of(prediction), prediction)
    ]

    # then
    assert values == [expected] * 3


@pytest.mark.parametrize(
    "image, detail",
    [
        (frame_image(), "no video metadata"),
        (frame_image(video_metadata=video(fps=None)), "positive fps"),
    ],
)
def test_frame_metadata_does_not_invent_values(image, detail) -> None:
    # when
    with pytest.raises(QueryRuntimeError) as error:
        _apply(
            {"type": "ExtractFrameMetadata", "property_name": "seconds_since_start"},
            image,
        )

    # then
    assert detail in error.value.detail


def test_compile_operations_requires_a_list() -> None:
    # when
    with pytest.raises(QueryDefinitionError) as error:
        compile_operations({"type": "ToString"}, path="$.steps[1].operations")

    # then
    assert error.value.path == "$.steps[1].operations"
    assert "must be a list" in error.value.detail


def test_compile_condition_prefixes_error_paths() -> None:
    # given
    definition = {
        "type": "StatementGroup",
        "statements": [
            {
                "type": "UnaryStatement",
                "operand": {"type": "DynamicOperand", "operand_name": "missing"},
                "operator": {"type": "Exists"},
            },
            {"type": "UnaryStatement", "operand": {"type": "StaticOperand"}},
        ],
    }

    # when
    with pytest.raises(QueryDefinitionError) as invalid:
        compile_condition(definition, path="$.condition_statement")
    with pytest.raises(QueryDefinitionError) as undeclared:
        compile_condition(
            {**definition, "statements": definition["statements"][:1]},
            path="$.condition_statement",
            declared_names=set(),
        )

    # then
    assert invalid.value.path == "$.condition_statement.statements[1].operand.value"
    assert undeclared.value.path == "$.condition_statement.statements[0].operand"

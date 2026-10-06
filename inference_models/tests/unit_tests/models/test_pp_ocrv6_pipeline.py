from __future__ import annotations

import numpy as np
import pytest
import torch

from inference_models.models.base.object_detection import Detections
from inference_models.models.pp_ocrv6.pp_ocrv6_pipeline import (
    PPOCRv6Pipeline,
    PPOCRv6PipelineResult,
)


def _detections(boxes: list, polygons: list) -> Detections:
    return Detections(
        xyxy=torch.tensor(boxes, dtype=torch.float32),
        class_id=torch.zeros((len(boxes),), dtype=torch.int64),
        confidence=torch.tensor([1.0] * len(boxes), dtype=torch.float32),
        bboxes_metadata=[{"polygon": polygon} for polygon in polygons],
    )


def _polygon(box: list) -> list:
    x_min, y_min, x_max, y_max = box
    return [
        [x_min, y_min],
        [x_max, y_min],
        [x_max, y_max],
        [x_min, y_max],
    ]


class _FakeDetector:
    """Detector stub returning canned detections, echoing polygon-derived text."""

    def __init__(self, detections: list):
        self._detections = detections
        self.calls = []

    def __call__(self, images, **kwargs):
        self.calls.append((images, kwargs))
        return self._detections


class _FakeRecognizer:
    """Recognizer stub returning one string per crop, encoding the crop size."""

    def __init__(self):
        self.crops = None

    def __call__(self, crops, **kwargs):
        self.crops = crops
        return [f"{crop.shape[1]}x{crop.shape[0]}" for crop in crops]


def test_infer_sorts_detections_into_reading_order() -> None:
    # Two lines out of order: bottom-left, top-right, top-left, bottom-right.
    boxes = [
        [0, 100, 40, 130],
        [60, 0, 100, 30],
        [0, 0, 40, 30],
        [60, 100, 100, 130],
    ]
    detector = _FakeDetector([_detections(boxes, [_polygon(box) for box in boxes])])
    recognizer = _FakeRecognizer()
    pipeline = PPOCRv6Pipeline(det_model=detector, rec_model=recognizer)
    image = np.zeros((200, 200, 3), dtype=np.uint8)

    result = pipeline(image)[0]

    reordered = result.detections.xyxy.tolist()
    assert reordered == [
        [0, 0, 40, 30],
        [60, 0, 100, 30],
        [0, 100, 40, 130],
        [60, 100, 100, 130],
    ]
    assert len(result.line_texts) == 4
    # Fragments on the same visual line join with spaces, lines with newlines.
    assert result.text == "40x30 40x30\n40x30 40x30"


def test_infer_joins_same_row_fragments_with_space() -> None:
    # One visual row split by the detector into two horizontally adjacent boxes
    # (mirrors "This is a test image for " + "OCR." from the e2e image).
    boxes = [
        [0, 0, 638, 90],
        [613, 0, 788, 72],
    ]
    detector = _FakeDetector([_detections(boxes, [_polygon(box) for box in boxes])])
    recognizer = _FakeRecognizer()
    pipeline = PPOCRv6Pipeline(det_model=detector, rec_model=recognizer)
    image = np.zeros((200, 800, 3), dtype=np.uint8)

    result = pipeline(image)[0]

    assert len(result.line_texts) == 2
    assert "\n" not in result.text
    assert result.text == "638x90 175x72"


def test_infer_perspective_crops_from_polygons() -> None:
    # A 120x30 box -> crop of width 120, height 30 (not rotated: ratio < 1.5).
    box = [10, 20, 130, 50]
    detector = _FakeDetector([_detections([box], [_polygon(box)])])
    recognizer = _FakeRecognizer()
    pipeline = PPOCRv6Pipeline(det_model=detector, rec_model=recognizer)
    image = np.zeros((200, 200, 3), dtype=np.uint8)

    result = pipeline(image)[0]

    assert len(recognizer.crops) == 1
    crop = recognizer.crops[0]
    assert crop.shape == (30, 120, 3)
    assert result.line_texts == ["120x30"]


def test_infer_rotates_tall_crops() -> None:
    # A 30x120 (portrait) box has height/width >= 1.5 and is rotated upright.
    box = [10, 20, 40, 140]
    detector = _FakeDetector([_detections([box], [_polygon(box)])])
    recognizer = _FakeRecognizer()
    pipeline = PPOCRv6Pipeline(det_model=detector, rec_model=recognizer)
    image = np.zeros((200, 200, 3), dtype=np.uint8)

    pipeline(image)

    crop = recognizer.crops[0]
    assert crop.shape[0] == 30 and crop.shape[1] == 120


def test_infer_empty_detections_yields_empty_result() -> None:
    detector = _FakeDetector([_detections([], [])])
    recognizer = _FakeRecognizer()
    pipeline = PPOCRv6Pipeline(det_model=detector, rec_model=recognizer)
    image = np.zeros((64, 64, 3), dtype=np.uint8)

    result = pipeline(image)[0]

    assert isinstance(result, PPOCRv6PipelineResult)
    assert result.text == ""
    assert result.line_texts == []
    assert len(result.detections.xyxy) == 0
    assert result.detections.bboxes_metadata == []
    assert recognizer.crops is None  # recognizer never invoked


def test_infer_returns_one_result_per_image() -> None:
    box = [0, 0, 40, 30]
    detector = _FakeDetector(
        [
            _detections([box], [_polygon(box)]),
            _detections([], []),
        ]
    )
    recognizer = _FakeRecognizer()
    pipeline = PPOCRv6Pipeline(det_model=detector, rec_model=recognizer)
    images = [
        np.zeros((64, 64, 3), dtype=np.uint8),
        np.zeros((64, 64, 3), dtype=np.uint8),
    ]

    results = pipeline(images)

    assert len(results) == 2
    assert results[0].line_texts == ["40x30"]
    assert results[1].line_texts == []


def test_infer_detect_only_populates_boxes_without_recognition() -> None:
    boxes = [
        [0, 100, 40, 130],
        [0, 0, 40, 30],
    ]
    detector = _FakeDetector([_detections(boxes, [_polygon(box) for box in boxes])])
    recognizer = _FakeRecognizer()
    pipeline = PPOCRv6Pipeline(det_model=detector, rec_model=None)
    image = np.zeros((200, 200, 3), dtype=np.uint8)

    result = pipeline(image)[0]

    assert result.text == ""
    assert result.line_texts == []
    # Detections are populated and reading-order sorted (top line first).
    assert result.detections.xyxy.tolist() == [
        [0, 0, 40, 30],
        [0, 100, 40, 130],
    ]
    assert recognizer.crops is None  # recognition never invoked


def test_infer_recognize_only_runs_on_full_images() -> None:
    class _RecordingRecognizer:
        def __init__(self):
            self.images = None

        def __call__(self, images, **kwargs):
            self.images = images
            return [f"line-{i}" for i in range(len(images))]

    detector = _FakeDetector([])  # would raise if indexed; must never be called
    recognizer = _RecordingRecognizer()
    pipeline = PPOCRv6Pipeline(det_model=None, rec_model=recognizer)
    images = [
        np.zeros((32, 100, 3), dtype=np.uint8),
        np.zeros((32, 120, 3), dtype=np.uint8),
    ]

    results = pipeline(images)

    assert detector.calls == []  # detection never invoked
    assert recognizer.images is images  # full images forwarded, not crops
    assert len(results) == 2
    assert results[0].text == "line-0"
    assert results[0].line_texts == ["line-0"]
    assert results[0].detections is None  # detection stage never ran
    assert results[1].text == "line-1"
    assert results[1].line_texts == ["line-1"]


def test_construct_without_any_model_raises() -> None:
    with pytest.raises(ValueError):
        PPOCRv6Pipeline(det_model=None, rec_model=None)


def test_with_models_builds_pipeline_from_model_list() -> None:
    det_model = object()
    rec_model = object()

    pipeline = PPOCRv6Pipeline.with_models([det_model, rec_model])

    assert pipeline._det_model is det_model
    assert pipeline._rec_model is rec_model


def test_with_models_rejects_wrong_model_count() -> None:
    with pytest.raises(ValueError):
        PPOCRv6Pipeline.with_models([object()])


def test_pipeline_registered_for_auto_model_pipeline() -> None:
    from inference_models.model_pipelines.auto_loaders.pipelines_registry import (
        DEFAULT_PIPELINES_PARAMETERS,
        resolve_pipeline_class,
    )

    assert resolve_pipeline_class("pp-ocrv6") is PPOCRv6Pipeline
    assert DEFAULT_PIPELINES_PARAMETERS["pp-ocrv6"] == [
        "pp-ocrv6-det/small",
        "pp-ocrv6-rec/small",
    ]

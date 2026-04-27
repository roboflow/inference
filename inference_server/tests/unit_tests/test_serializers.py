"""Tests for inference_model_manager.serializers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import orjson
import torch

from inference_server.serializers import serialize_json

# ---------------------------------------------------------------------------
# Fake dataclasses matching inference_models types
# ---------------------------------------------------------------------------


@dataclass
class FakeDetections:
    xyxy: torch.Tensor
    class_id: torch.Tensor
    confidence: torch.Tensor
    image_metadata: Optional[dict] = None
    bboxes_metadata: Optional[List[dict]] = None


@dataclass
class FakeClassificationPrediction:
    class_id: torch.Tensor
    confidence: torch.Tensor
    images_metadata: Optional[List[dict]] = None


@dataclass
class FakeInstanceDetections:
    xyxy: torch.Tensor
    class_id: torch.Tensor
    confidence: torch.Tensor
    mask: torch.Tensor
    image_metadata: Optional[dict] = None


@dataclass
class FakeSemanticSegResult:
    segmentation_map: torch.Tensor
    confidence: torch.Tensor
    image_metadata: Optional[dict] = None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSerializeDetections:
    def test_basic_detections(self):
        det = FakeDetections(
            xyxy=torch.tensor([[10.0, 20.0, 30.0, 40.0], [50.0, 60.0, 70.0, 80.0]]),
            class_id=torch.tensor([0, 1]),
            confidence=torch.tensor([0.9, 0.8]),
        )
        raw = serialize_json(det)
        obj = orjson.loads(raw)

        assert len(obj["xyxy"]) == 2
        assert len(obj["class_id"]) == 2
        assert len(obj["confidence"]) == 2
        assert obj["class_id"] == [0, 1]
        assert obj["xyxy"][0] == [10.0, 20.0, 30.0, 40.0]

    def test_empty_detections(self):
        det = FakeDetections(
            xyxy=torch.zeros(0, 4),
            class_id=torch.zeros(0, dtype=torch.long),
            confidence=torch.zeros(0),
        )
        raw = serialize_json(det)
        obj = orjson.loads(raw)

        assert obj["xyxy"] == []
        assert obj["class_id"] == []
        assert obj["confidence"] == []

    def test_with_metadata(self):
        det = FakeDetections(
            xyxy=torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
            class_id=torch.tensor([5]),
            confidence=torch.tensor([0.99]),
            image_metadata={"width": 640, "height": 480},
        )
        raw = serialize_json(det)
        obj = orjson.loads(raw)

        assert obj["image_metadata"]["width"] == 640

    def test_none_metadata_omitted(self):
        det = FakeDetections(
            xyxy=torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
            class_id=torch.tensor([0]),
            confidence=torch.tensor([0.5]),
        )
        raw = serialize_json(det)
        obj = orjson.loads(raw)

        assert obj["image_metadata"] is None
        assert obj["bboxes_metadata"] is None


class TestSerializeList:
    def test_list_of_detections(self):
        dets = [
            FakeDetections(
                xyxy=torch.tensor([[10.0, 20.0, 30.0, 40.0]]),
                class_id=torch.tensor([0]),
                confidence=torch.tensor([0.9]),
            ),
            FakeDetections(
                xyxy=torch.tensor([[50.0, 60.0, 70.0, 80.0]]),
                class_id=torch.tensor([1]),
                confidence=torch.tensor([0.8]),
            ),
        ]
        raw = serialize_json(dets)
        obj = orjson.loads(raw)

        assert len(obj) == 2
        assert obj[0]["class_id"] == [0]
        assert obj[1]["class_id"] == [1]


class TestSerializeClassification:
    def test_basic(self):
        pred = FakeClassificationPrediction(
            class_id=torch.tensor([3, 7]),
            confidence=torch.tensor([0.85, 0.12]),
        )
        raw = serialize_json(pred)
        obj = orjson.loads(raw)

        assert obj["class_id"] == [3, 7]
        assert len(obj["confidence"]) == 2


class TestSerializeInstanceSegmentation:
    def test_with_masks(self):
        det = FakeInstanceDetections(
            xyxy=torch.tensor([[10.0, 20.0, 30.0, 40.0]]),
            class_id=torch.tensor([0]),
            confidence=torch.tensor([0.9]),
            mask=torch.ones(1, 28, 28, dtype=torch.uint8),
        )
        raw = serialize_json(det)
        obj = orjson.loads(raw)

        assert obj["class_id"] == [0]
        assert len(obj["mask"]) == 1
        assert len(obj["mask"][0]) == 28


class TestSerializeSemanticSegmentation:
    def test_basic(self):
        result = FakeSemanticSegResult(
            segmentation_map=torch.zeros(480, 640, dtype=torch.long),
            confidence=torch.ones(480, 640),
        )
        raw = serialize_json(result)
        obj = orjson.loads(raw)

        assert len(obj["segmentation_map"]) == 480
        assert len(obj["segmentation_map"][0]) == 640


class TestSerializeRawTensor:
    def test_tensor(self):
        t = torch.rand(3, 256)
        raw = serialize_json(t)
        obj = orjson.loads(raw)

        assert len(obj) == 3
        assert len(obj[0]) == 256

    def test_numpy(self):
        a = np.random.rand(10, 4).astype(np.float32)
        raw = serialize_json(a)
        obj = orjson.loads(raw)

        assert len(obj) == 10


class TestSerializeTuple:
    def test_ocr_style_output(self):
        texts = ["hello", "world"]
        dets = FakeDetections(
            xyxy=torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]),
            class_id=torch.tensor([0, 0]),
            confidence=torch.tensor([0.9, 0.8]),
        )
        raw = serialize_json((texts, [dets]))
        obj = orjson.loads(raw)

        assert obj[0] == ["hello", "world"]
        assert len(obj[1]) == 1
        assert obj[1][0]["xyxy"] == [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]


class TestSerializePrimitives:
    def test_string(self):
        assert orjson.loads(serialize_json("hello")) == "hello"

    def test_none(self):
        assert orjson.loads(serialize_json(None)) is None

    def test_dict(self):
        d = {"foo": 1, "bar": [1, 2, 3]}
        assert orjson.loads(serialize_json(d)) == d

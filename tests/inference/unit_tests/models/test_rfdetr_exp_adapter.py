import sys
import types
from dataclasses import dataclass
from typing import List

import numpy as np
import pytest

from inference.models.rfdetr.rfdetr_exp import RFDetrExperimentalModel


@dataclass
class _FakeDetections:
    xyxy: np.ndarray
    class_id: np.ndarray
    confidence: np.ndarray


class _FakeModel:
    def __init__(self, class_names: List[str]):
        self.class_names = class_names

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        # class_names provided via kwargs for flexibility in tests
        class_names = kwargs.get("class_names", ["obj"])
        return cls(class_names=class_names)

    def __call__(self, images):
        # Return a single detection per image as default
        results = []
        for _ in images:
            xyxy = np.array([[10.0, 20.0, 30.0, 60.0]], dtype=np.float32)
            cls = np.array([0], dtype=np.int64)
            conf = np.array([0.9], dtype=np.float32)
            results.append(_FakeDetections(xyxy=xyxy, class_id=cls, confidence=conf))
        return results


def _install_fake_inference_exp(class_names: List[str] = ("obj",)):
    fake_module = types.ModuleType("inference_exp")

    class _AutoModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            kwargs = dict(kwargs)
            kwargs["class_names"] = list(class_names)
            return _FakeModel.from_pretrained(*args, **kwargs)

    fake_module.AutoModel = _AutoModel
    sys.modules["inference_exp"] = fake_module


def test_adapter_basic_inference(monkeypatch):
    _install_fake_inference_exp(class_names=["person"])  # ensure import works

    img = (np.random.rand(100, 120, 3) * 255).astype(np.uint8)
    model = RFDetrExperimentalModel("rfdetr-base", api_key=None)

    res = model.infer(img, confidence=0.5, iou_threshold=1.0)
    assert res is not None
    assert len(res.predictions) == 1
    pred = res.predictions[0]
    # Expected center/size from xyxy [10,20,30,60]
    assert np.isclose(pred.x, (10.0 + 30.0) / 2.0)
    assert np.isclose(pred.y, (20.0 + 60.0) / 2.0)
    assert np.isclose(pred.width, 20.0)
    assert np.isclose(pred.height, 40.0)
    assert pred.class_id == 0
    assert pred.class_name == "person"


def test_adapter_confidence_filter(monkeypatch):
    # Return two detections: one above, one below threshold
    class _TwoDetectionsModel(_FakeModel):
        def __call__(self, images):
            results = []
            for _ in images:
                xyxy = np.array(
                    [
                        [10.0, 20.0, 30.0, 60.0],  # high conf
                        [5.0, 5.0, 15.0, 15.0],  # low conf
                    ],
                    dtype=np.float32,
                )
                cls = np.array([0, 0], dtype=np.int64)
                conf = np.array([0.9, 0.3], dtype=np.float32)
                results.append(
                    _FakeDetections(xyxy=xyxy, class_id=cls, confidence=conf)
                )
            return results

    # Install fake module with custom AutoModel
    fake_module = types.ModuleType("inference_exp")

    class _AutoModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            m = _TwoDetectionsModel(class_names=["obj"])
            return m

    fake_module.AutoModel = _AutoModel
    sys.modules["inference_exp"] = fake_module

    img = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
    model = RFDetrExperimentalModel("rfdetr-base", api_key=None)
    res = model.infer(img, confidence=0.5, iou_threshold=1.0)

    assert len(res.predictions) == 1  # only high-confidence remains

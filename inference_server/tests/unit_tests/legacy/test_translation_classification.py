import base64
import io
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from inference_models.models.base.classification import ClassificationPrediction
from PIL import Image

from inference_server.legacy.bridge import Route
from inference_server.legacy.entities import (
    AnomalyDetectionResponse,
    ClassificationInferenceRequest,
    ClassificationInferenceResponse,
    SemanticSegmentationInferenceRequest,
)
from inference_server.legacy.translation import build_task_params, repack_prediction

IMG = {"type": "base64", "value": "x"}


def _anomaly_route():
    return Route(
        model_id="ds/1",
        registry_id="ds/1",
        task_type="classification",
        action="infer",
        class_names=["normal", "anomalous"],
    )


def test_anomaly_metadata_repacks_as_anomaly_detection_response():
    prediction = ClassificationPrediction(
        class_id=torch.tensor([1]),
        confidence=torch.tensor([[0.2, 0.8]]),
        images_metadata=[
            {
                "anomaly_score": 1.5,
                "anomaly_threshold": 1.0,
                "is_anomalous": True,
                "anomaly_map": np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
            }
        ],
    )
    resp = repack_prediction(
        "classification",
        "infer",
        [prediction],
        (2, 2),
        _anomaly_route(),
        ClassificationInferenceRequest(model_id="ds/1", image=IMG, confidence=0.9),
    )
    assert type(resp) is AnomalyDetectionResponse
    d = resp.model_dump(by_alias=True, exclude_none=True)
    assert d["top"] == "anomalous" and d["confidence"] == 0.8
    assert [p["class"] for p in d["predictions"]] == ["anomalous", "normal"]
    assert d["anomaly_score"] == 1.5
    assert d["anomaly_threshold"] == 1.0
    assert d["is_anomalous"] is True
    assert d["anomaly_map"] == [
        [pytest.approx(0.1), pytest.approx(0.2)],
        [pytest.approx(0.3), pytest.approx(0.4)],
    ]


def test_anomaly_metadata_without_map_leaves_anomaly_map_out():
    prediction = ClassificationPrediction(
        class_id=torch.tensor([0]),
        confidence=torch.tensor([[0.7, 0.3]]),
        images_metadata=[
            {"anomaly_score": 0.5, "anomaly_threshold": 1.0, "is_anomalous": False}
        ],
    )
    resp = repack_prediction(
        "classification",
        "infer",
        [prediction],
        (2, 2),
        _anomaly_route(),
        ClassificationInferenceRequest(model_id="ds/1", image=IMG),
    )
    d = resp.model_dump(by_alias=True, exclude_none=True)
    assert d["top"] == "normal" and d["is_anomalous"] is False
    assert "anomaly_map" not in d


def test_classification_without_metadata_stays_plain():
    prediction = ClassificationPrediction(
        class_id=torch.tensor([1]), confidence=torch.tensor([[0.2, 0.8]])
    )
    resp = repack_prediction(
        "classification",
        "infer",
        [prediction],
        (2, 2),
        _anomaly_route(),
        ClassificationInferenceRequest(model_id="ds/1", image=IMG),
    )
    assert type(resp) is ClassificationInferenceResponse


def test_build_task_params_forwards_include_anomaly_map_for_classification():
    request = ClassificationInferenceRequest(
        model_id="ds/1", image=IMG, confidence=0.3, include_anomaly_map=True
    )
    route = _anomaly_route()
    assert build_task_params("classification", "infer", request, route) == {
        "confidence": 0.3,
        "include_anomaly_map": True,
    }
    assert build_task_params("multi-label-classification", "infer", request, route) == {
        "confidence": 0.3
    }


def test_build_task_params_forwards_disable_preproc_flags_for_classification():
    request = ClassificationInferenceRequest(
        model_id="ds/1", image=IMG, disable_preproc_grayscale=True
    )
    route = _anomaly_route()
    for task_type in ("classification", "semantic-segmentation"):
        params = build_task_params(task_type, "infer", request, route)
        assert params["disable_preproc_grayscale"] is True
        assert "disable_preproc_contrast" not in params


def test_classification_sorted_and_thresholded():
    route = Route(
        model_id="ds/1",
        registry_id="ds/1",
        task_type="classification",
        action="infer",
        class_names=["a", "b", "c"],
    )
    pred = SimpleNamespace(confidence=np.array([0.2, 0.7, 0.1]), class_id=np.array([1]))
    resp = repack_prediction(
        "classification",
        "infer",
        [pred],
        (3, 3),
        route,
        ClassificationInferenceRequest(model_id="ds/1", image=IMG, confidence=0.15),
    )
    d = resp.model_dump(by_alias=True, exclude_none=True)
    assert d["top"] == "b"
    assert d["confidence"] == 0.7
    assert [p["class"] for p in d["predictions"]] == ["b", "a"]


def test_multilabel_classification():
    route = Route(
        model_id="ds/1",
        registry_id="ds/1",
        task_type="multi-label-classification",
        action="infer",
        class_names=["a", "b"],
    )
    pred = SimpleNamespace(confidence=np.array([0.9, 0.2]), class_ids=np.array([0]))
    resp = repack_prediction(
        "multi-label-classification",
        "infer",
        pred,
        (3, 3),
        route,
        ClassificationInferenceRequest(model_id="ds/1", image=IMG),
    )
    d = resp.model_dump(by_alias=True, exclude_none=True)
    assert d["predicted_classes"] == ["a"]
    assert d["predictions"]["a"] == {"confidence": 0.9, "class_id": 0}


def test_semantic_segmentation_png_masks():
    route = Route(
        model_id="ds/1",
        registry_id="ds/1",
        task_type="semantic-segmentation",
        action="infer",
        class_names=["bg", "fg"],
    )
    pred = SimpleNamespace(
        segmentation_map=np.array([[0, 2], [5, 0]]),
        confidence=np.array([[1.0, 0.5], [0.5, 1.0]]),
    )
    resp = repack_prediction(
        "semantic-segmentation",
        "infer",
        pred,
        (2, 2),
        route,
        SemanticSegmentationInferenceRequest(model_id="ds/1", image=IMG),
    )
    d = resp.model_dump(by_alias=True, exclude_none=True)
    decoded = Image.open(
        io.BytesIO(base64.b64decode(d["predictions"]["segmentation_mask"]))
    )
    assert decoded.size == (2, 2)
    assert d["predictions"]["class_map"] == {"0": "bg", "1": "fg"}
    assert d["predictions"]["present_class_ids"] == [0, 2, 5]

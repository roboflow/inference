import base64
import io
from types import SimpleNamespace

import numpy as np
from PIL import Image

from inference_server.legacy.bridge import Route
from inference_server.legacy.entities import (
    ClassificationInferenceRequest,
    SemanticSegmentationInferenceRequest,
)
from inference_server.legacy.translation import repack_prediction

IMG = {"type": "base64", "value": "x"}


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

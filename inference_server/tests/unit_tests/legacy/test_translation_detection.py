from types import SimpleNamespace

import numpy as np
import pytest

from inference_server.legacy.bridge import Route
from inference_server.legacy.entities import (
    InstanceSegmentationInferenceRequest,
    KeypointsDetectionInferenceRequest,
    ObjectDetectionInferenceRequest,
)
from inference_server.legacy.translation import (
    build_task_params,
    masks2poly,
    repack_prediction,
)

IMG = {"type": "base64", "value": "x"}
ROUTE = Route(
    model_id="ds/1",
    registry_id="ds/1",
    task_type="object-detection",
    action="infer",
    class_names=["cat", "dog"],
)


def _det(xyxy, conf, cls):
    return SimpleNamespace(
        xyxy=np.array(xyxy, dtype=float),
        confidence=np.array(conf),
        class_id=np.array(cls),
    )


def test_build_params_object_detection():
    req = ObjectDetectionInferenceRequest(
        model_id="ds/1",
        image=IMG,
        confidence=0.6,
        iou_threshold=0.4,
        max_detections=10,
        class_agnostic_nms=True,
    )
    assert build_task_params("object-detection", "infer", req, ROUTE) == {
        "confidence": 0.6,
        "iou_threshold": 0.4,
        "max_detections": 10,
        "class_agnostic_nms": True,
    }


def test_build_params_passes_best_confidence_through():
    req = ObjectDetectionInferenceRequest(model_id="ds/1", image=IMG, confidence="best")
    assert build_task_params("object-detection", "infer", req, ROUTE)["confidence"] == (
        "best"
    )


def test_repack_object_detection_matches_legacy_shape():
    req = ObjectDetectionInferenceRequest(model_id="ds/1", image=IMG)
    resp = repack_prediction(
        "object-detection",
        "infer",
        [_det([[10, 20, 30, 60]], [0.9], [1])],
        (100, 50),
        ROUTE,
        req,
    )
    dumped = resp.model_dump(by_alias=True, exclude_none=True)
    pred = dumped["predictions"][0]
    assert dumped["image"] == {"width": 100, "height": 50}
    assert (pred["x"], pred["y"], pred["width"], pred["height"]) == (
        20.0,
        40.0,
        20.0,
        40.0,
    )
    assert (
        pred["class"] == "dog" and pred["class_id"] == 1 and pred["confidence"] == 0.9
    )
    assert "detection_id" in pred and "class_confidence" not in pred


def test_repack_object_detection_applies_class_filter():
    req = ObjectDetectionInferenceRequest(
        model_id="ds/1", image=IMG, class_filter=["cat"]
    )
    resp = repack_prediction(
        "object-detection",
        "infer",
        _det([[0, 0, 1, 1], [0, 0, 2, 2]], [0.9, 0.8], [1, 0]),
        (4, 4),
        ROUTE,
        req,
    )
    assert [p.class_name for p in resp.predictions] == ["cat"]


def test_repack_instance_segmentation_polygon_and_rle():
    mask = np.zeros((2, 4, 4), dtype=bool)
    mask[0, 1:3, 1:3] = True
    mask[1, 0:2, 0:2] = True
    pred = SimpleNamespace(
        xyxy=np.array([[1, 1, 3, 3], [0, 0, 2, 2]], dtype=float),
        confidence=np.array([0.9, 0.7]),
        class_id=np.array([0, 1]),
        mask=mask,
    )
    route = Route(
        model_id="ds/1",
        registry_id="ds/1",
        task_type="instance-segmentation",
        action="infer",
        class_names=["cat", "dog"],
    )
    poly = repack_prediction(
        "instance-segmentation",
        "infer",
        pred,
        (4, 4),
        route,
        InstanceSegmentationInferenceRequest(model_id="ds/1", image=IMG),
    )
    assert poly.predictions[0].mask_format == "polygon"
    assert len(poly.predictions[0].points) >= 3
    rle = repack_prediction(
        "instance-segmentation",
        "infer",
        pred,
        (4, 4),
        route,
        InstanceSegmentationInferenceRequest(
            model_id="ds/1", image=IMG, response_mask_format="rle"
        ),
    )
    assert rle.predictions[0].mask_format == "rle"
    assert isinstance(rle.predictions[0].rle["counts"], str)


def test_repack_keypoints_from_wire_shape():
    kp = SimpleNamespace(
        xy=np.array([[[1.0, 2.0], [3.0, 4.0]]]),
        class_id=np.array([0]),
        confidence=np.array([[0.9, 0.0]]),
    )
    det = _det([[0, 0, 10, 10]], [0.8], [0])
    route = Route(
        model_id="ds/1",
        registry_id="ds/1",
        task_type="keypoint-detection",
        action="infer",
        class_names=["person"],
        key_points_classes=[["nose", "eye"]],
    )
    resp = repack_prediction(
        "keypoint-detection",
        "infer",
        ([kp], [det]),
        (10, 10),
        route,
        KeypointsDetectionInferenceRequest(model_id="ds/1", image=IMG),
    )
    assert [k.class_name for k in resp.predictions[0].keypoints] == ["nose"]


def test_masks2poly_returns_contours():
    m = np.zeros((1, 6, 6), dtype=np.uint8)
    m[0, 1:5, 1:5] = 1
    polys = masks2poly(m)
    assert len(polys) == 1 and polys[0].shape[1] == 2

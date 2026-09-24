import numpy as np
import pytest
import torch

from inference_models.models.base.object_detection import Detections
from inference_server.legacy.bridge import Route
from inference_server.workflows.models_provider import GatewayModelsProvider
from inference_server.workflows.tensor_native import native_params, numpy_to_tensors
from tests.unit_tests.workflows.test_models_provider import FakeSyncBridge


def _marshalled_detections():
    return Detections(
        xyxy=np.array([[0.0, 0.0, 2.0, 2.0]], dtype=np.float32),
        class_id=np.array([0]),
        confidence=np.array([0.9], dtype=np.float32),
    )


def test_numpy_to_tensors_is_inverse_of_marshalling():
    restored = numpy_to_tensors(_marshalled_detections(), device=None)
    assert isinstance(restored, Detections) and isinstance(restored.xyxy, torch.Tensor)
    assert restored.xyxy.detach().cpu().numpy().tolist() == [[0.0, 0.0, 2.0, 2.0]]
    assert (
        numpy_to_tensors({"a": [np.zeros(2)], "b": "keep"}, device=None)["b"] == "keep"
    )


def test_native_params_keeps_only_model_params():
    params = native_params(
        "object-detection",
        {
            "confidence": 0.4,
            "iou_threshold": 0.5,
            "class_agnostic_nms": True,
            "class_filter": ["a"],
            "max_detections": 10,
            "max_candidates": 3000,
            "disable_active_learning": True,
            "input_color_format": "rgb",
        },
    )
    assert params == {
        "confidence": 0.4,
        "iou_threshold": 0.5,
        "class_agnostic_nms": True,
        "max_detections": 10,
        "input_color_format": "rgb",
    }


def test_native_params_renames_keypoint_confidence():
    params = native_params("keypoint-detection", {"keypoint_confidence": 0.7})
    assert params == {"key_points_threshold": 0.7, "input_color_format": "bgr"}


def test_run_tensor_native_inference_returns_per_image_tensor_objects():
    bridge = FakeSyncBridge()
    bridge.routes["ds/1"] = Route(
        model_id="ds/1",
        registry_id="ds/1",
        task_type="object-detection",
        action="infer",
        actions={"infer"},
        class_names=["cat"],
    )
    bridge.predictions[("ds/1", "infer")] = _marshalled_detections()
    provider = GatewayModelsProvider(bridge, api_key="k")
    images = [np.zeros((4, 6, 3), np.uint8), torch.zeros((3, 4, 6), dtype=torch.uint8)]
    out = provider.run_tensor_native_inference(
        model_id="ds/1",
        images=images,
        input_color_format="rgb",
        confidence=0.3,
        class_filter=["cat"],
    )
    assert len(out) == 2 and all(
        isinstance(o, Detections) and isinstance(o.confidence, torch.Tensor)
        for o in out
    )
    assert (
        bridge.calls[0][2]["input_color_format"] == "rgb"
        and "class_filter" not in bridge.calls[0][2]
    )
    assert all(
        isinstance(img, np.ndarray) and img.shape == (4, 6, 3)
        for img in bridge.calls[0][3]
    )


def test_classification_is_one_batched_prediction():
    from inference_models.models.base.classification import ClassificationPrediction

    bridge = FakeSyncBridge()
    bridge.routes["c/1"] = Route(
        model_id="c/1",
        registry_id="c/1",
        task_type="classification",
        action="infer",
        actions={"infer"},
        class_names=["a", "b"],
    )
    bridge.predictions[("c/1", "infer")] = ClassificationPrediction(
        class_id=np.array([1]),
        confidence=np.array([[0.2, 0.8]], dtype=np.float32),
    )
    out = GatewayModelsProvider(bridge, api_key="k").run_tensor_native_inference(
        model_id="c/1",
        images=[np.zeros((2, 2, 3), np.uint8)] * 3,
        input_color_format="bgr",
    )
    assert (
        isinstance(out, ClassificationPrediction)
        and tuple(out.confidence.shape) == (3, 2)
        and tuple(out.class_id.shape) == (3,)
    )


def test_multi_label_classification_is_one_prediction_per_image():
    from inference_models.models.base.classification import (
        MultiLabelClassificationPrediction,
    )

    bridge = FakeSyncBridge()
    bridge.routes["m/1"] = Route(
        model_id="m/1",
        registry_id="m/1",
        task_type="multi-label-classification",
        action="infer",
        actions={"infer"},
        class_names=["a", "b"],
    )
    bridge.predictions[("m/1", "infer")] = MultiLabelClassificationPrediction(
        class_ids=np.array([1]),
        confidence=np.array([0.2, 0.8], dtype=np.float32),
    )
    out = GatewayModelsProvider(bridge, api_key="k").run_tensor_native_inference(
        model_id="m/1",
        images=[np.zeros((2, 2, 3), np.uint8)] * 2,
        input_color_format="bgr",
    )
    assert len(out) == 2 and all(
        isinstance(prediction, MultiLabelClassificationPrediction)
        and isinstance(prediction.confidence, torch.Tensor)
        and tuple(prediction.confidence.shape) == (2,)
        for prediction in out
    )


def test_keypoints_are_two_parallel_lists():
    from inference_models.models.base.keypoints_detection import KeyPoints

    bridge = FakeSyncBridge()
    bridge.routes["k/1"] = Route(
        model_id="k/1",
        registry_id="k/1",
        task_type="keypoint-detection",
        action="infer",
        actions={"infer"},
        key_points_classes=[["nose"]],
    )
    kp = KeyPoints(
        xy=np.zeros((1, 1, 2), np.float32),
        class_id=np.array([0]),
        confidence=np.ones((1, 1), np.float32),
    )
    bridge.predictions[("k/1", "infer")] = ([kp], [_marshalled_detections()])
    keypoints, detections = GatewayModelsProvider(
        bridge, api_key="k"
    ).run_tensor_native_inference(
        model_id="k/1", images=[np.zeros((2, 2, 3), np.uint8)] * 2
    )
    assert (
        len(keypoints) == 2
        and len(detections) == 2
        and isinstance(keypoints[0], KeyPoints)
        and isinstance(keypoints[0].xy, torch.Tensor)
    )


def test_keypoints_without_detections_keep_none():
    from inference_models.models.base.keypoints_detection import KeyPoints

    bridge = FakeSyncBridge()
    bridge.routes["k/1"] = Route(
        model_id="k/1",
        registry_id="k/1",
        task_type="keypoint-detection",
        action="infer",
        actions={"infer"},
    )
    kp = KeyPoints(
        xy=np.zeros((1, 1, 2), np.float32),
        class_id=np.array([0]),
        confidence=np.ones((1, 1), np.float32),
    )
    bridge.predictions[("k/1", "infer")] = ([kp], None)
    keypoints, detections = GatewayModelsProvider(
        bridge, api_key="k"
    ).run_tensor_native_inference(
        model_id="k/1", images=[np.zeros((2, 2, 3), np.uint8)]
    )
    assert len(keypoints) == 1 and detections == [None]


def test_dense_masks_flag_maps_to_mask_format():
    bridge = FakeSyncBridge()
    bridge.routes["s/1"] = Route(
        model_id="s/1",
        registry_id="s/1",
        task_type="instance-segmentation",
        action="infer",
        actions={"infer"},
    )
    bridge.predictions[("s/1", "infer")] = _marshalled_detections()
    GatewayModelsProvider(bridge, api_key="k").run_tensor_native_inference(
        model_id="s/1",
        images=[np.zeros((2, 2, 3), np.uint8)],
        enforce_dense_masks_in_inference_models=True,
    )
    assert bridge.calls[0][2]["mask_format"] == "dense"


def test_rle_masks_stay_the_default_for_instance_segmentation():
    bridge = FakeSyncBridge()
    bridge.routes["s/1"] = Route(
        model_id="s/1",
        registry_id="s/1",
        task_type="instance-segmentation",
        action="infer",
        actions={"infer"},
    )
    bridge.predictions[("s/1", "infer")] = _marshalled_detections()
    GatewayModelsProvider(bridge, api_key="k").run_tensor_native_inference(
        model_id="s/1",
        images=[np.zeros((2, 2, 3), np.uint8)],
        enforce_dense_masks_in_inference_models=False,
    )
    assert "mask_format" not in bridge.calls[0][2]


def test_run_tensor_native_inference_ships_npy_when_gateway_needs_bytes():
    bridge = FakeSyncBridge()
    bridge.accepts_ndarray = False
    bridge.routes["ds/1"] = Route(
        model_id="ds/1",
        registry_id="ds/1",
        task_type="object-detection",
        action="infer",
        actions={"infer"},
    )
    bridge.predictions[("ds/1", "infer")] = _marshalled_detections()
    GatewayModelsProvider(bridge, api_key="k").run_tensor_native_inference(
        model_id="ds/1", images=[np.zeros((4, 6, 3), np.uint8)]
    )
    assert bridge.calls[0][3][0][:6] == b"\x93NUMPY"


def test_unsupported_family_is_501():
    from inference_server.legacy.errors import LegacyHTTPError

    bridge = FakeSyncBridge()
    bridge.routes["v/1"] = Route(
        model_id="v/1",
        registry_id="v/1",
        task_type="vlm",
        action="prompt",
        actions={"prompt"},
    )
    with pytest.raises(LegacyHTTPError) as exc:
        GatewayModelsProvider(bridge, api_key="k").run_tensor_native_inference(
            model_id="v/1", images=[np.zeros((2, 2, 3), np.uint8)]
        )
    assert exc.value.status_code == 501

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


def test_native_params_keeps_explicit_none_color_format():
    params = native_params("object-detection", {"input_color_format": None})
    assert params == {"input_color_format": None}


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


def test_unknown_task_type_501():
    from inference_server.legacy.errors import LegacyHTTPError

    bridge = FakeSyncBridge()
    bridge.routes["v/1"] = Route(
        model_id="v/1",
        registry_id="v/1",
        task_type="made-up-task",
        action="infer",
        actions={"infer"},
    )
    with pytest.raises(LegacyHTTPError) as exc:
        GatewayModelsProvider(bridge, api_key="k").run_tensor_native_inference(
            model_id="v/1", images=[np.zeros((2, 2, 3), np.uint8)]
        )
    assert exc.value.status_code == 501


def _bridge_with_route(model_id, task_type, action, actions, **route_kwargs):
    bridge = FakeSyncBridge()
    bridge.routes[model_id] = Route(
        model_id=model_id,
        registry_id=model_id,
        task_type=task_type,
        action=action,
        actions=set(actions),
        **route_kwargs,
    )
    return bridge


def _image():
    return np.zeros((2, 2, 3), np.uint8)


def test_semantic_segmentation_returns_tensor_results_per_image():
    from inference_models.models.base.semantic_segmentation import (
        SemanticSegmentationResult,
    )

    bridge = _bridge_with_route("seg/1", "semantic-segmentation", "infer", {"infer"})
    bridge.predictions[("seg/1", "infer")] = SemanticSegmentationResult(
        segmentation_map=np.array([[0, 1], [1, 0]], dtype=np.int64),
        confidence=np.array([[0.9, 0.8], [0.7, 0.6]], dtype=np.float32),
    )
    out = GatewayModelsProvider(bridge, api_key="k").run_tensor_native_inference(
        model_id="seg/1",
        images=[_image(), _image()],
        input_color_format="rgb",
        confidence=0.4,
        class_filter=["a"],
    )
    assert bridge.calls[0][1] == "infer"
    assert bridge.calls[0][2] == {"confidence": 0.4, "input_color_format": "rgb"}
    assert len(out) == 2 and all(
        isinstance(result, SemanticSegmentationResult)
        and isinstance(result.segmentation_map, torch.Tensor)
        and isinstance(result.confidence, torch.Tensor)
        for result in out
    )
    assert out[0].segmentation_map.dtype == torch.int64
    assert out[0].confidence.dtype == torch.float32
    assert out[0].segmentation_map.tolist() == [[0, 1], [1, 0]]


def test_depth_estimation_keeps_depth_anything_maps_as_is():
    bridge = _bridge_with_route(
        "depth-anything-v2/small",
        "depth-estimation",
        "infer",
        {"infer"},
        model_class_name="DepthAnythingV2HF",
    )
    bridge.predictions[("depth-anything-v2/small", "infer")] = np.array(
        [[1.0, 2.0], [3.0, 4.0]], dtype=np.float32
    )
    out = GatewayModelsProvider(bridge, api_key="k").run_tensor_native_inference(
        "depth-anything-v2/small", images=[_image()], input_color_format="rgb"
    )
    assert bridge.calls[0][2] == {"input_color_format": "rgb"}
    assert len(out) == 1 and isinstance(out[0], torch.Tensor)
    assert out[0].dtype == torch.float32 and tuple(out[0].shape) == (2, 2)
    assert out[0].tolist() == [[1.0, 2.0], [3.0, 4.0]]


def test_depth_estimation_negates_yolo26_metric_depth():
    bridge = _bridge_with_route(
        "ds/3",
        "depth-estimation",
        "infer",
        {"infer"},
        model_class_name="YOLO26ForDepthEstimationOnnx",
    )
    bridge.predictions[("ds/3", "infer")] = np.array(
        [[1.0, 2.0], [3.0, 4.0]], dtype=np.float32
    )
    out = GatewayModelsProvider(bridge, api_key="k").run_tensor_native_inference(
        "ds/3", images=[_image()], input_color_format="bgr"
    )
    assert out[0].dtype == torch.float32
    assert out[0].tolist() == [[-1.0, -2.0], [-3.0, -4.0]]


def test_image_embedding_concatenates_one_row_per_image():
    bridge = _bridge_with_route(
        "clip/ViT-B-16", "embedding", "embed_images", {"embed_images", "embed_text"}
    )
    bridge.predictions[("clip/ViT-B-16", "embed_images")] = np.ones(
        (1, 3), dtype=np.float32
    )
    out = GatewayModelsProvider(bridge, api_key="k").run_tensor_native_inference(
        "clip/ViT-B-16",
        action="embed-image",
        images=[_image(), _image()],
        input_color_format="rgb",
    )
    assert bridge.calls[0][1] == "embed_images"
    assert bridge.calls[0][2] == {"input_color_format": "rgb"}
    assert isinstance(out, torch.Tensor) and tuple(out.shape) == (2, 3)
    assert out.dtype == torch.float32 and out.tolist() == [[1.0] * 3] * 2


def test_text_embedding_runs_without_images():
    bridge = _bridge_with_route(
        "clip/ViT-B-16", "embedding", "embed_images", {"embed_images", "embed_text"}
    )
    bridge.predictions[("clip/ViT-B-16", "embed_text")] = np.ones(
        (2, 3), dtype=np.float32
    )
    out = GatewayModelsProvider(bridge, api_key="k").run_tensor_native_inference(
        "clip/ViT-B-16", action="embed-text", texts=["a", "b"]
    )
    assert bridge.calls == [
        ("clip/ViT-B-16", "embed_text", {"texts": ["a", "b"]}, None)
    ]
    assert isinstance(out, torch.Tensor) and tuple(out.shape) == (2, 3)
    assert out.dtype == torch.float32


def test_structured_ocr_is_texts_and_detections_per_image():
    bridge = _bridge_with_route(
        "easy_ocr/english_g2", "structured-ocr", "infer", {"infer"}
    )
    detections = _marshalled_detections()
    detections.bboxes_metadata = [{"text": "hello"}]
    bridge.predictions[("easy_ocr/english_g2", "infer")] = (["hello"], [detections])
    texts, detections_batch = GatewayModelsProvider(
        bridge, api_key="k"
    ).run_tensor_native_inference(
        "easy_ocr/english_g2",
        images=[_image(), _image()],
        input_color_format="bgr",
        confidence=0.0,
    )
    assert bridge.calls[0][2] == {"confidence": 0.0, "input_color_format": "bgr"}
    assert texts == ["hello", "hello"]
    assert len(detections_batch) == 2
    assert isinstance(detections_batch[0], Detections)
    assert isinstance(detections_batch[0].xyxy, torch.Tensor)
    assert detections_batch[0].bboxes_metadata == [{"text": "hello"}]


def test_vlm_prompt_returns_one_answer_per_image():
    bridge = _bridge_with_route(
        "qwen/1", "vlm", "prompt", {"prompt"}, model_class_name="Qwen35HF"
    )
    bridge.predictions[("qwen/1", "prompt")] = "an answer"
    out = GatewayModelsProvider(bridge, api_key="k").run_tensor_native_inference(
        "qwen/1",
        images=[_image()],
        input_color_format="rgb",
        prompt="what<system_prompt>sys",
        enable_thinking=False,
        max_new_tokens=5,
    )
    assert bridge.calls[0][1] == "prompt"
    assert bridge.calls[0][2] == {
        "prompt": "what<system_prompt>sys",
        "enable_thinking": False,
        "max_new_tokens": 5,
        "input_color_format": "rgb",
    }
    assert out == ["an answer"]


@pytest.mark.parametrize(
    "prompt, task",
    [
        ("<OD>", "<OD>"),
        ("<CAPTION_TO_PHRASE_GROUNDING>a cat", "<CAPTION_TO_PHRASE_GROUNDING>"),
        ("describe the image", "describe the image>"),
    ],
)
def test_florence2_task_is_derived_from_the_prompt(prompt, task):
    bridge = _bridge_with_route(
        "florence/1",
        "vlm",
        "prompt",
        {"caption", "prompt", "detect"},
        model_class_name="Florence2HF",
    )
    bridge.predictions[("florence/1", "prompt")] = {task: {"labels": ["cat"]}}
    out = GatewayModelsProvider(bridge, api_key="k").run_tensor_native_inference(
        "florence/1", images=[_image()], input_color_format="rgb", prompt=prompt
    )
    assert bridge.calls[0][1] == "prompt"
    assert bridge.calls[0][2]["task"] == task
    assert bridge.calls[0][2]["prompt"] == prompt
    assert out == [{task: {"labels": ["cat"]}}]


def test_only_florence2_gets_a_task():
    bridge = _bridge_with_route(
        "smolvlm2/smolvlm-2.2b-instruct",
        "vlm",
        "prompt",
        {"prompt"},
        model_class_name="SmolVLMHF",
    )
    bridge.predictions[("smolvlm2/smolvlm-2.2b-instruct", "prompt")] = "text"
    GatewayModelsProvider(bridge, api_key="k").run_tensor_native_inference(
        "smolvlm2/smolvlm-2.2b-instruct",
        images=[_image()],
        input_color_format="rgb",
        prompt="<OD>",
    )
    assert "task" not in bridge.calls[0][2]


def test_moondream_runs_detect_with_classes():
    bridge = _bridge_with_route(
        "moondream2/2b",
        "vlm",
        "prompt",
        {"caption", "detect", "query", "point", "encode"},
        model_class_name="MoonDream2HF",
    )
    bridge.predictions[("moondream2/2b", "detect")] = _marshalled_detections()
    out = GatewayModelsProvider(bridge, api_key="k").run_tensor_native_inference(
        model_id="moondream2/2b",
        images=[_image()],
        input_color_format="rgb",
        classes=["dog"],
    )
    assert bridge.calls[0][1] == "detect"
    assert bridge.calls[0][2] == {"classes": ["dog"], "input_color_format": "rgb"}
    assert len(out) == 1 and isinstance(out[0], Detections)
    assert isinstance(out[0].xyxy, torch.Tensor)


def test_sam2_segment_ships_numpy_boxes_and_restores_tensors():
    from inference_models.models.sam2.entities import SAM2Prediction

    bridge = _bridge_with_route(
        "sam2/hiera_small",
        "interactive-instance-segmentation",
        "embed",
        {"embed", "segment"},
        model_class_name="SAM2Torch",
    )
    bridge.predictions[("sam2/hiera_small", "segment")] = SAM2Prediction(
        masks=np.array([[[1.0, -1.0], [-1.0, 1.0]]], np.float32),
        scores=np.array([0.7], np.float32),
        logits=np.zeros((1, 4, 4), np.float32),
    )
    boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
    out = GatewayModelsProvider(bridge, api_key="k").run_tensor_native_inference(
        "sam2/hiera_small",
        action="segment",
        images=[_image()],
        boxes=[boxes],
        multi_mask_output=False,
        input_color_format="rgb",
        return_logits=True,
    )
    params = bridge.calls[0][2]
    assert bridge.calls[0][1] == "segment"
    assert isinstance(params["boxes"][0], np.ndarray)
    assert params["boxes"][0].tolist() == [[0.0, 0.0, 1.0, 1.0]]
    assert params["multi_mask_output"] is False and params["return_logits"] is True
    assert params["input_color_format"] == "rgb" and "mask_format" not in params
    assert isinstance(out[0], SAM2Prediction)
    assert isinstance(out[0].masks, torch.Tensor)
    assert isinstance(out[0].scores, torch.Tensor)
    assert out[0].masks.dtype == torch.float32 and out[0].scores.dtype == torch.float32
    assert tuple(out[0].masks.shape) == (1, 2, 2) and tuple(out[0].scores.shape) == (1,)
    assert (out[0].masks >= 0.0).tolist() == [[[True, False], [False, True]]]
    assert out[0].scores.tolist() == pytest.approx([0.7])


def test_sam2_segment_drops_absent_boxes():
    from inference_models.models.sam2.entities import SAM2Prediction

    bridge = _bridge_with_route(
        "sam2/hiera_small",
        "interactive-instance-segmentation",
        "embed",
        {"embed", "segment"},
    )
    bridge.predictions[("sam2/hiera_small", "segment")] = SAM2Prediction(
        masks=np.zeros((1, 1, 2, 2), np.float32),
        scores=np.zeros((1, 1), np.float32),
        logits=np.zeros((1, 1, 4, 4), np.float32),
    )
    GatewayModelsProvider(bridge, api_key="k").run_tensor_native_inference(
        "sam2/hiera_small",
        action="segment",
        images=[_image()],
        boxes=None,
        multi_mask_output=True,
        input_color_format="bgr",
        return_logits=True,
    )
    assert "boxes" not in bridge.calls[0][2]


def test_sam3_interactive_segment_maps_to_visual_prompts():
    from inference_models.models.sam3.entities import SAM3Prediction

    bridge = _bridge_with_route(
        "sam3/sam3_interactive",
        "interactive-instance-segmentation",
        "embed",
        {"embed_images", "segment_with_visual_prompts", "segment_with_text_prompts"},
        model_class_name="SAM3Torch",
    )
    bridge.predictions[("sam3/sam3_interactive", "segment_with_visual_prompts")] = (
        SAM3Prediction(
            masks=np.array([[[2.0, -2.0], [-2.0, 2.0]]], np.float32),
            scores=np.array([0.9], np.float32),
            logits=np.zeros((1, 4, 4), np.float32),
        )
    )
    out = GatewayModelsProvider(bridge, api_key="k").run_tensor_native_inference(
        "sam3/sam3_interactive",
        action="segment",
        images=[_image()],
        point_coordinates=np.array([[[1, 1]]]),
        point_labels=np.array([[1]]),
        boxes=None,
        multi_mask_output=True,
        return_logits=True,
    )
    params = bridge.calls[0][2]
    assert bridge.calls[0][1] == "segment_with_visual_prompts"
    assert params["point_coordinates"].tolist() == [[[1, 1]]]
    assert params["point_labels"].tolist() == [[1]]
    assert "boxes" not in params and "input_color_format" not in params
    assert params["mask_format"] == "dense"
    assert params["multi_mask_output"] is True and params["return_logits"] is True
    assert isinstance(out[0], SAM3Prediction)
    assert isinstance(out[0].masks, torch.Tensor)
    assert out[0].masks.dtype == torch.float32 and out[0].scores.dtype == torch.float32
    assert tuple(out[0].masks.shape) == (1, 2, 2) and tuple(out[0].scores.shape) == (1,)
    assert (out[0].masks >= 0.0).tolist() == [[[True, False], [False, True]]]
    assert out[0].scores.detach().to("cpu").tolist() == pytest.approx([0.9])


def test_sam3_text_prompts_keep_numpy_masks():
    bridge = _bridge_with_route(
        "sam3/sam3_small",
        "interactive-instance-segmentation",
        "embed",
        {"embed_images", "segment_with_visual_prompts", "segment_with_text_prompts"},
        model_class_name="SAM3Torch",
    )
    bridge.predictions[("sam3/sam3_small", "segment_with_text_prompts")] = [
        {"prompt_index": 0, "masks": np.ones((1, 2, 2), np.uint8), "scores": [0.9]}
    ]
    out = GatewayModelsProvider(bridge, api_key="k").run_tensor_native_inference(
        "sam3/sam3_small",
        images=[_image()],
        prompts=[{"text": "cat"}],
        output_prob_thresh=0.3,
    )
    assert bridge.calls[0][1] == "segment_with_text_prompts"
    assert bridge.calls[0][2] == {
        "prompts": [{"text": "cat"}],
        "output_prob_thresh": 0.3,
        "mask_format": "dense",
    }
    assert len(out) == 1 and out[0][0]["prompt_index"] == 0
    assert isinstance(out[0][0]["masks"], np.ndarray)
    assert out[0][0]["scores"] == [0.9]

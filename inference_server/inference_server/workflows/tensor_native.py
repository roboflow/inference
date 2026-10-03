from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from inference_server.legacy.bridge import Route
from inference_server.legacy.common import ImagePayload, decode_inline_image
from inference_server.legacy.translation import MOONDREAM_MODEL_CLASS

FLORENCE2_MODEL_CLASS = "Florence2HF"
_METRIC_DEPTH_MODEL_CLASS_PREFIX = "YOLO26ForDepthEstimation"
_SAM3_SEGMENT_ACTIONS = frozenset(
    {"segment_with_visual_prompts", "segment_with_text_prompts"}
)

_ROBOFLOW_TASK_TYPES = frozenset(
    {
        "object-detection",
        "instance-segmentation",
        "keypoint-detection",
        "classification",
        "multi-label-classification",
        "semantic-segmentation",
        "depth-estimation",
    }
)

SUPPORTED_TASK_TYPES = _ROBOFLOW_TASK_TYPES | frozenset(
    {"embedding", "structured-ocr", "vlm", "interactive-instance-segmentation"}
)

_DETECTION_PARAMS = frozenset(
    {
        "confidence",
        "iou_threshold",
        "class_agnostic_nms",
        "max_detections",
        "key_points_threshold",
    }
)
_COLOR_FORMAT_PARAM = frozenset({"input_color_format"})

NATIVE_PARAMS: Dict[str, frozenset] = {
    "object-detection": _DETECTION_PARAMS | _COLOR_FORMAT_PARAM,
    "instance-segmentation": _DETECTION_PARAMS | _COLOR_FORMAT_PARAM,
    "keypoint-detection": _DETECTION_PARAMS | _COLOR_FORMAT_PARAM,
    "classification": _DETECTION_PARAMS | _COLOR_FORMAT_PARAM,
    "multi-label-classification": _DETECTION_PARAMS | _COLOR_FORMAT_PARAM,
    "semantic-segmentation": frozenset({"confidence"}) | _COLOR_FORMAT_PARAM,
    "depth-estimation": _COLOR_FORMAT_PARAM,
    "embedding": frozenset({"texts"}) | _COLOR_FORMAT_PARAM,
    "structured-ocr": frozenset({"confidence"}) | _COLOR_FORMAT_PARAM,
    "vlm": frozenset({"prompt", "classes", "enable_thinking", "max_new_tokens"})
    | _COLOR_FORMAT_PARAM,
    "interactive-instance-segmentation": frozenset(
        {
            "prompts",
            "output_prob_thresh",
            "boxes",
            "point_coordinates",
            "point_labels",
            "multi_mask_output",
            "return_logits",
        }
    )
    | _COLOR_FORMAT_PARAM,
}

NATIVE_ACTIONS: Dict[str, Dict[Optional[str], Tuple[str, ...]]] = {
    "embedding": {
        "embed-image": ("embed_images",),
        "embed-text": ("embed_text",),
    },
    "vlm": {None: ("prompt",)},
    "interactive-instance-segmentation": {
        None: ("segment_with_text_prompts",),
        "segment": ("segment", "segment_with_visual_prompts"),
        "embed": ("embed", "embed_images"),
    },
}


# NOTE: bfloat16 tensors were widened to float32 on the wire; the original precision is not restored.
def numpy_to_tensors(result: Any, device: Any) -> Any:
    if isinstance(result, np.ndarray):
        return torch.as_tensor(result, device=device)
    if dataclasses.is_dataclass(result) and not isinstance(result, type):
        for field in dataclasses.fields(result):
            object.__setattr__(
                result,
                field.name,
                numpy_to_tensors(getattr(result, field.name), device),
            )
        return result
    if isinstance(result, list):
        return [numpy_to_tensors(item, device) for item in result]
    if isinstance(result, tuple):
        return tuple(numpy_to_tensors(item, device) for item in result)
    if isinstance(result, dict):
        return {key: numpy_to_tensors(value, device) for key, value in result.items()}
    return result


def native_action(route: Route, action: Optional[str]) -> str:
    """Map a tensor block's action onto the model-manager action it runs as.

    Args:
        route: Resolved route of the model, with its registered actions.
        action: Action the block passed, or None when it passed none.

    Returns:
        The first candidate registered on the route, the first candidate when
        the route lists no actions, and the route's own action for task types
        whose blocks pass no action.
    """
    if route.task_type == "vlm" and route.model_class_name == MOONDREAM_MODEL_CLASS:
        return "detect"

    candidates = NATIVE_ACTIONS.get(route.task_type, {}).get(action)
    if candidates is None:
        return route.action

    for candidate in candidates:
        if candidate in route.actions:
            return candidate

    return candidates[0]


def native_params(
    task_type: str,
    kwargs: Dict[str, Any],
    *,
    action: Optional[str] = None,
    model_class_name: Optional[str] = None,
) -> Dict[str, Any]:
    source = dict(kwargs)
    keypoint_confidence = source.pop("keypoint_confidence", None)
    if keypoint_confidence is not None and source.get("key_points_threshold") is None:
        source["key_points_threshold"] = keypoint_confidence
    params = {
        name: _tensors_to_numpy(source[name])
        for name in NATIVE_PARAMS[task_type]
        if source.get(name) is not None
    }
    if task_type in _ROBOFLOW_TASK_TYPES:
        params["input_color_format"] = source.get("input_color_format", "bgr")
    if task_type == "instance-segmentation" and source.get(
        "enforce_dense_masks_in_inference_models"
    ):
        params["mask_format"] = "dense"
    if action in _SAM3_SEGMENT_ACTIONS:
        params["mask_format"] = "dense"
    if model_class_name == FLORENCE2_MODEL_CLASS:
        params["task"] = params.get("prompt", "").split(">")[0] + ">"
    return params


def _tensors_to_numpy(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, list):
        return [_tensors_to_numpy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_tensors_to_numpy(item) for item in value)
    if isinstance(value, dict):
        return {key: _tensors_to_numpy(item) for key, item in value.items()}
    return value


def native_image_payloads(images: List[Any], *, ndarray_ok: bool) -> List[ImagePayload]:
    payloads: List[ImagePayload] = []
    for image in images:
        array = image
        if isinstance(image, torch.Tensor):
            array = image.detach().cpu().permute(1, 2, 0).numpy()
        payloads.append(
            decode_inline_image(
                {"type": "numpy_object", "value": array}, ndarray_ok=ndarray_ok
            )
        )
    return payloads


def assemble_native_result(
    task_type: str,
    per_image: List[Any],
    device: Any,
    *,
    action: Optional[str] = None,
    model_class_name: Optional[str] = None,
) -> Any:
    if action == "segment_with_text_prompts":
        return list(per_image)

    restored = [numpy_to_tensors(item, device) for item in per_image]
    if task_type in ("keypoint-detection", "structured-ocr"):
        first = [_unwrap_single(item[0]) for item in restored]
        second = [_unwrap_single(item[1]) for item in restored]
        return first, second
    if task_type == "classification":
        return _batched_prediction(restored)
    if task_type == "embedding":
        return torch.cat(restored, dim=0)
    if task_type == "depth-estimation" and (model_class_name or "").startswith(
        _METRIC_DEPTH_MODEL_CLASS_PREFIX
    ):
        return [-depth_map for depth_map in restored]
    return restored


def _unwrap_single(component: Any) -> Any:
    if isinstance(component, (list, tuple)):
        return component[0] if component else None
    return component


def _batched_prediction(predictions: List[Any]) -> Any:
    first = predictions[0]
    if len(predictions) == 1:
        return first
    values: Dict[str, Any] = {}
    for field in dataclasses.fields(first):
        value = getattr(first, field.name)
        if isinstance(value, torch.Tensor):
            values[field.name] = torch.cat(
                [getattr(prediction, field.name) for prediction in predictions], dim=0
            )
        else:
            values[field.name] = value
    return type(first)(**values)

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List

import numpy as np
import torch

from inference_server.legacy.common import ImagePayload, decode_inline_image

SUPPORTED_TASK_TYPES = frozenset(
    {
        "object-detection",
        "instance-segmentation",
        "keypoint-detection",
        "classification",
        "multi-label-classification",
    }
)

_NATIVE_PARAM_NAMES = (
    "confidence",
    "iou_threshold",
    "class_agnostic_nms",
    "max_detections",
    "key_points_threshold",
)


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


def native_params(task_type: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    source = dict(kwargs)
    keypoint_confidence = source.pop("keypoint_confidence", None)
    if keypoint_confidence is not None and source.get("key_points_threshold") is None:
        source["key_points_threshold"] = keypoint_confidence
    params = {
        name: source[name]
        for name in _NATIVE_PARAM_NAMES
        if source.get(name) is not None
    }
    params["input_color_format"] = source.get("input_color_format", "bgr")
    if task_type == "instance-segmentation" and source.get(
        "enforce_dense_masks_in_inference_models"
    ):
        params["mask_format"] = "dense"
    return params


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


def assemble_native_result(task_type: str, per_image: List[Any], device: Any) -> Any:
    restored = [numpy_to_tensors(item, device) for item in per_image]
    if task_type == "keypoint-detection":
        key_points = [_unwrap_single(item[0]) for item in restored]
        detections = [_unwrap_single(item[1]) for item in restored]
        return key_points, detections
    if task_type == "classification":
        return _batched_prediction(restored)
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

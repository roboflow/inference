"""Typed serializers for model registry.

Each serializer: (raw_output, model_instance) → typed dict.
Output matches v2 API response type spec. Tensors stay as-is —
orjson + OPT_SERIALIZE_NUMPY handles conversion at response time.

These are pure functions. They don't import model classes — they inspect
output attributes at runtime (duck typing). This avoids circular imports
and stays decoupled from inference_models.
"""

from __future__ import annotations

from typing import Any


def _class_names(model: Any) -> list | None:
    return getattr(model, "class_names", None)


def _to_list(tensor_or_array: Any) -> Any:
    """Leave as-is — orjson serializes numpy/torch directly."""
    return tensor_or_array


# ---------------------------------------------------------------------------
# Object detection
# ---------------------------------------------------------------------------


def _unwrap_batch(output: Any) -> Any:
    """If output is a single-element list, unwrap it."""
    if isinstance(output, list) and len(output) == 1:
        return output[0]
    return output


def serialize_detections_compact(output: Any, model: Any) -> dict:
    """Detections → roboflow-object-detection-compact-v1"""
    output = _unwrap_batch(output)
    if isinstance(output, list):
        return {
            "type": "roboflow-object-detection-compact-v1",
            "class_names": _class_names(model),
            "batch": [
                {
                    "xyxy": _to_list(d.xyxy),
                    "class_id": _to_list(d.class_id),
                    "confidence": _to_list(d.confidence),
                }
                for d in output
            ],
        }
    return {
        "type": "roboflow-object-detection-compact-v1",
        "class_names": _class_names(model),
        "xyxy": _to_list(output.xyxy),
        "class_id": _to_list(output.class_id),
        "confidence": _to_list(output.confidence),
    }


# ---------------------------------------------------------------------------
# Classification (single-label)
# ---------------------------------------------------------------------------


def serialize_classification_compact(output: Any, model: Any) -> dict:
    """ClassificationPrediction → roboflow-classification-compact-v1"""
    return {
        "type": "roboflow-classification-compact-v1",
        "class_names": _class_names(model),
        "confidences": _to_list(output.confidence),
        "top_classes_ids": _to_list(output.class_id),
    }


# ---------------------------------------------------------------------------
# Classification (multi-label)
# ---------------------------------------------------------------------------


def serialize_multilabel_classification_compact(output: Any, model: Any) -> dict:
    """MultiLabelClassificationPrediction → roboflow-classification-compact-v1"""
    return {
        "type": "roboflow-classification-compact-v1",
        "class_names": _class_names(model),
        "confidences": _to_list(output.confidence),
        "detected_classes_ids": _to_list(output.class_ids),
    }


# ---------------------------------------------------------------------------
# Instance segmentation
# ---------------------------------------------------------------------------


def serialize_instance_segmentation_compact(output: Any, model: Any) -> dict:
    """InstanceDetections → roboflow-instance-segmentation-compact-v1

    NOTE: mask is passed as-is for now. Phase 36 will add cropped RLE encoding.
    """
    return {
        "type": "roboflow-instance-segmentation-compact-v1",
        "class_names": _class_names(model),
        "xyxy": _to_list(output.xyxy),
        "class_id": _to_list(output.class_id),
        "confidence": _to_list(output.confidence),
        "mask": _to_list(output.mask),
    }


# ---------------------------------------------------------------------------
# Semantic segmentation
# ---------------------------------------------------------------------------


def serialize_semantic_segmentation_compact(output: Any, model: Any) -> dict:
    """SemanticSegmentationResult → roboflow-semantic-segmentation-compact-v1"""
    return {
        "type": "roboflow-semantic-segmentation-compact-v1",
        "class_names": _class_names(model),
        "segmentation_map": _to_list(output.segmentation_map),
        "confidence": _to_list(output.confidence),
    }


# ---------------------------------------------------------------------------
# Keypoints detection
# ---------------------------------------------------------------------------


def serialize_keypoints_compact(output: Any, model: Any) -> dict:
    """KeyPoints → roboflow-keypoints-compact-v1"""
    return {
        "type": "roboflow-keypoints-compact-v1",
        "class_names": _class_names(model),
        "xy": _to_list(output.xy),
        "class_id": _to_list(output.class_id),
        "confidence": _to_list(output.confidence),
    }


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


def serialize_embeddings(output: Any, model: Any) -> dict:
    """Tensor/array embeddings → roboflow-embeddings-compact-v1"""
    return {
        "type": "roboflow-embeddings-compact-v1",
        "embeddings": _to_list(output),
    }


# ---------------------------------------------------------------------------
# Text output (captions, VLM responses, OCR)
# ---------------------------------------------------------------------------


def serialize_text(output: Any, model: Any) -> dict:
    """String or list of strings → roboflow-text-v1"""
    return {
        "type": "roboflow-text-v1",
        "text": output if isinstance(output, str) else str(output),
    }


# ---------------------------------------------------------------------------
# Depth estimation
# ---------------------------------------------------------------------------


def serialize_depth_compact(output: Any, model: Any) -> dict:
    """Depth map tensor → roboflow-depth-compact-v1"""
    return {
        "type": "roboflow-depth-compact-v1",
        "depth_map": _to_list(output),
    }


# ===========================================================================
# Rich serializers (per-object dicts, human-readable)
# ===========================================================================


def _to_py(val: Any) -> Any:
    """Convert tensor/numpy scalar to Python float/int for JSON."""
    if hasattr(val, "item"):
        return val.item()
    return val


def serialize_detections_rich(output: Any, model: Any) -> dict:
    """Detections → roboflow-object-detection-rich-v1"""
    output = _unwrap_batch(output)
    names = _class_names(model)

    def _det(d, i):
        xyxy = d.xyxy[i]
        det = {
            "left_top": [_to_py(xyxy[0]), _to_py(xyxy[1])],
            "right_bottom": [_to_py(xyxy[2]), _to_py(xyxy[3])],
            "confidence": _to_py(d.confidence[i]),
            "class_id": _to_py(d.class_id[i]),
        }
        if names and int(det["class_id"]) < len(names):
            det["class_name"] = names[int(det["class_id"])]
        return det

    if isinstance(output, list):
        return {
            "type": "roboflow-object-detection-rich-v1",
            "batch": [
                {"detections": [_det(d, i) for i in range(len(d.xyxy))]} for d in output
            ],
        }
    return {
        "type": "roboflow-object-detection-rich-v1",
        "detections": [_det(output, i) for i in range(len(output.xyxy))],
    }


def serialize_classification_rich(output: Any, model: Any) -> dict:
    """ClassificationPrediction → roboflow-classification-rich-v1"""
    names = _class_names(model)
    candidates = []
    for i in range(len(output.confidence)):
        c = {
            "class_id": (
                _to_py(output.class_id[i])
                if hasattr(output.class_id, "__len__")
                else _to_py(output.class_id)
            ),
            "confidence": _to_py(output.confidence[i]),
        }
        cid = int(c["class_id"])
        if names and cid < len(names):
            c["class_name"] = names[cid]
        candidates.append(c)
    candidates.sort(key=lambda x: x["confidence"], reverse=True)
    return {
        "type": "roboflow-classification-rich-v1",
        "candidates": candidates,
        "top": candidates[:1],
    }


def serialize_instance_segmentation_rich(output: Any, model: Any) -> dict:
    """InstanceDetections → roboflow-instance-segmentation-rich-v1"""
    names = _class_names(model)
    detections = []
    for i in range(len(output.xyxy)):
        xyxy = output.xyxy[i]
        det = {
            "left_top": [_to_py(xyxy[0]), _to_py(xyxy[1])],
            "right_bottom": [_to_py(xyxy[2]), _to_py(xyxy[3])],
            "confidence": _to_py(output.confidence[i]),
            "class_id": _to_py(output.class_id[i]),
        }
        if names and int(det["class_id"]) < len(names):
            det["class_name"] = names[int(det["class_id"])]
        if hasattr(output, "mask") and output.mask is not None:
            det["mask"] = _to_list(output.mask[i])
        detections.append(det)
    return {
        "type": "roboflow-instance-segmentation-rich-v1",
        "detections": detections,
    }


# ---------------------------------------------------------------------------
# Generic passthrough (for unregistered models)
# ---------------------------------------------------------------------------


def serialize_passthrough(output: Any, model: Any) -> dict:
    """Raw output wrapped in generic envelope."""
    return {
        "type": "roboflow-generic-v1",
        "data": output,
    }

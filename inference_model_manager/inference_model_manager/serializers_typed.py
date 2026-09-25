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

import numpy as np


def _class_names(model: Any) -> list | None:
    return getattr(model, "class_names", None)


def _detections_class_names(model: Any, output: Any) -> list | None:
    names = _class_names(model)
    if names is not None:
        return names
    detections = output if isinstance(output, list) else [output]
    for detection in detections:
        metadata = getattr(detection, "image_metadata", None)
        if metadata:
            names = metadata.get("class_names")
            if names is not None:
                return names
    return None


def _to_list(tensor_or_array: Any) -> Any:
    """Leave as-is — orjson serializes numpy/torch directly."""
    return tensor_or_array


def _mask_to_json(mask: Any) -> Any:
    """Dense masks pass through; RLE masks become per-detection COCO RLE dicts."""
    if hasattr(mask, "to_coco_rle_masks"):
        return [
            {
                "format": "rle",
                "size": m["size"],
                "counts": (
                    m["counts"].decode("utf-8")
                    if isinstance(m["counts"], bytes)
                    else m["counts"]
                ),
            }
            for m in mask.to_coco_rle_masks()
        ]
    return _to_list(mask)


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
            "class_names": _detections_class_names(model, output),
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
        "class_names": _detections_class_names(model, output),
        "xyxy": _to_list(output.xyxy),
        "class_id": _to_list(output.class_id),
        "confidence": _to_list(output.confidence),
    }


# ---------------------------------------------------------------------------
# Classification (single-label)
# ---------------------------------------------------------------------------


def serialize_classification_compact(output: Any, model: Any) -> dict:
    """ClassificationPrediction → roboflow-classification-compact-v1"""
    output = _unwrap_batch(output)
    if isinstance(output, list):
        return {
            "type": "roboflow-classification-compact-v1",
            "class_names": _class_names(model),
            "batch": [
                {
                    "confidences": _to_list(o.confidence),
                    "top_classes_ids": _to_list(o.class_id),
                }
                for o in output
            ],
        }
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
    output = _unwrap_batch(output)
    if isinstance(output, list):
        return {
            "type": "roboflow-classification-compact-v1",
            "class_names": _class_names(model),
            "batch": [
                {
                    "confidences": _to_list(o.confidence),
                    "detected_classes_ids": _to_list(o.class_ids),
                }
                for o in output
            ],
        }
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
    output = _unwrap_batch(output)
    if isinstance(output, list):
        return {
            "type": "roboflow-instance-segmentation-compact-v1",
            "class_names": _class_names(model),
            "batch": [
                {
                    "xyxy": _to_list(o.xyxy),
                    "class_id": _to_list(o.class_id),
                    "confidence": _to_list(o.confidence),
                    "mask": _mask_to_json(o.mask),
                }
                for o in output
            ],
        }
    return {
        "type": "roboflow-instance-segmentation-compact-v1",
        "class_names": _class_names(model),
        "xyxy": _to_list(output.xyxy),
        "class_id": _to_list(output.class_id),
        "confidence": _to_list(output.confidence),
        "mask": _mask_to_json(output.mask),
    }


def serialize_sam_segmentation_compact(output: Any, model: Any) -> dict:
    """SAMPrediction/SAM2Prediction → roboflow-sam-segmentation-compact-v1"""
    output = _unwrap_batch(output)
    if isinstance(output, list):
        return {
            "type": "roboflow-sam-segmentation-compact-v1",
            "batch": [
                {"masks": _to_list(o.masks), "scores": _to_list(o.scores)}
                for o in output
            ],
        }
    return {
        "type": "roboflow-sam-segmentation-compact-v1",
        "masks": _to_list(output.masks),
        "scores": _to_list(output.scores),
    }


# ---------------------------------------------------------------------------
# Semantic segmentation
# ---------------------------------------------------------------------------


def serialize_semantic_segmentation_compact(output: Any, model: Any) -> dict:
    """SemanticSegmentationResult → roboflow-semantic-segmentation-compact-v1"""
    output = _unwrap_batch(output)
    if isinstance(output, list):
        return {
            "type": "roboflow-semantic-segmentation-compact-v1",
            "class_names": _class_names(model),
            "batch": [
                {
                    "segmentation_map": _to_list(o.segmentation_map),
                    "confidence": _to_list(o.confidence),
                }
                for o in output
            ],
        }
    return {
        "type": "roboflow-semantic-segmentation-compact-v1",
        "class_names": _class_names(model),
        "segmentation_map": _to_list(output.segmentation_map),
        "confidence": _to_list(output.confidence),
    }


# ---------------------------------------------------------------------------
# Keypoints detection
# ---------------------------------------------------------------------------


def _split_keypoints(item: Any) -> Any:
    """Split a (keypoints, detections) pair, unwrapping each half."""
    if isinstance(item, tuple) and len(item) == 2:
        keypoints, detections = item
        return _unwrap_batch(keypoints), _unwrap_batch(detections)
    return item, None


def _keypoints_fields(keypoints: Any, detections: Any) -> dict:
    """xy/class_id/confidence, plus boxes when detections accompany the keypoints."""
    fields = {
        "xy": _to_list(keypoints.xy),
        "class_id": _to_list(keypoints.class_id),
        "confidence": _to_list(keypoints.confidence),
    }
    if detections is not None:
        fields["boxes"] = {
            "xyxy": _to_list(detections.xyxy),
            "class_id": _to_list(detections.class_id),
            "confidence": _to_list(detections.confidence),
        }
    return fields


def serialize_keypoints_compact(output: Any, model: Any) -> dict:
    """KeyPoints → roboflow-keypoints-compact-v1"""
    output = _unwrap_batch(output)
    if isinstance(output, list):
        return {
            "type": "roboflow-keypoints-compact-v1",
            "class_names": _class_names(model),
            "batch": [_keypoints_fields(*_split_keypoints(item)) for item in output],
        }
    keypoints, detections = _split_keypoints(output)
    if isinstance(keypoints, list):
        detections_list = (
            detections if isinstance(detections, list) else [None] * len(keypoints)
        )
        return {
            "type": "roboflow-keypoints-compact-v1",
            "class_names": _class_names(model),
            "batch": [
                _keypoints_fields(kp, det)
                for kp, det in zip(keypoints, detections_list)
            ],
        }
    return {
        "type": "roboflow-keypoints-compact-v1",
        "class_names": _class_names(model),
        **_keypoints_fields(keypoints, detections),
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
    if isinstance(output, list):
        return {
            "type": "roboflow-text-v1",
            "batch": [{"text": t if isinstance(t, str) else str(t)} for t in output],
        }
    return {
        "type": "roboflow-text-v1",
        "text": output if isinstance(output, str) else str(output),
    }


def serialize_structured_ocr_compact(output: Any, model: Any) -> dict:
    """(texts, detections) tuple → roboflow-structured-ocr-compact-v1"""
    if isinstance(output, tuple) and len(output) == 2:
        texts, detections = output
    else:
        texts, detections = output, [None] * len(output)

    def _regions(det: Any) -> Any:
        if det is None:
            return None
        meta = getattr(det, "bboxes_metadata", None)
        return {
            "xyxy": _to_list(det.xyxy),
            "class_id": _to_list(det.class_id),
            "confidence": _to_list(det.confidence),
            "texts": [m.get("text") for m in meta] if meta else None,
        }

    return {
        "type": "roboflow-structured-ocr-compact-v1",
        "class_names": _class_names(model),
        "batch": [
            {"text": t, "regions": _regions(d)} for t, d in zip(texts, detections)
        ],
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


def serialize_gaze_compact(output: Any, model: Any) -> dict:
    """L2CSGazeDetection → roboflow-gaze-compact-v1"""
    output = _unwrap_batch(output)
    if isinstance(output, list):
        return {
            "type": "roboflow-gaze-compact-v1",
            "batch": [
                {"yaw": _to_list(o.yaw), "pitch": _to_list(o.pitch)} for o in output
            ],
        }
    return {
        "type": "roboflow-gaze-compact-v1",
        "yaw": _to_list(output.yaw),
        "pitch": _to_list(output.pitch),
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
    names = _detections_class_names(model, output)

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


def _classification_rich_row(row: Any, top_id: Any, names: Any) -> dict:
    """Candidates for one confidence row; top matches top_id, else the sorted first."""
    candidates = []
    for j in range(len(row)):
        c = {"class_id": j, "confidence": float(row[j])}
        if names and j < len(names):
            c["class_name"] = names[j]
        candidates.append(c)
    candidates.sort(key=lambda x: x["confidence"], reverse=True)
    if top_id is not None:
        top = [c for c in candidates if c["class_id"] == int(top_id)][:1]
    else:
        top = candidates[:1]
    return {"candidates": candidates, "top": top}


def serialize_classification_rich(output: Any, model: Any) -> dict:
    """ClassificationPrediction → roboflow-classification-rich-v1"""
    output = _unwrap_batch(output)
    if isinstance(output, list):
        return {
            "type": "roboflow-classification-rich-v1",
            "batch": [
                {
                    "candidates": (one := serialize_classification_rich(o, model))[
                        "candidates"
                    ],
                    "top": one["top"],
                }
                for o in output
            ],
        }
    names = _class_names(model)
    confidence = np.asarray(output.confidence)
    if confidence.ndim == 1:
        confidence = confidence[None, :]
    class_id = np.asarray(output.class_id).reshape(-1)
    bs = confidence.shape[0]
    top_ids = class_id if len(class_id) == bs else [None] * bs
    if bs > 1:
        return {
            "type": "roboflow-classification-rich-v1",
            "batch": [
                _classification_rich_row(confidence[i], top_ids[i], names)
                for i in range(bs)
            ],
        }
    return {
        "type": "roboflow-classification-rich-v1",
        **_classification_rich_row(confidence[0], top_ids[0], names),
    }


def serialize_instance_segmentation_rich(output: Any, model: Any) -> dict:
    """InstanceDetections → roboflow-instance-segmentation-rich-v1"""
    names = _class_names(model)
    masks = None
    if getattr(output, "mask", None) is not None:
        masks = _mask_to_json(output.mask)
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
        if masks is not None:
            det["mask"] = _to_list(masks[i])
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

"""Serialize Workflow outputs to rosbridge JSON message payloads.

Each serializer returns a list of :class:`OutboundMessage` envelopes — most
output kinds map to a single envelope, but segmentation and keypoints emit a
bundle (mask + labels + boxes / json + markers) on companion topics.

Pure-Python; safe to import without the ``ros`` extra installed.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from inference.core.workflows.core_steps.common.rosbridge.encoding import (
    encode_compressed_image,
    encode_label_image,
    now_stamp,
)


@dataclass
class OutboundMessage:
    """A single rosbridge advertise+publish target."""

    topic_suffix: str  # "" for the configured root topic; "/foo" appended otherwise
    message_type: str
    payload: Dict[str, Any]
    latch: bool = False


@dataclass
class SerializerContext:
    frame_id: str = "inference"
    ros_version: int = 2
    label_info_suffix: str = "/label_info"
    instances_suffix: str = "/instances"
    classes_suffix: str = "/classes"
    detections_suffix: str = "/detections"
    markers_suffix: str = "/markers"
    jpeg_quality: int = 90


# Message type constants (short ROS1-style; the connection layer normalizes to
# ROS2 if requested per-bridge).
MSG_DETECTION2D_ARRAY = "vision_msgs/Detection2DArray"
MSG_CLASSIFICATION = "vision_msgs/Classification"
MSG_LABEL_INFO = "vision_msgs/LabelInfo"
MSG_IMAGE = "sensor_msgs/Image"
MSG_COMPRESSED_IMAGE = "sensor_msgs/CompressedImage"
MSG_STRING = "std_msgs/String"
MSG_INT32 = "std_msgs/Int32"
MSG_FLOAT64 = "std_msgs/Float64"
MSG_BOOL = "std_msgs/Bool"
MSG_MARKER_ARRAY = "visualization_msgs/MarkerArray"


# ---------------------------------------------------------------------------
# Public dispatch
# ---------------------------------------------------------------------------


SUPPORTED_MESSAGE_TYPES: Tuple[str, ...] = (
    MSG_DETECTION2D_ARRAY,
    "semantic_segmentation",
    "instance_segmentation",
    MSG_CLASSIFICATION,
    "keypoints",
    MSG_COMPRESSED_IMAGE,
    MSG_STRING,
    MSG_INT32,
    MSG_FLOAT64,
    MSG_BOOL,
    "custom",
)


def serialize(
    message_type: str,
    value: Any,
    ctx: SerializerContext,
) -> List[OutboundMessage]:
    """Dispatch ``value`` to the serializer for ``message_type``."""
    short = _short(message_type)
    if short == MSG_DETECTION2D_ARRAY:
        return [_detections_to_array(value, ctx)]
    if short == "semantic_segmentation" or short == "vision_msgs/Segmentation":
        return _semantic_seg(value, ctx)
    if short == "instance_segmentation":
        return _instance_seg(value, ctx)
    if short == MSG_CLASSIFICATION:
        return [_classification(value, ctx)]
    if short == "keypoints":
        return _keypoints(value, ctx)
    if short == MSG_COMPRESSED_IMAGE:
        return [_compressed_image(value, ctx)]
    if short == MSG_STRING:
        return [_scalar_string(value, ctx)]
    if short == MSG_INT32:
        return [_scalar_int(value, ctx)]
    if short == MSG_FLOAT64:
        return [_scalar_float(value, ctx)]
    if short == MSG_BOOL:
        return [_scalar_bool(value, ctx)]
    if short == "custom":
        return [
            OutboundMessage(
                topic_suffix="",
                message_type=MSG_STRING,
                payload={"data": json.dumps(_to_jsonable(value))},
            )
        ]
    raise ValueError(f"unsupported message_type for serialization: {message_type}")


# ---------------------------------------------------------------------------
# Serializer implementations
# ---------------------------------------------------------------------------


def _detections_to_array(value: Any, ctx: SerializerContext) -> OutboundMessage:
    detections = _coerce_detections(value)
    items: List[Dict[str, Any]] = []
    n = len(detections.xyxy) if detections.xyxy is not None else 0
    class_names = _detection_class_names(detections)
    detection_ids = _detection_ids(detections, n)
    for i in range(n):
        x1, y1, x2, y2 = (float(v) for v in detections.xyxy[i])
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = float(x2 - x1)
        h = float(y2 - y1)
        class_id = (
            int(detections.class_id[i]) if detections.class_id is not None else -1
        )
        confidence = (
            float(detections.confidence[i])
            if detections.confidence is not None
            else 0.0
        )
        items.append(
            {
                "header": _header(ctx),
                "results": [
                    {
                        "hypothesis": {
                            "class_id": class_names[i],
                            "score": confidence,
                        },
                        "pose": _zero_pose_with_covariance(),
                    }
                ],
                "bbox": {
                    "center": {
                        "position": {"x": cx, "y": cy, "z": 0.0},
                        "theta": 0.0,
                    },
                    "size_x": w,
                    "size_y": h,
                },
                "id": detection_ids[i],
            }
        )
        # int class id is preserved separately on the outer message via
        # detection_id mapping to make joining with mask images straightforward.
    return OutboundMessage(
        topic_suffix="",
        message_type=MSG_DETECTION2D_ARRAY,
        payload={
            "header": _header(ctx),
            "detections": items,
        },
    )


def _semantic_seg(value: Any, ctx: SerializerContext) -> List[OutboundMessage]:
    detections = _coerce_detections(value)
    label_image, class_id_to_name = _paint_semantic_label_image(detections)
    out = [
        OutboundMessage(
            topic_suffix="",
            message_type=MSG_IMAGE,
            payload=encode_label_image(
                label_image, ros_version=ctx.ros_version, frame_id=ctx.frame_id
            ),
        ),
        OutboundMessage(
            topic_suffix=ctx.label_info_suffix,
            message_type=MSG_LABEL_INFO,
            payload=_label_info_payload(class_id_to_name, ctx),
            latch=True,
        ),
    ]
    return out


def _instance_seg(value: Any, ctx: SerializerContext) -> List[OutboundMessage]:
    detections = _coerce_detections(value)
    instance_image, class_image, class_id_to_name, det_ids = _paint_instance_images(
        detections
    )
    detections_msg = _detections_to_array(detections, ctx)
    # Override the auto-generated detection ids in the boxes message with the
    # same instance ids we baked into the mask, so consumers can join.
    array_payload = detections_msg.payload
    for det, inst_id in zip(array_payload["detections"], det_ids):
        det["id"] = str(inst_id)
    return [
        OutboundMessage(
            topic_suffix=ctx.instances_suffix,
            message_type=MSG_IMAGE,
            payload=encode_label_image(
                instance_image, ros_version=ctx.ros_version, frame_id=ctx.frame_id
            ),
        ),
        OutboundMessage(
            topic_suffix=ctx.classes_suffix,
            message_type=MSG_IMAGE,
            payload=encode_label_image(
                class_image, ros_version=ctx.ros_version, frame_id=ctx.frame_id
            ),
        ),
        OutboundMessage(
            topic_suffix=ctx.detections_suffix,
            message_type=MSG_DETECTION2D_ARRAY,
            payload=array_payload,
        ),
        OutboundMessage(
            topic_suffix=ctx.label_info_suffix,
            message_type=MSG_LABEL_INFO,
            payload=_label_info_payload(class_id_to_name, ctx),
            latch=True,
        ),
    ]


def _classification(value: Any, ctx: SerializerContext) -> OutboundMessage:
    if not isinstance(value, dict):
        raise TypeError(
            f"Classification serializer expects a dict, got {type(value).__name__}"
        )
    results: List[Dict[str, Any]] = []
    predictions = value.get("predictions")
    if isinstance(predictions, list):
        for p in predictions:
            results.append(
                {
                    "hypothesis": {
                        "class_id": str(p.get("class_name", p.get("class_id", ""))),
                        "score": float(p.get("confidence", 0.0)),
                    }
                }
            )
    elif isinstance(predictions, dict):
        for class_name, p in predictions.items():
            results.append(
                {
                    "hypothesis": {
                        "class_id": str(class_name),
                        "score": float(p.get("confidence", 0.0)),
                    }
                }
            )
    return OutboundMessage(
        topic_suffix="",
        message_type=MSG_CLASSIFICATION,
        payload={
            "header": _header(ctx),
            "results": results,
        },
    )


def _keypoints(value: Any, ctx: SerializerContext) -> List[OutboundMessage]:
    detections = _coerce_detections(value)
    n = len(detections.xyxy) if detections.xyxy is not None else 0
    class_names = _detection_class_names(detections)
    kp_xy = _data_field(detections, "keypoints_xy", default=[None] * n)
    kp_conf = _data_field(detections, "keypoints_confidence", default=[None] * n)
    kp_names = _data_field(detections, "keypoints_class_name", default=[None] * n)
    out_detections: List[Dict[str, Any]] = []
    markers: List[Dict[str, Any]] = []
    for i in range(n):
        confidence = (
            float(detections.confidence[i])
            if detections.confidence is not None
            else 0.0
        )
        kps = kp_xy[i] if i < len(kp_xy) else None
        confs = kp_conf[i] if i < len(kp_conf) else None
        names = kp_names[i] if i < len(kp_names) else None
        keypoints: List[Dict[str, Any]] = []
        points_payload: List[Dict[str, float]] = []
        if kps is not None:
            kps_arr = np.asarray(kps)
            for j, (x, y) in enumerate(kps_arr.tolist()):
                kp = {"x": float(x), "y": float(y)}
                kp["confidence"] = (
                    float(confs[j]) if confs is not None and j < len(confs) else 1.0
                )
                kp["name"] = (
                    str(names[j]) if names is not None and j < len(names) else str(j)
                )
                keypoints.append(kp)
                points_payload.append({"x": float(x), "y": float(y), "z": 0.0})
        out_detections.append(
            {
                "class_name": class_names[i],
                "confidence": confidence,
                "keypoints": keypoints,
            }
        )
        markers.append(_keypoint_marker(i, points_payload, ctx))
    payload_json = json.dumps({"detections": out_detections})
    return [
        OutboundMessage(
            topic_suffix="",
            message_type=MSG_STRING,
            payload={"data": payload_json},
        ),
        OutboundMessage(
            topic_suffix=ctx.markers_suffix,
            message_type=MSG_MARKER_ARRAY,
            payload={"markers": markers},
        ),
    ]


def _compressed_image(value: Any, ctx: SerializerContext) -> OutboundMessage:
    arr = _coerce_image(value)
    return OutboundMessage(
        topic_suffix="",
        message_type=MSG_COMPRESSED_IMAGE,
        payload=encode_compressed_image(
            arr,
            image_format="jpeg",
            jpeg_quality=ctx.jpeg_quality,
            ros_version=ctx.ros_version,
            frame_id=ctx.frame_id,
        ),
    )


def _scalar_string(value: Any, ctx: SerializerContext) -> OutboundMessage:
    return OutboundMessage(
        topic_suffix="",
        message_type=MSG_STRING,
        payload={"data": str(value)},
    )


def _scalar_int(value: Any, ctx: SerializerContext) -> OutboundMessage:
    return OutboundMessage(
        topic_suffix="",
        message_type=MSG_INT32,
        payload={"data": int(value)},
    )


def _scalar_float(value: Any, ctx: SerializerContext) -> OutboundMessage:
    return OutboundMessage(
        topic_suffix="",
        message_type=MSG_FLOAT64,
        payload={"data": float(value)},
    )


def _scalar_bool(value: Any, ctx: SerializerContext) -> OutboundMessage:
    return OutboundMessage(
        topic_suffix="",
        message_type=MSG_BOOL,
        payload={"data": bool(value)},
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _coerce_detections(value: Any):
    try:
        import supervision as sv  # type: ignore
    except ImportError as e:
        raise ImportError("supervision is required for detection serializers") from e
    if isinstance(value, sv.Detections):
        return value
    raise TypeError(
        f"detection serializers expect sv.Detections, got {type(value).__name__}"
    )


def _coerce_image(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    img = getattr(value, "numpy_image", None)
    if isinstance(img, np.ndarray):
        return img
    raise TypeError(
        f"image serializer expects ndarray or WorkflowImageData, got "
        f"{type(value).__name__}"
    )


def _detection_class_names(detections: Any) -> List[str]:
    n = len(detections.xyxy) if detections.xyxy is not None else 0
    raw = _data_field(detections, "class_name", default=None)
    if raw is None:
        if detections.class_id is None:
            return [""] * n
        return [str(int(c)) for c in detections.class_id]
    return [str(v) for v in raw]


def _detection_ids(detections: Any, n: int) -> List[str]:
    raw = _data_field(detections, "detection_id", default=None)
    if raw is None:
        return [str(uuid.uuid4()) for _ in range(n)]
    return [str(v) for v in raw]


def _data_field(detections: Any, key: str, default=None):
    data = getattr(detections, "data", None) or {}
    if key in data:
        return data[key]
    return default


def _short(message_type: str) -> str:
    parts = message_type.split("/")
    if len(parts) == 3 and parts[1] == "msg":
        return f"{parts[0]}/{parts[2]}"
    return message_type


def _header(ctx: SerializerContext) -> Dict[str, Any]:
    return {
        "stamp": now_stamp(ctx.ros_version),
        "frame_id": ctx.frame_id,
    }


def _zero_pose_with_covariance() -> Dict[str, Any]:
    return {
        "pose": {
            "position": {"x": 0.0, "y": 0.0, "z": 0.0},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        },
        "covariance": [0.0] * 36,
    }


def _label_info_payload(
    class_id_to_name: Dict[int, str],
    ctx: SerializerContext,
) -> Dict[str, Any]:
    return {
        "header": _header(ctx),
        "class_map": [
            {"class_id": int(class_id), "class_name": str(class_name)}
            for class_id, class_name in sorted(class_id_to_name.items())
        ],
        "threshold": 0.0,
    }


def _paint_semantic_label_image(detections: Any) -> Tuple[np.ndarray, Dict[int, str]]:
    height, width = _detections_dims(detections)
    n = len(detections.class_id) if detections.class_id is not None else 0
    if n == 0:
        return np.zeros((height, width), dtype=np.uint8), {}
    max_class = int(np.max(detections.class_id)) if n else 0
    dtype = np.uint8 if max_class < 256 else np.uint16
    label_image = np.zeros((height, width), dtype=dtype)
    class_id_to_name: Dict[int, str] = {}
    class_names = _detection_class_names(detections)
    for i in range(n):
        class_id = int(detections.class_id[i])
        class_id_to_name.setdefault(class_id, class_names[i])
        mask = _decode_detection_mask(detections, i, height, width)
        if mask is None:
            continue
        label_image[mask] = class_id
    return label_image, class_id_to_name


def _paint_instance_images(
    detections: Any,
) -> Tuple[np.ndarray, np.ndarray, Dict[int, str], List[int]]:
    height, width = _detections_dims(detections)
    n = len(detections.xyxy) if detections.xyxy is not None else 0
    instance_image = np.zeros((height, width), dtype=np.uint16)
    max_class = (
        int(np.max(detections.class_id)) if n and detections.class_id is not None else 0
    )
    class_dtype = np.uint8 if max_class < 256 else np.uint16
    class_image = np.zeros((height, width), dtype=class_dtype)
    class_id_to_name: Dict[int, str] = {}
    class_names = _detection_class_names(detections)
    instance_ids: List[int] = []
    for i in range(n):
        instance_id = i + 1  # reserve 0 for background
        instance_ids.append(instance_id)
        class_id = int(detections.class_id[i]) if detections.class_id is not None else 0
        class_id_to_name.setdefault(class_id, class_names[i])
        mask = _decode_detection_mask(detections, i, height, width)
        if mask is None:
            continue
        instance_image[mask] = instance_id
        class_image[mask] = class_id
    return instance_image, class_image, class_id_to_name, instance_ids


def _detections_dims(detections: Any) -> Tuple[int, int]:
    img_dims = _data_field(detections, "image_dimensions", default=None)
    if img_dims is not None and len(img_dims) > 0:
        h, w = img_dims[0]
        return int(h), int(w)
    if detections.mask is not None and len(detections.mask) > 0:
        h, w = detections.mask[0].shape[-2:]
        return int(h), int(w)
    rle = _data_field(detections, "rle_mask", default=None)
    if rle is not None and len(rle) > 0:
        first = rle[0]
        if isinstance(first, dict) and "size" in first:
            h, w = first["size"]
            return int(h), int(w)
    raise ValueError(
        "cannot infer image dimensions from sv.Detections — no mask, "
        "rle_mask, or image_dimensions present"
    )


def _decode_detection_mask(
    detections: Any, idx: int, height: int, width: int
) -> Optional[np.ndarray]:
    if detections.mask is not None and len(detections.mask) > idx:
        m = np.asarray(detections.mask[idx])
        if m.dtype != np.bool_:
            m = m > 0
        return m
    rle = _data_field(detections, "rle_mask", default=None)
    if rle is None or len(rle) <= idx:
        return None
    try:
        import supervision as sv  # type: ignore
    except ImportError:
        return None
    entry = rle[idx]
    # Inference emits COCO-style dicts: {"size": [H, W], "counts": "..."}.
    # Supervision's sv.mask_to_rle emits a bare list[int] or compressed str.
    if isinstance(entry, dict) and "counts" in entry:
        size = entry.get("size") or [height, width]
        h, w = int(size[0]), int(size[1])
        counts = entry["counts"]
        decoded = sv.rle_to_mask(counts, resolution_wh=(w, h))
    else:
        decoded = sv.rle_to_mask(entry, resolution_wh=(width, height))
    return np.asarray(decoded).astype(bool)


def _keypoint_marker(
    instance_idx: int,
    points: List[Dict[str, float]],
    ctx: SerializerContext,
) -> Dict[str, Any]:
    return {
        "header": _header(ctx),
        "ns": "keypoints",
        "id": int(instance_idx),
        "type": 8,  # POINTS
        "action": 0,  # ADD
        "pose": {
            "position": {"x": 0.0, "y": 0.0, "z": 0.0},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        },
        "scale": {"x": 4.0, "y": 4.0, "z": 0.0},
        "color": {"r": 0.0, "g": 1.0, "b": 0.0, "a": 1.0},
        "points": points,
    }


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value

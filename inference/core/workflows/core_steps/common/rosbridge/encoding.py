"""ROS message <-> ndarray encoding helpers for the rosbridge JSON protocol.

Pure-Python; safe to import without the ``ros`` extra installed.
"""

from __future__ import annotations

import base64
import time
from typing import Any, Dict, Tuple

import cv2
import numpy as np

# Mapping from sensor_msgs/Image.encoding -> (dtype, channels, swap_to_bgr).
# Only the encodings we actually emit/consume are handled; uncommon encodings
# fall through to the "raw" path which preserves bytes but does not reshape
# beyond a 2D array.
_IMAGE_ENCODINGS: Dict[str, Tuple[np.dtype, int, bool]] = {
    "rgb8": (np.dtype("uint8"), 3, True),
    "rgba8": (np.dtype("uint8"), 4, True),
    "bgr8": (np.dtype("uint8"), 3, False),
    "bgra8": (np.dtype("uint8"), 4, False),
    "mono8": (np.dtype("uint8"), 1, False),
    "mono16": (np.dtype("uint16"), 1, False),
    "16uc1": (np.dtype("uint16"), 1, False),
    "32fc1": (np.dtype("float32"), 1, False),
}


def now_stamp(ros_version: int = 2) -> Dict[str, int]:
    """Return a ROS time stamp dict in the version-appropriate field shape."""
    t = time.time()
    secs = int(t)
    nsecs = int((t - secs) * 1e9)
    if ros_version == 1:
        return {"secs": secs, "nsecs": nsecs}
    return {"sec": secs, "nanosec": nsecs}


def _coerce_data_bytes(data: Any) -> bytes:
    """rosbridge delivers binary fields as base64-encoded str (default JSON
    transport) or raw bytes (when subscribed with compression=cbor-raw).
    Normalize both."""
    if isinstance(data, (bytes, bytearray, memoryview)):
        return bytes(data)
    if isinstance(data, str):
        return base64.b64decode(data)
    raise TypeError(f"unsupported data field type: {type(data).__name__}")


def decode_image_message(
    msg: Dict[str, Any],
    message_type: str,
) -> np.ndarray:
    """Decode a rosbridge JSON Image / CompressedImage payload to ndarray.

    Returns a BGR ndarray (matching cv2 conventions throughout inference).
    """
    short_type = _short_type(message_type)
    if short_type == "sensor_msgs/CompressedImage":
        raw = _coerce_data_bytes(msg["data"])
        buf = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("cv2.imdecode failed on CompressedImage payload")
        return img
    if short_type == "sensor_msgs/Image":
        return _decode_raw_image(msg)
    raise ValueError(f"unsupported image message type: {message_type}")


def _decode_raw_image(msg: Dict[str, Any]) -> np.ndarray:
    encoding = str(msg.get("encoding", "")).lower()
    height = int(msg["height"])
    width = int(msg["width"])
    raw = _coerce_data_bytes(msg["data"])

    spec = _IMAGE_ENCODINGS.get(encoding)
    if spec is None:
        # Fall back: best-effort 2D mono8 reshape.
        arr = np.frombuffer(raw, dtype=np.uint8)
        if arr.size == height * width:
            return arr.reshape(height, width)
        raise ValueError(f"unsupported sensor_msgs/Image encoding: {encoding}")

    dtype, channels, needs_bgr_swap = spec
    expected = height * width * channels * dtype.itemsize
    if len(raw) < expected:
        raise ValueError(
            f"sensor_msgs/Image data truncated: expected {expected} bytes, "
            f"got {len(raw)} for {width}x{height}x{channels} {dtype}"
        )
    arr = np.frombuffer(raw[:expected], dtype=dtype)
    if channels == 1:
        arr = arr.reshape(height, width)
    else:
        arr = arr.reshape(height, width, channels)
    if needs_bgr_swap and channels >= 3:
        if channels == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        else:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)
    return arr


def encode_compressed_image(
    bgr: np.ndarray,
    image_format: str = "jpeg",
    jpeg_quality: int = 90,
    ros_version: int = 2,
    frame_id: str = "inference",
) -> Dict[str, Any]:
    """Build a ``sensor_msgs/CompressedImage`` payload from a BGR ndarray."""
    fmt = image_format.lower()
    if fmt in ("jpeg", "jpg"):
        ext = ".jpg"
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
        format_str = "jpeg"
    elif fmt == "png":
        ext = ".png"
        params = []
        format_str = "png"
    else:
        raise ValueError(f"unsupported image_format: {image_format}")
    ok, buf = cv2.imencode(ext, bgr, params)
    if not ok:
        raise ValueError(f"cv2.imencode failed for format {image_format}")
    return {
        "header": {
            "stamp": now_stamp(ros_version),
            "frame_id": frame_id,
        },
        "format": format_str,
        "data": base64.b64encode(buf.tobytes()).decode("ascii"),
    }


def encode_label_image(
    label_image: np.ndarray,
    ros_version: int = 2,
    frame_id: str = "inference",
) -> Dict[str, Any]:
    """Build a ``sensor_msgs/Image`` payload carrying a single-channel label
    image (mono8 if dtype=uint8, mono16 if dtype=uint16)."""
    if label_image.ndim != 2:
        raise ValueError("label_image must be 2D (H, W)")
    if label_image.dtype == np.uint8:
        encoding = "mono8"
    elif label_image.dtype == np.uint16:
        encoding = "mono16"
    else:
        raise ValueError(
            f"label_image must be uint8 or uint16, got {label_image.dtype}"
        )
    height, width = label_image.shape
    step = width * label_image.dtype.itemsize
    return {
        "header": {
            "stamp": now_stamp(ros_version),
            "frame_id": frame_id,
        },
        "height": int(height),
        "width": int(width),
        "encoding": encoding,
        "is_bigendian": 0,
        "step": int(step),
        "data": base64.b64encode(label_image.tobytes()).decode("ascii"),
    }


def _short_type(message_type: str) -> str:
    """Collapse ROS2 ``pkg/msg/Msg`` to the canonical short ``pkg/Msg`` form
    for routing decisions."""
    parts = message_type.split("/")
    if len(parts) == 3 and parts[1] == "msg":
        return f"{parts[0]}/{parts[2]}"
    return message_type

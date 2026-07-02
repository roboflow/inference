from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np
import pyclipper
import torch
import torch.nn.functional as F
import yaml
from shapely.geometry import Polygon

from inference_models.errors import ModelInputError

DEFAULT_LIMIT_SIDE_LEN = 736
DEFAULT_LIMIT_TYPE = "min"
DEFAULT_MEAN = (0.485, 0.456, 0.406)
DEFAULT_STD = (0.229, 0.224, 0.225)
DEFAULT_SCALE = 1.0 / 255.0
DEFAULT_THRESH = 0.3
DEFAULT_BOX_THRESH = 0.6
DEFAULT_UNCLIP_RATIO = 1.5
DEFAULT_MAX_CANDIDATES = 1000
MIN_BOX_SIDE = 3


@dataclass
class DBNetConfig:
    limit_side_len: int
    limit_type: str
    mean: Tuple[float, float, float]
    std: Tuple[float, float, float]
    scale: float
    to_rgb: bool
    thresh: float
    box_thresh: float
    unclip_ratio: float
    max_candidates: int


def load_detection_config(config_path: str) -> DBNetConfig:
    with open(config_path, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ModelInputError(
            message=f"Could not parse PP-OCRv6 detection config: {config_path}",
            help_url="https://inference-models.roboflow.com/errors/input-validation/#modelinputerror",
        )
    transform_ops = (config.get("PreProcess") or {}).get("transform_ops") or []
    normalize_op = _find_op(transform_ops, "NormalizeImage") or {}
    resize_op = _find_op(transform_ops, "DetResizeForTest") or {}
    decode_op = _find_op(transform_ops, "DecodeImage") or {}
    post_process = config.get("PostProcess") or {}
    return DBNetConfig(
        limit_side_len=int(resize_op.get("limit_side_len", DEFAULT_LIMIT_SIDE_LEN)),
        limit_type=str(resize_op.get("limit_type", DEFAULT_LIMIT_TYPE)),
        mean=tuple(normalize_op.get("mean", DEFAULT_MEAN)),
        std=tuple(normalize_op.get("std", DEFAULT_STD)),
        scale=_parse_scale(normalize_op.get("scale")),
        to_rgb=str(decode_op.get("img_mode", "BGR")).upper() == "RGB",
        thresh=float(post_process.get("thresh", DEFAULT_THRESH)),
        box_thresh=float(post_process.get("box_thresh", DEFAULT_BOX_THRESH)),
        unclip_ratio=float(post_process.get("unclip_ratio", DEFAULT_UNCLIP_RATIO)),
        max_candidates=int(post_process.get("max_candidates", DEFAULT_MAX_CANDIDATES)),
    )


def _find_op(transform_ops: List[Any], op_name: str) -> Optional[dict]:
    for op in transform_ops:
        if isinstance(op, dict) and op_name in op:
            return op[op_name] or {}
    return None


def _parse_scale(value: Any) -> float:
    if value is None:
        return DEFAULT_SCALE
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if "/" in text:
        numerator, denominator = text.split("/", 1)
        return float(numerator) / float(denominator)
    return float(text)


def resize_for_detection(
    image: np.ndarray, limit_side_len: int, limit_type: str
) -> Tuple[np.ndarray, float, float]:
    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        raise ModelInputError(
            message="Cannot run PP-OCRv6 detection on an empty image.",
            help_url="https://inference-models.roboflow.com/errors/input-validation/#modelinputerror",
        )
    if limit_type == "max":
        ratio = (
            limit_side_len / max(height, width)
            if max(height, width) > limit_side_len
            else 1.0
        )
    elif limit_type == "min":
        ratio = (
            limit_side_len / min(height, width)
            if min(height, width) < limit_side_len
            else 1.0
        )
    else:
        ratio = limit_side_len / max(height, width)
    resize_h = max(int(round(height * ratio / 32) * 32), 32)
    resize_w = max(int(round(width * ratio / 32) * 32), 32)
    resized = cv2.resize(image, (resize_w, resize_h))
    return resized, resize_h / float(height), resize_w / float(width)


def normalize_detection_image(image_bgr: np.ndarray, config: DBNetConfig) -> np.ndarray:
    image = image_bgr[:, :, ::-1] if config.to_rgb else image_bgr
    mean = np.array(config.mean, dtype="float32")
    std = np.array(config.std, dtype="float32")
    normalized = (image.astype("float32") * config.scale - mean) / std
    return np.ascontiguousarray(np.transpose(normalized, (2, 0, 1))[np.newaxis, ...])


def resize_for_detection_torch(
    image: torch.Tensor, limit_side_len: int, limit_type: str
) -> torch.Tensor:
    """Device-native counterpart of ``resize_for_detection``.

    ``image`` is a ``CHW`` float tensor in ``[0, 255]``; returns the resized
    ``NCHW`` tensor (multiple of 32 on each side) on the same device, without a
    numpy/cv2 round-trip.
    """
    _, height, width = image.shape
    if height <= 0 or width <= 0:
        raise ModelInputError(
            message="Cannot run PP-OCRv6 detection on an empty image.",
            help_url="https://inference-models.roboflow.com/errors/input-validation/#modelinputerror",
        )
    if limit_type == "max":
        ratio = (
            limit_side_len / max(height, width)
            if max(height, width) > limit_side_len
            else 1.0
        )
    elif limit_type == "min":
        ratio = (
            limit_side_len / min(height, width)
            if min(height, width) < limit_side_len
            else 1.0
        )
    else:
        ratio = limit_side_len / max(height, width)
    resize_h = max(int(round(height * ratio / 32) * 32), 32)
    resize_w = max(int(round(width * ratio / 32) * 32), 32)
    return F.interpolate(
        image.unsqueeze(0),
        size=(resize_h, resize_w),
        mode="bilinear",
        align_corners=False,
    )


def normalize_detection_image_torch(
    image_bgr: torch.Tensor, config: DBNetConfig
) -> torch.Tensor:
    """Device-native counterpart of ``normalize_detection_image``.

    ``image_bgr`` is an ``NCHW`` BGR float tensor in ``[0, 255]``; returns the
    normalized ``NCHW`` tensor on the same device.
    """
    image = image_bgr.flip(1) if config.to_rgb else image_bgr
    mean = torch.tensor(config.mean, dtype=image.dtype, device=image.device).view(
        1, 3, 1, 1
    )
    std = torch.tensor(config.std, dtype=image.dtype, device=image.device).view(
        1, 3, 1, 1
    )
    return (image * config.scale - mean) / std


def boxes_from_probability_map(
    probability_map: np.ndarray,
    source_height: int,
    source_width: int,
    config: DBNetConfig,
) -> List[Tuple[np.ndarray, float]]:
    """Decode a DBNet probability map into (quadrilateral, score) detections.

    Each quadrilateral is a (4, 2) int32 array of corner points in the original
    image coordinate system.
    """
    bitmap = (probability_map > config.thresh).astype(np.uint8)
    map_height, map_width = bitmap.shape
    contours, _ = cv2.findContours(bitmap * 255, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    detections: List[Tuple[np.ndarray, float]] = []
    for contour in contours[: config.max_candidates]:
        points, min_side = _min_area_quad(contour)
        if min_side < MIN_BOX_SIDE:
            continue
        score = _box_score(probability_map, points)
        if score < config.box_thresh:
            continue
        expanded = _unclip(points, config.unclip_ratio)
        if expanded is None:
            continue
        box, min_side = _min_area_quad(expanded.reshape(-1, 1, 2))
        if min_side < MIN_BOX_SIDE + 2:
            continue
        box[:, 0] = np.clip(box[:, 0] / map_width * source_width, 0, source_width)
        box[:, 1] = np.clip(box[:, 1] / map_height * source_height, 0, source_height)
        # Drop sub-line-sized boxes in source coordinates, matching PaddleOCR's
        # text-detector `filter_tag_det_res`, which discards any box whose width or
        # height is <= 3px in the original image. The earlier `_min_area_quad`
        # checks are in bitmap coordinates, so a blob can still scale down to a
        # few source pixels and surface as a spurious (non-text) detection.
        box_width = int(np.linalg.norm(box[0] - box[1]))
        box_height = int(np.linalg.norm(box[0] - box[3]))
        if box_width <= MIN_BOX_SIDE or box_height <= MIN_BOX_SIDE:
            continue
        detections.append((box.astype(np.int32), float(score)))
    return detections


def _min_area_quad(contour: np.ndarray) -> Tuple[np.ndarray, float]:
    rectangle = cv2.minAreaRect(contour)
    points = sorted(cv2.boxPoints(rectangle).tolist(), key=lambda point: point[0])
    if points[1][1] > points[0][1]:
        index_1, index_4 = 0, 1
    else:
        index_1, index_4 = 1, 0
    if points[3][1] > points[2][1]:
        index_2, index_3 = 2, 3
    else:
        index_2, index_3 = 3, 2
    quad = np.array(
        [points[index_1], points[index_2], points[index_3], points[index_4]],
        dtype="float32",
    )
    return quad, float(min(rectangle[1]))


def _box_score(probability_map: np.ndarray, quad: np.ndarray) -> float:
    height, width = probability_map.shape[:2]
    box = quad.copy()
    x_min = int(np.clip(np.floor(box[:, 0].min()), 0, width - 1))
    x_max = int(np.clip(np.ceil(box[:, 0].max()), 0, width - 1))
    y_min = int(np.clip(np.floor(box[:, 1].min()), 0, height - 1))
    y_max = int(np.clip(np.ceil(box[:, 1].max()), 0, height - 1))
    mask = np.zeros((y_max - y_min + 1, x_max - x_min + 1), dtype=np.uint8)
    box[:, 0] -= x_min
    box[:, 1] -= y_min
    cv2.fillPoly(mask, [box.reshape(-1, 2).astype(np.int32)], 1)
    return cv2.mean(probability_map[y_min : y_max + 1, x_min : x_max + 1], mask)[0]


def _unclip(quad: np.ndarray, unclip_ratio: float) -> Optional[np.ndarray]:
    polygon = Polygon(quad)
    if polygon.length == 0:
        return None
    distance = polygon.area * unclip_ratio / polygon.length
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(
        [tuple(point) for point in quad],
        pyclipper.JT_ROUND,
        pyclipper.ET_CLOSEDPOLYGON,
    )
    expanded = offset.Execute(distance)
    if not expanded:
        return None
    largest = max(
        expanded, key=lambda path: cv2.contourArea(np.array(path, dtype=np.float32))
    )
    return np.array(largest, dtype="float32")

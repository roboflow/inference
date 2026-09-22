from __future__ import annotations

import io
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel

from inference_server.legacy.bridge import Route
from inference_server.legacy.common import ImagePayload
from inference_server.legacy.errors import LegacyHTTPError

DEFAULT_COLOR_PALETTE = [
    "#4892EA",
    "#00EEC3",
    "#FE4EF0",
    "#F4004E",
    "#FA7200",
    "#EEEE17",
    "#90FF00",
    "#78C1D2",
    "#8C29FF",
]

_FALLBACK_COLOR = "#4892EA"
_JPEG_QUALITY = 90
_NPY_MAGIC = b"\x93NUMPY"
_IMAGE_ERROR = "Could not load valid image from request."
_DETECTION_TASK_TYPES = frozenset(
    [
        "object-detection",
        "instance-segmentation",
        "keypoint-detection",
        "open-vocabulary-object-detection",
    ]
)
_CLASSIFICATION_TASK_TYPES = frozenset(["classification", "multi-label-classification"])


def render_visualization(
    route: Route, request: Any, response: BaseModel, payload: ImagePayload
) -> bytes:
    colors = class_colors(route)
    stroke_width = getattr(request, "visualization_stroke_width", None) or 1
    if route.task_type in _DETECTION_TASK_TYPES:
        return draw_detection_predictions(
            image=payload_to_rgb(payload),
            predictions=response.predictions,
            colors=colors,
            thickness=stroke_width,
            labels=bool(getattr(request, "visualization_labels", False)),
        )
    if route.task_type in _CLASSIFICATION_TASK_TYPES:
        return draw_classification_predictions(
            image=payload_to_rgb(payload),
            predictions=response.predictions,
            colors=colors,
            thickness=stroke_width,
        )
    raise LegacyHTTPError(
        501,
        f"Visualization is not supported for task type '{route.task_type}'.",
    )


def class_colors(route: Route) -> Dict[str, str]:
    # NOTE: the legacy per-model colour fetch from the Roboflow API is not ported.
    if route.class_colors:
        return route.class_colors
    return default_color_mapping(class_names=route.class_names or [])


def default_color_mapping(class_names: List[str]) -> Dict[str, str]:
    return {
        class_name: DEFAULT_COLOR_PALETTE[i % len(DEFAULT_COLOR_PALETTE)]
        for i, class_name in enumerate(class_names)
    }


def payload_to_rgb(payload: ImagePayload) -> np.ndarray:
    data = payload.data
    if isinstance(data, np.ndarray):
        image_bgr = data
    elif data[:6] == _NPY_MAGIC:
        image_bgr = np.load(io.BytesIO(data), allow_pickle=False)
    else:
        image_bgr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise LegacyHTTPError(400, _IMAGE_ERROR)
    return cv2.cvtColor(np.asarray(image_bgr, dtype=np.uint8), cv2.COLOR_BGR2RGB)


def draw_detection_predictions(
    image: np.ndarray,
    predictions: List[Any],
    colors: Dict[str, str],
    thickness: int,
    labels: bool,
) -> bytes:
    for box in predictions:
        color = tuple(
            int(colors.get(box.class_name, _FALLBACK_COLOR)[i : i + 2], 16)
            for i in (1, 3, 5)
        )
        image = draw_bbox(image=image, box=box, color=color, thickness=thickness)
        if hasattr(box, "points"):
            image = draw_instance_segmentation_points(
                image=image, points=box.points, color=color, thickness=thickness
            )
        if hasattr(box, "keypoints"):
            draw_keypoints(
                image=image,
                keypoints=box.keypoints,
                color=color,
                thickness=thickness,
            )
        if labels:
            image = draw_labels(image=image, box=box, color=color)
    return encode_image_to_jpeg_bytes(image=image[:, :, ::-1])


def draw_bbox(
    image: np.ndarray, box: Any, color: Tuple[int, ...], thickness: int
) -> np.ndarray:
    left_top, right_bottom = bbox_to_points(box=box)
    return cv2.rectangle(
        image,
        left_top,
        right_bottom,
        color=color,
        thickness=thickness,
    )


def draw_instance_segmentation_points(
    image: np.ndarray, points: List[Any], color: Tuple[int, ...], thickness: int
) -> np.ndarray:
    points_array = np.array([(int(p.x), int(p.y)) for p in points], np.int32)
    if len(points) > 2:
        image = cv2.polylines(
            image,
            [points_array],
            isClosed=True,
            color=color,
            thickness=thickness,
        )
    return image


def draw_keypoints(
    image: np.ndarray, keypoints: List[Any], color: Tuple[int, ...], thickness: int
) -> None:
    for keypoint in keypoints:
        center_coordinates = (round(keypoint.x), round(keypoint.y))
        image = cv2.circle(
            image,
            center_coordinates,
            thickness,
            color,
            -1,
        )


def draw_labels(image: np.ndarray, box: Any, color: Tuple[int, ...]) -> np.ndarray:
    (x1, y1), _ = bbox_to_points(box=box)
    text = f"{box.class_name} {box.confidence:.2f}"
    (text_width, text_height), _ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    )
    button_size = (text_width + 20, text_height + 20)
    button_img = np.full(
        (button_size[1], button_size[0], 3), color[::-1], dtype=np.uint8
    )
    cv2.putText(
        button_img,
        text,
        (10, 10 + text_height),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )
    end_x = min(x1 + button_size[0], image.shape[1])
    end_y = min(y1 + button_size[1], image.shape[0])
    image[y1:end_y, x1:end_x] = button_img[: end_y - y1, : end_x - x1]
    return image


def bbox_to_points(box: Any) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    x1 = int(box.x - box.width / 2)
    x2 = int(box.x + box.width / 2)
    y1 = int(box.y - box.height / 2)
    y2 = int(box.y + box.height / 2)
    return (x1, y1), (x2, y2)


def draw_classification_predictions(
    image: np.ndarray,
    predictions: Any,
    colors: Dict[str, str],
    thickness: int,
) -> bytes:
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)
    font = ImageFont.load_default()
    if isinstance(predictions, list):
        if predictions:
            prediction = predictions[0]
            color = colors.get(prediction.class_name, _FALLBACK_COLOR)
            draw.rectangle(
                [0, 0, pil_image.size[1], pil_image.size[0]],
                outline=color,
                width=thickness,
            )
            text = (
                f"{prediction.class_id} - {prediction.class_name} "
                f"{prediction.confidence:.2f}"
            )
            _paste_label(pil_image, font, text, color, row=0)
    else:
        if len(predictions) > 0:
            draw.rectangle(
                [0, 0, pil_image.size[1], pil_image.size[0]],
                outline=_FALLBACK_COLOR,
                width=thickness,
            )
        row = 0
        ordered = sorted(
            predictions.items(), key=lambda item: item[1].confidence, reverse=True
        )
        for class_name, prediction in ordered:
            color = colors.get(class_name, _FALLBACK_COLOR)
            text = f"{class_name} {prediction.confidence:.2f}"
            row += _paste_label(pil_image, font, text, color, row=row)
    buffered = io.BytesIO()
    pil_image = pil_image.convert("RGB")
    pil_image.save(buffered, format="JPEG")
    return buffered.getvalue()


def _paste_label(image: Image.Image, font: Any, text: str, color: str, row: int) -> int:
    text_size = font.getbbox(text)
    button_size = (text_size[2] + 20, text_size[3] + 20)
    button_img = Image.new("RGBA", button_size, color)
    button_draw = ImageDraw.Draw(button_img)
    button_draw.text((10, 10), text, font=font, fill=(255, 255, 255, 255))
    image.paste(button_img, (0, row))
    return button_size[1]


def encode_image_to_jpeg_bytes(image: np.ndarray) -> bytes:
    encoding_param = [int(cv2.IMWRITE_JPEG_QUALITY), _JPEG_QUALITY]
    _, img_encoded = cv2.imencode(".jpg", image, encoding_param)
    return np.array(img_encoded).tobytes()

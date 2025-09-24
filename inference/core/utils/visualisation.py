from typing import Dict, List, Tuple, Union

import cv2
import numpy as np

from inference.core.entities.requests.inference import (
    InstanceSegmentationInferenceRequest,
    KeypointsDetectionInferenceRequest,
    ObjectDetectionInferenceRequest,
)
from inference.core.entities.responses.inference import (
    InstanceSegmentationPrediction,
    Keypoint,
    KeypointsPrediction,
    ObjectDetectionInferenceResponse,
    ObjectDetectionPrediction,
    Point,
)
from inference.core.utils.image_utils import encode_image_to_jpeg_bytes, load_image_rgb


def draw_detection_predictions(
    inference_request: Union[
        ObjectDetectionInferenceRequest,
        InstanceSegmentationInferenceRequest,
        KeypointsDetectionInferenceRequest,
    ],
    inference_response: Union[
        ObjectDetectionInferenceResponse,
        InstanceSegmentationPrediction,
        KeypointsPrediction,
    ],
    colors: Dict[str, str],
) -> bytes:
    image = load_image_rgb(inference_request.image)
    for box in inference_response.predictions:
        color = tuple(
            int(colors.get(box.class_name, "#4892EA")[i : i + 2], 16) for i in (1, 3, 5)
        )
        image = draw_bbox(
            image=image,
            box=box,
            color=color,
            thickness=inference_request.visualization_stroke_width,
        )
        if hasattr(box, "points"):
            image = draw_instance_segmentation_points(
                image=image,
                points=box.points,
                color=color,
                thickness=inference_request.visualization_stroke_width,
            )
        if hasattr(box, "keypoints"):
            draw_keypoints(
                image=image,
                keypoints=box.keypoints,
                color=color,
                thickness=inference_request.visualization_stroke_width,
            )
        if inference_request.visualization_labels:
            image = draw_labels(
                image=image,
                box=box,
                color=color,
            )
    image_bgr = image[:, :, ::-1]
    return encode_image_to_jpeg_bytes(image=image_bgr)


def draw_bbox(
    image: np.ndarray,
    box: ObjectDetectionPrediction,
    color: Tuple[int, ...],
    thickness: int,
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
    image: np.ndarray,
    points: List[Point],
    color: Tuple[int, ...],
    thickness: int,
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
    image: np.ndarray,
    keypoints: List[Keypoint],
    color: Tuple[int, ...],
    thickness: int,
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


def draw_labels(
    image: np.ndarray,
    box: Union[ObjectDetectionPrediction, InstanceSegmentationPrediction],
    color: Tuple[int, ...],
) -> np.ndarray:
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


def bbox_to_points(
    box: Union[ObjectDetectionPrediction, InstanceSegmentationPrediction],
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    x1 = int(box.x - box.width / 2)
    x2 = int(box.x + box.width / 2)
    y1 = int(box.y - box.height / 2)
    y2 = int(box.y + box.height / 2)
    return (x1, y1), (x2, y2)

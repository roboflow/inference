from datetime import datetime
from typing import Tuple, Optional, Callable

import cv2
import numpy as np
import supervision as sv
from inference.core.interfaces.camera.entities import VideoFrame
from inference.core.utils.preprocess import letterbox_image

DEFAULT_ANNOTATOR = sv.BoxAnnotator()
DEFAULT_FPS_MONITOR = sv.FPSMonitor()


def display_image(image: np.ndarray) -> None:
    cv2.imshow("Predictions", image)
    cv2.waitKey(1)


def render_predictions(
    video_frame: VideoFrame,
    predictions: dict,
    annotator: sv.BoxAnnotator = DEFAULT_ANNOTATOR,
    display_size: Optional[Tuple[int, int]] = (1280, 720),
    fps_monitor: Optional[sv.FPSMonitor] = DEFAULT_FPS_MONITOR,
    display_statistics: bool = False,
    on_frame_rendered: Callable[[np.ndarray], None] = display_image,
) -> None:
    fps_value = None
    if fps_monitor is not None:
        fps_monitor.tick()
        fps_value = fps_monitor()
    labels = [p["class"] for p in predictions["predictions"]]
    detections = sv.Detections.from_roboflow(predictions)
    image = annotator.annotate(
        scene=video_frame.image, detections=detections, labels=labels
    )
    if display_size is not None:
        image = letterbox_image(image, desired_size=display_size)
    if display_statistics:
        image = render_statistics(
            image=image, frame_timestamp=video_frame.frame_timestamp, fps=fps_value
        )
    on_frame_rendered(image)


def render_statistics(
    image: np.ndarray, frame_timestamp: datetime, fps: Optional[float]
) -> np.ndarray:
    latency = round((datetime.now() - frame_timestamp).total_seconds() * 1000, 2)
    image_height = image.shape[0]
    image = cv2.putText(
        image,
        f"LATENCY: {latency} ms",
        (10, image_height - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )
    if fps is not None:
        fps = round(fps, 2)
        image = cv2.putText(
            image,
            f"THROUGHPUT: {fps}",
            (10, image_height - 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
    return image

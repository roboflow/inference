import json
from datetime import datetime
import socket
from typing import Tuple, Optional, Callable, List

import cv2
import numpy as np
import supervision as sv

from inference.core import logger
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


class UDPSink:

    @classmethod
    def init(cls, ip_address: str, port: int) -> "UDPSink":
        udp_socket = socket.socket(
            family=socket.AF_INET, type=socket.SOCK_DGRAM
        )
        udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1)
        udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 65536)
        return cls(
            ip_address=ip_address,
            port=port,
            udp_socket=udp_socket,
        )

    def __init__(self, ip_address: str, port: int, udp_socket: socket.socket):
        self._ip_address = ip_address
        self._port = port
        self._socket = udp_socket

    def send_predictions(
        self,
        video_frame: VideoFrame,
        predictions: dict,
    ) -> None:
        inference_metadata = {
            "frame_id": video_frame.frame_id,
            "frame_decoding_time": video_frame.frame_timestamp.isoformat(),
            "emission_time": datetime.now().isoformat()
        }
        predictions["inference_metadata"] = inference_metadata
        serialised_predictions = json.dumps(predictions).encode("utf-8")
        self._socket.sendto(
            serialised_predictions,
            (
                self._ip_address,
                self._port,
            ),
        )


def multi_sink(
    video_frame: VideoFrame,
    predictions: dict,
    sinks: List[Callable[[VideoFrame, dict], None]],
) -> None:
    for sink in sinks:
        try:
            sink(video_frame, predictions)
        except Exception as error:
            logger.error(
                f"Could not sent prediction with frame_id={video_frame.frame_id} to sink {sink.__name__} "
                f"due to error: {error}."
            )
import json
import socket
from datetime import datetime
from functools import partial
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
import supervision as sv

from inference.core import logger
from inference.core.active_learning.middlewares import ActiveLearningMiddleware
from inference.core.interfaces.camera.entities import VideoFrame
from inference.core.utils.preprocess import letterbox_image

DEFAULT_ANNOTATOR = sv.BoxAnnotator()
DEFAULT_FPS_MONITOR = sv.FPSMonitor()


def display_image(image: np.ndarray) -> None:
    cv2.imshow("Predictions", image)
    cv2.waitKey(1)


def render_boxes(
    predictions: dict,
    video_frame: VideoFrame,
    annotator: sv.BoxAnnotator = DEFAULT_ANNOTATOR,
    display_size: Optional[Tuple[int, int]] = (1280, 720),
    fps_monitor: Optional[sv.FPSMonitor] = DEFAULT_FPS_MONITOR,
    display_statistics: bool = False,
    on_frame_rendered: Callable[[np.ndarray], None] = display_image,
) -> None:
    """
    Helper tool to render object detection predictions on top of video frame. It is designed
    to be used with `InferencePipeline`, as sink for predictions. By default, it uses standard `sv.BoxAnnotator()`
    to draw bounding boxes and resizes prediction to 1280x720 (keeping aspect ratio and adding black padding).
    One may configure default behaviour, for instance to display latency and throughput statistics.

    This sink is only partially compatible with stubs and classification models (it will not fail,
    although predictions will not be displayed).

    Args:
        predictions (dict): Roboflow object detection predictions with Bounding Boxes
        video_frame (VideoFrame): frame of video with its basic metadata emitted by `VideoSource`
        annotator (sv.BoxAnnotator): Annotator used to draw Bounding Boxes - if custom object is not passed,
            default is used.
        display_size (Tuple[int, int]): tuple in format (width, height) to resize visualisation output
        fps_monitor (Optional[sv.FPSMonitor]): FPS monitor used to monitor throughput
        display_statistics (bool): Flag to decide if throughput and latency can be displayed in the result image,
            if enabled, throughput will only be presented if `fps_monitor` is not None
        on_frame_rendered (Callable[[np.ndarray], None]): callback to be called once frame is rendered - by default,
            function will display OpenCV window.

    Returns: None
    Side effects: on_frame_rendered() is called against the np.ndarray produced from video frame
        and predictions.

    Example:
        ```python
        from functools import partial
        import cv2
        from inference import InferencePipeline
        from inference.core.interfaces.stream.sinks import render_boxes

        output_size = (640, 480)
        video_sink = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 25.0, output_size)
        on_prediction = partial(render_boxes, display_size=output_size, on_frame_rendered=video_sink.write)

        pipeline = InferencePipeline.init(
             model_id="your-model/3",
             video_reference="./some_file.mp4",
             on_prediction=on_prediction,
        )
        pipeline.start()
        pipeline.join()
        video_sink.release()
        ```

        In this example, `render_boxes()` is used as a sink for `InferencePipeline` predictions - making frames with
        predictions displayed to be saved into video file.
    """
    fps_value = None
    if fps_monitor is not None:
        fps_monitor.tick()
        fps_value = fps_monitor()
    try:
        labels = [p["class"] for p in predictions["predictions"]]
        detections = sv.Detections.from_roboflow(predictions)
        image = annotator.annotate(
            scene=video_frame.image.copy(), detections=detections, labels=labels
        )
    except (TypeError, KeyError):
        logger.warning(
            f"Used `render_boxes(...)` sink, but predictions that were provided do not match the expected format "
            f"of object detection prediction that could be accepted by `supervision.Detection.from_roboflow(...)"
        )
        image = video_frame.image.copy()
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
        """
        Creates `InferencePipeline` predictions sink capable of sending model predictions over network
        using UDP socket.

        As an `inference` user, please use .init() method instead of constructor to instantiate objects.
        Args:
            ip_address (str): IP address to send predictions
            port (int): Port to send predictions

        Returns: Initialised object of `UDPSink` class.
        """
        udp_socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
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
        predictions: dict,
        video_frame: VideoFrame,
    ) -> None:
        """
        Method to send predictions via UDP socket. Useful in combination with `InferencePipeline` as
        a sink for predictions.

        Args:
            predictions (dict): Roboflow object detection predictions with Bounding Boxes
            video_frame (VideoFrame): frame of video with its basic metadata emitted by `VideoSource`

        Returns: None
        Side effects: Sends serialised `predictions` and `video_frame` metadata via the UDP socket as
            JSON string. It adds key named "inference_metadata" into `predictions` dict (mutating its
            state). "inference_metadata" contain id of the frame, frame grabbing timestamp and message
            emission time in datetime iso format.

        Example:
            ```python
            import cv2
            from inference.core.interfaces.stream.inference_pipeline import InferencePipeline
            from inference.core.interfaces.stream.sinks import UDPSink

            udp_sink = UDPSink.init(ip_address="127.0.0.1", port=9090)

            pipeline = InferencePipeline.init(
                 model_id="your-model/3",
                 video_reference="./some_file.mp4",
                 on_prediction=udp_sink.send_predictions,
            )
            pipeline.start()
            pipeline.join()
            ```
            `UDPSink` used in this way will emit predictions to receiver automatically.
        """
        inference_metadata = {
            "frame_id": video_frame.frame_id,
            "frame_decoding_time": video_frame.frame_timestamp.isoformat(),
            "emission_time": datetime.now().isoformat(),
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
    predictions: dict,
    video_frame: VideoFrame,
    sinks: List[Callable[[dict, VideoFrame], None]],
) -> None:
    """
    Helper util useful to combine multiple sinks together, while using `InferencePipeline`.

    Args:
        video_frame (VideoFrame): frame of video with its basic metadata emitted by `VideoSource`
        predictions (dict): Roboflow object detection predictions with Bounding Boxes
        sinks (List[Callable[[VideoFrame, dict], None]]): list of sinks to be used. Each will be executed
            one-by-one in the order pointed in input list, all errors will be caught and reported via logger,
            without re-raising.

    Returns: None
    Side effects: Uses all sinks in context if (video_frame, predictions) input.

    Example:
        ```python
        from functools import partial
        import cv2
        from inference import InferencePipeline
        from inference.core.interfaces.stream.sinks import UDPSink, render_boxes

        udp_sink = UDPSink(ip_address="127.0.0.1", port=9090)
        on_prediction = partial(multi_sink, sinks=[udp_sink.send_predictions, render_boxes])

        pipeline = InferencePipeline.init(
            model_id="your-model/3",
            video_reference="./some_file.mp4",
            on_prediction=on_prediction,
        )
        pipeline.start()
        pipeline.join()
        ```

        As a result, predictions will both be sent via UDP socket and displayed in the screen.
    """
    for sink in sinks:
        try:
            sink(predictions, video_frame)
        except Exception as error:
            logger.error(
                f"Could not sent prediction with frame_id={video_frame.frame_id} to sink "
                f"due to error: {error}."
            )


def active_learning_sink(
    predictions: dict,
    video_frame: VideoFrame,
    active_learning_middleware: ActiveLearningMiddleware,
    model_type: str,
    disable_preproc_auto_orient: bool = False,
) -> None:
    active_learning_middleware.register(
        inference_input=video_frame.image,
        prediction=predictions,
        prediction_type=model_type,
        disable_preproc_auto_orient=disable_preproc_auto_orient,
    )


class VideoFileSink:
    @classmethod
    def init(
        cls,
        video_file_name: str,
        annotator: sv.BoxAnnotator = DEFAULT_ANNOTATOR,
        display_size: Optional[Tuple[int, int]] = (1280, 720),
        fps_monitor: Optional[sv.FPSMonitor] = DEFAULT_FPS_MONITOR,
        display_statistics: bool = False,
        output_fps: int = 25,
        quiet: bool = False,
    ) -> "VideoFileSink":
        """
        Creates `InferencePipeline` predictions sink capable of saving model predictions into video file.

        As an `inference` user, please use .init() method instead of constructor to instantiate objects.
        Args:
            video_file_name (str): name of the video file to save predictions
            render_boxes (Callable[[dict, VideoFrame], None]): callable to render predictions on top of video frame

        Attributes:
            on_prediction (Callable[[dict, VideoFrame], None]): callable to be used as a sink for predictions

        Returns: Initialized object of `VideoFileSink` class.

        Example:
            ```python
            import cv2
            from inference import InferencePipeline
            from inference.core.interfaces.stream.sinks import VideoFileSink

            video_sink = VideoFileSink.init(video_file_name="output.avi")

            pipeline = InferencePipeline.init(
                model_id="your-model/3",
                video_reference="./some_file.mp4",
                on_prediction=video_sink.on_prediction,
            )
            pipeline.start()
            pipeline.join()
            video_sink.release()
            ```

            `VideoFileSink` used in this way will save predictions to video file automatically.
        """
        return cls(
            video_file_name=video_file_name,
            annotator=annotator,
            display_size=display_size,
            fps_monitor=fps_monitor,
            display_statistics=display_statistics,
            output_fps=output_fps,
            quiet=quiet,
        )

    def __init__(
        self,
        video_file_name: str,
        annotator: sv.BoxAnnotator,
        display_size: Optional[Tuple[int, int]],
        fps_monitor: Optional[sv.FPSMonitor],
        display_statistics: bool,
        output_fps: int,
        quiet: bool,
    ):
        self._video_file_name = video_file_name
        self._annotator = annotator
        self._display_size = display_size
        self._fps_monitor = fps_monitor
        self._display_statistics = display_statistics
        self._output_fps = output_fps
        self._quiet = quiet
        self._frame_idx = 0

        self._video_writer = cv2.VideoWriter(
            self._video_file_name,
            cv2.VideoWriter_fourcc(*"MJPG"),
            self._output_fps,
            self._display_size,
        )

        self.on_prediction = partial(
            render_boxes,
            annotator=self._annotator,
            display_size=self._display_size,
            fps_monitor=self._fps_monitor,
            display_statistics=self._display_statistics,
            on_frame_rendered=self._save_predictions,
        )

    def _save_predictions(
        self,
        frame: np.ndarray,
    ) -> None:
        """ """
        self._video_writer.write(frame)
        if not self._quiet:
            print(f"Writing frame {self._frame_idx}", end="\r")
        self._frame_idx += 1

    def release(self) -> None:
        """
        Releases VideoWriter object.
        """
        self._video_writer.release()

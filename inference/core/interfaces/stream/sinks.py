import json
import socket
from collections import deque
from datetime import datetime
from functools import partial
from typing import Callable, Dict, List, Optional, Set, Tuple, Union

import cv2
import numpy as np
import supervision as sv
from supervision.annotators.base import BaseAnnotator

from inference.core import logger
from inference.core.active_learning.middlewares import ActiveLearningMiddleware
from inference.core.interfaces.camera.entities import VideoFrame
from inference.core.interfaces.stream.entities import SinkHandler
from inference.core.interfaces.stream.utils import wrap_in_list
from inference.core.utils.drawing import create_tiles
from inference.core.utils.preprocess import letterbox_image

DEFAULT_BBOX_ANNOTATOR = sv.BoundingBoxAnnotator()
DEFAULT_LABEL_ANNOTATOR = sv.LabelAnnotator()
DEFAULT_FPS_MONITOR = sv.FPSMonitor()

ImageWithSourceID = Tuple[Optional[int], np.ndarray]


def display_image(image: Union[ImageWithSourceID, List[ImageWithSourceID]]) -> None:
    if issubclass(type(image), list):
        tiles = create_tiles(images=[i[1] for i in image])
        cv2.imshow("Predictions - tiles", tiles)
    else:
        source_id, picture_to_display = image
        if source_id is None:
            source_id = "N/A"
        cv2.imshow(f"Predictions - video: {source_id}", picture_to_display)
    cv2.waitKey(1)


def render_boxes(
    predictions: Union[dict, List[Optional[dict]]],
    video_frame: Union[VideoFrame, List[Optional[VideoFrame]]],
    annotator: Union[BaseAnnotator, List[BaseAnnotator]] = None,
    display_size: Optional[Tuple[int, int]] = (1280, 720),
    fps_monitor: Optional[sv.FPSMonitor] = DEFAULT_FPS_MONITOR,
    display_statistics: bool = False,
    on_frame_rendered: Callable[
        [Union[ImageWithSourceID, List[ImageWithSourceID]]], None
    ] = display_image,
) -> None:
    """
    Helper tool to render object detection predictions on top of video frame. It is designed
    to be used with `InferencePipeline`, as sink for predictions. By default it uses
    standard `sv.BoundingBoxAnnotator()` chained with `sv.LabelAnnotator()`
    to draw bounding boxes and resizes prediction to 1280x720 (keeping aspect ratio and adding black padding).
    One may configure default behaviour, for instance to display latency and throughput statistics.
    In batch mode it will display tiles of frames and overlay predictions.

    This sink is only partially compatible with stubs and classification models (it will not fail,
    although predictions will not be displayed).

    Since version `0.9.18`, when multi-source InferencePipeline was introduced - it support batch input, without
    changes to old functionality when single (predictions, video_frame) is used.

    Args:
        predictions (Union[dict, List[Optional[dict]]]): Roboflow predictions, the function support single prediction
            processing and batch processing since version `0.9.18`. Batch predictions elements are optional, but
            should occur at the same position as `video_frame` list. Order is expected to match with `video_frame`.
        video_frame (Union[VideoFrame, List[Optional[VideoFrame]]]): frame of video with its basic metadata emitted
            by `VideoSource` or list of frames from (it is possible for empty batch frames at corresponding positions
            to `predictions` list). Order is expected to match with `predictions`
        annotator (Union[BaseAnnotator, List[BaseAnnotator]]): instance of class inheriting from supervision BaseAnnotator
            or list of such instances. If nothing is passed chain of `sv.BoundingBoxAnnotator()` and `sv.LabelAnnotator()` is used.
        display_size (Tuple[int, int]): tuple in format (width, height) to resize visualisation output
        fps_monitor (Optional[sv.FPSMonitor]): FPS monitor used to monitor throughput
        display_statistics (bool): Flag to decide if throughput and latency can be displayed in the result image,
            if enabled, throughput will only be presented if `fps_monitor` is not None
        on_frame_rendered (Callable[[Union[ImageWithSourceID, List[ImageWithSourceID]]], None]): callback to be
            called once frame is rendered - by default, function will display OpenCV window. It expects optional integer
            identifier with np.ndarray or list of those elements. Identifier is supposed to refer to either source_id
            (for sequential input) or position in the batch (from 0 to batch_size-1).

    Returns: None
    Side effects: on_frame_rendered() is called against the tuple (stream_id, np.ndarray) produced from video
        frame and predictions.

    Example:
        ```python
        from functools import partial
        import cv2
        from inference import InferencePipeline
        from inference.core.interfaces.stream.sinks import render_boxes

        output_size = (640, 480)
        video_sink = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 25.0, output_size)
        on_prediction = partial(
            render_boxes,
            display_size=output_size,
            on_frame_rendered=lambda frame_data: video_sink.write(frame_data[1])
        )

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
        predictions displayed to be saved into video file. Please note that this is oversimplified example of usage
        which will not be robust against multiple streams - better implementation available in `VideoFileSink` class.
    """
    sequential_input_provided = False
    if not isinstance(video_frame, list):
        sequential_input_provided = True
    video_frame = wrap_in_list(element=video_frame)
    predictions = wrap_in_list(element=predictions)
    if annotator is None:
        annotator = [
            DEFAULT_BBOX_ANNOTATOR,
            DEFAULT_LABEL_ANNOTATOR,
        ]
    fps_value = None
    if fps_monitor is not None:
        ticks = sum(f is not None for f in video_frame)
        for _ in range(ticks):
            fps_monitor.tick()
        if hasattr(fps_monitor, "fps"):
            fps_value = fps_monitor.fps
        else:
            fps_value = fps_monitor()
    images: List[ImageWithSourceID] = []
    annotators = annotator if isinstance(annotator, list) else [annotator]
    for idx, (single_frame, frame_prediction) in enumerate(
        zip(video_frame, predictions)
    ):
        image = _handle_frame_rendering(
            frame=single_frame,
            prediction=frame_prediction,
            annotators=annotators,
            display_size=display_size,
            display_statistics=display_statistics,
            fps_value=fps_value,
        )
        images.append((idx, image))
    if sequential_input_provided:
        on_frame_rendered((video_frame[0].source_id, images[0][1]))
    else:
        on_frame_rendered(images)


def _handle_frame_rendering(
    frame: Optional[VideoFrame],
    prediction: dict,
    annotators: List[BaseAnnotator],
    display_size: Optional[Tuple[int, int]],
    display_statistics: bool,
    fps_value: Optional[float],
) -> np.ndarray:
    if frame is None:
        image = np.zeros((256, 256, 3), dtype=np.uint8)
    else:
        try:
            labels = [p["class"] for p in prediction["predictions"]]
            if hasattr(sv.Detections, "from_inference"):
                detections = sv.Detections.from_inference(prediction)
            else:
                detections = sv.Detections.from_inference(prediction)
            image = frame.image.copy()
            for annotator in annotators:
                kwargs = {
                    "scene": image,
                    "detections": detections,
                }
                if isinstance(annotator, sv.LabelAnnotator):
                    kwargs["labels"] = labels
                image = annotator.annotate(**kwargs)
        except (TypeError, KeyError):
            logger.warning(
                f"Used `render_boxes(...)` sink, but predictions that were provided do not match the expected "
                f"format of object detection prediction that could be accepted by "
                f"`supervision.Detection.from_inference(...)"
            )
            image = frame.image.copy()
    if display_size is not None:
        image = letterbox_image(image, desired_size=display_size)
    if display_statistics:
        image = render_statistics(
            image=image,
            frame_timestamp=(frame.frame_timestamp if frame is not None else None),
            fps=fps_value,
        )
    return image


def render_statistics(
    image: np.ndarray, frame_timestamp: Optional[datetime], fps: Optional[float]
) -> np.ndarray:
    image_height = image.shape[0]
    if frame_timestamp is not None:
        latency = round((datetime.now() - frame_timestamp).total_seconds() * 1000, 2)
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
        predictions: Union[dict, List[Optional[dict]]],
        video_frame: Union[VideoFrame, List[Optional[VideoFrame]]],
    ) -> None:
        """
        Method to send predictions via UDP socket. Useful in combination with `InferencePipeline` as
        a sink for predictions.

        Args:
            predictions (Union[dict, List[Optional[dict]]]): Roboflow predictions, the function support single prediction
                processing and batch processing since version `0.9.18`. Batch predictions elements are optional, but
                should occur at the same position as `video_frame` list. Order is expected to match with `video_frame`.
            video_frame (Union[VideoFrame, List[Optional[VideoFrame]]]): frame of video with its basic metadata emitted
                by `VideoSource` or list of frames from (it is possible for empty batch frames at corresponding positions
                to `predictions` list). Order is expected to match with `predictions`

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
        video_frame = wrap_in_list(element=video_frame)
        predictions = wrap_in_list(element=predictions)
        for single_frame, frame_predictions in zip(video_frame, predictions):
            if single_frame is None:
                continue
            inference_metadata = {
                "source_id": single_frame.source_id,
                "frame_id": single_frame.frame_id,
                "frame_decoding_time": single_frame.frame_timestamp.isoformat(),
                "emission_time": datetime.now().isoformat(),
            }
            frame_predictions["inference_metadata"] = inference_metadata
            serialised_predictions = json.dumps(frame_predictions).encode("utf-8")
            self._socket.sendto(
                serialised_predictions,
                (
                    self._ip_address,
                    self._port,
                ),
            )


def multi_sink(
    predictions: Union[dict, List[Optional[dict]]],
    video_frame: Union[VideoFrame, List[Optional[VideoFrame]]],
    sinks: List[SinkHandler],
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
                f"Could not sent prediction with to sink due to error: {error}."
            )


def active_learning_sink(
    predictions: Union[dict, List[Optional[dict]]],
    video_frame: Union[VideoFrame, List[Optional[VideoFrame]]],
    active_learning_middleware: ActiveLearningMiddleware,
    model_type: str,
    disable_preproc_auto_orient: bool = False,
) -> None:
    """
    Function to serve as Active Learning sink for InferencePipeline.

    Args:
        predictions (Union[dict, List[Optional[dict]]]): Roboflow predictions, the function support single prediction
            processing and batch processing since version `0.9.18`. Batch predictions elements are optional, but
            should occur at the same position as `video_frame` list. Order is expected to match with `video_frame`.
        video_frame (Union[VideoFrame, List[Optional[VideoFrame]]]): frame of video with its basic metadata emitted
            by `VideoSource` or list of frames from (it is possible for empty batch frames at corresponding positions
            to `predictions` list). Order is expected to match with `predictions`
        active_learning_middleware (ActiveLearningMiddleware): instance of middleware to register data.
        model_type (str): Type of Roboflow model in use
        disable_preproc_auto_orient (bool): Flag to denote how image is preprocessed which is important in
            Active Learning.

    Returns: None
    Side effects: Can register data and predictions in Roboflow backend if that's the evaluation of sampling engine.
    """
    video_frame = wrap_in_list(element=video_frame)
    predictions = wrap_in_list(element=predictions)
    images = [f.image for f in video_frame if f is not None]
    predictions = [p for p in predictions if p is not None]
    active_learning_middleware.register_batch(
        inference_inputs=images,
        predictions=predictions,
        prediction_type=model_type,
        disable_preproc_auto_orient=disable_preproc_auto_orient,
    )


class VideoFileSink:
    @classmethod
    def init(
        cls,
        video_file_name: str,
        annotator: Optional[Union[BaseAnnotator, List[BaseAnnotator]]] = None,
        display_size: Optional[Tuple[int, int]] = (1280, 720),
        fps_monitor: Optional[sv.FPSMonitor] = DEFAULT_FPS_MONITOR,
        display_statistics: bool = False,
        output_fps: int = 25,
        quiet: bool = False,
        video_frame_size: Tuple[int, int] = (1280, 720),
    ) -> "VideoFileSink":
        """
        Creates `InferencePipeline` predictions sink capable of saving model predictions into video file.
        It works both for pipelines with single input video and multiple ones.

        As an `inference` user, please use .init() method instead of constructor to instantiate objects.
        Args:
            video_file_name (str): name of the video file to save predictions
            annotator (Union[BaseAnnotator, List[BaseAnnotator]]): instance of class inheriting from supervision BaseAnnotator
                or list of such instances. If nothing is passed chain of `sv.BoundingBoxAnnotator()` and `sv.LabelAnnotator()` is used.
            display_size (Tuple[int, int]): tuple in format (width, height) to resize visualisation output. Should
                be set to the same value as `display_size` for InferencePipeline with single video source, otherwise
                it represents the size of single visualisation tile (whole tiles mosaic will be scaled to
                `video_frame_size`)
            fps_monitor (Optional[sv.FPSMonitor]): FPS monitor used to monitor throughput
            display_statistics (bool): Flag to decide if throughput and latency can be displayed in the result image,
                if enabled, throughput will only be presented if `fps_monitor` is not None
            output_fps (int): desired FPS of output file
            quiet (bool): Flag to decide whether to log progress
            video_frame_size (Tuple[int, int]): The size of frame in target video file.

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
            video_frame_size=video_frame_size,
        )

    def __init__(
        self,
        video_file_name: str,
        annotator: Union[BaseAnnotator, List[BaseAnnotator]],
        display_size: Optional[Tuple[int, int]],
        fps_monitor: Optional[sv.FPSMonitor],
        display_statistics: bool,
        output_fps: int,
        quiet: bool,
        video_frame_size: Tuple[int, int],
    ):
        self._video_file_name = video_file_name
        self._annotator = annotator
        self._display_size = display_size
        self._fps_monitor = fps_monitor
        self._display_statistics = display_statistics
        self._output_fps = output_fps
        self._quiet = quiet
        self._frame_idx = 0
        self._video_frame_size = video_frame_size
        self._video_writer: Optional[cv2.VideoWriter] = None
        self.on_prediction = partial(
            render_boxes,
            annotator=self._annotator,
            display_size=self._display_size,
            fps_monitor=self._fps_monitor,
            display_statistics=self._display_statistics,
            on_frame_rendered=self._save_predictions,
        )

    def release(self) -> None:
        """
        Releases VideoWriter object.
        """
        if self._video_writer is not None and self._video_writer.isOpened():
            self._video_writer.release()

    def _save_predictions(
        self,
        frame: Union[ImageWithSourceID, List[ImageWithSourceID]],
    ) -> None:
        if self._video_writer is None:
            self._initialise_sink()
        if issubclass(type(frame), list):
            frame = create_tiles(images=[i[1] for i in frame])
        else:
            frame = frame[1]
        if (frame.shape[1], frame.shape[0]) != self._video_frame_size:
            frame = letterbox_image(image=frame, desired_size=self._video_frame_size)
        self._video_writer.write(frame)
        if not self._quiet:
            print(f"Writing frame {self._frame_idx}", end="\r")
        self._frame_idx += 1

    def _initialise_sink(self) -> None:
        self._video_writer = cv2.VideoWriter(
            self._video_file_name,
            cv2.VideoWriter_fourcc(*"MJPG"),
            self._output_fps,
            self._video_frame_size,
        )

    def __enter__(self) -> "VideoFileSink":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()


class InMemoryBufferSink:

    @classmethod
    def init(cls, queue_size: int):
        return cls(queue_size=queue_size)

    def __init__(self, queue_size: int):
        self._buffer = deque(maxlen=queue_size)

    def on_prediction(
        self,
        predictions: Union[dict, List[Optional[dict]]],
        video_frame: Union[VideoFrame, List[Optional[VideoFrame]]],
    ) -> None:
        if not isinstance(predictions, list):
            predictions = [predictions]
        if not isinstance(video_frame, list):
            video_frame = [video_frame]
        self._buffer.append((predictions, video_frame))

    def empty(self) -> bool:
        return len(self._buffer) == 0

    def consume_prediction(
        self,
    ) -> Tuple[List[Optional[dict]], List[Optional[VideoFrame]]]:
        return self._buffer.popleft()

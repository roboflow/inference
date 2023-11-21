from datetime import datetime
from queue import Queue
from threading import Thread
from typing import Union, Optional, Callable, Tuple, Generator

import numpy as np

from inference.core.interfaces.camera.entities import FrameTimestamp, FrameID
from inference.core.interfaces.camera.utils import get_video_frames_generator
from inference.core.interfaces.camera.video_source import VideoSource
from inference.core.models.roboflow import OnnxRoboflowInferenceModel
from inference.models.utils import get_roboflow_model

COMMAND_QUEUE_SIZE = 64
PREDICTIONS_QUEUE_SIZE = 256


class InferencePipeline:
    @classmethod
    def init(
        cls,
        api_key: str,
        model_id: str,
        video_reference: Union[str, int],
        on_prediction: Callable,
        max_fps: Optional[float] = None,
    ) -> "InferencePipeline":
        model = get_roboflow_model(model_id=model_id, api_key=api_key)
        video_source = VideoSource.init(video_reference=video_reference)
        command_queue = Queue(maxsize=COMMAND_QUEUE_SIZE)
        predictions_queue = Queue(maxsize=PREDICTIONS_QUEUE_SIZE)
        return cls(
            model=model,
            video_source=video_source,
            on_prediction=on_prediction,
            max_fps=max_fps,
            command_queue=command_queue,
            predictions_queue=predictions_queue,
        )

    def __init__(
        self,
        model: OnnxRoboflowInferenceModel,
        video_source: VideoSource,
        on_prediction: Callable,
        max_fps: Optional[float],
        command_queue: Queue,
        predictions_queue: Queue,
    ):
        self._model = model
        self._video_source = video_source
        self._on_prediction = on_prediction
        self._max_fps = max_fps
        self._command_queue = command_queue
        self._predictions_queue = predictions_queue
        self._command_handler_thread: Optional[Thread] = None
        self._inference_thread: Optional[Thread] = None
        self._dispatching_thread: Optional[Thread] = None
        self._stop = False
        self._camera_restart_ongoing = False

    def start(self) -> None:
        self._command_handler_thread = Thread(target=self._handle_commands)
        self._command_handler_thread.start()
        self._inference_thread = Thread(target=self._execute_inference)
        self._inference_thread.start()
        self._dispatching_thread = Thread(target=self._dispatching_thread)
        self._dispatching_thread.start()

    def stop(self) -> None:
        pass

    def join(self) -> None:
        self._command_handler_thread.join()
        self._inference_thread.join()
        self._dispatching_thread.join()

    def _handle_commands(self) -> None:
        while True:
            command = self._command_queue.get()
            if command is None:
                break

    def _execute_inference(self) -> None:
        try:
            for timestamp, frame_id, frame in self._generate_frames():
                preprocessed_image, preprocessing_metadata = self._model.preprocess(
                    frame
                )
                predictions = self._model.predict(preprocessed_image)
                predictions = self._model.postprocess(
                    predictions,
                    preprocessing_metadata,
                    confidence=0.5,
                    iou_threshold=0.5,
                )
                predictions = self._model.make_response(
                    predictions, preprocessing_metadata
                )[0].dict(
                    by_alias=True,
                    exclude_none=True,
                )
                self._predictions_queue.put((timestamp, frame_id, frame, predictions))
        finally:
            self._predictions_queue.put(None)

    def _dispatch_inference_results(self) -> None:
        while True:
            inference_results: Optional[
                Tuple[datetime, int, np.ndarray, dict]
            ] = self._command_queue.get()
            if inference_results is None:
                break
            timestamp, frame_id, frame, predictions = inference_results
            try:
                self._on_prediction(predictions, frame)
            except Exception:
                pass

    def _generate_frames(
        self,
    ) -> Generator[Tuple[FrameTimestamp, FrameID, np.ndarray], None, None]:
        self._video_source.start()
        while not self._stop:
            allow_reconnect = (
                not self._video_source.describe_source().stream_properties.is_file
            )
            yield from get_video_frames_generator(
                stream=self._video_source, max_fps=self._max_fps
            )
            if not allow_reconnect:
                self._command_queue.put(None)
                break
            self._video_source.restart()

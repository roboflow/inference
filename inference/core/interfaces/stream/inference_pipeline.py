import logging
import time
from datetime import datetime
from queue import Queue
from threading import Thread
from typing import Callable, Generator, Optional, Tuple, Union

import numpy as np

from inference.core.interfaces.camera.entities import FrameID, FrameTimestamp
from inference.core.interfaces.camera.exceptions import SourceConnectionError
from inference.core.interfaces.camera.utils import get_video_frames_generator
from inference.core.interfaces.camera.video_source import VideoSource
from inference.core.interfaces.stream.watchdog import (
    NullPipelineWatchdog,
    PipelineWatchDog,
)
from inference.core.models.roboflow import OnnxRoboflowInferenceModel
from inference.models.utils import get_roboflow_model

PREDICTIONS_QUEUE_SIZE = 256
RESTART_ATTEMPT_DELAY = 1


class InferencePipeline:
    @classmethod
    def init(
        cls,
        api_key: str,
        model_id: str,
        video_reference: Union[str, int],
        on_prediction: Callable,
        max_fps: Optional[float] = None,
        watchdog: Optional[PipelineWatchDog] = None,
    ) -> "InferencePipeline":
        model = get_roboflow_model(model_id=model_id, api_key=api_key)
        if watchdog is None:
            watchdog = NullPipelineWatchdog()
        video_source = VideoSource.init(
            video_reference=video_reference,
            status_update_handlers=[watchdog.on_video_source_status_update],
        )
        watchdog.register_video_source(video_source=video_source)
        predictions_queue = Queue(maxsize=PREDICTIONS_QUEUE_SIZE)
        return cls(
            model=model,
            video_source=video_source,
            on_prediction=on_prediction,
            max_fps=max_fps,
            predictions_queue=predictions_queue,
            watchdog=watchdog,
        )

    def __init__(
        self,
        model: OnnxRoboflowInferenceModel,
        video_source: VideoSource,
        on_prediction: Callable,
        max_fps: Optional[float],
        predictions_queue: Queue,
        watchdog: PipelineWatchDog,
    ):
        self._model = model
        self._video_source = video_source
        self._on_prediction = on_prediction
        self._max_fps = max_fps
        self._predictions_queue = predictions_queue
        self._watchdog = watchdog
        self._command_handler_thread: Optional[Thread] = None
        self._inference_thread: Optional[Thread] = None
        self._dispatching_thread: Optional[Thread] = None
        self._stop = False
        self._camera_restart_ongoing = False

    def start(self, use_main_thread: bool = True) -> None:
        self._inference_thread = Thread(target=self._execute_inference)
        self._inference_thread.start()
        if use_main_thread:
            self._dispatch_inference_results()
        else:
            self._dispatching_thread = Thread(target=self._dispatch_inference_results)
            self._dispatching_thread.start()

    def terminate(self) -> None:
        self._stop = True
        self._video_source.terminate()

    def pause_stream(self) -> None:
        self._video_source.pause()

    def mute_stream(self) -> None:
        self._video_source.mute()

    def resume_stream(self) -> None:
        self._video_source.resume()

    def join(self) -> None:
        if self._inference_thread is not None:
            self._inference_thread.join()
        if self._dispatching_thread is not None:
            self._dispatching_thread.join()

    def _execute_inference(self) -> None:
        try:
            for timestamp, frame_id, frame in self._generate_frames():
                self._watchdog.on_model_preprocessing_started(
                    frame_timestamp=timestamp, frame_id=frame_id
                )
                preprocessed_image, preprocessing_metadata = self._model.preprocess(
                    frame
                )
                self._watchdog.on_model_inference_started(
                    frame_timestamp=timestamp, frame_id=frame_id
                )
                predictions = self._model.predict(preprocessed_image)
                self._watchdog.on_model_postprocessing_started(
                    frame_timestamp=timestamp, frame_id=frame_id
                )
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
                self._watchdog.on_model_prediction_ready(
                    frame_timestamp=timestamp, frame_id=frame_id
                )
                self._predictions_queue.put((timestamp, frame_id, frame, predictions))
        finally:
            self._predictions_queue.put(None)

    def _dispatch_inference_results(self) -> None:
        while True:
            inference_results: Optional[
                Tuple[datetime, int, np.ndarray, dict]
            ] = self._predictions_queue.get()
            if inference_results is None:
                break
            timestamp, frame_id, frame, predictions = inference_results
            try:
                self._on_prediction(timestamp, frame_id, frame, predictions)
            except Exception as error:
                self._watchdog.on_error(context="predictions_dispatcher", error=error)
                logging.warning(f"Error in results dispatching - {error}")

    def _generate_frames(
        self,
    ) -> Generator[Tuple[FrameTimestamp, FrameID, np.ndarray], None, None]:
        self._video_source.start()
        while True:
            allow_reconnect = (
                not self._video_source.describe_source().source_properties.is_file
            )
            yield from get_video_frames_generator(
                video=self._video_source, max_fps=self._max_fps
            )
            if not allow_reconnect:
                self.terminate()
                break
            if self._stop:
                break
            self._attempt_restart()

    def _attempt_restart(self) -> None:
        succeeded = False
        while not self._stop and not succeeded:
            try:
                self._video_source.restart()
                succeeded = True
            except SourceConnectionError as error:
                self._watchdog.on_error(context="stream_frames_generator", error=error)
                time.sleep(RESTART_ATTEMPT_DELAY)

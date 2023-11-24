import logging
import os
import time
from datetime import datetime
from queue import Queue
from threading import Thread
from typing import Callable, Generator, Optional, Union, List

from inference.core import logger
from inference.core.env import API_KEY_ENV_NAMES
from inference.core.exceptions import MissingApiKeyError
from inference.core.interfaces.camera.entities import (
    VideoFrame,
    StatusUpdate,
    UpdateSeverity,
)
from inference.core.interfaces.camera.exceptions import SourceConnectionError
from inference.core.interfaces.camera.utils import get_video_frames_generator
from inference.core.interfaces.camera.video_source import (
    VideoSource,
    BufferFillingStrategy,
    BufferConsumptionStrategy,
)
from inference.core.interfaces.stream.entities import (
    ObjectDetectionPrediction,
    ObjectDetectionInferenceConfig,
)
from inference.core.interfaces.stream.watchdog import (
    NullPipelineWatchdog,
    PipelineWatchDog,
)
from inference.core.models.roboflow import OnnxRoboflowInferenceModel
from inference.models.utils import get_roboflow_model

PREDICTIONS_QUEUE_SIZE = 256
RESTART_ATTEMPT_DELAY = 1
INFERENCE_PIPELINE_CONTEXT = "inference_pipeline"
SOURCE_CONNECTION_ATTEMPT_FAILED_EVENT = "SOURCE_CONNECTION_ATTEMPT_FAILED"
SOURCE_CONNECTION_LOST_EVENT = "SOURCE_CONNECTION_LOST"
INFERENCE_RESULTS_DISPATCHING_ERROR_EVENT = "INFERENCE_RESULTS_DISPATCHING_ERROR"
INFERENCE_THREAD_STARTED_EVENT = "INFERENCE_THREAD_STARTED"
INFERENCE_THREAD_FINISHED_EVENT = "INFERENCE_THREAD_FINISHED"
INFERENCE_COMPLETED_EVENT = "INFERENCE_COMPLETED"


class InferencePipeline:
    @classmethod
    def init(
        cls,
        model_id: str,
        video_reference: Union[str, int],
        on_prediction: Callable[[VideoFrame, ObjectDetectionPrediction], None],
        api_key: Optional[str] = None,
        max_fps: Optional[float] = None,
        watchdog: Optional[PipelineWatchDog] = None,
        status_update_handlers: Optional[List[Callable[[StatusUpdate], None]]] = None,
        source_buffer_filling_strategy: Optional[BufferFillingStrategy] = None,
        source_buffer_consumption_strategy: Optional[BufferConsumptionStrategy] = None,
        class_agnostic_nms: Optional[bool] = None,
        confidence: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        max_candidates: Optional[int] = None,
        max_detections: Optional[int] = None,
    ) -> "InferencePipeline":
        if api_key is None:
            api_key = os.environ.get(API_KEY_ENV_NAMES[0], None) or os.environ.get(
                API_KEY_ENV_NAMES[1], None
            )
        if api_key is None:
            raise MissingApiKeyError(
                "Could not initialise InferencePipeline, as API key is missing either in initializer parameters "
                f"or in one one of allowed env variables: {API_KEY_ENV_NAMES}."
            )
        if status_update_handlers is None:
            status_update_handlers = []
        inference_config = ObjectDetectionInferenceConfig.init(
            class_agnostic_nms=class_agnostic_nms,
            confidence=confidence,
            iou_threshold=iou_threshold,
            max_candidates=max_candidates,
            max_detections=max_detections,
        )
        model = get_roboflow_model(model_id=model_id, api_key=api_key)
        if watchdog is None:
            watchdog = NullPipelineWatchdog()
        status_update_handlers.append(watchdog.on_status_update)
        video_source = VideoSource.init(
            video_reference=video_reference,
            status_update_handlers=status_update_handlers,
            buffer_filling_strategy=source_buffer_filling_strategy,
            buffer_consumption_strategy=source_buffer_consumption_strategy,
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
            status_update_handlers=status_update_handlers,
            inference_config=inference_config,
        )

    def __init__(
        self,
        model: OnnxRoboflowInferenceModel,
        video_source: VideoSource,
        on_prediction: Callable[[VideoFrame, ObjectDetectionPrediction], None],
        max_fps: Optional[float],
        predictions_queue: Queue,
        watchdog: PipelineWatchDog,
        status_update_handlers: List[Callable[[StatusUpdate], None]],
        inference_config: ObjectDetectionInferenceConfig,
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
        self._status_update_handlers = status_update_handlers
        self._inference_config = inference_config

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
        send_inference_pipeline_status_update(
            severity=UpdateSeverity.INFO,
            event_type=INFERENCE_THREAD_STARTED_EVENT,
            status_update_handlers=self._status_update_handlers,
        )
        logger.info(f"Inference thread started")
        try:
            for video_frame in self._generate_frames():
                self._watchdog.on_model_preprocessing_started(
                    frame_timestamp=video_frame.frame_timestamp,
                    frame_id=video_frame.frame_id,
                )
                preprocessed_image, preprocessing_metadata = self._model.preprocess(
                    video_frame.image
                )
                self._watchdog.on_model_inference_started(
                    frame_timestamp=video_frame.frame_timestamp,
                    frame_id=video_frame.frame_id,
                )
                predictions = self._model.predict(preprocessed_image)
                self._watchdog.on_model_postprocessing_started(
                    frame_timestamp=video_frame.frame_timestamp,
                    frame_id=video_frame.frame_id,
                )
                predictions = self._model.postprocess(
                    predictions,
                    preprocessing_metadata,
                    class_agnostic_nms=self._inference_config.class_agnostic_nms,
                    confidence=self._inference_config.confidence,
                    iou_threshold=self._inference_config.iou_threshold,
                    max_candidates=self._inference_config.max_candidates,
                    max_detections=self._inference_config.max_detections,
                )
                predictions = self._model.make_response(
                    predictions, preprocessing_metadata
                )[0].dict(
                    by_alias=True,
                    exclude_none=True,
                )
                self._watchdog.on_model_prediction_ready(
                    frame_timestamp=video_frame.frame_timestamp,
                    frame_id=video_frame.frame_id,
                )
                self._predictions_queue.put((video_frame, predictions))
                send_inference_pipeline_status_update(
                    severity=UpdateSeverity.DEBUG,
                    event_type=INFERENCE_COMPLETED_EVENT,
                    payload={
                        "frame_id": video_frame.frame_id,
                        "frame_timestamp": video_frame.frame_timestamp,
                    },
                    status_update_handlers=self._status_update_handlers,
                )
        finally:
            self._predictions_queue.put(None)
            send_inference_pipeline_status_update(
                severity=UpdateSeverity.INFO,
                event_type=INFERENCE_THREAD_FINISHED_EVENT,
                status_update_handlers=self._status_update_handlers,
            )
            logger.info(f"Inference thread finished")

    def _dispatch_inference_results(self) -> None:
        while True:
            inference_results: Optional[VideoFrame] = self._predictions_queue.get()
            if inference_results is None:
                break
            video_frame, predictions = inference_results
            try:
                self._on_prediction(video_frame, predictions)
            except Exception as error:
                payload = {
                    "error_type": error.__class__.__name__,
                    "error_message": str(error),
                    "error_context": "inference_results_dispatching",
                }
                send_inference_pipeline_status_update(
                    severity=UpdateSeverity.ERROR,
                    event_type=INFERENCE_RESULTS_DISPATCHING_ERROR_EVENT,
                    payload=payload,
                    status_update_handlers=self._status_update_handlers,
                )
                logger.warning(f"Error in results dispatching - {error}")

    def _generate_frames(
        self,
    ) -> Generator[VideoFrame, None, None]:
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
            logger.warning(f"Lost connection with video source.")
            send_inference_pipeline_status_update(
                severity=UpdateSeverity.WARNING,
                event_type=SOURCE_CONNECTION_LOST_EVENT,
                payload={
                    "source_reference": self._video_source.describe_source().source_reference
                },
                status_update_handlers=self._status_update_handlers,
            )
            self._attempt_restart()

    def _attempt_restart(self) -> None:
        succeeded = False
        while not self._stop and not succeeded:
            try:
                self._video_source.restart()
                succeeded = True
            except SourceConnectionError as error:
                payload = {
                    "error_type": error.__class__.__name__,
                    "error_message": str(error),
                    "error_context": "video_frames_generator",
                }
                send_inference_pipeline_status_update(
                    severity=UpdateSeverity.WARNING,
                    event_type=SOURCE_CONNECTION_ATTEMPT_FAILED_EVENT,
                    payload=payload,
                    status_update_handlers=self._status_update_handlers,
                )
                logger.warning(
                    f"Could not connect to video source. Retrying in {RESTART_ATTEMPT_DELAY}s..."
                )
                time.sleep(RESTART_ATTEMPT_DELAY)


def send_inference_pipeline_status_update(
    severity: UpdateSeverity,
    event_type: str,
    status_update_handlers: List[Callable[[StatusUpdate], None]],
    payload: Optional[dict] = None,
    sub_context: Optional[str] = None,
) -> None:
    if payload is None:
        payload = {}
    context = INFERENCE_PIPELINE_CONTEXT
    if sub_context is not None:
        context = f"{context}.{sub_context}"
    status_update = StatusUpdate(
        timestamp=datetime.now(),
        severity=severity,
        event_type=event_type,
        payload=payload,
        context=context,
    )
    for handler in status_update_handlers:
        try:
            handler(status_update)
        except Exception as error:
            logger.warning(f"Could not execute handler update. Cause: {error}")

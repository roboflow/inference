import time
from datetime import datetime
from functools import partial
from queue import Queue
from threading import Thread
from typing import Callable, Generator, List, Optional, Tuple, Union

from inference.core import logger
from inference.core.active_learning.middlewares import (
    NullActiveLearningMiddleware,
    ThreadingActiveLearningMiddleware,
)
from inference.core.cache import cache
from inference.core.env import (
    ACTIVE_LEARNING_ENABLED,
    API_KEY,
    DISABLE_PREPROC_AUTO_ORIENT,
    PREDICTIONS_QUEUE_SIZE,
    RESTART_ATTEMPT_DELAY,
)
from inference.core.interfaces.camera.entities import (
    StatusUpdate,
    UpdateSeverity,
    VideoFrame,
)
from inference.core.interfaces.camera.exceptions import SourceConnectionError
from inference.core.interfaces.camera.utils import get_video_frames_generator
from inference.core.interfaces.camera.video_source import (
    BufferConsumptionStrategy,
    BufferFillingStrategy,
    VideoSource,
)
from inference.core.interfaces.stream.entities import (
    ModelConfig,
    ObjectDetectionPrediction,
)
from inference.core.interfaces.stream.model_functions import (
    init_yolo_world_model,
    process_frame,
    process_frame_yolo_world,
)
from inference.core.interfaces.stream.sinks import active_learning_sink, multi_sink
from inference.core.interfaces.stream.watchdog import (
    NullPipelineWatchdog,
    PipelineWatchDog,
)
from inference.core.models.roboflow import OnnxRoboflowInferenceModel
from inference.models.utils import get_model

INFERENCE_PIPELINE_CONTEXT = "inference_pipeline"
SOURCE_CONNECTION_ATTEMPT_FAILED_EVENT = "SOURCE_CONNECTION_ATTEMPT_FAILED"
SOURCE_CONNECTION_LOST_EVENT = "SOURCE_CONNECTION_LOST"
INFERENCE_RESULTS_DISPATCHING_ERROR_EVENT = "INFERENCE_RESULTS_DISPATCHING_ERROR"
INFERENCE_THREAD_STARTED_EVENT = "INFERENCE_THREAD_STARTED"
INFERENCE_THREAD_FINISHED_EVENT = "INFERENCE_THREAD_FINISHED"
INFERENCE_COMPLETED_EVENT = "INFERENCE_COMPLETED"
INFERENCE_ERROR_EVENT = "INFERENCE_ERROR"

CORE_MODEL_CONFIG = {"yolo-world": None}


try:
    from inference.models import YOLOWorld

    CORE_MODEL_CONFIG["yolo_world"] = {
        "model_class": YOLOWorld,
        "process_frame_func": process_frame_yolo_world,
        "init_model_func": init_yolo_world_model,
    }
except:
    pass


class InferencePipeline:
    @classmethod
    def init(
        cls,
        model_id: str,
        video_reference: Union[str, int],
        on_prediction: Callable[[ObjectDetectionPrediction, VideoFrame], None],
        api_key: Optional[str] = None,
        max_fps: Optional[Union[float, int]] = None,
        watchdog: Optional[PipelineWatchDog] = None,
        status_update_handlers: Optional[List[Callable[[StatusUpdate], None]]] = None,
        source_buffer_filling_strategy: Optional[BufferFillingStrategy] = None,
        source_buffer_consumption_strategy: Optional[BufferConsumptionStrategy] = None,
        class_agnostic_nms: Optional[bool] = None,
        confidence: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        max_candidates: Optional[int] = None,
        max_detections: Optional[int] = None,
        mask_decode_mode: Optional[str] = "accurate",
        tradeoff_factor: Optional[float] = 0.0,
        active_learning_enabled: Optional[bool] = None,
        top_k: Optional[int] = None,
        **kwargs,
    ) -> "InferencePipeline":
        """
        This class creates the abstraction for making inferences from CV models against video stream.
        It allows to choose Object Detection model from Roboflow platform and run predictions against
        video streams - just by the price of specifying which model to use and what to do with predictions.

        It allows to set the model post-processing parameters (via .init() or env) and intercept updates
        related to state of pipeline via `PipelineWatchDog` abstraction (although that is something probably
        useful only for advanced use-cases).

        For maximum efficiency, all separate chunks of processing: video decoding, inference, results dispatching
        are handled by separate threads.

        Given that reference to stream is passed and connectivity is lost - it attempts to re-connect with delay.

        Since version 0.9.11 it works not only for object detection models but is also compatible with stubs,
        classification, instance-segmentation and keypoint-detection models.

        Args:
            model_id (str): Name and version of model at Roboflow platform (example: "my-model/3")
            video_reference (Union[str, int]): Reference of source to be used to make predictions against.
                It can be video file path, stream URL and device (like camera) id (we handle whatever cv2 handles).
            on_prediction (Callable[ObjectDetectionPrediction, VideoFrame], None]): Function to be called
                once prediction is ready - passing both decoded frame, their metadata and dict with standard
                Roboflow Object Detection prediction.
            api_key (Optional[str]): Roboflow API key - if not passed - will be looked in env under "ROBOFLOW_API_KEY"
                and "API_KEY" variables. API key, passed in some form is required.
            max_fps (Optional[Union[float, int]]): Specific value passed as this parameter will be used to
                dictate max FPS of processing. It can be useful if we wanted to run concurrent inference pipelines
                on single machine making tradeoff between number of frames and number of streams handled. Disabled
                by default.
            watchdog (Optional[PipelineWatchDog]): Implementation of class that allows profiling of
                inference pipeline - if not given null implementation (doing nothing) will be used.
            status_update_handlers (Optional[List[Callable[[StatusUpdate], None]]]): List of handlers to intercept
                status updates of all elements of the pipeline. Should be used only if detailed inspection of
                pipeline behaviour in time is needed. Please point out that handlers should be possible to be executed
                fast - otherwise they will impair pipeline performance. All errors will be logged as warnings
                without re-raising. Default: None.
            source_buffer_filling_strategy (Optional[BufferFillingStrategy]): Parameter dictating strategy for
                video stream decoding behaviour. By default - tweaked to the type of source given.
                Please find detailed explanation in docs of [`VideoSource`](../camera/video_source.py)
            source_buffer_consumption_strategy (Optional[BufferConsumptionStrategy]): Parameter dictating strategy for
                video stream frames consumption. By default - tweaked to the type of source given.
                Please find detailed explanation in docs of [`VideoSource`](../camera/video_source.py)
            class_agnostic_nms (Optional[bool]): Parameter of model post-processing. If not given - value checked in
                env variable "CLASS_AGNOSTIC_NMS" with default "False"
            confidence (Optional[float]): Parameter of model post-processing. If not given - value checked in
                env variable "CONFIDENCE" with default "0.5"
            iou_threshold (Optional[float]): Parameter of model post-processing. If not given - value checked in
                env variable "IOU_THRESHOLD" with default "0.5"
            max_candidates (Optional[int]): Parameter of model post-processing. If not given - value checked in
                env variable "MAX_CANDIDATES" with default "3000"
            max_detections (Optional[int]): Parameter of model post-processing. If not given - value checked in
                env variable "MAX_DETECTIONS" with default "300"
            mask_decode_mode: (Optional[str]): Parameter of model post-processing. If not given - model "accurate" is
                used. Applicable for instance segmentation models
            tradeoff_factor (Optional[float]): Parameter of model post-processing. If not 0.0 - model default is used.
                Applicable for instance segmentation models
            active_learning_enabled (Optional[bool]): Flag to enable / disable Active Learning middleware (setting it
                true does not guarantee any data to be collected, as data collection is controlled by Roboflow backend -
                it just enables middleware intercepting predictions). If not given, env variable
                `ACTIVE_LEARNING_ENABLED` will be used. Please point out that Active Learning will be forcefully
                disabled in a scenario when Roboflow API key is not given, as Roboflow account is required
                for this feature to be operational.
            top_k (Optional[int]): Sets the maximum number of predictions to be returned by the model. If not given,
                no limit is set.

        Other ENV variables involved in low-level configuration:
        * INFERENCE_PIPELINE_PREDICTIONS_QUEUE_SIZE - size of buffer for predictions that are ready for dispatching
        * INFERENCE_PIPELINE_RESTART_ATTEMPT_DELAY - delay for restarts on stream connection drop
        * ACTIVE_LEARNING_ENABLED - controls Active Learning middleware if explicit parameter not given

        Returns: Instance of InferencePipeline

        Throws:
            * SourceConnectionError if source cannot be connected at start, however it attempts to reconnect
                always if connection to stream is lost.
        """
        if api_key is None:
            api_key = API_KEY
        if status_update_handlers is None:
            status_update_handlers = []
        inference_config = ModelConfig.init(
            class_agnostic_nms=class_agnostic_nms,
            confidence=confidence,
            iou_threshold=iou_threshold,
            max_candidates=max_candidates,
            max_detections=max_detections,
            mask_decode_mode=mask_decode_mode,
            tradeoff_factor=tradeoff_factor,
        )
        project, _ = model_id.split("/")
        if project in CORE_MODEL_CONFIG:
            if CORE_MODEL_CONFIG[project] is None:
                raise ValueError(
                    f"Model {project} is not installed or not available in this environment"
                )
            ModelClass = CORE_MODEL_CONFIG[project]["model_class"]
            process_frame_func = CORE_MODEL_CONFIG[project]["process_frame_func"]
            model = ModelClass(api_key=api_key, model_id=model_id)
            init_model_func = CORE_MODEL_CONFIG[project].get("init_model_func")
            if init_model_func is not None:
                model = init_model_func(model, **kwargs)
        else:
            model = get_model(model_id=model_id, api_key=api_key)
            process_frame_func = process_frame
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
        active_learning_middleware = NullActiveLearningMiddleware()
        if active_learning_enabled is None:
            logger.info(
                f"`active_learning_enabled` parameter not set - using env `ACTIVE_LEARNING_ENABLED` "
                f"with value: {ACTIVE_LEARNING_ENABLED}"
            )
            active_learning_enabled = ACTIVE_LEARNING_ENABLED
        if api_key is None:
            logger.info(
                f"Roboflow API key not given - Active Learning is forced to be disabled."
            )
            active_learning_enabled = False
        if active_learning_enabled is True and project in CORE_MODEL_CONFIG:
            logger.info(
                f"Active Learning middleware is not available for model {project} - forcing it to be disabled."
            )
            active_learning_enabled = False
        if active_learning_enabled is True:
            active_learning_middleware = ThreadingActiveLearningMiddleware.init(
                api_key=api_key,
                model_id=model_id,
                cache=cache,
            )
            al_sink = partial(
                active_learning_sink,
                active_learning_middleware=active_learning_middleware,
                model_type=model.task_type,
                disable_preproc_auto_orient=DISABLE_PREPROC_AUTO_ORIENT,
            )
            logger.info(
                "AL enabled - wrapping `on_prediction` with multi_sink() and active_learning_sink()"
            )
            on_prediction = partial(multi_sink, sinks=[on_prediction, al_sink])
        return cls(
            model=model,
            process_frame_func=process_frame_func,
            video_source=video_source,
            on_prediction=on_prediction,
            max_fps=max_fps,
            predictions_queue=predictions_queue,
            watchdog=watchdog,
            status_update_handlers=status_update_handlers,
            inference_config=inference_config,
            active_learning_middleware=active_learning_middleware,
            top_k=top_k,
        )

    def __init__(
        self,
        model: OnnxRoboflowInferenceModel,
        video_source: VideoSource,
        on_prediction: Callable[[ObjectDetectionPrediction, VideoFrame], None],
        max_fps: Optional[float],
        predictions_queue: Queue,
        watchdog: PipelineWatchDog,
        status_update_handlers: List[Callable[[StatusUpdate], None]],
        inference_config: ModelConfig,
        active_learning_middleware: Union[
            NullActiveLearningMiddleware, ThreadingActiveLearningMiddleware
        ],
        process_frame_func: Callable = process_frame,
        top_k: Optional[int] = None,
    ):
        self._model = model
        self._process_frame_func = process_frame_func
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
        self._active_learning_middleware = active_learning_middleware
        self._top_k = top_k

    def _init_yolo_world_model(self, classes: List[str], **kwargs) -> None:
        self._model.set_classes(classes)

    def start(self, use_main_thread: bool = True) -> None:
        self._stop = False
        self._inference_thread = Thread(target=self._execute_inference)
        self._inference_thread.start()
        if self._active_learning_middleware is not None:
            self._active_learning_middleware.start_registration_thread()
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
            self._inference_thread = None
        if self._dispatching_thread is not None:
            self._dispatching_thread.join()
            self._dispatching_thread = None
        if self._active_learning_middleware is not None:
            self._active_learning_middleware.stop_registration_thread()

    def _execute_inference(self) -> None:
        send_inference_pipeline_status_update(
            severity=UpdateSeverity.INFO,
            event_type=INFERENCE_THREAD_STARTED_EVENT,
            status_update_handlers=self._status_update_handlers,
        )
        logger.info(f"Inference thread started")
        try:
            for video_frame in self._generate_frames():
                predictions = self._process_frame_func(
                    video_frame, self._model, self._inference_config, self._watchdog
                )
                if (
                    predictions
                    and self._top_k
                    and len(predictions["predictions"]) > self._top_k
                ):
                    preds = sorted(
                        predictions["predictions"],
                        key=lambda x: x["confidence"],
                        reverse=True,
                    )
                    predictions["predictions"] = preds[: self._top_k]

                self._predictions_queue.put((predictions, video_frame))
                send_inference_pipeline_status_update(
                    severity=UpdateSeverity.DEBUG,
                    event_type=INFERENCE_COMPLETED_EVENT,
                    payload={
                        "frame_id": video_frame.frame_id,
                        "frame_timestamp": video_frame.frame_timestamp,
                    },
                    status_update_handlers=self._status_update_handlers,
                )

        except Exception as error:
            payload = {
                "error_type": error.__class__.__name__,
                "error_message": str(error),
                "error_context": "inference_thread",
            }
            send_inference_pipeline_status_update(
                severity=UpdateSeverity.ERROR,
                event_type=INFERENCE_ERROR_EVENT,
                payload=payload,
                status_update_handlers=self._status_update_handlers,
            )
            logger.exception(f"Encountered inference error: {error}")
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
            inference_results: Optional[Tuple[dict, VideoFrame]] = (
                self._predictions_queue.get()
            )
            if inference_results is None:
                self._predictions_queue.task_done()
                break
            predictions, video_frame = inference_results
            try:
                self._on_prediction(predictions, video_frame)
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
            finally:
                self._predictions_queue.task_done()

    def _generate_frames(
        self,
    ) -> Generator[VideoFrame, None, None]:
        self._video_source.start()
        while True:
            source_properties = self._video_source.describe_source().source_properties
            if source_properties is None:
                break
            allow_reconnect = not source_properties.is_file
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

import time
from datetime import datetime
from functools import partial
from queue import Queue
from threading import Thread
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

from fastapi import BackgroundTasks

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
    MAX_ACTIVE_MODELS,
    PREDICTIONS_QUEUE_SIZE,
    RESTART_ATTEMPT_DELAY,
)
from inference.core.exceptions import CannotInitialiseModelError
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
from inference.core.interfaces.stream.entities import AnyPrediction, ModelConfig
from inference.core.interfaces.stream.model_handlers.roboflow_models import (
    default_process_frame,
)
from inference.core.interfaces.stream.sinks import active_learning_sink, multi_sink
from inference.core.interfaces.stream.watchdog import (
    NullPipelineWatchdog,
    PipelineWatchDog,
)
from inference.core.managers.active_learning import BackgroundTaskActiveLearningManager
from inference.core.managers.decorators.fixed_size_cache import WithFixedSizeCache
from inference.core.registries.roboflow import RoboflowModelRegistry
from inference.models.utils import ROBOFLOW_MODEL_TYPES, get_model

INFERENCE_PIPELINE_CONTEXT = "inference_pipeline"
SOURCE_CONNECTION_ATTEMPT_FAILED_EVENT = "SOURCE_CONNECTION_ATTEMPT_FAILED"
SOURCE_CONNECTION_LOST_EVENT = "SOURCE_CONNECTION_LOST"
INFERENCE_RESULTS_DISPATCHING_ERROR_EVENT = "INFERENCE_RESULTS_DISPATCHING_ERROR"
INFERENCE_THREAD_STARTED_EVENT = "INFERENCE_THREAD_STARTED"
INFERENCE_THREAD_FINISHED_EVENT = "INFERENCE_THREAD_FINISHED"
INFERENCE_COMPLETED_EVENT = "INFERENCE_COMPLETED"
INFERENCE_ERROR_EVENT = "INFERENCE_ERROR"


class InferencePipeline:
    @classmethod
    def init(
        cls,
        video_reference: Union[str, int],
        model_id: str,
        on_prediction: Optional[Callable[[AnyPrediction, VideoFrame], None]] = None,
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
        video_source_properties: Optional[Dict[str, float]] = None,
    ) -> "InferencePipeline":
        """
        This class creates the abstraction for making inferences from Roboflow models against video stream.
        It allows to choose model from Roboflow platform and run predictions against
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
            on_prediction (Callable[AnyPrediction, VideoFrame], None]): Function to be called
                once prediction is ready - passing both decoded frame, their metadata and dict with standard
                Roboflow model prediction (different for specific types of models).
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
            video_source_properties (Optional[dict[str, float]]): Optional source properties to set up the video source,
                corresponding to cv2 VideoCapture properties cv2.CAP_PROP_*. If not given, defaults for the video source
                will be used.
                Example valid properties are: {"frame_width": 1920, "frame_height": 1080, "fps": 30.0}


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
        model = get_model(model_id=model_id, api_key=api_key)
        on_video_frame = partial(
            default_process_frame, model=model, inference_config=inference_config
        )
        if watchdog is None:
            watchdog = NullPipelineWatchdog()
        status_update_handlers.append(watchdog.on_status_update)
        video_source = VideoSource.init(
            video_reference=video_reference,
            status_update_handlers=status_update_handlers,
            buffer_filling_strategy=source_buffer_filling_strategy,
            buffer_consumption_strategy=source_buffer_consumption_strategy,
            video_source_properties=video_source_properties,
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
        on_pipeline_start = active_learning_middleware.start_registration_thread
        on_pipeline_end = active_learning_middleware.stop_registration_thread
        return cls(
            on_video_frame=on_video_frame,
            video_source=video_source,
            on_prediction=on_prediction,
            max_fps=max_fps,
            predictions_queue=predictions_queue,
            watchdog=watchdog,
            status_update_handlers=status_update_handlers,
            on_pipeline_start=on_pipeline_start,
            on_pipeline_end=on_pipeline_end,
        )

    # we shall prob. generalise to zero-shot models and possibly create aliases
    # for easier use without knowledge of model ids
    @classmethod
    def init_with_yolo_world(
        cls,
        video_reference: Union[str, int],
        classes: List[str],
        model_size: str = "s",
        on_prediction: Optional[Callable[[AnyPrediction, VideoFrame], None]] = None,
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
        video_source_properties: Optional[Dict[str, float]] = None,
    ) -> "InferencePipeline":
        """
        This class creates the abstraction for making inferences from YoloWorld against video stream.
        The way of how `InferencePipeline` works is displayed in `InferencePipeline.init(...)` initializer
        method.

        Args:
            video_reference (Union[str, int]): Reference of source to be used to make predictions against.
                It can be video file path, stream URL and device (like camera) id (we handle whatever cv2 handles).
            classes (List[str]): List of classes to execute zero-shot detection against
            model_size (str): version of model - to be chosen from `s`, `m`, `l`
            on_prediction (Callable[AnyPrediction, VideoFrame], None]): Function to be called
                once prediction is ready - passing both decoded frame, their metadata and dict with standard
                Roboflow Object Detection prediction.
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
            video_source_properties (Optional[dict[str, float]]): Optional source properties to set up the video source,
                corresponding to cv2 VideoCapture properties cv2.CAP_PROP_*. If not given, defaults for the video source
                will be used.
                Example valid properties are: {"frame_width": 1920, "frame_height": 1080, "fps": 30.0}


        Other ENV variables involved in low-level configuration:
        * INFERENCE_PIPELINE_PREDICTIONS_QUEUE_SIZE - size of buffer for predictions that are ready for dispatching
        * INFERENCE_PIPELINE_RESTART_ATTEMPT_DELAY - delay for restarts on stream connection drop

        Returns: Instance of InferencePipeline

        Throws:
            * SourceConnectionError if source cannot be connected at start, however it attempts to reconnect
                always if connection to stream is lost.
        """
        inference_config = ModelConfig.init(
            class_agnostic_nms=class_agnostic_nms,
            confidence=confidence,
            iou_threshold=iou_threshold,
            max_candidates=max_candidates,
            max_detections=max_detections,
        )
        try:
            from inference.core.interfaces.stream.model_handlers.yolo_world import (
                build_yolo_world_inference_function,
            )

            on_video_frame = build_yolo_world_inference_function(
                model_id=f"yolo_world/{model_size}",
                classes=classes,
                inference_config=inference_config,
            )
        except ImportError as error:
            raise CannotInitialiseModelError(
                f"Could not initialise yolo_world/{model_size} due to lack of sufficient dependencies. "
                f"Use pip install inference[yolo-world] to install missing dependencies and try again."
            ) from error
        if watchdog is None:
            watchdog = NullPipelineWatchdog()
        if status_update_handlers is None:
            status_update_handlers = []
        status_update_handlers.append(watchdog.on_status_update)
        video_source = VideoSource.init(
            video_reference=video_reference,
            status_update_handlers=status_update_handlers,
            buffer_filling_strategy=source_buffer_filling_strategy,
            buffer_consumption_strategy=source_buffer_consumption_strategy,
            video_source_properties=video_source_properties,
        )
        watchdog.register_video_source(video_source=video_source)
        predictions_queue = Queue(maxsize=PREDICTIONS_QUEUE_SIZE)
        return cls(
            on_video_frame=on_video_frame,
            video_source=video_source,
            on_prediction=on_prediction,
            max_fps=max_fps,
            predictions_queue=predictions_queue,
            watchdog=watchdog,
            status_update_handlers=status_update_handlers,
        )

    @classmethod
    def init_with_workflow(
        cls,
        video_reference: Union[str, int],
        workflow_specification: dict,
        api_key: Optional[str] = None,
        image_input_name: str = "image",
        workflows_parameters: Optional[Dict[str, Any]] = None,
        on_prediction: Optional[Callable[[AnyPrediction, VideoFrame], None]] = None,
        max_fps: Optional[Union[float, int]] = None,
        watchdog: Optional[PipelineWatchDog] = None,
        status_update_handlers: Optional[List[Callable[[StatusUpdate], None]]] = None,
        source_buffer_filling_strategy: Optional[BufferFillingStrategy] = None,
        source_buffer_consumption_strategy: Optional[BufferConsumptionStrategy] = None,
        video_source_properties: Optional[Dict[str, float]] = None,
    ) -> "InferencePipeline":
        """
        This class creates the abstraction for making inferences from given workflow against video stream.
        The way of how `InferencePipeline` works is displayed in `InferencePipeline.init(...)` initializer
        method.

        Args:
            video_reference (Union[str, int]): Reference of source to be used to make predictions against.
                It can be video file path, stream URL and device (like camera) id (we handle whatever cv2 handles).
            workflow_specification (dict): Valid specification of workflow. See [workflow docs](https://github.com/roboflow/inference/tree/main/inference/enterprise/workflows)
            api_key (Optional[str]): Roboflow API key - if not passed - will be looked in env under "ROBOFLOW_API_KEY"
                and "API_KEY" variables. API key, passed in some form is required.
            image_input_name (str): Name of input image defined in `workflow_specification`. `InferencePipeline` will be
                injecting video frames to workflow through that parameter name.
            workflows_parameters (Optional[Dict[str, Any]]): Dictionary with additional parameters that can be
                defined within `workflow_specification`.
            on_prediction (Callable[AnyPrediction, VideoFrame], None]): Function to be called
                once prediction is ready - passing both decoded frame, their metadata and dict with workflow output.
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
            video_source_properties (Optional[dict[str, float]]): Optional source properties to set up the video source,
                corresponding to cv2 VideoCapture properties cv2.CAP_PROP_*. If not given, defaults for the video source
                will be used.
                Example valid properties are: {"frame_width": 1920, "frame_height": 1080, "fps": 30.0}


        Other ENV variables involved in low-level configuration:
        * INFERENCE_PIPELINE_PREDICTIONS_QUEUE_SIZE - size of buffer for predictions that are ready for dispatching
        * INFERENCE_PIPELINE_RESTART_ATTEMPT_DELAY - delay for restarts on stream connection drop

        Returns: Instance of InferencePipeline

        Throws:
            * SourceConnectionError if source cannot be connected at start, however it attempts to reconnect
                always if connection to stream is lost.
        """
        try:
            from inference.core.interfaces.stream.model_handlers.workflows import (
                run_video_frame_through_workflow,
            )
            from inference.enterprise.workflows.complier.steps_executors.active_learning_middlewares import (
                WorkflowsActiveLearningMiddleware,
            )

            workflows_active_learning_middleware = WorkflowsActiveLearningMiddleware(
                cache=cache,
            )
            model_registry = RoboflowModelRegistry(ROBOFLOW_MODEL_TYPES)
            model_manager = BackgroundTaskActiveLearningManager(
                model_registry=model_registry, cache=cache
            )
            model_manager = WithFixedSizeCache(
                model_manager,
                max_size=MAX_ACTIVE_MODELS,
            )
            if api_key is None:
                api_key = API_KEY
            background_tasks = BackgroundTasks()
            on_video_frame = partial(
                run_video_frame_through_workflow,
                workflow_specification=workflow_specification,
                model_manager=model_manager,
                image_input_name=image_input_name,
                workflows_parameters=workflows_parameters,
                api_key=api_key,
                workflows_active_learning_middleware=workflows_active_learning_middleware,
                background_tasks=background_tasks,
            )
        except ImportError as error:
            raise CannotInitialiseModelError(
                f"Could not initialise workflow processing due to lack of dependencies required. "
                f"Please provide an issue report under https://github.com/roboflow/inference/issues"
            ) from error
        if watchdog is None:
            watchdog = NullPipelineWatchdog()
        if status_update_handlers is None:
            status_update_handlers = []
        status_update_handlers.append(watchdog.on_status_update)
        video_source = VideoSource.init(
            video_reference=video_reference,
            status_update_handlers=status_update_handlers,
            buffer_filling_strategy=source_buffer_filling_strategy,
            buffer_consumption_strategy=source_buffer_consumption_strategy,
            video_source_properties=video_source_properties,
        )
        watchdog.register_video_source(video_source=video_source)
        predictions_queue = Queue(maxsize=PREDICTIONS_QUEUE_SIZE)
        return cls(
            on_video_frame=on_video_frame,
            video_source=video_source,
            predictions_queue=predictions_queue,
            watchdog=watchdog,
            status_update_handlers=status_update_handlers,
            on_prediction=on_prediction,
            max_fps=max_fps,
        )

    @classmethod
    def init_with_custom_logic(
        cls,
        video_reference: Union[str, int],
        on_video_frame: Callable[[VideoFrame], AnyPrediction],
        on_prediction: Optional[Callable[[AnyPrediction, VideoFrame], None]] = None,
        on_pipeline_start: Optional[Callable[[], None]] = None,
        on_pipeline_end: Optional[Callable[[], None]] = None,
        max_fps: Optional[Union[float, int]] = None,
        watchdog: Optional[PipelineWatchDog] = None,
        status_update_handlers: Optional[List[Callable[[StatusUpdate], None]]] = None,
        source_buffer_filling_strategy: Optional[BufferFillingStrategy] = None,
        source_buffer_consumption_strategy: Optional[BufferConsumptionStrategy] = None,
        video_source_properties: Optional[Dict[str, float]] = None,
    ) -> "InferencePipeline":
        """
        This class creates the abstraction for making inferences from given workflow against video stream.
        The way of how `InferencePipeline` works is displayed in `InferencePipeline.init(...)` initialiser
        method.

        Args:
            video_reference (Union[str, int]): Reference of source to be used to make predictions against.
                It can be video file path, stream URL and device (like camera) id (we handle whatever cv2 handles).
            on_video_frame (Callable[[VideoFrame], AnyPrediction]): function supposed to make prediction (or do another
                kind of custom processing according to your will). Accept `VideoFrame` object and is supposed
                to return dictionary with results of any kind.
            on_prediction (Callable[AnyPrediction, VideoFrame], None]): Function to be called
                once prediction is ready - passing both decoded frame, their metadata and dict with output from your
                custom callable `on_video_frame(...)`. Logic here must be adjusted to the output of `on_video_frame`.
            on_pipeline_start (Optional[Callable[[], None]]): Optional (parameter-free) function to be called
                whenever pipeline starts
            on_pipeline_end (Optional[Callable[[], None]]): Optional (parameter-free) function to be called
                whenever pipeline ends
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
            video_source_properties (Optional[dict[str, float]]): Optional source properties to set up the video source,
                corresponding to cv2 VideoCapture properties cv2.CAP_PROP_*. If not given, defaults for the video source
                will be used.
                Example valid properties are: {"frame_width": 1920, "frame_height": 1080, "fps": 30.0}


        Other ENV variables involved in low-level configuration:
        * INFERENCE_PIPELINE_PREDICTIONS_QUEUE_SIZE - size of buffer for predictions that are ready for dispatching
        * INFERENCE_PIPELINE_RESTART_ATTEMPT_DELAY - delay for restarts on stream connection drop

        Returns: Instance of InferencePipeline

        Throws:
            * SourceConnectionError if source cannot be connected at start, however it attempts to reconnect
                always if connection to stream is lost.
        """
        if watchdog is None:
            watchdog = NullPipelineWatchdog()
        if status_update_handlers is None:
            status_update_handlers = []
        status_update_handlers.append(watchdog.on_status_update)
        video_source = VideoSource.init(
            video_reference=video_reference,
            status_update_handlers=status_update_handlers,
            buffer_filling_strategy=source_buffer_filling_strategy,
            buffer_consumption_strategy=source_buffer_consumption_strategy,
            video_source_properties=video_source_properties,
        )
        watchdog.register_video_source(video_source=video_source)
        predictions_queue = Queue(maxsize=PREDICTIONS_QUEUE_SIZE)
        return cls(
            on_video_frame=on_video_frame,
            video_source=video_source,
            predictions_queue=predictions_queue,
            watchdog=watchdog,
            status_update_handlers=status_update_handlers,
            on_prediction=on_prediction,
            max_fps=max_fps,
            on_pipeline_start=on_pipeline_start,
            on_pipeline_end=on_pipeline_end,
        )

    def __init__(
        self,
        on_video_frame: Callable[[VideoFrame], AnyPrediction],
        video_source: VideoSource,
        predictions_queue: Queue,
        watchdog: PipelineWatchDog,
        status_update_handlers: List[Callable[[StatusUpdate], None]],
        on_prediction: Optional[Callable[[AnyPrediction, VideoFrame], None]] = None,
        on_pipeline_start: Optional[Callable[[], None]] = None,
        on_pipeline_end: Optional[Callable[[], None]] = None,
        max_fps: Optional[float] = None,
    ):
        self._on_video_frame = on_video_frame
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
        self._on_pipeline_start = on_pipeline_start
        self._on_pipeline_end = on_pipeline_end

    def start(self, use_main_thread: bool = True) -> None:
        self._stop = False
        self._inference_thread = Thread(target=self._execute_inference)
        self._inference_thread.start()
        if self._on_pipeline_start is not None:
            self._on_pipeline_start()
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
        if self._on_pipeline_end is not None:
            self._on_pipeline_end()

    def _execute_inference(self) -> None:
        send_inference_pipeline_status_update(
            severity=UpdateSeverity.INFO,
            event_type=INFERENCE_THREAD_STARTED_EVENT,
            status_update_handlers=self._status_update_handlers,
        )
        logger.info(f"Inference thread started")
        try:
            for video_frame in self._generate_frames():
                self._watchdog.on_model_inference_started(
                    frame_timestamp=video_frame.frame_timestamp,
                    frame_id=video_frame.frame_id,
                )
                predictions = self._on_video_frame(video_frame)
                self._watchdog.on_model_prediction_ready(
                    frame_timestamp=video_frame.frame_timestamp,
                    frame_id=video_frame.frame_id,
                )
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
            if self._on_prediction is not None:
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

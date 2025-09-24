from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from enum import Enum
from functools import partial
from queue import Queue
from threading import Thread
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

from inference.core import logger
from inference.core.active_learning.middlewares import (
    NullActiveLearningMiddleware,
    ThreadingActiveLearningMiddleware,
)
from inference.core.cache import cache
from inference.core.env import (
    ACTIVE_LEARNING_ENABLED,
    API_KEY,
    DEFAULT_BUFFER_SIZE,
    DISABLE_PREPROC_AUTO_ORIENT,
    ENABLE_FRAME_DROP_ON_VIDEO_FILE_RATE_LIMITING,
    ENABLE_WORKFLOWS_PROFILING,
    MAX_ACTIVE_MODELS,
    PREDICTIONS_QUEUE_SIZE,
    WORKFLOWS_PROFILER_BUFFER_SIZE,
)
from inference.core.exceptions import CannotInitialiseModelError, MissingApiKeyError
from inference.core.interfaces.camera.entities import (
    StatusUpdate,
    UpdateSeverity,
    VideoFrame,
    VideoSourceIdentifier,
)
from inference.core.interfaces.camera.utils import multiplex_videos
from inference.core.interfaces.camera.video_source import (
    BufferConsumptionStrategy,
    BufferFillingStrategy,
    VideoSource,
)
from inference.core.interfaces.stream.entities import (
    AnyPrediction,
    InferenceHandler,
    ModelConfig,
    SinkHandler,
)
from inference.core.interfaces.stream.model_handlers.roboflow_models import (
    default_process_frame,
)
from inference.core.interfaces.stream.sinks import active_learning_sink, multi_sink
from inference.core.interfaces.stream.utils import (
    on_pipeline_end,
    prepare_video_sources,
)
from inference.core.interfaces.stream.watchdog import (
    NullPipelineWatchdog,
    PipelineWatchDog,
)
from inference.core.managers.active_learning import BackgroundTaskActiveLearningManager
from inference.core.managers.decorators.fixed_size_cache import WithFixedSizeCache
from inference.core.registries.roboflow import RoboflowModelRegistry
from inference.core.utils.function import experimental
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.profiling.core import (
    BaseWorkflowsProfiler,
    NullWorkflowsProfiler,
)
from inference.models.aliases import resolve_roboflow_model_alias
from inference.models.utils import ROBOFLOW_MODEL_TYPES, get_model

INFERENCE_PIPELINE_CONTEXT = "inference_pipeline"
SOURCE_CONNECTION_ATTEMPT_FAILED_EVENT = "SOURCE_CONNECTION_ATTEMPT_FAILED"
SOURCE_CONNECTION_LOST_EVENT = "SOURCE_CONNECTION_LOST"
INFERENCE_RESULTS_DISPATCHING_ERROR_EVENT = "INFERENCE_RESULTS_DISPATCHING_ERROR"
INFERENCE_THREAD_STARTED_EVENT = "INFERENCE_THREAD_STARTED"
INFERENCE_THREAD_FINISHED_EVENT = "INFERENCE_THREAD_FINISHED"
INFERENCE_COMPLETED_EVENT = "INFERENCE_COMPLETED"
INFERENCE_ERROR_EVENT = "INFERENCE_ERROR"


class SinkMode(Enum):
    ADAPTIVE = "adaptive"
    BATCH = "batch"
    SEQUENTIAL = "sequential"


class InferencePipeline:
    @classmethod
    def init(
        cls,
        video_reference: Union[VideoSourceIdentifier, List[VideoSourceIdentifier]],
        model_id: str,
        on_prediction: SinkHandler = None,
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
        video_source_properties: Optional[
            Union[Dict[str, float], List[Optional[Dict[str, float]]]]
        ] = None,
        active_learning_target_dataset: Optional[str] = None,
        batch_collection_timeout: Optional[float] = None,
        sink_mode: SinkMode = SinkMode.ADAPTIVE,
        predictions_queue_size: int = PREDICTIONS_QUEUE_SIZE,
        decoding_buffer_size: int = DEFAULT_BUFFER_SIZE,
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

        Since version 0.9.18, `InferencePipeline` is capable of handling multiple video sources at once. If multiple
        sources are provided - source multiplexing will happen. One of the change introduced in that release is switch
        from `get_video_frames_generator(...)` as video frames provider into `multiplex_videos(...)`. For a single
        video source, the behaviour of `InferencePipeline` is remained unchanged when default parameters are used.
        For multiple videos - frames are multiplexed, and we can adjust the pipeline behaviour using new configuration
        options. `batch_collection_timeout` is one of the new option - it is the parameter of `multiplex_videos(...)`
        that dictates how long the batch frames collection process may wait for all sources to provide video frame.
        It can be set infinite (None) or with specific value representing fraction of second. We advise that value to
        be set in production solutions to avoid processing slow-down caused by source with unstable latency spikes.
        For more information on multiplexing process - please visit `multiplex_videos(...)` function docs.
        Another change is the way on how sinks work. They can work in `SinkMode.ADAPTIVE` - which means that
        video frames and predictions will be either provided to sink as list of objects, or specific elements -
        and the determining factor is number of sources (it will behave SEQUENTIAL for one source and BATCH if multiple
        ones are provided). All old sinks were adjusted to work in both modes, custom ones should be migrated
        to reflect changes in sink function signature.

        Args:
            model_id (str): Name and version of model on the Roboflow platform (example: "my-model/3")
            video_reference (Union[str, int, List[Union[str, int]]]): Reference of source or sources to be used to make
                predictions against. It can be video file path, stream URL and device (like camera) id
                (we handle whatever cv2 handles). It can also be a list of references (since v0.9.18) - and then
                it will trigger parallel processing of multiple sources. It has some implication on sinks. See:
                `sink_mode` parameter comments.
            on_prediction (Callable[AnyPrediction, VideoFrame], None]): Function to be called
                once prediction is ready - passing both decoded frame, their metadata and dict with standard
                Roboflow model prediction (different for specific types of models).
            api_key (Optional[str]): Roboflow API key - if not passed - will be looked in env under "ROBOFLOW_API_KEY"
                and "API_KEY" variables. API key, passed in some form is required.
            max_fps (Optional[Union[float, int]]): Specific value passed as this parameter will be used to
                dictate max FPS of each video source.
                The implementation details of this option has been changed in release `v0.26.0`. Prior to the release
                this value, when applied to video files caused the processing to wait `1 / max_fps` seconds before next
                frame is processed - the new implementation drops the intermediate frames, which seems to be more
                aligned with peoples expectations.
                New behaviour is now enabled in experimental mode, by setting environmental variable flag
                `ENABLE_FRAME_DROP_ON_VIDEO_FILE_RATE_LIMITING=True`. Please note that the new behaviour will
                be the default one end of Q4 2024!
            watchdog (Optional[PipelineWatchDog]): Implementation of class that allows profiling of
                inference pipeline - if not given null implementation (doing nothing) will be used.
            status_update_handlers (Optional[List[Callable[[StatusUpdate], None]]]): List of handlers to intercept
                status updates of all elements of the pipeline. Should be used only if detailed inspection of
                pipeline behaviour in time is needed. Please point out that handlers should be possible to be executed
                fast - otherwise they will impair pipeline performance. All errors will be logged as warnings
                without re-raising. Default: None.
            source_buffer_filling_strategy (Optional[BufferFillingStrategy]): Parameter dictating strategy for
                video stream decoding behaviour. By default - tweaked to the type of source given.
                Please find detailed explanation in docs of [`VideoSource`](/reference/inference/core/interfaces/camera/video_source/#inference.core.interfaces.camera.video_source.VideoSource)
            source_buffer_consumption_strategy (Optional[BufferConsumptionStrategy]): Parameter dictating strategy for
                video stream frames consumption. By default - tweaked to the type of source given.
                Please find detailed explanation in docs of [`VideoSource`](/reference/inference/core/interfaces/camera/video_source/#inference.core.interfaces.camera.video_source.VideoSource)
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
            video_source_properties (Optional[Union[Dict[str, float], List[Optional[Dict[str, float]]]]]):
                Optional source properties to set up the video source, corresponding to cv2 VideoCapture properties
                cv2.CAP_PROP_*. If not given, defaults for the video source will be used.
                It is optional and if provided can be provided as single dict (applicable for all sources) or
                as list of configs. Then the list must be of length of `video_reference` and may also contain None
                values to denote that specific source should remain not configured.
                Example valid properties are: {"frame_width": 1920, "frame_height": 1080, "fps": 30.0}
            active_learning_target_dataset (Optional[str]): Parameter to be used when Active Learning data registration
                should happen against different dataset than the one pointed by model_id
            batch_collection_timeout (Optional[float]): Parameter of multiplex_videos(...) dictating how long process
                to grab frames from multiple sources can wait for batch to be filled before yielding already collected
                frames. Please set this value in PRODUCTION to avoid performance drops when specific sources shows
                unstable latency. Visit `multiplex_videos(...)` for more information about multiplexing process.
            sink_mode (SinkMode): Parameter that controls how video frames and predictions will be passed to sink
                handler. With SinkMode.SEQUENTIAL - each frame and prediction triggers separate call for sink,
                in case of SinkMode.BATCH - list of frames and predictions will be provided to sink, always aligned
                in the order of video sources - with None values in the place of vide_frames / predictions that
                were skipped due to `batch_collection_timeout`.
                `SinkMode.ADAPTIVE` is a middle ground (and default mode) - all old sources will work in that mode
                against a single video input, as the pipeline will behave as if running in `SinkMode.SEQUENTIAL`.
                To handle multiple videos - sink needs to accept `predictions: List[Optional[dict]]` and
                `video_frame: List[Optional[VideoFrame]]`. It is also possible to process multiple videos using
                old sinks - but then `SinkMode.SEQUENTIAL` is to be used, causing sink to be called on each
                prediction element.
            predictions_queue_size int: Size of buffer for predictions that are ready for dispatching
                default value is taken from INFERENCE_PIPELINE_PREDICTIONS_QUEUE_SIZE env variable
            decoding_buffer_size (int): size of video source decoding buffer
                default value is taken from VIDEO_SOURCE_BUFFER_SIZE env variable

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
            resolved_model_id = resolve_roboflow_model_alias(model_id=model_id)
            target_dataset = (
                active_learning_target_dataset or resolved_model_id.split("/")[0]
            )
            active_learning_middleware = ThreadingActiveLearningMiddleware.init(
                api_key=api_key,
                target_dataset=target_dataset,
                model_id=resolved_model_id,
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
        return cls.init_with_custom_logic(
            video_reference=video_reference,
            on_video_frame=on_video_frame,
            on_prediction=on_prediction,
            on_pipeline_start=on_pipeline_start,
            on_pipeline_end=on_pipeline_end,
            max_fps=max_fps,
            watchdog=watchdog,
            status_update_handlers=status_update_handlers,
            source_buffer_filling_strategy=source_buffer_filling_strategy,
            source_buffer_consumption_strategy=source_buffer_consumption_strategy,
            video_source_properties=video_source_properties,
            batch_collection_timeout=batch_collection_timeout,
            sink_mode=sink_mode,
            predictions_queue_size=predictions_queue_size,
            decoding_buffer_size=decoding_buffer_size,
        )

    @classmethod
    def init_with_yolo_world(
        cls,
        video_reference: Union[str, int, List[Union[str, int]]],
        classes: List[str],
        model_size: str = "s",
        on_prediction: SinkHandler = None,
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
        batch_collection_timeout: Optional[float] = None,
        sink_mode: SinkMode = SinkMode.ADAPTIVE,
        predictions_queue_size: int = PREDICTIONS_QUEUE_SIZE,
        decoding_buffer_size: int = DEFAULT_BUFFER_SIZE,
    ) -> "InferencePipeline":
        """
        This class creates the abstraction for making inferences from YoloWorld against video stream.
        The way of how `InferencePipeline` works is displayed in `InferencePipeline.init(...)` initializer
        method.

        Args:
            video_reference (Union[str, int, List[Union[str, int]]]): Reference of source or sources to be used to make
                predictions against. It can be video file path, stream URL and device (like camera) id
                (we handle whatever cv2 handles). It can also be a list of references (since v0.9.18) - and then
                it will trigger parallel processing of multiple sources. It has some implication on sinks. See:
                `sink_mode` parameter comments.
            classes (List[str]): List of classes to execute zero-shot detection against
            model_size (str): version of model - to be chosen from `s`, `m`, `l`
            on_prediction (Callable[AnyPrediction, VideoFrame], None]): Function to be called
                once prediction is ready - passing both decoded frame, their metadata and dict with standard
                Roboflow Object Detection prediction.
            max_fps (Optional[Union[float, int]]): Specific value passed as this parameter will be used to
                dictate max FPS of each video source.
                The implementation details of this option has been changed in release `v0.26.0`. Prior to the release
                this value, when applied to video files caused the processing to wait `1 / max_fps` seconds before next
                frame is processed - the new implementation drops the intermediate frames, which seems to be more
                aligned with peoples expectations.
                New behaviour is now enabled in experimental mode, by setting environmental variable flag
                `ENABLE_FRAME_DROP_ON_VIDEO_FILE_RATE_LIMITING=True`. Please note that the new behaviour will
                be the default one end of Q4 2024!
            watchdog (Optional[PipelineWatchDog]): Implementation of class that allows profiling of
                inference pipeline - if not given null implementation (doing nothing) will be used.
            status_update_handlers (Optional[List[Callable[[StatusUpdate], None]]]): List of handlers to intercept
                status updates of all elements of the pipeline. Should be used only if detailed inspection of
                pipeline behaviour in time is needed. Please point out that handlers should be possible to be executed
                fast - otherwise they will impair pipeline performance. All errors will be logged as warnings
                without re-raising. Default: None.
            source_buffer_filling_strategy (Optional[BufferFillingStrategy]): Parameter dictating strategy for
                video stream decoding behaviour. By default - tweaked to the type of source given.
                Please find detailed explanation in docs of [`VideoSource`](/reference/inference/core/interfaces/camera/video_source/#inference.core.interfaces.camera.video_source.VideoSource)
            source_buffer_consumption_strategy (Optional[BufferConsumptionStrategy]): Parameter dictating strategy for
                video stream frames consumption. By default - tweaked to the type of source given.
                Please find detailed explanation in docs of [`VideoSource`](/reference/inference/core/interfaces/camera/video_source/#inference.core.interfaces.camera.video_source.VideoSource)
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
            video_source_properties (Optional[Union[Dict[str, float], List[Optional[Dict[str, float]]]]]):
                Optional source properties to set up the video source, corresponding to cv2 VideoCapture properties
                cv2.CAP_PROP_*. If not given, defaults for the video source will be used.
                It is optional and if provided can be provided as single dict (applicable for all sources) or
                as list of configs. Then the list must be of length of `video_reference` and may also contain None
                values to denote that specific source should remain not configured.
                Example valid properties are: {"frame_width": 1920, "frame_height": 1080, "fps": 30.0}
            batch_collection_timeout (Optional[float]): Parameter of multiplex_videos(...) dictating how long process
                to grab frames from multiple sources can wait for batch to be filled before yielding already collected
                frames. Please set this value in PRODUCTION to avoid performance drops when specific sources shows
                unstable latency. Visit `multiplex_videos(...)` for more information about multiplexing process.
            sink_mode (SinkMode): Parameter that controls how video frames and predictions will be passed to sink
                handler. With SinkMode.SEQUENTIAL - each frame and prediction triggers separate call for sink,
                in case of SinkMode.BATCH - list of frames and predictions will be provided to sink, always aligned
                in the order of video sources - with None values in the place of vide_frames / predictions that
                were skipped due to `batch_collection_timeout`.
                `SinkMode.ADAPTIVE` is a middle ground (and default mode) - all old sources will work in that mode
                against a single video input, as the pipeline will behave as if running in `SinkMode.SEQUENTIAL`.
                To handle multiple videos - sink needs to accept `predictions: List[Optional[dict]]` and
                `video_frame: List[Optional[VideoFrame]]`. It is also possible to process multiple videos using
                old sinks - but then `SinkMode.SEQUENTIAL` is to be used, causing sink to be called on each
                prediction element.
            predictions_queue_size int: Size of buffer for predictions that are ready for dispatching
                default value is taken from INFERENCE_PIPELINE_PREDICTIONS_QUEUE_SIZE env variable
            decoding_buffer_size (int): size of video source decoding buffer
                default value is taken from VIDEO_SOURCE_BUFFER_SIZE env variable

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
        return cls.init_with_custom_logic(
            video_reference=video_reference,
            on_video_frame=on_video_frame,
            on_prediction=on_prediction,
            on_pipeline_start=None,
            on_pipeline_end=None,
            max_fps=max_fps,
            watchdog=watchdog,
            status_update_handlers=status_update_handlers,
            source_buffer_filling_strategy=source_buffer_filling_strategy,
            source_buffer_consumption_strategy=source_buffer_consumption_strategy,
            video_source_properties=video_source_properties,
            batch_collection_timeout=batch_collection_timeout,
            sink_mode=sink_mode,
            predictions_queue_size=predictions_queue_size,
            decoding_buffer_size=decoding_buffer_size,
        )

    @classmethod
    @experimental(
        reason="Usage of workflows with `InferencePipeline` is an experimental feature. Please report any issues "
        "here: https://github.com/roboflow/inference/issues"
    )
    def init_with_workflow(
        cls,
        video_reference: Union[str, int, List[Union[str, int]]],
        workflow_specification: Optional[dict] = None,
        workspace_name: Optional[str] = None,
        workflow_id: Optional[str] = None,
        api_key: Optional[str] = None,
        image_input_name: str = "image",
        workflows_parameters: Optional[Dict[str, Any]] = None,
        on_prediction: SinkHandler = None,
        max_fps: Optional[Union[float, int]] = None,
        watchdog: Optional[PipelineWatchDog] = None,
        status_update_handlers: Optional[List[Callable[[StatusUpdate], None]]] = None,
        source_buffer_filling_strategy: Optional[BufferFillingStrategy] = None,
        source_buffer_consumption_strategy: Optional[BufferConsumptionStrategy] = None,
        video_source_properties: Optional[Dict[str, float]] = None,
        workflow_init_parameters: Optional[Dict[str, Any]] = None,
        workflows_thread_pool_workers: int = 4,
        cancel_thread_pool_tasks_on_exit: bool = True,
        video_metadata_input_name: str = "video_metadata",
        batch_collection_timeout: Optional[float] = None,
        profiling_directory: str = "./inference_profiling",
        use_workflow_definition_cache: bool = True,
        serialize_results: bool = False,
        predictions_queue_size: int = PREDICTIONS_QUEUE_SIZE,
        decoding_buffer_size: int = DEFAULT_BUFFER_SIZE,
    ) -> "InferencePipeline":
        """
        This class creates the abstraction for making inferences from given workflow against video stream.
        The way of how `InferencePipeline` works is displayed in `InferencePipeline.init(...)` initializer
        method.

        Args:
            video_reference (Union[str, int, List[Union[str, int]]]): Reference of source to be used to make predictions
                against. It can be video file path, stream URL and device (like camera) id
                (we handle whatever cv2 handles). It can also be a list of references (since v0.13.0) - and then
                it will trigger parallel processing of multiple sources. It has some implication on sinks. See:
                `sink_mode` parameter comments.
            workflow_specification (Optional[dict]): Valid specification of workflow. See [workflow docs](https://github.com/roboflow/inference/tree/main/inference/enterprise/workflows).
                It can be provided optionally, but if not given, both `workspace_name` and `workflow_id`
                must be provided.
            workspace_name (Optional[str]): When using registered workflows - Roboflow workspace name needs to be given.
            workflow_id (Optional[str]): When using registered workflows - Roboflow workflow id needs to be given.
            api_key (Optional[str]): Roboflow API key - if not passed - will be looked in env under "ROBOFLOW_API_KEY"
                and "API_KEY" variables. API key, passed in some form is required.
            image_input_name (str): Name of input image defined in `workflow_specification` or Workflow definition saved
                on the Roboflow Platform. `InferencePipeline` will be injecting video frames to workflow through that
                parameter name.
            workflows_parameters (Optional[Dict[str, Any]]): Dictionary with additional parameters that can be
                defined within `workflow_specification`.
            on_prediction (Callable[AnyPrediction, VideoFrame], None]): Function to be called
                once prediction is ready - passing both decoded frame, their metadata and dict with workflow output.
            max_fps (Optional[Union[float, int]]): Specific value passed as this parameter will be used to
                dictate max FPS of each video source.
                The implementation details of this option has been changed in release `v0.26.0`. Prior to the release
                this value, when applied to video files caused the processing to wait `1 / max_fps` seconds before next
                frame is processed - the new implementation drops the intermediate frames, which seems to be more
                aligned with peoples expectations.
                New behaviour is now enabled in experimental mode, by setting environmental variable flag
                `ENABLE_FRAME_DROP_ON_VIDEO_FILE_RATE_LIMITING=True`. Please note that the new behaviour will
                be the default one end of Q4 2024!
            watchdog (Optional[PipelineWatchDog]): Implementation of class that allows profiling of
                inference pipeline - if not given null implementation (doing nothing) will be used.
            status_update_handlers (Optional[List[Callable[[StatusUpdate], None]]]): List of handlers to intercept
                status updates of all elements of the pipeline. Should be used only if detailed inspection of
                pipeline behaviour in time is needed. Please point out that handlers should be possible to be executed
                fast - otherwise they will impair pipeline performance. All errors will be logged as warnings
                without re-raising. Default: None.
            source_buffer_filling_strategy (Optional[BufferFillingStrategy]): Parameter dictating strategy for
                video stream decoding behaviour. By default - tweaked to the type of source given.
                Please find detailed explanation in docs of [`VideoSource`](/reference/inference/core/interfaces/camera/video_source/#inference.core.interfaces.camera.video_source.VideoSource)
            source_buffer_consumption_strategy (Optional[BufferConsumptionStrategy]): Parameter dictating strategy for
                video stream frames consumption. By default - tweaked to the type of source given.
                Please find detailed explanation in docs of [`VideoSource`](/reference/inference/core/interfaces/camera/video_source/#inference.core.interfaces.camera.video_source.VideoSource)
            video_source_properties (Optional[dict[str, float]]): Optional source properties to set up the video source,
                corresponding to cv2 VideoCapture properties cv2.CAP_PROP_*. If not given, defaults for the video source
                will be used.
                Example valid properties are: {"frame_width": 1920, "frame_height": 1080, "fps": 30.0}
            workflow_init_parameters (Optional[Dict[str, Any]]): Additional init parameters to be used by
                workflows Execution Engine to init steps of your workflow - may be required when running workflows
                with custom plugins.
            workflows_thread_pool_workers (int): Number of workers for workflows thread pool which is used
                by workflows blocks to run background tasks.
            cancel_thread_pool_tasks_on_exit (bool): Flag to decide if unstated background tasks should be
                canceled at the end of InferencePipeline processing. By default, when video file ends or
                pipeline is stopped, tasks that has not started will be cancelled.
            video_metadata_input_name (str): Name of input for video metadata defined in `workflow_specification` or
                Workflow definition saved  on the Roboflow Platform. `InferencePipeline` will be injecting video frames
                metadata to workflows through that parameter name.
            batch_collection_timeout (Optional[float]): Parameter of multiplex_videos(...) dictating how long process
                to grab frames from multiple sources can wait for batch to be filled before yielding already collected
                frames. Please set this value in PRODUCTION to avoid performance drops when specific sources shows
                unstable latency. Visit `multiplex_videos(...)` for more information about multiplexing process.
            profiling_directory (str): Directory where workflows profiler traces will be dumped. To enable profiling
                export `ENABLE_WORKFLOWS_PROFILING=True` environmental variable. You may specify number of workflow
                runs in a buffer with environmental variable `WORKFLOWS_PROFILER_BUFFER_SIZE=n` - making last `n`
                frames to be present in buffer on processing end.
            use_workflow_definition_cache (bool): Controls usage of cache for workflow definitions. Set this to False
                when you frequently modify definition saved in Roboflow app and want to fetch the
                newest version for the request. Only applies for Workflows definitions saved on Roboflow platform.
            serialize_results (bool): Boolean flag to decide if ExecutionEngine run should serialize workflow
                results for each frame. If that is set true, sinks will receive serialized workflow responses.
            predictions_queue_size int: Size of buffer for predictions that are ready for dispatching
                default value is taken from INFERENCE_PIPELINE_PREDICTIONS_QUEUE_SIZE env variable
            decoding_buffer_size (int): size of video source decoding buffer
                default value is taken from VIDEO_SOURCE_BUFFER_SIZE env variable

        Other ENV variables involved in low-level configuration:
        * INFERENCE_PIPELINE_PREDICTIONS_QUEUE_SIZE - size of buffer for predictions that are ready for dispatching
        * INFERENCE_PIPELINE_RESTART_ATTEMPT_DELAY - delay for restarts on stream connection drop

        Returns: Instance of InferencePipeline

        Throws:
            * SourceConnectionError if source cannot be connected at start, however it attempts to reconnect
                always if connection to stream is lost.
            * ValueError if workflow specification not provided and registered workflow not pointed out
            * NotImplementedError if workflow used against multiple videos which is not supported yet
            * MissingApiKeyError - if API key is not provided in situation when retrieving workflow definition
                from Roboflow API is needed
        """
        if ENABLE_WORKFLOWS_PROFILING:
            profiler = BaseWorkflowsProfiler.init(
                max_runs_in_buffer=WORKFLOWS_PROFILER_BUFFER_SIZE
            )
        else:
            profiler = NullWorkflowsProfiler.init()
        if api_key is None:
            api_key = API_KEY
        named_workflow_specified = (workspace_name is not None) and (
            workflow_id is not None
        )
        if not (named_workflow_specified != (workflow_specification is not None)):
            raise ValueError(
                "Parameters (`workspace_name`, `workflow_id`) can be used mutually exclusive with "
                "`workflow_specification`, but at least one must be set."
            )
        try:
            from inference.core.interfaces.stream.model_handlers.workflows import (
                WorkflowRunner,
            )
            from inference.core.roboflow_api import get_workflow_specification
            from inference.core.workflows.execution_engine.core import ExecutionEngine

            if workflow_specification is None:
                if api_key is None:
                    raise MissingApiKeyError(
                        "Roboflow API key needs to be provided either as parameter or via env variable "
                        "ROBOFLOW_API_KEY. If you do not know how to get API key - visit "
                        "https://docs.roboflow.com/api-reference/authentication#retrieve-an-api-key to learn how to "
                        "retrieve one."
                    )
                with profiler.profile_execution_phase(
                    name="workflow_definition_fetching",
                    categories=["inference_package_operation"],
                ):
                    workflow_specification = get_workflow_specification(
                        api_key=api_key,
                        workspace_id=workspace_name,
                        workflow_id=workflow_id,
                        use_cache=use_workflow_definition_cache,
                    )
            model_registry = RoboflowModelRegistry(ROBOFLOW_MODEL_TYPES)
            model_manager = BackgroundTaskActiveLearningManager(
                model_registry=model_registry, cache=cache
            )
            model_manager = WithFixedSizeCache(
                model_manager,
                max_size=MAX_ACTIVE_MODELS,
            )
            if workflow_init_parameters is None:
                workflow_init_parameters = {}
            thread_pool_executor = ThreadPoolExecutor(
                max_workers=workflows_thread_pool_workers
            )
            workflow_init_parameters["workflows_core.model_manager"] = model_manager
            workflow_init_parameters["workflows_core.api_key"] = api_key
            workflow_init_parameters["workflows_core.thread_pool_executor"] = (
                thread_pool_executor
            )
            execution_engine = ExecutionEngine.init(
                workflow_definition=workflow_specification,
                init_parameters=workflow_init_parameters,
                workflow_id=workflow_id,
                profiler=profiler,
            )
            workflow_runner = WorkflowRunner()
            on_video_frame = partial(
                workflow_runner.run_workflow,
                workflows_parameters=workflows_parameters,
                execution_engine=execution_engine,
                image_input_name=image_input_name,
                video_metadata_input_name=video_metadata_input_name,
                serialize_results=serialize_results,
            )
        except ImportError as error:
            raise CannotInitialiseModelError(
                f"Could not initialise workflow processing due to lack of dependencies required. "
                f"Please provide an issue report under https://github.com/roboflow/inference/issues"
            ) from error
        on_pipeline_end_closure = partial(
            on_pipeline_end,
            thread_pool_executor=thread_pool_executor,
            cancel_thread_pool_tasks_on_exit=cancel_thread_pool_tasks_on_exit,
            profiler=profiler,
            profiling_directory=profiling_directory,
        )
        return cls.init_with_custom_logic(
            video_reference=video_reference,
            on_video_frame=on_video_frame,
            on_prediction=on_prediction,
            on_pipeline_start=None,
            on_pipeline_end=on_pipeline_end_closure,
            max_fps=max_fps,
            watchdog=watchdog,
            status_update_handlers=status_update_handlers,
            source_buffer_filling_strategy=source_buffer_filling_strategy,
            source_buffer_consumption_strategy=source_buffer_consumption_strategy,
            video_source_properties=video_source_properties,
            batch_collection_timeout=batch_collection_timeout,
            predictions_queue_size=predictions_queue_size,
            decoding_buffer_size=decoding_buffer_size,
        )

    @classmethod
    def init_with_custom_logic(
        cls,
        video_reference: Union[VideoSourceIdentifier, List[VideoSourceIdentifier]],
        on_video_frame: InferenceHandler,
        on_prediction: SinkHandler = None,
        on_pipeline_start: Optional[Callable[[], None]] = None,
        on_pipeline_end: Optional[Callable[[], None]] = None,
        max_fps: Optional[Union[float, int]] = None,
        watchdog: Optional[PipelineWatchDog] = None,
        status_update_handlers: Optional[List[Callable[[StatusUpdate], None]]] = None,
        source_buffer_filling_strategy: Optional[BufferFillingStrategy] = None,
        source_buffer_consumption_strategy: Optional[BufferConsumptionStrategy] = None,
        video_source_properties: Optional[Dict[str, float]] = None,
        batch_collection_timeout: Optional[float] = None,
        sink_mode: SinkMode = SinkMode.ADAPTIVE,
        predictions_queue_size: int = PREDICTIONS_QUEUE_SIZE,
        decoding_buffer_size: int = DEFAULT_BUFFER_SIZE,
    ) -> "InferencePipeline":
        """
        This class creates the abstraction for making inferences from given workflow against video stream.
        The way of how `InferencePipeline` works is displayed in `InferencePipeline.init(...)` initialiser
        method.

        Args:
            video_reference (Union[str, int, List[Union[str, int]]]): Reference of source or sources to be used to make
                predictions against. It can be video file path, stream URL and device (like camera) id
                (we handle whatever cv2 handles). It can also be a list of references (since v0.9.18) - and then
                it will trigger parallel processing of multiple sources. It has some implication on sinks. See:
                `sink_mode` parameter comments.
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
                dictate max FPS of each video source.
                The implementation details of this option has been changed in release `v0.26.0`. Prior to the release
                this value, when applied to video files caused the processing to wait `1 / max_fps` seconds before next
                frame is processed - the new implementation drops the intermediate frames, which seems to be more
                aligned with peoples expectations.
                New behaviour is now enabled in experimental mode, by setting environmental variable flag
                `ENABLE_FRAME_DROP_ON_VIDEO_FILE_RATE_LIMITING=True`. Please note that the new behaviour will
                be the default one end of Q4 2024!
            watchdog (Optional[PipelineWatchDog]): Implementation of class that allows profiling of
                inference pipeline - if not given null implementation (doing nothing) will be used.
            status_update_handlers (Optional[List[Callable[[StatusUpdate], None]]]): List of handlers to intercept
                status updates of all elements of the pipeline. Should be used only if detailed inspection of
                pipeline behaviour in time is needed. Please point out that handlers should be possible to be executed
                fast - otherwise they will impair pipeline performance. All errors will be logged as warnings
                without re-raising. Default: None.
            source_buffer_filling_strategy (Optional[BufferFillingStrategy]): Parameter dictating strategy for
                video stream decoding behaviour. By default - tweaked to the type of source given.
                Please find detailed explanation in docs of [`VideoSource`](/reference/inference/core/interfaces/camera/video_source/#inference.core.interfaces.camera.video_source.VideoSource)
            source_buffer_consumption_strategy (Optional[BufferConsumptionStrategy]): Parameter dictating strategy for
                video stream frames consumption. By default - tweaked to the type of source given.
                Please find detailed explanation in docs of [`VideoSource`](/reference/inference/core/interfaces/camera/video_source/#inference.core.interfaces.camera.video_source.VideoSource)
            video_source_properties (Optional[Union[Dict[str, float], List[Optional[Dict[str, float]]]]]):
                Optional source properties to set up the video source, corresponding to cv2 VideoCapture properties
                cv2.CAP_PROP_*. If not given, defaults for the video source will be used.
                It is optional and if provided can be provided as single dict (applicable for all sources) or
                as list of configs. Then the list must be of length of `video_reference` and may also contain None
                values to denote that specific source should remain not configured.
                Example valid properties are: {"frame_width": 1920, "frame_height": 1080, "fps": 30.0}
            batch_collection_timeout (Optional[float]): Parameter of multiplex_videos(...) dictating how long process
                to grab frames from multiple sources can wait for batch to be filled before yielding already collected
                frames. Please set this value in PRODUCTION to avoid performance drops when specific sources shows
                unstable latency. Visit `multiplex_videos(...)` for more information about multiplexing process.
            sink_mode (SinkMode): Parameter that controls how video frames and predictions will be passed to sink
                handler. With SinkMode.SEQUENTIAL - each frame and prediction triggers separate call for sink,
                in case of SinkMode.BATCH - list of frames and predictions will be provided to sink, always aligned
                in the order of video sources - with None values in the place of vide_frames / predictions that
                were skipped due to `batch_collection_timeout`.
                `SinkMode.ADAPTIVE` is a middle ground (and default mode) - all old sources will work in that mode
                against a single video input, as the pipeline will behave as if running in `SinkMode.SEQUENTIAL`.
                To handle multiple videos - sink needs to accept `predictions: List[Optional[dict]]` and
                `video_frame: List[Optional[VideoFrame]]`. It is also possible to process multiple videos using
                old sinks - but then `SinkMode.SEQUENTIAL` is to be used, causing sink to be called on each
                prediction element.
            predictions_queue_size int: Size of buffer for predictions that are ready for dispatching
                default value is taken from INFERENCE_PIPELINE_PREDICTIONS_QUEUE_SIZE env variable
            decoding_buffer_size (int): size of video source decoding buffer
                default value is taken from VIDEO_SOURCE_BUFFER_SIZE env variable

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
        desired_source_fps = None
        if ENABLE_FRAME_DROP_ON_VIDEO_FILE_RATE_LIMITING:
            desired_source_fps = max_fps
        video_sources = prepare_video_sources(
            video_reference=video_reference,
            video_source_properties=video_source_properties,
            status_update_handlers=status_update_handlers,
            source_buffer_filling_strategy=source_buffer_filling_strategy,
            source_buffer_consumption_strategy=source_buffer_consumption_strategy,
            desired_source_fps=desired_source_fps,
            decoding_buffer_size=decoding_buffer_size,
        )
        watchdog.register_video_sources(video_sources=video_sources)
        try:
            predictions_queue_size = int(predictions_queue_size)
        except ValueError:
            predictions_queue_size = 512
        predictions_queue = Queue(maxsize=predictions_queue_size)
        return cls(
            on_video_frame=on_video_frame,
            video_sources=video_sources,
            predictions_queue=predictions_queue,
            watchdog=watchdog,
            status_update_handlers=status_update_handlers,
            on_prediction=on_prediction,
            max_fps=max_fps,
            on_pipeline_start=on_pipeline_start,
            on_pipeline_end=on_pipeline_end,
            batch_collection_timeout=batch_collection_timeout,
            sink_mode=sink_mode,
        )

    def __init__(
        self,
        on_video_frame: InferenceHandler,
        video_sources: List[VideoSource],
        predictions_queue: Queue,
        watchdog: PipelineWatchDog,
        status_update_handlers: List[Callable[[StatusUpdate], None]],
        on_prediction: SinkHandler = None,
        on_pipeline_start: Optional[Callable[[], None]] = None,
        on_pipeline_end: Optional[Callable[[], None]] = None,
        max_fps: Optional[float] = None,
        batch_collection_timeout: Optional[float] = None,
        sink_mode: SinkMode = SinkMode.ADAPTIVE,
    ):
        self._on_video_frame = on_video_frame
        self._video_sources = video_sources
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
        self._batch_collection_timeout = batch_collection_timeout
        self._sink_mode = sink_mode

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
        for video_source in self._video_sources:
            video_source.terminate(
                wait_on_frames_consumption=False, purge_frames_buffer=True
            )

    def pause_stream(self, source_id: Optional[int] = None) -> None:
        for video_source in self._video_sources:
            if video_source.source_id == source_id or source_id is None:
                video_source.pause()

    def mute_stream(self, source_id: Optional[int] = None) -> None:
        for video_source in self._video_sources:
            if video_source.source_id == source_id or source_id is None:
                video_source.mute()

    def resume_stream(self, source_id: Optional[int] = None) -> None:
        for video_source in self._video_sources:
            if video_source.source_id == source_id or source_id is None:
                video_source.resume()

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
            for video_frames in self._generate_frames():
                self._watchdog.on_model_inference_started(
                    frames=video_frames,
                )
                predictions = self._on_video_frame(video_frames)
                self._watchdog.on_model_prediction_ready(
                    frames=video_frames,
                )
                self._predictions_queue.put((predictions, video_frames))
                send_inference_pipeline_status_update(
                    severity=UpdateSeverity.DEBUG,
                    event_type=INFERENCE_COMPLETED_EVENT,
                    payload={
                        "frames_ids": [f.frame_id for f in video_frames],
                        "frames_timestamps": [f.frame_timestamp for f in video_frames],
                        "sources_id": [f.source_id for f in video_frames],
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
            inference_results: Optional[
                Tuple[List[AnyPrediction], List[VideoFrame]]
            ] = self._predictions_queue.get()
            if inference_results is None:
                self._predictions_queue.task_done()
                break
            predictions, video_frames = inference_results
            if self._on_prediction is not None:
                self._handle_predictions_dispatching(
                    predictions=predictions,
                    video_frames=video_frames,
                )
            self._predictions_queue.task_done()

    def _handle_predictions_dispatching(
        self,
        predictions: List[AnyPrediction],
        video_frames: List[VideoFrame],
    ) -> None:
        if self._should_use_batch_sink():
            self._use_batch_sink(predictions, video_frames)
            return None
        for frame_predictions, video_frame in zip(predictions, video_frames):
            self._use_sink(frame_predictions, video_frame)

    def _should_use_batch_sink(self) -> bool:
        return self._sink_mode is SinkMode.BATCH or (
            self._sink_mode is SinkMode.ADAPTIVE and len(self._video_sources) > 1
        )

    def _use_batch_sink(
        self,
        predictions: List[AnyPrediction],
        video_frames: List[VideoFrame],
    ) -> None:
        # This function makes it possible to always call sinks with payloads aligned to order of
        # video sources - marking empty frames as None
        results_by_source_id = {
            video_frame.source_id: (frame_predictions, video_frame)
            for frame_predictions, video_frame in zip(predictions, video_frames)
        }
        source_id_aligned_sink_payload = [
            results_by_source_id.get(video_source.source_id, (None, None))
            for video_source in self._video_sources
        ]
        source_id_aligned_predictions = [e[0] for e in source_id_aligned_sink_payload]
        source_id_aligned_frames = [e[1] for e in source_id_aligned_sink_payload]
        self._use_sink(
            predictions=source_id_aligned_predictions,
            video_frames=source_id_aligned_frames,
        )

    def _use_sink(
        self,
        predictions: Union[AnyPrediction, List[Optional[AnyPrediction]]],
        video_frames: Union[VideoFrame, List[Optional[VideoFrame]]],
    ) -> None:
        try:
            self._on_prediction(predictions, video_frames)
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
            logger.exception(f"Error in results dispatching - {error}")

    def _generate_frames(
        self,
    ) -> Generator[List[VideoFrame], None, None]:
        for video_source in self._video_sources:
            video_source.start()
        max_fps = None
        if not ENABLE_FRAME_DROP_ON_VIDEO_FILE_RATE_LIMITING:
            max_fps = self._max_fps
        yield from multiplex_videos(
            videos=self._video_sources,
            max_fps=max_fps,
            batch_collection_timeout=self._batch_collection_timeout,
            should_stop=lambda: self._stop,
        )


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

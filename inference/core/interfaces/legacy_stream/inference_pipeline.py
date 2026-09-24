"""The `inference` flavour of `InferencePipeline`: models, model managers and the platform.

`InferencePipeline` here subclasses the host-neutral pipeline
(`inference.core.interfaces.stream.pipeline`) and keeps the historical public
entry points - `init(model_id=...)`, `init_with_yolo_world(...)` and
`init_with_workflow(model_manager=...)` - with their exact signatures and
behaviour: model loading, Active Learning, API-key fallback, workflow
definition fetching and the default model manager all happen here, never in
the neutral pipeline.

Every factory constructs `cls`, so a pipeline created through this class is
an instance of both this class and the neutral one; a pipeline created
through the neutral class is not an instance of this one.

`inference.core.interfaces.stream.inference_pipeline` is this very module
(see that facade), so module attributes patched through the historical name
- `get_model`, `API_KEY`, `ACTIVE_LEARNING_ENABLED`, `ENABLE_WORKFLOWS_PROFILING`,
`BaseWorkflowsProfiler` and so on - are the ones read here. The composition
points the neutral pipeline reads through `cls` (`prepare_video_sources`,
`ENABLE_FRAME_DROP_ON_VIDEO_FILE_RATE_LIMITING`,
`ENABLE_TENSOR_DATA_REPRESENTATION`) are forwarded from this module too.
"""

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from functools import partial
from queue import Queue
from threading import Thread
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

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
    ENABLE_TENSOR_DATA_REPRESENTATION,
    ENABLE_WORKFLOWS_PROFILING,
    MAX_ACTIVE_MODELS,
    PREDICTIONS_QUEUE_SIZE,
    WORKFLOWS_PROFILER_BUFFER_SIZE,
)
from inference.core.exceptions import CannotInitialiseModelError, MissingApiKeyError
from inference.core.interfaces.camera.collection_policy import (
    FRESHEST_MODE_BATCH_COLLECTION_TIMEOUT,
    STALENESS_DROP_CAUSE,
    CollectionPolicy,
    VideoProcessingMode,
    resolve_video_processing_mode,
)
from inference.core.interfaces.camera.entities import (
    StatusUpdate,
    UpdateSeverity,
    VideoFrame,
    VideoSourceIdentifier,
)
from inference.core.interfaces.camera.utils import multiplex_videos
from inference.core.interfaces.camera.video_source import (
    FRAME_DROPPED_EVENT,
    BufferConsumptionStrategy,
    BufferFillingStrategy,
    VideoSource,
)
from inference.core.interfaces.legacy_stream.model_handlers.roboflow_models import (
    default_process_frame,
)
from inference.core.interfaces.roboflow_platform_client import (
    install_workflows_platform_bindings,
)
from inference.core.interfaces.stream.entities import (
    AnyPrediction,
    InferenceHandler,
    InferenceHandlerResult,
    ModelConfig,
    SinkHandler,
)
from inference.core.interfaces.stream.pipeline import (
    INFERENCE_COMPLETED_EVENT,
    INFERENCE_ERROR_EVENT,
    INFERENCE_PIPELINE_CONTEXT,
    INFERENCE_RESULTS_DISPATCHING_ERROR_EVENT,
    INFERENCE_THREAD_FINISHED_EVENT,
    INFERENCE_THREAD_STARTED_EVENT,
    SOURCE_CONNECTION_ATTEMPT_FAILED_EVENT,
    SOURCE_CONNECTION_LOST_EVENT,
)
from inference.core.interfaces.stream.pipeline import (
    InferencePipeline as HostNeutralInferencePipeline,
)
from inference.core.interfaces.stream.pipeline import (
    SinkMode,
    _resolve_prediction_futures,
    _rfdetr_stream_pipeline_enabled,
    send_inference_pipeline_status_update,
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
from inference.core.interfaces.workflows_models_provider import (
    ModelManagerModelsProvider,
    bind_model_manager_to_workflows,
)
from inference.core.managers.active_learning import BackgroundTaskActiveLearningManager
from inference.core.managers.base import ModelManager
from inference.core.managers.decorators.fixed_size_cache import WithFixedSizeCache
from inference.core.registries.roboflow import RoboflowModelRegistry
from inference.core.utils.function import experimental
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.profiling.core import (
    BaseWorkflowsProfiler,
    NullWorkflowsProfiler,
    WorkflowsProfiler,
)
from inference.core.workflows.execution_engine.v1.executor.utils import resolve_futures
from inference.models.aliases import resolve_roboflow_model_alias
from inference.models.utils import ROBOFLOW_MODEL_TYPES, get_model
from inference.usage_tracking.stream_session import (
    mint_stream_session_id,
    stream_session_id,
)

# The historical logger name: log configuration keyed on it keeps matching the
# records this module emits (the pipeline runtime itself logs through
# `inference.core.interfaces.stream.pipeline`).
logger = logging.getLogger("inference.core.interfaces.stream.inference_pipeline")

PREDICTIONS_QUEUE_SIZE_ENV = "INFERENCE_PIPELINE_PREDICTIONS_QUEUE_SIZE"


@dataclass(frozen=True)
class PreparedWorkflow:
    """Everything the host-neutral workflow constructor needs from `inference`.

    Attributes:
        workflow_specification: The inline or fetched workflow definition.
        workflow_init_parameters: Execution Engine init parameters, with the
            models provider, API key, execution observer and platform, codec
            and configuration bindings in place.
        step_error_handler: The effective step error handler.
    """

    workflow_specification: dict
    workflow_init_parameters: Dict[str, Any]
    step_error_handler: Any


def prepare_workflow_for_pipeline(
    workflow_specification: Optional[dict],
    workspace_name: Optional[str],
    workflow_id: Optional[str],
    workflow_version_id: Optional[str],
    api_key: Optional[str],
    use_workflow_definition_cache: bool,
    workflow_init_parameters: Optional[Dict[str, Any]],
    model_manager: Optional[ModelManager],
    profiler: WorkflowsProfiler,
) -> PreparedWorkflow:
    """Resolve a workflow and its Execution Engine bindings the way `inference` does.

    Shared by `InferencePipeline.init_with_workflow` and the stream manager's
    legacy host so both keep the same precedence:

    * a missing `api_key` falls back to the `API_KEY` environment setting;
    * a named workflow is fetched from the Roboflow API (inside the
      `workflow_definition_fetching` phase of `profiler`) unless an inline
      specification is given;
    * `model_manager` is used as-is when supplied; otherwise the default
      `WithFixedSizeCache(BackgroundTaskActiveLearningManager)` stack is built;
    * `workflow_init_parameters` is updated in place (a new dict when `None`):
      the namespaced API key and usage-tracking execution observer are always
      overwritten, then `bind_model_manager_to_workflows` always installs a
      fresh `ModelManagerModelsProvider` around `model_manager` and fills the
      platform, codec and configuration bindings only where the caller left
      them unset. The binder is called explicitly - `model_manager` needs no
      `__workflows_bind__` hook.

    Args:
        workflow_specification: Inline workflow definition, if any.
        workspace_name: Workspace of a registered workflow.
        workflow_id: Identifier of a registered workflow.
        workflow_version_id: Version of a registered workflow.
        api_key: Roboflow API key; `API_KEY` is used when `None`.
        use_workflow_definition_cache: Whether a fetched definition may come
            from cache.
        workflow_init_parameters: Caller-supplied Execution Engine init
            parameters.
        model_manager: Model manager the workflow blocks run models through.
        profiler: Profiler recording the definition fetch.

    Returns:
        The specification, init parameters and effective step error handler.

    Raises:
        ValueError: Neither an inline specification nor a workspace name and
            workflow id are given.
        MissingApiKeyError: A registered workflow must be fetched without an
            API key.
        CannotInitialiseModelError: A dependency of workflow processing cannot
            be imported.
    """
    if api_key is None:
        api_key = API_KEY
    named_workflow_specified = (workspace_name is not None) and (
        workflow_id is not None
    )
    if not named_workflow_specified and not workflow_specification:
        raise ValueError(
            "Either (`workspace_name`, `workflow_id`) or `workflow_specification` must be provided."
        )
    try:
        from inference.core.interfaces.workflows_execution_observer import (
            UsageTrackingExecutionObserver,
        )
        from inference.core.interfaces.workflows_step_error_handlers import (
            resolve_step_error_handler,
        )
        from inference.core.roboflow_api import get_workflow_specification

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
                    workflow_version_id=workflow_version_id,
                    use_cache=use_workflow_definition_cache,
                )
        if model_manager is None:
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
        workflow_init_parameters["workflows_core.api_key"] = api_key
        workflow_init_parameters["workflows_core.execution_observer"] = (
            UsageTrackingExecutionObserver()
        )
        # Installs the provider, then the platform, codec and configuration
        # bindings the caller left unset. A caller-supplied configuration is
        # validated before the process-wide codec is touched, and still
        # reaches `ExecutionEngine.init` unchanged.
        step_error_handler = bind_model_manager_to_workflows(
            model_manager=model_manager,
            init_parameters=workflow_init_parameters,
            step_error_handler=resolve_step_error_handler(),
        )
    except ImportError as error:
        raise CannotInitialiseModelError(
            f"Could not initialise workflow processing due to lack of dependencies required. "
            f"Please provide an issue report under https://github.com/roboflow/inference/issues"
        ) from error
    return PreparedWorkflow(
        workflow_specification=workflow_specification,
        workflow_init_parameters=workflow_init_parameters,
        step_error_handler=step_error_handler,
    )


class InferencePipeline(HostNeutralInferencePipeline):
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
        video_processing_mode: Optional[Union[str, VideoProcessingMode]] = None,
        max_staleness: Optional[float] = None,
        sink_mode: SinkMode = SinkMode.ADAPTIVE,
        predictions_queue_size: int = PREDICTIONS_QUEUE_SIZE,
        decoding_buffer_size: int = DEFAULT_BUFFER_SIZE,
        exec_session_id: Optional[str] = None,
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
            video_processing_mode (Optional[Union[str, VideoProcessingMode]]): High-level intent for live
                multi-source consumption: "auto" (FIFO with a staleness budget and self-tuning collection
                window), "every_frame" (strict FIFO) or "freshest" (legacy latest-wins with a small fixed
                collection timeout). Defaults to "auto" when ENABLE_TENSOR_DATA_REPRESENTATION is set,
                otherwise the legacy collection behavior is preserved unchanged; pass "legacy" to force
                the legacy behavior explicitly (the escape hatch from the flag-driven default). File
                sources always keep every-frame semantics regardless of mode. See
                `inference.core.interfaces.camera.collection_policy` for details.
            max_staleness (Optional[float]): Staleness budget (seconds) of "auto" mode - live frames older
                than this are dropped (reported as FRAME_DROPPED status updates with cause
                STALENESS_BUDGET_EXCEEDED) instead of being served late. Default: 0.5.
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
            exec_session_id (Optional[str]): Usage session identifier for this pipeline. If empty or omitted,
                a unique identifier is generated for the pipeline.

        Other ENV variables involved in low-level configuration:
        * INFERENCE_PIPELINE_PREDICTIONS_QUEUE_SIZE - size of buffer for predictions that are ready for dispatching
        * INFERENCE_PIPELINE_RESTART_ATTEMPT_DELAY - delay for restarts on stream connection drop
        * ACTIVE_LEARNING_ENABLED - controls Active Learning middleware if explicit parameter not given

        Returns: Instance of InferencePipeline.

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
            default_process_frame,
            model=model,
            inference_config=inference_config,
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
            video_processing_mode=video_processing_mode,
            max_staleness=max_staleness,
            sink_mode=sink_mode,
            predictions_queue_size=predictions_queue_size,
            decoding_buffer_size=decoding_buffer_size,
            exec_session_id=exec_session_id,
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
        video_processing_mode: Optional[Union[str, VideoProcessingMode]] = None,
        max_staleness: Optional[float] = None,
        sink_mode: SinkMode = SinkMode.ADAPTIVE,
        predictions_queue_size: int = PREDICTIONS_QUEUE_SIZE,
        decoding_buffer_size: int = DEFAULT_BUFFER_SIZE,
        exec_session_id: Optional[str] = None,
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
            video_processing_mode (Optional[Union[str, VideoProcessingMode]]): High-level intent for live
                multi-source consumption: "auto" (FIFO with a staleness budget and self-tuning collection
                window), "every_frame" (strict FIFO) or "freshest" (legacy latest-wins with a small fixed
                collection timeout). Defaults to "auto" when ENABLE_TENSOR_DATA_REPRESENTATION is set,
                otherwise the legacy collection behavior is preserved unchanged; pass "legacy" to force
                the legacy behavior explicitly (the escape hatch from the flag-driven default). File
                sources always keep every-frame semantics regardless of mode. See
                `inference.core.interfaces.camera.collection_policy` for details.
            max_staleness (Optional[float]): Staleness budget (seconds) of "auto" mode - live frames older
                than this are dropped (reported as FRAME_DROPPED status updates with cause
                STALENESS_BUDGET_EXCEEDED) instead of being served late. Default: 0.5.
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
            exec_session_id (Optional[str]): Usage session identifier for this pipeline. If empty or omitted,
                a unique identifier is generated for the pipeline.

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
            from inference.core.interfaces.legacy_stream.model_handlers.yolo_world import (
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
            video_processing_mode=video_processing_mode,
            max_staleness=max_staleness,
            sink_mode=sink_mode,
            predictions_queue_size=predictions_queue_size,
            decoding_buffer_size=decoding_buffer_size,
            exec_session_id=exec_session_id,
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
        disable_sinks: bool = False,
        workflows_thread_pool_workers: int = 4,
        execution_engine_thread_pool_workers: int = 4,
        cancel_thread_pool_tasks_on_exit: bool = True,
        video_metadata_input_name: str = "video_metadata",
        batch_collection_timeout: Optional[float] = None,
        video_processing_mode: Optional[Union[str, VideoProcessingMode]] = None,
        max_staleness: Optional[float] = None,
        profiling_directory: str = "./inference_profiling",
        use_workflow_definition_cache: bool = True,
        serialize_results: bool = False,
        predictions_queue_size: int = PREDICTIONS_QUEUE_SIZE,
        decoding_buffer_size: int = DEFAULT_BUFFER_SIZE,
        model_manager: Optional[ModelManager] = None,
        _is_preview: bool = False,
        workflow_version_id: Optional[str] = None,
        exec_session_id: Optional[str] = None,
        workflows_dependencies_pre_init: Optional[List[str]] = None,
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
            disable_sinks (bool): Whether to disable sink writes and outbound notifications/uploads.
            workflows_thread_pool_workers (int): Number of workers for workflows thread pool which is used
                by workflows blocks and sinks to run background tasks (fire-and-forget dispatch of
                notifications, uploads and other side effects).
            execution_engine_thread_pool_workers (int): Number of workers for the thread pool used
                exclusively by the workflows Execution Engine to run workflow steps. Kept separate
                from `workflows_thread_pool_workers` so that slow background sink tasks cannot
                starve step execution.
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
            video_processing_mode (Optional[Union[str, VideoProcessingMode]]): High-level intent for live
                multi-source consumption: "auto" (FIFO with a staleness budget and self-tuning collection
                window), "every_frame" (strict FIFO) or "freshest" (legacy latest-wins with a small fixed
                collection timeout). Defaults to "auto" when ENABLE_TENSOR_DATA_REPRESENTATION is set,
                otherwise the legacy collection behavior is preserved unchanged; pass "legacy" to force
                the legacy behavior explicitly (the escape hatch from the flag-driven default). File
                sources always keep every-frame semantics regardless of mode. See
                `inference.core.interfaces.camera.collection_policy` for details.
            max_staleness (Optional[float]): Staleness budget (seconds) of "auto" mode - live frames older
                than this are dropped (reported as FRAME_DROPPED status updates with cause
                STALENESS_BUDGET_EXCEEDED) instead of being served late. Default: 0.5.
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
            model_manager (Optional[ModelManager]): Model manager to be used by InferencePipeline, defaults to
                BackgroundTaskActiveLearningManager with WithFixedSizeCache
            exec_session_id (Optional[str]): Usage session identifier for this pipeline. If empty or omitted,
                a unique identifier is generated for the pipeline.
            workflows_dependencies_pre_init (Optional[List[str]]): Opt-in pre-loading of dependent
                resources declared by workflow blocks (`discover_dependent_resources()`). Pass a list
                of dependent-resource type names to pre-load — `"roboflow_platform_model"` is the only
                supported value for now. When enabled, Roboflow models declared with concrete ids are
                registered in the model manager at pipeline init (weights fetched upfront, giving
                predictable startup instead of lazy loading on the first frame); model ids fed from
                workflow inputs are resolved and registered on the first frame. Pre-loading honours
                the effective step execution mode — nothing is fetched when steps execute remotely.
                Defaults to None — no pre-loading.

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
        # Built from this module's globals, as before the split, so the
        # profiler classes and settings patched through the historical module
        # name steer the one profiler that records the definition fetch and
        # every Execution Engine run.
        if ENABLE_WORKFLOWS_PROFILING:
            profiler = BaseWorkflowsProfiler.init(
                max_runs_in_buffer=WORKFLOWS_PROFILER_BUFFER_SIZE,
            )
        else:
            profiler = NullWorkflowsProfiler.init()
        prepared_workflow = prepare_workflow_for_pipeline(
            workflow_specification=workflow_specification,
            workspace_name=workspace_name,
            workflow_id=workflow_id,
            workflow_version_id=workflow_version_id,
            api_key=api_key,
            use_workflow_definition_cache=use_workflow_definition_cache,
            workflow_init_parameters=workflow_init_parameters,
            model_manager=model_manager,
            profiler=profiler,
        )
        return super().init_with_workflow(
            video_reference=video_reference,
            workflow_specification=prepared_workflow.workflow_specification,
            workflow_init_parameters=prepared_workflow.workflow_init_parameters,
            step_error_handler=prepared_workflow.step_error_handler,
            workflow_id=workflow_id,
            image_input_name=image_input_name,
            workflows_parameters=workflows_parameters,
            on_prediction=on_prediction,
            max_fps=max_fps,
            watchdog=watchdog,
            status_update_handlers=status_update_handlers,
            source_buffer_filling_strategy=source_buffer_filling_strategy,
            source_buffer_consumption_strategy=source_buffer_consumption_strategy,
            video_source_properties=video_source_properties,
            disable_sinks=disable_sinks,
            workflows_thread_pool_workers=workflows_thread_pool_workers,
            execution_engine_thread_pool_workers=execution_engine_thread_pool_workers,
            cancel_thread_pool_tasks_on_exit=cancel_thread_pool_tasks_on_exit,
            video_metadata_input_name=video_metadata_input_name,
            batch_collection_timeout=batch_collection_timeout,
            video_processing_mode=video_processing_mode,
            max_staleness=max_staleness,
            profiling_directory=profiling_directory,
            serialize_results=serialize_results,
            predictions_queue_size=predictions_queue_size,
            decoding_buffer_size=decoding_buffer_size,
            _is_preview=_is_preview,
            exec_session_id=exec_session_id,
            workflows_dependencies_pre_init=workflows_dependencies_pre_init,
            profiler=profiler,
        )

    @classmethod
    def _predictions_queue_size_set_explicitly(cls) -> bool:
        # Checked when each pipeline is built, as it always was - not frozen
        # when `inference.core` configured the stream runtime.
        return PREDICTIONS_QUEUE_SIZE_ENV in os.environ

    # The neutral pipeline looks these up through `cls`; reading this
    # module's globals keeps patches made through the historical module name
    # effective for pipelines built by this class.
    @classmethod
    def _frame_drop_on_video_file_rate_limiting_enabled(cls) -> bool:
        return ENABLE_FRAME_DROP_ON_VIDEO_FILE_RATE_LIMITING

    @classmethod
    def _tensor_frames_enabled(cls) -> bool:
        return ENABLE_TENSOR_DATA_REPRESENTATION

    @classmethod
    def _prepare_video_sources(cls, **kwargs) -> List[VideoSource]:
        return prepare_video_sources(**kwargs)

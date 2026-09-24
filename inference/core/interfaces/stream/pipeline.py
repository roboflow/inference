"""The host-neutral `InferencePipeline`: frame collection, inference and dispatch.

This class knows nothing about models, model managers, API keys or the Roboflow
platform. Its workflow constructor receives an already-resolved workflow
specification, the Execution Engine init parameters (including the models
provider) and the effective step error handler; the host that supplies them
decides how they are fetched and bound. The legacy `inference` entry points
(`InferencePipeline.init(model_id=...)`, `init_with_workflow(model_manager=...)`)
live in `inference.core.interfaces.legacy_stream.inference_pipeline`, a subclass
of this class.

Settings come from the installed `StreamsConfiguration`
(`inference.core.interfaces.stream.environment`), never from `os.environ` -
except `RFDETR_PIPELINE_DEPTH`, which is deliberately read when it is used.
"""

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from enum import Enum
from functools import partial
from queue import Queue
from threading import Event, Thread, current_thread
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

from roboflow_workflows.execution_engine.profiling.core import (
    BaseWorkflowsProfiler,
    NullWorkflowsProfiler,
    WorkflowsProfiler,
)
from roboflow_workflows.execution_engine.v1.executor.utils import resolve_futures

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
from inference.core.interfaces.stream.entities import (
    AnyPrediction,
    InferenceHandler,
    InferenceHandlerResult,
    SinkHandler,
)
from inference.core.interfaces.stream.environment import (
    DEFAULT_BUFFER_SIZE,
    ENABLE_FRAME_DROP_ON_VIDEO_FILE_RATE_LIMITING,
    ENABLE_TENSOR_DATA_REPRESENTATION,
    ENABLE_WORKFLOWS_PROFILING,
    PREDICTIONS_QUEUE_SIZE,
    PREDICTIONS_QUEUE_SIZE_EXPLICIT,
    WORKFLOWS_PROFILER_BUFFER_SIZE,
)
from inference.core.interfaces.stream.exceptions import CannotInitialiseModelError
from inference.core.interfaces.stream.session import (
    mint_stream_session_id,
    stream_session_id,
)
from inference.core.interfaces.stream.utils import (
    on_pipeline_end,
    prepare_video_sources,
)
from inference.core.interfaces.stream.watchdog import (
    NullPipelineWatchdog,
    PipelineWatchDog,
)

logger = logging.getLogger(__name__)

INFERENCE_PIPELINE_CONTEXT = "inference_pipeline"
SOURCE_CONNECTION_ATTEMPT_FAILED_EVENT = "SOURCE_CONNECTION_ATTEMPT_FAILED"
SOURCE_CONNECTION_LOST_EVENT = "SOURCE_CONNECTION_LOST"
INFERENCE_RESULTS_DISPATCHING_ERROR_EVENT = "INFERENCE_RESULTS_DISPATCHING_ERROR"
INFERENCE_THREAD_STARTED_EVENT = "INFERENCE_THREAD_STARTED"
INFERENCE_THREAD_FINISHED_EVENT = "INFERENCE_THREAD_FINISHED"
INFERENCE_COMPLETED_EVENT = "INFERENCE_COMPLETED"
INFERENCE_ERROR_EVENT = "INFERENCE_ERROR"


def build_workflows_profiler(
    enabled: bool,
    max_runs_in_buffer: int,
) -> WorkflowsProfiler:
    """Create the profiler a workflow pipeline records its Execution Engine runs in.

    Hosts that do work of their own before the pipeline exists (fetching the
    workflow definition, for example) create the profiler with this function,
    profile that work with it and pass the same instance to
    `InferencePipeline.init_with_workflow(profiler=...)`, so a single trace
    covers both.

    Args:
        enabled: Whether to record anything; a null profiler is returned if not.
        max_runs_in_buffer: Number of most recent workflow runs kept in the trace.

    Returns:
        A `BaseWorkflowsProfiler` when enabled, a `NullWorkflowsProfiler` otherwise.
    """
    if enabled:
        return BaseWorkflowsProfiler.init(max_runs_in_buffer=max_runs_in_buffer)
    return NullWorkflowsProfiler.init()


class SinkMode(Enum):
    ADAPTIVE = "adaptive"
    BATCH = "batch"
    SEQUENTIAL = "sequential"


class InferencePipeline:
    @classmethod
    def init_with_workflow(
        cls,
        video_reference: Union[str, int, List[Union[str, int]]],
        *,
        workflow_specification: dict,
        workflow_init_parameters: Dict[str, Any],
        step_error_handler: Any,
        workflow_id: Optional[str] = None,
        image_input_name: str = "image",
        workflows_parameters: Optional[Dict[str, Any]] = None,
        on_prediction: SinkHandler = None,
        max_fps: Optional[Union[float, int]] = None,
        watchdog: Optional[PipelineWatchDog] = None,
        status_update_handlers: Optional[List[Callable[[StatusUpdate], None]]] = None,
        source_buffer_filling_strategy: Optional[BufferFillingStrategy] = None,
        source_buffer_consumption_strategy: Optional[BufferConsumptionStrategy] = None,
        video_source_properties: Optional[Dict[str, float]] = None,
        disable_sinks: bool = False,
        workflows_thread_pool_workers: int = 4,
        execution_engine_thread_pool_workers: int = 4,
        cancel_thread_pool_tasks_on_exit: bool = True,
        video_metadata_input_name: str = "video_metadata",
        batch_collection_timeout: Optional[float] = None,
        video_processing_mode: Optional[Union[str, VideoProcessingMode]] = None,
        max_staleness: Optional[float] = None,
        profiling_directory: str = "./inference_profiling",
        serialize_results: bool = False,
        predictions_queue_size: int = PREDICTIONS_QUEUE_SIZE,
        decoding_buffer_size: int = DEFAULT_BUFFER_SIZE,
        _is_preview: bool = False,
        exec_session_id: Optional[str] = None,
        workflows_dependencies_pre_init: Optional[List[str]] = None,
        profiler: Optional[WorkflowsProfiler] = None,
    ) -> "InferencePipeline":
        """Create a pipeline running an already-resolved workflow against video.

        The host resolves everything platform-specific first: it fetches the
        workflow definition, builds the Execution Engine init parameters
        (models provider, API key, execution observer and the other
        `workflows_core.*` services) and picks the step error handler. This
        method adds only what the pipeline owns - its two thread pools, the
        sink policy and the profiler - and then builds the pipeline exactly as
        `init_with_custom_logic` does.

        Args:
            video_reference: Reference of the source or sources to process.
            workflow_specification: The resolved workflow definition.
            workflow_init_parameters: Execution Engine init parameters built by
                the host. The pipeline sets `workflows_core.thread_pool_executor`
                and `workflows_core.disable_sinks` in this same dict and passes
                it, together with every value the host placed in it (the models
                provider object included), to `ExecutionEngine.init` unchanged.
            step_error_handler: The effective step error handler, passed to
                `ExecutionEngine.init` as-is.
            workflow_id: Identifier of a registered workflow, used by the
                Execution Engine for reporting.
            image_input_name: Workflow input receiving the video frames.
            workflows_parameters: Additional workflow input values.
            on_prediction: Sink receiving the workflow outputs and frames.
            max_fps: Maximum FPS of each video source.
            watchdog: Pipeline watchdog; a null implementation when omitted.
            status_update_handlers: Handlers of pipeline status updates.
            source_buffer_filling_strategy: Decoding buffer filling strategy.
            source_buffer_consumption_strategy: Decoding buffer consumption
                strategy.
            video_source_properties: cv2 capture properties of the sources.
            disable_sinks: Whether to disable sink writes and outbound
                notifications/uploads.
            workflows_thread_pool_workers: Workers of the pool the workflow
                blocks and sinks use for background tasks.
            execution_engine_thread_pool_workers: Workers of the separate pool
                the Execution Engine runs workflow steps in.
            cancel_thread_pool_tasks_on_exit: Whether to cancel background
                tasks that have not started when the pipeline ends.
            video_metadata_input_name: Workflow input receiving frame metadata.
            batch_collection_timeout: Multi-source batch collection timeout.
            video_processing_mode: Multi-source consumption mode.
            max_staleness: Staleness budget of the "auto" processing mode.
            profiling_directory: Directory the profiler trace is saved in.
            serialize_results: Whether to serialize each frame's workflow results.
            predictions_queue_size: Size of the buffer of results awaiting dispatch.
            decoding_buffer_size: Size of the video source decoding buffer.
            _is_preview: Whether the workflow runs as a preview.
            exec_session_id: Usage session identifier; minted when empty.
            workflows_dependencies_pre_init: Dependent-resource types the
                Execution Engine pre-loads at init.
            profiler: Profiler to record the workflow runs in, typically the one
                the host already profiled its own preparation with. Built from
                the stream configuration when omitted.

        Returns:
            Instance of the class this method is called on.

        Raises:
            CannotInitialiseModelError: A dependency of workflow processing
                cannot be imported.
        """
        if profiler is None:
            profiler = build_workflows_profiler(
                enabled=ENABLE_WORKFLOWS_PROFILING,
                max_runs_in_buffer=WORKFLOWS_PROFILER_BUFFER_SIZE,
            )
        try:
            from roboflow_workflows.execution_engine.core import ExecutionEngine

            from inference.core.interfaces.stream.model_handlers.workflows import (
                WorkflowRunner,
                wrap_workflow_runner_for_stream_pipeline,
            )

            thread_pool_executor = ThreadPoolExecutor(
                max_workers=workflows_thread_pool_workers
            )
            # Deliberately a separate pool: sharing one executor between
            # fire-and-forget sink tasks and step execution lets slow sinks
            # block the whole pipeline.
            execution_engine_thread_pool_executor = ThreadPoolExecutor(
                max_workers=execution_engine_thread_pool_workers
            )
            workflow_init_parameters["workflows_core.thread_pool_executor"] = (
                thread_pool_executor
            )
            workflow_init_parameters["workflows_core.disable_sinks"] = disable_sinks
            execution_engine = ExecutionEngine.init(
                workflow_definition=workflow_specification,
                init_parameters=workflow_init_parameters,
                workflow_id=workflow_id,
                profiler=profiler,
                executor=execution_engine_thread_pool_executor,
                dependencies_pre_init=workflows_dependencies_pre_init,
                step_error_handler=step_error_handler,
            )
            workflow_runner = WorkflowRunner(
                workflows_parameters=workflows_parameters,
                execution_engine=execution_engine,
                image_input_name=image_input_name,
                video_metadata_input_name=video_metadata_input_name,
                serialize_results=serialize_results,
                _is_preview=_is_preview,
            )
            on_video_frame = wrap_workflow_runner_for_stream_pipeline(
                workflow_runner=workflow_runner,
                execution_engine=execution_engine,
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
            execution_engine_thread_pool_executor=execution_engine_thread_pool_executor,
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
            video_processing_mode=video_processing_mode,
            max_staleness=max_staleness,
            predictions_queue_size=predictions_queue_size,
            decoding_buffer_size=decoding_buffer_size,
            allow_tensor_frames=cls._tensor_frames_enabled(),
            exec_session_id=exec_session_id,
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
        video_processing_mode: Optional[Union[str, VideoProcessingMode]] = None,
        max_staleness: Optional[float] = None,
        sink_mode: SinkMode = SinkMode.ADAPTIVE,
        predictions_queue_size: int = PREDICTIONS_QUEUE_SIZE,
        decoding_buffer_size: int = DEFAULT_BUFFER_SIZE,
        exec_session_id: Optional[str] = None,
        allow_tensor_frames: bool = False,
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
        if watchdog is None:
            watchdog = NullPipelineWatchdog()
        status_update_handlers = list(status_update_handlers or [])
        status_update_handlers.append(watchdog.on_status_update)
        resolved_processing_mode = resolve_video_processing_mode(
            explicit_mode=video_processing_mode
        )
        collection_policy = None
        if resolved_processing_mode is VideoProcessingMode.FRESHEST:
            if source_buffer_consumption_strategy is None:
                source_buffer_consumption_strategy = BufferConsumptionStrategy.EAGER
            if batch_collection_timeout is None:
                batch_collection_timeout = FRESHEST_MODE_BATCH_COLLECTION_TIMEOUT
        elif resolved_processing_mode is not None:
            if source_buffer_consumption_strategy is None:
                source_buffer_consumption_strategy = BufferConsumptionStrategy.LAZY

            def _report_stale_frame_dropped(frame: VideoFrame) -> None:
                send_inference_pipeline_status_update(
                    severity=UpdateSeverity.DEBUG,
                    event_type=FRAME_DROPPED_EVENT,
                    status_update_handlers=status_update_handlers,
                    payload={
                        "source_id": frame.source_id,
                        "frame_id": frame.frame_id,
                        "cause": STALENESS_DROP_CAUSE,
                    },
                )

            collection_policy = CollectionPolicy(
                mode=resolved_processing_mode,
                max_staleness=max_staleness,
                on_frame_dropped=_report_stale_frame_dropped,
            )
        desired_source_fps = None
        if cls._frame_drop_on_video_file_rate_limiting_enabled():
            desired_source_fps = max_fps
        video_sources = cls._prepare_video_sources(
            video_reference=video_reference,
            video_source_properties=video_source_properties,
            status_update_handlers=status_update_handlers,
            source_buffer_filling_strategy=source_buffer_filling_strategy,
            source_buffer_consumption_strategy=source_buffer_consumption_strategy,
            desired_source_fps=desired_source_fps,
            decoding_buffer_size=decoding_buffer_size,
            allow_tensor_frames=allow_tensor_frames,
        )
        watchdog.register_video_sources(video_sources=video_sources)
        try:
            predictions_queue_size = int(predictions_queue_size)
        except ValueError:
            predictions_queue_size = 512
        if (
            _rfdetr_stream_pipeline_enabled()
            and not cls._predictions_queue_size_set_explicitly()
        ):
            # Stream-pipelined RF-DETR returns async response futures. Letting
            # the producer queue hundreds of full-resolution VideoFrame objects
            # can exhaust host memory on 4K videos before dispatch catches up.
            predictions_queue_size = min(predictions_queue_size, 4)
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
            collection_policy=collection_policy,
            exec_session_id=exec_session_id,
        )

    @classmethod
    def _predictions_queue_size_set_explicitly(cls) -> bool:
        # The stream-pipelined RF-DETR cap applies only to an omitted size,
        # never to an explicit one - even one equal to the default - so the
        # configuration records presence separately from the resolved value.
        return PREDICTIONS_QUEUE_SIZE_EXPLICIT

    # Like the queue-size check above, the settings and the source factory
    # below are looked up through `cls`, so a host subclass can read them from
    # its own module - the legacy `inference` pipeline reads the ones patched
    # through its historical module name.
    @classmethod
    def _frame_drop_on_video_file_rate_limiting_enabled(cls) -> bool:
        return ENABLE_FRAME_DROP_ON_VIDEO_FILE_RATE_LIMITING

    @classmethod
    def _tensor_frames_enabled(cls) -> bool:
        return ENABLE_TENSOR_DATA_REPRESENTATION

    @classmethod
    def _prepare_video_sources(cls, **kwargs) -> List[VideoSource]:
        return prepare_video_sources(**kwargs)

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
        collection_policy: Optional[CollectionPolicy] = None,
        exec_session_id: Optional[str] = None,
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
        self._stream_session_id = exec_session_id or mint_stream_session_id()
        self._collection_policy = collection_policy
        # The inference thread starts the sources; `terminate()` stops only
        # those it did start, once it has finished starting them.
        self._sources_startup_finished = Event()
        self._started_sources: List[VideoSource] = []
        # Set while the inference thread runs without anything consuming its
        # results - between its start and the dispatcher's - so that `join()`
        # discards them if the dispatcher never came to be.
        self._results_consumer_missing = False

    def start(self, use_main_thread: bool = True) -> None:
        self._stop = False
        self._sources_startup_finished = Event()
        self._started_sources = []
        self._inference_thread = Thread(target=self._execute_inference)
        try:
            self._inference_thread.start()
        except Exception:
            # A thread that never started cannot be joined.
            self._inference_thread = None
            raise
        self._results_consumer_missing = True
        if self._on_pipeline_start is not None:
            self._on_pipeline_start()
        if use_main_thread:
            self._results_consumer_missing = False
            self._dispatch_inference_results()
        else:
            self._dispatching_thread = Thread(target=self._dispatch_inference_results)
            try:
                self._dispatching_thread.start()
            except Exception:
                self._dispatching_thread = None
                raise
            self._results_consumer_missing = False

    def terminate(self) -> None:
        self._stop = True
        if (
            self._inference_thread is not None
            and self._inference_thread.is_alive()
            and self._inference_thread is not current_thread()
        ):
            # A source failing to start ends the startup: the sources after it
            # are never started, so there is nothing of them to stop.
            self._sources_startup_finished.wait()
        # Each source is stopped once, so a call repeated after an error only
        # stops those still running.
        while self._started_sources:
            self._started_sources[0].terminate(
                wait_on_frames_consumption=False, purge_frames_buffer=True
            )
            self._started_sources.pop(0)

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
            if self._results_consumer_missing:
                self._discard_inference_results()
            self._inference_thread.join()
            self._inference_thread = None
        if self._dispatching_thread is not None:
            self._dispatching_thread.join()
            self._dispatching_thread = None
        if self._on_pipeline_end is not None:
            self._on_pipeline_end()

    def _discard_inference_results(self) -> None:
        # The dispatcher never started, so a full results queue would block
        # the inference thread for good: its results - never meant for the
        # sinks - are dropped up to its final sentinel.
        while self._predictions_queue.get() is not None:
            self._predictions_queue.task_done()
        self._predictions_queue.task_done()
        self._results_consumer_missing = False

    def _execute_inference(self) -> None:
        session_token = stream_session_id.set(self._stream_session_id)
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
                if _rfdetr_stream_pipeline_enabled():
                    self._queue_inference_result(
                        inference_result=predictions,
                        fallback_video_frames=video_frames,
                    )
                    continue
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
            if _rfdetr_stream_pipeline_enabled():
                self._drain_inference_handler()

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
            self._sources_startup_finished.set()
            if _rfdetr_stream_pipeline_enabled():
                self._close_inference_handler()
            self._predictions_queue.put(None)
            send_inference_pipeline_status_update(
                severity=UpdateSeverity.INFO,
                event_type=INFERENCE_THREAD_FINISHED_EVENT,
                status_update_handlers=self._status_update_handlers,
            )
            logger.info(f"Inference thread finished")
            stream_session_id.reset(session_token)

    def _dispatch_inference_results(self) -> None:
        while True:
            inference_results: Optional[
                Tuple[List[AnyPrediction], List[VideoFrame]]
            ] = self._predictions_queue.get()
            if inference_results is None:
                self._predictions_queue.task_done()
                break
            predictions, video_frames = inference_results
            if _rfdetr_stream_pipeline_enabled():
                predictions = _resolve_prediction_futures(predictions)
            # Older duck-typed watchdogs need not implement completion telemetry.
            on_completed = getattr(
                self._watchdog, "on_model_prediction_completed", None
            )
            if on_completed is not None:
                on_completed(frames=video_frames)
            if self._on_prediction is not None:
                self._handle_predictions_dispatching(
                    predictions=predictions,
                    video_frames=video_frames,
                )
            self._predictions_queue.task_done()

    def _queue_inference_result(
        self,
        inference_result: Optional[Union[List[AnyPrediction], InferenceHandlerResult]],
        fallback_video_frames: List[VideoFrame],
    ) -> None:
        normalised_result = self._normalise_inference_result(
            inference_result=inference_result,
            fallback_video_frames=fallback_video_frames,
        )
        if normalised_result is None:
            return None
        predictions, video_frames = normalised_result
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

    def _normalise_inference_result(
        self,
        inference_result: Optional[Union[List[AnyPrediction], InferenceHandlerResult]],
        fallback_video_frames: List[VideoFrame],
    ) -> Optional[Tuple[List[AnyPrediction], List[VideoFrame]]]:
        if inference_result is None:
            return None
        if isinstance(inference_result, InferenceHandlerResult):
            video_frames = (
                inference_result.video_frames
                if inference_result.video_frames is not None
                else fallback_video_frames
            )
            if len(video_frames) == 0:
                return None
            return inference_result.predictions, video_frames
        if len(fallback_video_frames) == 0:
            return None
        return inference_result, fallback_video_frames

    def _drain_inference_handler(self) -> None:
        flush_fn = getattr(self._on_video_frame, "flush", None)
        if not callable(flush_fn):
            return None
        flush_result = flush_fn()
        if flush_result is None:
            return None
        if isinstance(flush_result, list) and all(
            isinstance(result, InferenceHandlerResult) for result in flush_result
        ):
            for result in flush_result:
                self._queue_inference_result(
                    inference_result=result,
                    fallback_video_frames=[],
                )
            return None
        self._queue_inference_result(
            inference_result=flush_result,
            fallback_video_frames=[],
        )

    def _close_inference_handler(self) -> None:
        close_fn = getattr(self._on_video_frame, "close", None)
        if not callable(close_fn):
            return None
        try:
            close_fn()
        except Exception as error:
            logger.warning(f"Could not close inference handler. Cause: {error}")

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
            # Frames are handed to the sink AS-IS: under
            # ENABLE_TENSOR_DATA_REPRESENTATION that is the original on-device
            # tensor frame (no per-frame device-to-host materialisation here).
            # Pixel-consuming sinks materialise at their own boundary via
            # stream.utils.materialise_video_frame_for_sink.
            self._on_prediction(
                predictions,
                video_frames,
            )
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
        try:
            for video_source in self._video_sources:
                video_source.start()
                self._started_sources.append(video_source)
        finally:
            self._sources_startup_finished.set()
        max_fps = None
        if not self._frame_drop_on_video_file_rate_limiting_enabled():
            max_fps = self._max_fps
        yield from multiplex_videos(
            videos=self._video_sources,
            max_fps=max_fps,
            batch_collection_timeout=self._batch_collection_timeout,
            should_stop=lambda: self._stop,
            collection_policy=self._collection_policy,
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


def _resolve_prediction_futures(value: Any) -> Any:
    return resolve_futures(
        value=value,
        context="inference_pipeline | prediction_dispatch",
    )


def _rfdetr_stream_pipeline_enabled() -> bool:
    try:
        return int(os.getenv("RFDETR_PIPELINE_DEPTH", "1").strip()) > 1
    except ValueError:
        return False

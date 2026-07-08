from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

from inference.core import logger
from inference.core.env import WORKFLOWS_STREAM_LOOKAHEAD_DEPTH
from inference.core.interfaces.camera.entities import VideoFrame
from inference.core.interfaces.stream.entities import InferenceHandlerResult
from inference.core.workflows.execution_engine.core import ExecutionEngine
from inference.core.workflows.execution_engine.entities.base import VideoMetadata
from inference.core.workflows.execution_engine.v1.executor.core import (
    compute_async_stream_step_selectors,
    compute_stream_lookahead_frontier,
)

if TYPE_CHECKING:
    from inference.core.workflows.execution_engine.v1.executor.execution_data_manager.manager import (
        ExecutionDataManager,
    )


@dataclass(frozen=True)
class _StreamPipelineStep:
    step: Any
    name: Optional[str] = None


class WorkflowRunner:
    def __init__(
        self,
        workflows_parameters: Optional[Dict[str, Any]],
        execution_engine: ExecutionEngine,
        image_input_name: str,
        video_metadata_input_name: str,
        serialize_results: bool = False,
        _is_preview: bool = False,
    ):
        self._workflows_parameters = workflows_parameters
        self._execution_engine = execution_engine
        self._image_input_name = image_input_name
        self._video_metadata_input_name = video_metadata_input_name
        self._serialize_results = serialize_results
        self._is_preview = _is_preview

    def __call__(self, video_frames: List[VideoFrame]) -> List[dict]:
        return self._run_workflow(video_frames=video_frames)

    def _run_workflow(
        self,
        video_frames: List[VideoFrame],
        defer_stream_pipeline_flush: bool = False,
        resolve_output_futures: bool = True,
    ) -> List[dict]:
        workflows_parameters, fps = self._build_workflows_parameters(
            video_frames=video_frames
        )
        return self._execution_engine.run(
            runtime_parameters=workflows_parameters,
            fps=fps,
            serialize_results=self._serialize_results,
            _is_preview=self._is_preview,
            defer_stream_pipeline_flush=defer_stream_pipeline_flush,
            resolve_output_futures=resolve_output_futures,
        )

    def _flush_stream_pipeline(self, video_frames: List[VideoFrame]) -> List[dict]:
        workflows_parameters, fps = self._build_workflows_parameters(
            video_frames=video_frames
        )
        return self._execution_engine.flush_stream_pipeline(
            runtime_parameters=workflows_parameters,
            fps=fps,
            serialize_results=self._serialize_results,
            _is_preview=self._is_preview,
        )

    def _run_stream_lookahead(
        self,
        video_frames: List[VideoFrame],
        frontier_step_selectors: Set[str],
        async_step_selectors: Set[str],
        lookahead_executor: "ThreadPoolExecutor",
    ) -> "ExecutionDataManager":
        workflows_parameters, fps = self._build_workflows_parameters(
            video_frames=video_frames
        )
        return self._execution_engine.run_stream_lookahead(
            runtime_parameters=workflows_parameters,
            frontier_step_selectors=frontier_step_selectors,
            async_step_selectors=async_step_selectors,
            lookahead_executor=lookahead_executor,
            fps=fps,
            _is_preview=self._is_preview,
        )

    def _resume_stream_lookahead(
        self,
        execution_data_manager: "ExecutionDataManager",
        frontier_step_selectors: Set[str],
    ) -> List[dict]:
        return self._execution_engine.resume_stream_lookahead(
            execution_data_manager=execution_data_manager,
            frontier_step_selectors=frontier_step_selectors,
            serialize_results=self._serialize_results,
        )

    def _build_workflows_parameters(
        self,
        video_frames: List[VideoFrame],
    ) -> tuple[Dict[str, Any], float]:
        workflows_parameters: Dict[str, Any] = dict(self._workflows_parameters or {})
        # TODO: pass fps reflecting each stream to workflows_parameters
        fps = video_frames[0].fps
        if video_frames[0].measured_fps:
            fps = video_frames[0].measured_fps
        if fps is None:
            # for FPS reporting we expect 0 when FPS cannot be determined
            fps = 0
        video_metadata_for_images = [
            VideoMetadata(
                video_identifier=(
                    str(video_frame.source_id)
                    if video_frame.source_id is not None
                    else "default_source"
                ),
                frame_number=video_frame.frame_id,
                frame_timestamp=video_frame.frame_timestamp,
                fps=video_frame.fps,
                measured_fps=video_frame.measured_fps,
                comes_from_video_file=video_frame.comes_from_video_file,
            )
            for video_frame in video_frames
        ]
        workflows_parameters[self._image_input_name] = [
            {
                "type": "numpy_object",
                "value": video_frame.image,
                "video_metadata": video_metadata,
            }
            for video_frame, video_metadata in zip(
                video_frames, video_metadata_for_images
            )
        ]
        workflows_parameters[self._video_metadata_input_name] = (
            video_metadata_for_images
        )
        return workflows_parameters, fps


class PipelinedWorkflowRunner:
    def __init__(
        self,
        workflow_runner: WorkflowRunner,
        stream_steps: List[_StreamPipelineStep],
    ) -> None:
        self._workflow_runner = workflow_runner
        self._stream_steps = stream_steps
        self._pending_video_frames: List[List[VideoFrame]] = []

    def __call__(
        self, video_frames: List[VideoFrame]
    ) -> Optional[InferenceHandlerResult]:
        # Resolving RF-DETR output futures here serializes postprocess before the
        # next frame can launch. The stream dispatcher resolves them after the
        # frame buffer delay, when they should already be ready.
        predictions = self._workflow_runner._run_workflow(
            video_frames=video_frames,
            defer_stream_pipeline_flush=True,
            resolve_output_futures=self._workflow_runner._serialize_results,
        )
        stream_buffer_depth = self._stream_buffer_depth()
        if stream_buffer_depth <= 0:
            self._pending_video_frames.clear()
            return InferenceHandlerResult(
                predictions=predictions,
                video_frames=video_frames,
            )
        self._pending_video_frames.append(video_frames)
        if len(self._pending_video_frames) <= stream_buffer_depth:
            return None
        emit_video_frames = self._pending_video_frames.pop(0)
        return InferenceHandlerResult(
            predictions=predictions,
            video_frames=emit_video_frames,
        )

    def flush(self) -> Optional[List[InferenceHandlerResult]]:
        stream_steps = self._stream_steps
        if not stream_steps:
            self._pending_video_frames.clear()
            return None
        if not self._pending_video_frames:
            return None
        if len(stream_steps) != 1:
            raise RuntimeError("Stream pipeline flushing supports one pipelined step")
        results = []
        for pending_video_frames in list(self._pending_video_frames):
            prediction = self._workflow_runner._flush_stream_pipeline(
                video_frames=pending_video_frames,
            )
            emit_video_frames = self._pending_video_frames.pop(0)
            results.append(
                InferenceHandlerResult(
                    predictions=prediction,
                    video_frames=emit_video_frames,
                )
            )
        return results

    def close(self) -> None:
        _close_stream_steps(stream_steps=self._stream_steps)

    def _stream_buffer_depth(self) -> int:
        return _max_stream_buffer_depth(stream_steps=self._stream_steps)


class LookaheadPipelinedWorkflowRunner:
    """Pipelined runner for steps that defer downstream execution.

    Each frame gets a deferred pass that executes only the workflow's
    stream-lookahead frontier (stateless steps; pipelined model steps launch
    their remote requests and register future-bearing outputs), and the
    frame's live execution state is buffered. Once the buffer exceeds the
    pipeline depth, the oldest frame is emitted by resuming its execution
    state: the remaining steps run with that frame's own inputs, in frame
    order, with in-flight futures resolved at input assembly.
    """

    def __init__(
        self,
        workflow_runner: WorkflowRunner,
        frontier_step_selectors: Set[str],
        async_step_selectors: Set[str],
        buffer_depth: int,
    ) -> None:
        self._workflow_runner = workflow_runner
        self._frontier_step_selectors = frontier_step_selectors
        self._async_step_selectors = async_step_selectors
        self._buffer_depth = buffer_depth
        self._lookahead_executor = ThreadPoolExecutor(
            max_workers=max(1, (buffer_depth + 1) * len(async_step_selectors)),
            thread_name_prefix="workflows_stream_lookahead",
        )
        self._pending_frames: List[Tuple[List[VideoFrame], "ExecutionDataManager"]] = []

    def __call__(
        self, video_frames: List[VideoFrame]
    ) -> Optional[InferenceHandlerResult]:
        execution_data_manager = self._workflow_runner._run_stream_lookahead(
            video_frames=video_frames,
            frontier_step_selectors=self._frontier_step_selectors,
            async_step_selectors=self._async_step_selectors,
            lookahead_executor=self._lookahead_executor,
        )
        self._pending_frames.append((video_frames, execution_data_manager))
        if len(self._pending_frames) <= self._buffer_depth:
            return None
        return self._emit_oldest_frame()

    def flush(self) -> Optional[List[InferenceHandlerResult]]:
        if not self._pending_frames:
            return None
        results = []
        while self._pending_frames:
            try:
                results.append(self._emit_oldest_frame())
            except Exception as error:
                # One failed frame must not drop the rest of the buffered
                # frames whose requests already completed.
                logger.exception(
                    "Failed to emit a buffered frame during stream-lookahead "
                    "drain: %s",
                    error,
                )
        return results

    def close(self) -> None:
        self._lookahead_executor.shutdown(wait=False)

    def _emit_oldest_frame(self) -> InferenceHandlerResult:
        emit_video_frames, execution_data_manager = self._pending_frames.pop(0)
        predictions = self._workflow_runner._resume_stream_lookahead(
            execution_data_manager=execution_data_manager,
            frontier_step_selectors=self._frontier_step_selectors,
        )
        return InferenceHandlerResult(
            predictions=predictions,
            video_frames=emit_video_frames,
        )


def wrap_workflow_runner_for_stream_pipeline(
    workflow_runner: WorkflowRunner,
    execution_engine: ExecutionEngine,
    allow_lookahead: bool = True,
):
    stream_steps = _stream_pipeline_steps(execution_engine=execution_engine)
    compiled_workflow = _compiled_workflow_with_graph(
        execution_engine=execution_engine
    )
    async_step_selectors = (
        compute_async_stream_step_selectors(workflow=compiled_workflow)
        if compiled_workflow is not None and WORKFLOWS_STREAM_LOOKAHEAD_DEPTH > 1
        else set()
    )
    if not async_step_selectors:
        if stream_steps:
            return PipelinedWorkflowRunner(
                workflow_runner=workflow_runner,
                stream_steps=stream_steps,
            )
        return workflow_runner
    if not allow_lookahead:
        # Multi-source pipelines feed frame batches larger than one image;
        # buffering them would add latency without any overlap.
        logger.warning(
            "Stream lookahead is disabled: it currently supports "
            "single-source pipelines only. Falling back to sequential "
            "execution."
        )
        return workflow_runner
    if stream_steps:
        # A workflow mixing async lookahead steps with a local stream
        # pipeline (RF-DETR) would leave the latter's flush queue undrained
        # under the lookahead runner.
        logger.warning(
            "Stream lookahead is disabled for this workflow: it mixes async "
            "steps with a locally stream-pipelined model. Falling back to "
            "sequential execution."
        )
        return workflow_runner
    frontier_step_selectors = compute_stream_lookahead_frontier(
        workflow=compiled_workflow
    )
    if not async_step_selectors <= frontier_step_selectors:
        logger.warning(
            "Stream lookahead is disabled for this workflow. Every async "
            "step must sit in the stream-lookahead frontier: fed only by "
            "steps that are stateless for video processing and not by "
            "another async step. Falling back to sequential execution."
        )
        return workflow_runner
    return LookaheadPipelinedWorkflowRunner(
        workflow_runner=workflow_runner,
        frontier_step_selectors=frontier_step_selectors,
        async_step_selectors=async_step_selectors,
        buffer_depth=WORKFLOWS_STREAM_LOOKAHEAD_DEPTH - 1,
    )


def _compiled_workflow_with_graph(execution_engine: ExecutionEngine) -> Optional[Any]:
    engine = getattr(execution_engine, "_engine", None)
    compiled_workflow = getattr(engine, "_compiled_workflow", None)
    if getattr(compiled_workflow, "execution_graph", None) is None:
        return None
    return compiled_workflow


def _stream_pipeline_steps(
    execution_engine: ExecutionEngine,
) -> List[_StreamPipelineStep]:
    engine = getattr(execution_engine, "_engine", None)
    compiled_workflow = getattr(engine, "_compiled_workflow", None)
    steps = getattr(compiled_workflow, "steps", {})
    stream_steps = []
    for step_name, initialised_step in steps.items():
        step_instance = getattr(initialised_step, "step", None)
        if _is_stream_pipeline_step(step_instance=step_instance):
            stream_steps.append(_StreamPipelineStep(step=step_instance, name=step_name))
    return stream_steps


def _is_stream_pipeline_step(step_instance: Any) -> bool:
    is_stream_pipelined = getattr(step_instance, "is_stream_pipelined", None)
    if callable(is_stream_pipelined) and is_stream_pipelined():
        return True
    can_activate_pipeline = getattr(step_instance, "can_activate_stream_pipeline", None)
    return callable(can_activate_pipeline) and can_activate_pipeline()


def _close_stream_steps(stream_steps: List[_StreamPipelineStep]) -> None:
    for stream_step in stream_steps:
        close_fn = getattr(stream_step.step, "close_stream_pipeline", None)
        if not callable(close_fn):
            continue
        try:
            close_fn()
        except Exception as error:
            logger.exception("Failed to close stream pipeline step: %s", error)


def _max_stream_buffer_depth(stream_steps: List[_StreamPipelineStep]) -> int:
    return max(
        (_stream_step_depth(stream_step) for stream_step in stream_steps),
        default=0,
    )


def _stream_step_depth(stream_step: _StreamPipelineStep) -> int:
    get_depth = getattr(stream_step.step, "stream_pipeline_depth", None)
    if not callable(get_depth):
        return 0
    return max(0, int(get_depth()))



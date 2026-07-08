from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from inference.core import logger
from inference.core.interfaces.camera.entities import VideoFrame
from inference.core.interfaces.stream.entities import InferenceHandlerResult
from inference.core.workflows.execution_engine.core import ExecutionEngine
from inference.core.workflows.execution_engine.entities.base import VideoMetadata
from inference.core.workflows.execution_engine.v1.compiler.utils import (
    construct_step_selector,
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
        for stream_step in self._stream_steps:
            close_fn = getattr(stream_step.step, "close_stream_pipeline", None)
            if callable(close_fn):
                close_fn()

    def _stream_buffer_depth(self) -> int:
        return max(
            (_stream_step_depth(stream_step) for stream_step in self._stream_steps),
            default=0,
        )


class FlushEmittingPipelinedWorkflowRunner(PipelinedWorkflowRunner):
    """Pipelined runner for steps that defer downstream execution.

    The deferred pass only launches the pipelined steps (e.g. remote model
    HTTP requests) — downstream steps are skipped by the execution engine, so
    the deferred run's outputs are incomplete and discarded. Each buffered
    frame is emitted through the stream-pipeline flush instead, which
    materializes every pipelined step's oldest pending result and runs the
    downstream steps with that frame's own inputs, preserving frame order
    for stateful consumers like trackers.
    """

    def __call__(
        self, video_frames: List[VideoFrame]
    ) -> Optional[InferenceHandlerResult]:
        self._workflow_runner._run_workflow(
            video_frames=video_frames,
            defer_stream_pipeline_flush=True,
            resolve_output_futures=False,
        )
        self._pending_video_frames.append(video_frames)
        if len(self._pending_video_frames) <= self._stream_buffer_depth():
            return None
        emit_video_frames = self._pending_video_frames.pop(0)
        predictions = self._workflow_runner._flush_stream_pipeline(
            video_frames=emit_video_frames,
        )
        return InferenceHandlerResult(
            predictions=predictions,
            video_frames=emit_video_frames,
        )

    def flush(self) -> Optional[List[InferenceHandlerResult]]:
        # Unlike the base runner, multiple independent pipelined steps are
        # supported: each flush pass drains one pending request from every
        # pipelined step for the emitted frame.
        if not self._stream_steps:
            self._pending_video_frames.clear()
            return None
        if not self._pending_video_frames:
            return None
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


def wrap_workflow_runner_for_stream_pipeline(
    workflow_runner: WorkflowRunner,
    execution_engine: ExecutionEngine,
):
    stream_steps = _stream_pipeline_steps(execution_engine=execution_engine)
    if not stream_steps:
        return workflow_runner
    deferring_steps = [
        stream_step
        for stream_step in stream_steps
        if _step_defers_downstream_execution(stream_step=stream_step)
    ]
    if not deferring_steps:
        return PipelinedWorkflowRunner(
            workflow_runner=workflow_runner,
            stream_steps=stream_steps,
        )
    if not _workflow_supports_downstream_deferral(
        execution_engine=execution_engine,
        deferring_steps=deferring_steps,
    ):
        logger.warning(
            "Remote stream pipelining is disabled for this workflow. It requires "
            "independent pipelined model steps (none consuming another's output) "
            "with every other step downstream of them; falling back to "
            "sequential execution."
        )
        return workflow_runner
    return FlushEmittingPipelinedWorkflowRunner(
        workflow_runner=workflow_runner,
        stream_steps=stream_steps,
    )


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


def _stream_step_depth(stream_step: _StreamPipelineStep) -> int:
    get_depth = getattr(stream_step.step, "stream_pipeline_depth", None)
    if not callable(get_depth):
        return 0
    return max(0, int(get_depth()))


def _step_defers_downstream_execution(stream_step: _StreamPipelineStep) -> bool:
    defers_fn = getattr(stream_step.step, "defers_downstream_execution", None)
    return callable(defers_fn) and defers_fn()


def _workflow_supports_downstream_deferral(
    execution_engine: ExecutionEngine,
    deferring_steps: List[_StreamPipelineStep],
) -> bool:
    # Flush-based emission reruns only the steps downstream of the pipelined
    # steps, so every other step's output would be missing from emitted
    # results. Restrict activation to shapes where that cannot happen: one or
    # more independent pipelined steps (none consuming another's output) with
    # every remaining step downstream of at least one of them.
    deferring_step_names = [deferring_step.name for deferring_step in deferring_steps]
    engine = getattr(execution_engine, "_engine", None)
    compiled_workflow = getattr(engine, "_compiled_workflow", None)
    execution_graph = getattr(compiled_workflow, "execution_graph", None)
    if execution_graph is None or any(
        step_name is None for step_name in deferring_step_names
    ):
        return len(deferring_steps) == 1
    step_selectors = {
        construct_step_selector(step_name=step_name)
        for step_name in getattr(compiled_workflow, "steps", {})
    }
    deferring_selectors = {
        construct_step_selector(step_name=step_name)
        for step_name in deferring_step_names
    }
    downstream_selectors = set()
    nodes_to_visit = list(deferring_selectors)
    while nodes_to_visit:
        node = nodes_to_visit.pop(0)
        for successor in execution_graph.successors(node):
            if successor in downstream_selectors:
                continue
            if successor in step_selectors:
                downstream_selectors.add(successor)
            nodes_to_visit.append(successor)
    if deferring_selectors & downstream_selectors:
        # A pipelined step consuming another's output would block on its
        # upstream future; chained pipelined steps are unsupported.
        return False
    return step_selectors - deferring_selectors <= downstream_selectors

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import networkx as nx

from inference.core.interfaces.camera.entities import VideoFrame
from inference.core.interfaces.stream.entities import InferenceHandlerResult
from inference.core.workflows.execution_engine.constants import (
    NODE_COMPILATION_OUTPUT_PROPERTY,
)
from inference.core.workflows.execution_engine.core import ExecutionEngine
from inference.core.workflows.execution_engine.entities.base import VideoMetadata
from inference.core.workflows.execution_engine.v1.compiler.entities import (
    DynamicStepInputDefinition,
    NodeCategory,
    StepInputDefinition,
    StepNode,
)
from inference.core.workflows.execution_engine.v1.compiler.utils import (
    get_step_selector_from_its_output,
)


@dataclass(frozen=True)
class _StreamPipelineStep:
    selector: str
    step: Any


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
                    if video_frame.source_id
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


def wrap_workflow_runner_for_stream_pipeline(
    workflow_runner: WorkflowRunner,
    execution_engine: ExecutionEngine,
):
    stream_steps = _stream_pipeline_steps(execution_engine=execution_engine)
    if not stream_steps:
        return workflow_runner
    if _workflow_requires_synchronous_stream_steps(
        execution_engine=execution_engine,
        stream_steps=stream_steps,
    ):
        _disable_stream_pipeline_steps(stream_steps=stream_steps)
        return workflow_runner
    return PipelinedWorkflowRunner(
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
            stream_steps.append(
                _StreamPipelineStep(
                    selector=f"$steps.{step_name}",
                    step=step_instance,
                )
            )
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


def _workflow_requires_synchronous_stream_steps(
    execution_engine: ExecutionEngine,
    stream_steps: List[_StreamPipelineStep],
) -> bool:
    engine = getattr(execution_engine, "_engine", None)
    compiled_workflow = getattr(engine, "_compiled_workflow", None)
    execution_graph = getattr(compiled_workflow, "execution_graph", None)
    if execution_graph is None:
        return False
    stream_step_selectors = {stream_step.selector for stream_step in stream_steps}
    safe_step_selectors = set(stream_step_selectors)
    downstream_step_selectors = set()
    for stream_step_selector in stream_step_selectors:
        if stream_step_selector not in execution_graph:
            continue
        downstream_step_selectors.update(
            node
            for node in nx.descendants(execution_graph, stream_step_selector)
            if _is_step_node(execution_graph=execution_graph, node=node)
        )
    for node in nx.topological_sort(execution_graph):
        if node not in downstream_step_selectors:
            continue
        step_node = _get_step_node(execution_graph=execution_graph, node=node)
        if step_node is None:
            continue
        if _step_has_unsafe_stream_dependency(
            step_node=step_node,
            safe_step_selectors=safe_step_selectors,
        ):
            return True
        safe_step_selectors.add(node)
    return False


def _disable_stream_pipeline_steps(stream_steps: List[_StreamPipelineStep]) -> None:
    for stream_step in stream_steps:
        disable_fn = getattr(
            stream_step.step,
            "disable_stream_pipeline_for_workflow",
            None,
        )
        if callable(disable_fn):
            disable_fn()


def _step_has_unsafe_stream_dependency(
    step_node: StepNode,
    safe_step_selectors: set[str],
) -> bool:
    for input_definition in step_node.input_data.values():
        for leaf_definition in _iter_step_input_definitions(input_definition):
            if leaf_definition.is_static_value():
                continue
            if leaf_definition.points_to_input():
                return True
            if not isinstance(leaf_definition, DynamicStepInputDefinition):
                continue
            producer_step_selector = get_step_selector_from_its_output(
                step_output_selector=leaf_definition.selector,
            )
            if producer_step_selector not in safe_step_selectors:
                return True
    return False


def _iter_step_input_definitions(input_definition: Any):
    iterate_definitions = getattr(input_definition, "iterate_through_definitions", None)
    if callable(iterate_definitions):
        yield from iterate_definitions()
        return
    if isinstance(input_definition, StepInputDefinition):
        yield input_definition


def _is_step_node(execution_graph: Any, node: str) -> bool:
    step_node = _get_step_node(execution_graph=execution_graph, node=node)
    return step_node is not None


def _get_step_node(execution_graph: Any, node: str) -> Optional[StepNode]:
    node_data = execution_graph.nodes[node].get(NODE_COMPILATION_OUTPUT_PROPERTY)
    if getattr(node_data, "node_category", None) is not NodeCategory.STEP_NODE:
        return None
    if not isinstance(node_data, StepNode):
        return None
    return node_data

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from inference.core.interfaces.camera.entities import VideoFrame
from inference.core.interfaces.stream.entities import InferenceHandlerResult
from inference.core.workflows.execution_engine.core import ExecutionEngine
from inference.core.workflows.execution_engine.entities.base import VideoMetadata


@dataclass(frozen=True)
class _StreamPipelineStep:
    step: Any
    depth: int


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

    def _run_workflow(self, video_frames: List[VideoFrame]) -> List[dict]:
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
        return self._execution_engine.run(
            runtime_parameters=workflows_parameters,
            fps=fps,
            serialize_results=self._serialize_results,
            _is_preview=self._is_preview,
        )


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
        predictions = self._workflow_runner(video_frames=video_frames)
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
        flush_fn = getattr(stream_steps[0].step, "flush_stream_pipeline", None)
        if not callable(flush_fn):
            raise RuntimeError(
                "Stream-pipelined workflow step must implement flush_stream_pipeline()"
            )
        predictions = flush_fn()
        if predictions is None:
            predictions = []
        if len(predictions) != len(self._pending_video_frames):
            raise RuntimeError(
                "Stream pipeline flush returned a different number of prediction "
                "batches than pending video-frame batches"
            )
        results = []
        for prediction in predictions:
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
        stream_steps = self._stream_steps
        if not stream_steps:
            return 0
        return max(stream_step.depth for stream_step in stream_steps)


def wrap_workflow_runner_for_stream_pipeline(
    workflow_runner: WorkflowRunner,
    execution_engine: ExecutionEngine,
):
    stream_steps = _stream_pipeline_steps(execution_engine=execution_engine)
    if not stream_steps:
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
    for initialised_step in steps.values():
        step_instance = getattr(initialised_step, "step", None)
        is_stream_pipelined = getattr(step_instance, "is_stream_pipelined", None)
        if callable(is_stream_pipelined) and is_stream_pipelined():
            get_depth = getattr(step_instance, "stream_pipeline_depth", None)
            if callable(get_depth):
                depth = int(get_depth())
            else:
                depth = 1
            if depth > 0:
                stream_steps.append(
                    _StreamPipelineStep(step=step_instance, depth=depth)
                )
    return stream_steps

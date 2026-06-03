from typing import Any, Dict, List, Optional

from inference.core.interfaces.camera.entities import VideoFrame
from inference.core.interfaces.stream.entities import InferenceHandlerResult
from inference.core.interfaces.stream.model_handlers.workflows_context import (
    workflow_stream_flush_context,
)
from inference.core.workflows.execution_engine.core import ExecutionEngine
from inference.core.workflows.execution_engine.entities.base import VideoMetadata


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
        self._pending_video_frames: List[List[VideoFrame]] = []

    def __call__(
        self, video_frames: List[VideoFrame]
    ) -> Optional[InferenceHandlerResult]:
        predictions = self._run_workflow(video_frames=video_frames)
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
        if self._stream_buffer_depth() <= 0:
            self._pending_video_frames.clear()
            return None
        if not self._pending_video_frames:
            return None
        results = []
        while self._pending_video_frames:
            emit_video_frames = self._pending_video_frames.pop(0)
            with workflow_stream_flush_context():
                predictions = self._run_workflow(video_frames=emit_video_frames)
            results.append(
                InferenceHandlerResult(
                    predictions=predictions,
                    video_frames=emit_video_frames,
                )
            )
        return results

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

    def _uses_stream_buffering(self) -> bool:
        return self._stream_buffer_depth() > 0

    def _stream_buffer_depth(self) -> int:
        engine = getattr(self._execution_engine, "_engine", None)
        compiled_workflow = getattr(engine, "_compiled_workflow", None)
        steps = getattr(compiled_workflow, "steps", {})
        stream_buffer_depth = 0
        for initialised_step in steps.values():
            step_instance = getattr(initialised_step, "step", None)
            is_stream_pipelined = getattr(step_instance, "is_stream_pipelined", None)
            if callable(is_stream_pipelined) and is_stream_pipelined():
                get_depth = getattr(step_instance, "stream_pipeline_depth", None)
                if callable(get_depth):
                    stream_buffer_depth = max(stream_buffer_depth, int(get_depth()))
                else:
                    stream_buffer_depth = max(stream_buffer_depth, 1)
        return stream_buffer_depth

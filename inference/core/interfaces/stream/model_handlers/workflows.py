import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Union

import torch

from inference.core.env import ENABLE_TENSOR_DATA_REPRESENTATION
from inference.core.interfaces.camera.entities import VideoFrame
from inference.core.interfaces.stream.entities import InferenceHandlerResult
from inference.core.workflows.execution_engine.core import ExecutionEngine
from inference.core.workflows.execution_engine.entities.base import (
    VideoMetadata,
    WorkflowBatchInput,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _StreamPipelineStep:
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
        self._clamp_warned_keys: Set[str] = set()
        self._batch_input_names: Set[str] = (
            _declared_batch_input_names(execution_engine=execution_engine)
            - {image_input_name, video_metadata_input_name}
            if _is_preview
            else set()
        )

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
        if self._is_preview and self._batch_input_names:
            # Preview block-cache may pass full per-video lists via
            # workflows_parameters while the stream runs one frame / small
            # batch at a time. Index those lists by frame_id so
            # WorkflowBatchInput length matches the current image batch.
            # Restricted to inputs the workflow declares as WorkflowBatchInput -
            # WorkflowParameter values (zones, class lists, ...) are ordinary
            # parameters of arbitrary length and must pass through untouched.
            workflows_parameters = _index_list_parameters_by_frame_id(
                workflows_parameters,
                video_frames,
                batch_input_names=self._batch_input_names,
                warned_keys=self._clamp_warned_keys,
            )
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
                # GPU-tensor decoding is best-effort under the tensor flag
                # (the cv2 CPU fallback emits numpy frames, and sources may
                # mix within one batch), so each frame declares its actual
                # payload type instead of a fixed flag-derived one.
                "type": (
                    "tensor"
                    if ENABLE_TENSOR_DATA_REPRESENTATION
                    and isinstance(video_frame.image, torch.Tensor)
                    else "numpy_object"
                ),
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


def _declared_batch_input_names(execution_engine: ExecutionEngine) -> Set[str]:
    # Only inputs declared as WorkflowBatchInput may hold per-frame caches.
    # The Workflow Builder preview injects each cached block output by adding
    # `{"type": "WorkflowBatchInput", "name": <cached input>, ...}` to the
    # definition it sends and rewriting `$steps.<step>.<prop>` selectors to
    # `$inputs.<cached input>` (roboflow app: cachedBlocks.ts,
    # buildBatchInputsForHits / rewriteStepRefsToInputs), so the declaration is
    # what marks a runtime value as a cache. Ordinary WorkflowParameters
    # (polygon zones, class filters, ...) are never declared this way.
    # Reached defensively - the engine internals are not part of a public
    # contract and an engine version may not expose a compiled workflow.
    engine = getattr(execution_engine, "_engine", None)
    compiled_workflow = getattr(engine, "_compiled_workflow", None)
    workflow_definition = getattr(compiled_workflow, "workflow_definition", None)
    declared_inputs = getattr(workflow_definition, "inputs", None) or []
    return {
        declared_input.name
        for declared_input in declared_inputs
        if isinstance(declared_input, WorkflowBatchInput)
    }


def _index_list_parameters_by_frame_id(
    workflows_parameters: Dict[str, Any],
    video_frames: List[VideoFrame],
    batch_input_names: Set[str],
    warned_keys: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    # Cached lists are keyed by raw frame_id. The producer (Workflow Builder
    # preview, roboflow app: cachedBlocks.ts) records outputs into a map keyed
    # by the streamed frame_id and materialises a dense array of length
    # max(frame_id) + 1 (populateCachedBlocksFromVideoFrames), so index 0 is
    # unused for the 1-based ids both VideoSource and the webrtc worker emit,
    # and gaps from dropped frames are backfilled with the nearest earlier
    # recorded value. It only serves a cached run when the recorded frame
    # count covers the expected one and ids stay within its replay bound, so
    # alignment is enforced there - it cannot be re-checked here, where the
    # previewed video's total frame count is unknown.
    # A misaligned cache would clamp to the nearest cached element and log a
    # warning once per key - the workflow keeps running, on a neighbouring
    # cached value, rather than having the engine treat the cache length as
    # the batch size.
    # Only finite video files qualify: on a live stream frame ids grow
    # unbounded and indexing would pin every frame to the last cached element.
    if not video_frames:
        return workflows_parameters
    if any(not frame.comes_from_video_file for frame in video_frames):
        return workflows_parameters
    frame_ids = [frame.frame_id for frame in video_frames]
    if any(not isinstance(frame_id, int) for frame_id in frame_ids):
        return workflows_parameters
    batch_size = len(video_frames)
    if warned_keys is None:
        warned_keys = set()
    indexed: Dict[str, Any] = {}
    for key, value in workflows_parameters.items():
        if key not in batch_input_names:
            indexed[key] = value
            continue
        if not isinstance(value, list) or len(value) in (0, 1, batch_size):
            indexed[key] = value
            continue
        # Out-of-range frame ids clamp to the nearest cached element. Passing
        # the full list through would make the execution engine treat its
        # length as the batch size and broadcast the single image across it.
        if key not in warned_keys and any(
            frame_id < 0 or frame_id >= len(value) for frame_id in frame_ids
        ):
            warned_keys.add(key)
            logger.warning(
                "Parameter '%s' holds %d cached elements but frame ids reach %d - "
                "clamping to the nearest cached value.",
                key,
                len(value),
                max(frame_ids),
            )
        indexed[key] = [
            value[min(max(frame_id, 0), len(value) - 1)] for frame_id in frame_ids
        ]
    return indexed


class PipelinedWorkflowRunner:
    def __init__(
        self,
        workflow_runner: WorkflowRunner,
        stream_steps: List[_StreamPipelineStep],
    ) -> None:
        self._workflow_runner = workflow_runner
        self._stream_steps = stream_steps
        self._pending_video_frames: List[List[VideoFrame]] = []
        self._last_video_frames: Optional[List[VideoFrame]] = None

    def __call__(
        self, video_frames: List[VideoFrame]
    ) -> Optional[InferenceHandlerResult]:
        self._last_video_frames = video_frames
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

    def flush(
        self,
    ) -> Optional[Union[InferenceHandlerResult, List[InferenceHandlerResult]]]:
        stream_steps = self._stream_steps
        if not stream_steps:
            self._pending_video_frames.clear()
            self._last_video_frames = None
            return None
        if self._last_video_frames is None:
            return None
        if len(stream_steps) != 1:
            raise RuntimeError("Stream pipeline flushing supports one pipelined step")
        if not self._pending_video_frames:
            video_frames = self._last_video_frames
            self._last_video_frames = None
            prediction = self._workflow_runner._flush_stream_pipeline(
                video_frames=video_frames,
            )
            return InferenceHandlerResult(
                predictions=prediction,
                video_frames=video_frames,
            )
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
        self._last_video_frames = None
        return results

    def close(self) -> None:
        self._last_video_frames = None
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
        if _is_stream_pipeline_step(step_instance=step_instance):
            stream_steps.append(_StreamPipelineStep(step=step_instance))
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

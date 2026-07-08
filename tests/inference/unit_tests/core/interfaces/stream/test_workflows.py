from concurrent.futures import Future
from datetime import datetime
from types import SimpleNamespace
from typing import Optional

import numpy as np
import pytest

from inference.core.interfaces.camera.entities import VideoFrame
from inference.core.interfaces.stream.model_handlers.workflows import (
    FlushEmittingPipelinedWorkflowRunner,
    PipelinedWorkflowRunner,
    WorkflowRunner,
    wrap_workflow_runner_for_stream_pipeline,
)
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.models.roboflow.instance_segmentation.v3 import (
    RoboflowInstanceSegmentationModelBlockV3,
)
from inference.core.workflows.execution_engine.entities.base import Batch, VideoMetadata
from inference_models.models.base.async_handoff import attach_async_response_future


class _FakePipelinedStep:
    def __init__(self, stream_buffer_depth: int) -> None:
        self._stream_buffer_depth = stream_buffer_depth
        self.flush_calls = 0
        self.close_calls = 0

    def is_stream_pipelined(self) -> bool:
        return self._stream_buffer_depth > 0

    def stream_pipeline_depth(self) -> int:
        return self._stream_buffer_depth

    def flush_stream_pipeline(self):
        self.flush_calls += 1
        return [[{"predictions": "frame-2"}]]

    def close_stream_pipeline(self) -> None:
        self.close_calls += 1


class _FakeExecutionEngine:
    def __init__(self, stream_buffer_depth: int) -> None:
        self._stream_buffer_depth = stream_buffer_depth
        self.step = _FakePipelinedStep(stream_buffer_depth=stream_buffer_depth)
        self._engine = SimpleNamespace(
            _compiled_workflow=SimpleNamespace(
                steps={"segmentation": SimpleNamespace(step=self.step)}
            )
        )
        self.calls = []

    def run(
        self,
        runtime_parameters,
        fps,
        serialize_results,
        _is_preview,
        defer_stream_pipeline_flush=False,
        resolve_output_futures=True,
    ):
        frame_number = runtime_parameters["image"][0]["video_metadata"].frame_number
        self.calls.append(
            {
                "frame_number": frame_number,
                "fps": fps,
                "serialize_results": serialize_results,
                "is_preview": _is_preview,
                "defer_stream_pipeline_flush": defer_stream_pipeline_flush,
                "resolve_output_futures": resolve_output_futures,
            }
        )
        prediction_frame = frame_number - self._stream_buffer_depth
        return [{"predictions": f"frame-{prediction_frame}"}]

    def flush_stream_pipeline(
        self,
        runtime_parameters,
        fps,
        serialize_results,
        _is_preview,
    ):
        return self.step.flush_stream_pipeline()[0]


class _FakeLateActivatingStep:
    def __init__(self, active_stream_buffer_depth: int) -> None:
        self._active_stream_buffer_depth = active_stream_buffer_depth
        self._stream_buffer_depth = 0
        self.flush_calls = 0

    def can_activate_stream_pipeline(self) -> bool:
        return True

    def is_stream_pipelined(self) -> bool:
        return self._stream_buffer_depth > 0

    def stream_pipeline_depth(self) -> int:
        return self._stream_buffer_depth

    def activate(self) -> None:
        self._stream_buffer_depth = self._active_stream_buffer_depth

    def flush_stream_pipeline(self):
        self.flush_calls += 1
        return [[{"predictions": "frame-2"}]]


class _FakeLateActivatingExecutionEngine:
    def __init__(self, active_stream_buffer_depth: int) -> None:
        self.step = _FakeLateActivatingStep(
            active_stream_buffer_depth=active_stream_buffer_depth
        )
        self._engine = SimpleNamespace(
            _compiled_workflow=SimpleNamespace(
                steps={"segmentation": SimpleNamespace(step=self.step)}
            )
        )

    def run(
        self,
        runtime_parameters,
        fps,
        serialize_results,
        _is_preview,
        defer_stream_pipeline_flush=False,
        resolve_output_futures=True,
    ):
        frame_number = runtime_parameters["image"][0]["video_metadata"].frame_number
        # Real RF-DETR workflow steps only know whether stream pipelining is
        # active after the first model request has loaded the concrete model.
        self.step.activate()
        prediction_frame = frame_number - self.step.stream_pipeline_depth()
        return [{"predictions": f"frame-{prediction_frame}"}]

    def flush_stream_pipeline(
        self,
        runtime_parameters,
        fps,
        serialize_results,
        _is_preview,
    ):
        return self.step.flush_stream_pipeline()[0]


class _FakeDownstreamPipelinedExecutionEngine:
    def __init__(self) -> None:
        self.step = _FakePipelinedStep(stream_buffer_depth=1)
        self._engine = SimpleNamespace(
            _compiled_workflow=SimpleNamespace(
                steps={"segmentation": SimpleNamespace(step=self.step)}
            )
        )
        self.calls = []
        self.flush_calls = []

    def run(
        self,
        runtime_parameters,
        fps,
        serialize_results,
        _is_preview,
        defer_stream_pipeline_flush=False,
        resolve_output_futures=True,
    ):
        frame_number = runtime_parameters["image"][0]["video_metadata"].frame_number
        self.calls.append(
            {
                "frame_number": frame_number,
                "defer_stream_pipeline_flush": defer_stream_pipeline_flush,
                "resolve_output_futures": resolve_output_futures,
            }
        )
        if defer_stream_pipeline_flush:
            prediction_frame = frame_number - self.step.stream_pipeline_depth()
        else:
            # This represents the reviewed regression: flushing before downstream
            # consumers makes the workflow output belong to the current frame.
            prediction_frame = frame_number
        return [{"result": f"downstream-frame-{prediction_frame}"}]

    def flush_stream_pipeline(
        self,
        runtime_parameters,
        fps,
        serialize_results,
        _is_preview,
    ):
        frame_number = runtime_parameters["image"][0]["video_metadata"].frame_number
        self.flush_calls.append(frame_number)
        return [{"result": f"downstream-frame-{frame_number}"}]


class _FakeDeferringStep:
    def __init__(self, stream_buffer_depth: int, defers: bool = True) -> None:
        self._stream_buffer_depth = stream_buffer_depth
        self._defers = defers
        self.close_calls = 0

    def is_stream_pipelined(self) -> bool:
        return self._stream_buffer_depth > 0

    def defers_downstream_execution(self) -> bool:
        return self._defers

    def stream_pipeline_depth(self) -> int:
        return self._stream_buffer_depth

    def close_stream_pipeline(self) -> None:
        self.close_calls += 1


class _FakeDeferringExecutionEngine:
    def __init__(self, stream_buffer_depth: int, defers: bool = True) -> None:
        self.step = _FakeDeferringStep(
            stream_buffer_depth=stream_buffer_depth, defers=defers
        )
        self._engine = SimpleNamespace(
            _compiled_workflow=SimpleNamespace(
                steps={"detection": SimpleNamespace(step=self.step)}
            )
        )
        self.calls = []
        self.flush_calls = []

    def run(
        self,
        runtime_parameters,
        fps,
        serialize_results,
        _is_preview,
        defer_stream_pipeline_flush=False,
        resolve_output_futures=True,
    ):
        frame_number = runtime_parameters["image"][0]["video_metadata"].frame_number
        self.calls.append(
            {
                "frame_number": frame_number,
                "defer_stream_pipeline_flush": defer_stream_pipeline_flush,
                "resolve_output_futures": resolve_output_futures,
            }
        )
        # The deferred pass only launches the pipelined step; downstream steps are
        # skipped, so this output is incomplete and discarded by the runner.
        return [{"result": f"deferred-frame-{frame_number}"}]

    def flush_stream_pipeline(
        self,
        runtime_parameters,
        fps,
        serialize_results,
        _is_preview,
    ):
        frame_number = runtime_parameters["image"][0]["video_metadata"].frame_number
        self.flush_calls.append(frame_number)
        return [{"result": f"flushed-frame-{frame_number}"}]


class _ImmediateExecutor:
    def submit(self, fn, *args, **kwargs) -> Future:
        future = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except BaseException as error:  # pragma: no cover - defensive
            future.set_exception(error)
        return future


class _FakeWorkflowImage:
    def __init__(
        self,
        tag: str,
        width: int = 8,
        height: int = 8,
        frame_number: Optional[int] = None,
    ) -> None:
        self.tag = tag
        self._image = np.zeros((height, width, 3), dtype=np.uint8)
        if frame_number is None and tag.rsplit("-", maxsplit=1)[-1].isdigit():
            frame_number = int(tag.rsplit("-", maxsplit=1)[-1])
        if frame_number is None:
            frame_number = 1
        self._video_metadata = VideoMetadata(
            video_identifier=f"video-{tag}",
            frame_number=frame_number,
            frame_timestamp=datetime.now(),
        )

    @property
    def video_metadata(self) -> VideoMetadata:
        return self._video_metadata

    @property
    def numpy_image(self):
        return self._image

    def to_inference_format(self, numpy_preferred: bool):
        assert numpy_preferred is True
        return {
            "type": "numpy_object",
            "value": self._image,
        }


class _FakeResponse:
    def __init__(
        self,
        tag: str,
        width: int = 8,
        height: int = 8,
    ) -> None:
        self.tag = tag
        self.width = width
        self.height = height

    def model_dump(self, by_alias: bool, exclude_none: bool):
        assert by_alias is True
        assert exclude_none is True
        return {
            "tag": self.tag,
            "image": {"width": self.width, "height": self.height},
            "predictions": [],
        }


class _FakeStreamModel:
    _pipeline_depth = 2

    def __init__(self) -> None:
        self.flush_calls = 0
        self.shutdown_calls = 0

    def flush(self):
        self.flush_calls += 1
        return [_FakeResponse("tail-final")]

    def shutdown_pipeline(self) -> None:
        self.shutdown_calls += 1


class _FakeModelManager:
    def __init__(self, inference_results) -> None:
        self._inference_results = list(inference_results)
        self.model = _FakeStreamModel()
        self.add_model_calls = []
        self.infer_calls = 0

    def add_model(self, model_id: str, api_key: str) -> None:
        self.add_model_calls.append((model_id, api_key))

    def infer_from_request_sync(self, model_id: str, request):
        self.infer_calls += 1
        return self._inference_results.pop(0)

    def __contains__(self, model_id: str) -> bool:
        return model_id == "model"

    def __getitem__(self, model_id: str):
        assert model_id == "model"
        return self.model


class _ContextAwareModelManager(_FakeModelManager):
    def __init__(self, mode: str) -> None:
        super().__init__(inference_results=[])
        self.mode = mode
        self.stream_pipeline_context_ids = []

    def infer_from_request_sync(self, model_id: str, request):
        self.infer_calls += 1
        assert request.source_info is None
        self.stream_pipeline_context_ids.append(request.stream_pipeline_context_id)
        if self.infer_calls == 1:
            return [_FakeResponse("priming", width=8, height=8)]
        if self.mode == "previous":
            return [
                _make_async_placeholder(
                    "first-final",
                    context_id=self.stream_pipeline_context_ids[0],
                    response_width=8,
                    response_height=8,
                )
            ]
        if self.mode == "current-with-old-size":
            return [
                _make_async_placeholder(
                    "first-final",
                    context_id=request.stream_pipeline_context_id,
                    response_width=8,
                    response_height=8,
                )
            ]
        return [
            _make_async_placeholder(
                "first-final",
                context_id="missing-context",
                response_width=8,
                response_height=8,
            )
        ]


def _make_async_placeholder(
    response_tag: str,
    context_id: Optional[str] = None,
    response_width: int = 8,
    response_height: int = 8,
) -> _FakeResponse:
    future = Future()
    future.set_result(
        [_FakeResponse(response_tag, width=response_width, height=response_height)]
    )
    response = _FakeResponse("placeholder")
    attach_async_response_future(
        response=response,
        response_future=future,
        context_id=context_id,
    )
    return response


def _make_frame(frame_id: int) -> VideoFrame:
    return VideoFrame(
        image=np.zeros((8, 8, 3), dtype=np.uint8),
        frame_id=frame_id,
        frame_timestamp=datetime.fromtimestamp(frame_id),
        fps=30.0,
        measured_fps=None,
        source_id=0,
        comes_from_video_file=True,
    )


def test_workflow_runner_without_stream_buffering_returns_current_frame() -> None:
    engine = _FakeExecutionEngine(stream_buffer_depth=0)
    runner = WorkflowRunner(
        workflows_parameters=None,
        execution_engine=engine,
        image_input_name="image",
        video_metadata_input_name="video_metadata",
        serialize_results=True,
        _is_preview=True,
    )
    frame = _make_frame(1)

    result = runner([frame])

    assert result == [{"predictions": "frame-1"}]
    assert engine.calls == [
        {
            "frame_number": 1,
            "fps": 30.0,
            "serialize_results": True,
            "is_preview": True,
            "defer_stream_pipeline_flush": False,
            "resolve_output_futures": True,
        }
    ]


def test_wrap_workflow_runner_leaves_non_pipelined_workflows_unchanged() -> None:
    engine = _FakeExecutionEngine(stream_buffer_depth=0)
    runner = WorkflowRunner(
        workflows_parameters=None,
        execution_engine=engine,
        image_input_name="image",
        video_metadata_input_name="video_metadata",
    )

    assert (
        wrap_workflow_runner_for_stream_pipeline(
            workflow_runner=runner,
            execution_engine=engine,
        )
        is runner
    )


def test_workflow_runner_buffers_frames_until_delayed_prediction_arrives() -> None:
    engine = _FakeExecutionEngine(stream_buffer_depth=1)
    workflow_runner = WorkflowRunner(
        workflows_parameters=None,
        execution_engine=engine,
        image_input_name="image",
        video_metadata_input_name="video_metadata",
    )
    runner = wrap_workflow_runner_for_stream_pipeline(
        workflow_runner=workflow_runner,
        execution_engine=engine,
    )
    assert isinstance(runner, PipelinedWorkflowRunner)
    frame_1 = _make_frame(1)
    frame_2 = _make_frame(2)

    first_result = runner([frame_1])
    second_result = runner([frame_2])
    flushed_results = runner.flush()

    assert first_result is None
    assert second_result is not None
    assert second_result.predictions == [{"predictions": "frame-1"}]
    assert second_result.video_frames == [frame_1]
    assert flushed_results is not None
    assert len(flushed_results) == 1
    assert flushed_results[0].predictions == [{"predictions": "frame-2"}]
    assert flushed_results[0].video_frames == [frame_2]
    assert engine.step.flush_calls == 1
    runner.close()
    assert engine.step.close_calls == 1
    assert engine.calls == [
        {
            "frame_number": 1,
            "fps": 30.0,
            "serialize_results": False,
            "is_preview": False,
            "defer_stream_pipeline_flush": True,
            "resolve_output_futures": False,
        },
        {
            "frame_number": 2,
            "fps": 30.0,
            "serialize_results": False,
            "is_preview": False,
            "defer_stream_pipeline_flush": True,
            "resolve_output_futures": False,
        },
    ]


def test_workflow_runner_buffers_when_stream_pipeline_activates_after_first_run() -> (
    None
):
    engine = _FakeLateActivatingExecutionEngine(active_stream_buffer_depth=1)
    workflow_runner = WorkflowRunner(
        workflows_parameters=None,
        execution_engine=engine,
        image_input_name="image",
        video_metadata_input_name="video_metadata",
    )
    runner = wrap_workflow_runner_for_stream_pipeline(
        workflow_runner=workflow_runner,
        execution_engine=engine,
    )
    assert isinstance(runner, PipelinedWorkflowRunner)
    assert engine.step.stream_pipeline_depth() == 0
    frame_1 = _make_frame(1)
    frame_2 = _make_frame(2)

    first_result = runner([frame_1])
    second_result = runner([frame_2])
    flushed_results = runner.flush()

    assert first_result is None
    assert second_result is not None
    assert second_result.predictions == [{"predictions": "frame-1"}]
    assert second_result.video_frames == [frame_1]
    assert flushed_results is not None
    assert len(flushed_results) == 1
    assert flushed_results[0].predictions == [{"predictions": "frame-2"}]
    assert flushed_results[0].video_frames == [frame_2]
    assert engine.step.flush_calls == 1


def test_pipelined_workflow_runner_preserves_frame_alignment_for_downstream_steps() -> (
    None
):
    engine = _FakeDownstreamPipelinedExecutionEngine()
    workflow_runner = WorkflowRunner(
        workflows_parameters=None,
        execution_engine=engine,
        image_input_name="image",
        video_metadata_input_name="video_metadata",
    )
    runner = wrap_workflow_runner_for_stream_pipeline(
        workflow_runner=workflow_runner,
        execution_engine=engine,
    )
    assert isinstance(runner, PipelinedWorkflowRunner)
    frame_1 = _make_frame(1)
    frame_2 = _make_frame(2)

    first_result = runner([frame_1])
    second_result = runner([frame_2])
    flushed_results = runner.flush()

    assert first_result is None
    assert second_result is not None
    assert second_result.predictions == [{"result": "downstream-frame-1"}]
    assert second_result.video_frames == [frame_1]
    assert flushed_results is not None
    assert len(flushed_results) == 1
    assert flushed_results[0].predictions == [{"result": "downstream-frame-2"}]
    assert flushed_results[0].video_frames == [frame_2]
    assert engine.calls == [
        {
            "frame_number": 1,
            "defer_stream_pipeline_flush": True,
            "resolve_output_futures": False,
        },
        {
            "frame_number": 2,
            "defer_stream_pipeline_flush": True,
            "resolve_output_futures": False,
        },
    ]
    assert engine.flush_calls == [2]


def test_wrap_returns_flush_emitting_runner_when_step_defers_downstream_execution() -> (
    None
):
    engine = _FakeDeferringExecutionEngine(stream_buffer_depth=1)
    workflow_runner = WorkflowRunner(
        workflows_parameters=None,
        execution_engine=engine,
        image_input_name="image",
        video_metadata_input_name="video_metadata",
    )

    runner = wrap_workflow_runner_for_stream_pipeline(
        workflow_runner=workflow_runner,
        execution_engine=engine,
    )

    assert isinstance(runner, FlushEmittingPipelinedWorkflowRunner)


def test_wrap_returns_plain_pipelined_runner_when_step_does_not_defer() -> None:
    engine = _FakeDeferringExecutionEngine(stream_buffer_depth=1, defers=False)
    workflow_runner = WorkflowRunner(
        workflows_parameters=None,
        execution_engine=engine,
        image_input_name="image",
        video_metadata_input_name="video_metadata",
    )

    runner = wrap_workflow_runner_for_stream_pipeline(
        workflow_runner=workflow_runner,
        execution_engine=engine,
    )

    assert type(runner) is PipelinedWorkflowRunner


def test_wrap_returns_plain_pipelined_runner_when_defer_marker_absent() -> None:
    engine = _FakeExecutionEngine(stream_buffer_depth=1)
    workflow_runner = WorkflowRunner(
        workflows_parameters=None,
        execution_engine=engine,
        image_input_name="image",
        video_metadata_input_name="video_metadata",
    )

    runner = wrap_workflow_runner_for_stream_pipeline(
        workflow_runner=workflow_runner,
        execution_engine=engine,
    )

    assert type(runner) is PipelinedWorkflowRunner


def test_flush_emitting_runner_emits_from_flush_path_with_buffer_depth_one() -> None:
    engine = _FakeDeferringExecutionEngine(stream_buffer_depth=1)
    workflow_runner = WorkflowRunner(
        workflows_parameters=None,
        execution_engine=engine,
        image_input_name="image",
        video_metadata_input_name="video_metadata",
    )
    runner = wrap_workflow_runner_for_stream_pipeline(
        workflow_runner=workflow_runner,
        execution_engine=engine,
    )
    assert isinstance(runner, FlushEmittingPipelinedWorkflowRunner)
    frame_1 = _make_frame(1)
    frame_2 = _make_frame(2)

    first_result = runner([frame_1])
    second_result = runner([frame_2])
    flushed_results = runner.flush()

    assert first_result is None
    assert second_result is not None
    # The emitted predictions come from the flush path, not the discarded run.
    assert second_result.predictions == [{"result": "flushed-frame-1"}]
    assert second_result.video_frames == [frame_1]
    assert flushed_results is not None
    assert len(flushed_results) == 1
    assert flushed_results[0].predictions == [{"result": "flushed-frame-2"}]
    assert flushed_results[0].video_frames == [frame_2]
    assert engine.flush_calls == [1, 2]
    assert engine.calls == [
        {
            "frame_number": 1,
            "defer_stream_pipeline_flush": True,
            "resolve_output_futures": False,
        },
        {
            "frame_number": 2,
            "defer_stream_pipeline_flush": True,
            "resolve_output_futures": False,
        },
    ]
    runner.close()
    assert engine.step.close_calls == 1


def test_flush_emitting_runner_preserves_emission_order_with_buffer_depth_two() -> None:
    engine = _FakeDeferringExecutionEngine(stream_buffer_depth=2)
    workflow_runner = WorkflowRunner(
        workflows_parameters=None,
        execution_engine=engine,
        image_input_name="image",
        video_metadata_input_name="video_metadata",
    )
    runner = wrap_workflow_runner_for_stream_pipeline(
        workflow_runner=workflow_runner,
        execution_engine=engine,
    )
    assert isinstance(runner, FlushEmittingPipelinedWorkflowRunner)
    frames = [_make_frame(frame_id) for frame_id in range(1, 5)]

    results = [runner([frame]) for frame in frames]
    flushed_results = runner.flush()

    assert results[0] is None
    assert results[1] is None
    assert results[2].predictions == [{"result": "flushed-frame-1"}]
    assert results[2].video_frames == [frames[0]]
    assert results[3].predictions == [{"result": "flushed-frame-2"}]
    assert results[3].video_frames == [frames[1]]
    assert flushed_results is not None
    assert [result.video_frames for result in flushed_results] == [
        [frames[2]],
        [frames[3]],
    ]
    assert flushed_results[0].predictions == [{"result": "flushed-frame-3"}]
    assert flushed_results[1].predictions == [{"result": "flushed-frame-4"}]
    # Every frame is emitted through the flush path, in strict frame order.
    assert engine.flush_calls == [1, 2, 3, 4]


def test_instance_segmentation_stream_pipeline_activation_requires_depth_above_one(
    monkeypatch,
) -> None:
    block = RoboflowInstanceSegmentationModelBlockV3(
        model_manager=_FakeModelManager(inference_results=[]),
        api_key="api-key",
        step_execution_mode=StepExecutionMode.LOCAL,
    )

    monkeypatch.setattr(
        "inference.core.workflows.core_steps.models.roboflow.instance_segmentation.v3.get_rfdetr_pipeline_depth",
        lambda: 1,
    )
    assert block.can_activate_stream_pipeline() is False

    monkeypatch.setattr(
        "inference.core.workflows.core_steps.models.roboflow.instance_segmentation.v3.get_rfdetr_pipeline_depth",
        lambda: 2,
    )
    assert block.can_activate_stream_pipeline() is True


def test_instance_segmentation_stream_flush_drains_model_without_rerunning_workflow() -> (
    None
):
    manager = _FakeModelManager(
        inference_results=[
            [_FakeResponse("priming")],
            [_make_async_placeholder("first-final")],
        ]
    )
    block = RoboflowInstanceSegmentationModelBlockV3(
        model_manager=manager,
        api_key="api-key",
        step_execution_mode=StepExecutionMode.LOCAL,
    )
    block._get_stream_response_executor = lambda: _ImmediateExecutor()
    block._post_process_result = lambda images, predictions, class_filter, model_id: [
        {
            "predictions": f"{images[0].tag}:{predictions[0]['tag']}",
            "class_filter": class_filter,
            "model_id": model_id,
        }
    ]

    first_result = block.run_locally(
        images=[_FakeWorkflowImage("frame-1")],
        model_id="model",
        class_agnostic_nms=None,
        class_filter=["car"],
        confidence=0.4,
        iou_threshold=None,
        max_detections=None,
        max_candidates=None,
        mask_decode_mode="accurate",
        tradeoff_factor=None,
        disable_active_learning=None,
        active_learning_target_dataset=None,
        enforce_dense_masks_in_inference_models=False,
    )
    second_result = block.run_locally(
        images=[_FakeWorkflowImage("frame-2")],
        model_id="model",
        class_agnostic_nms=None,
        class_filter=["car"],
        confidence=0.4,
        iou_threshold=None,
        max_detections=None,
        max_candidates=None,
        mask_decode_mode="accurate",
        tradeoff_factor=None,
        disable_active_learning=None,
        active_learning_target_dataset=None,
        enforce_dense_masks_in_inference_models=False,
    )
    flushed_results = block.flush_stream_pipeline()

    assert first_result == [
        {
            "predictions": "frame-1:priming",
            "class_filter": ["car"],
            "model_id": "model",
        }
    ]
    assert second_result[0]["predictions"].result() == "frame-1:first-final"
    assert flushed_results == [
        [
            {
                "predictions": "frame-2:tail-final",
                "class_filter": ["car"],
                "model_id": "model",
            }
        ]
    ]
    assert manager.infer_calls == 2
    assert manager.model.flush_calls == 1
    block.close_stream_pipeline()
    assert manager.model.shutdown_calls == 1


def test_instance_segmentation_stream_pipeline_uses_response_context_id() -> None:
    manager = _ContextAwareModelManager(mode="previous")
    block = RoboflowInstanceSegmentationModelBlockV3(
        model_manager=manager,
        api_key="api-key",
        step_execution_mode=StepExecutionMode.LOCAL,
    )
    block._get_stream_response_executor = lambda: _ImmediateExecutor()
    block._post_process_result = lambda images, predictions, class_filter, model_id: [
        {
            "predictions": f"{images[0].tag}:{predictions[0]['tag']}",
            "class_filter": class_filter,
            "model_id": model_id,
        }
    ]

    block.run_locally(
        images=[_FakeWorkflowImage("frame-1", width=8, height=8)],
        model_id="model",
        class_agnostic_nms=None,
        class_filter=["car"],
        confidence=0.4,
        iou_threshold=None,
        max_detections=None,
        max_candidates=None,
        mask_decode_mode="accurate",
        tradeoff_factor=None,
        disable_active_learning=None,
        active_learning_target_dataset=None,
        enforce_dense_masks_in_inference_models=False,
    )
    second_result = block.run_locally(
        images=[_FakeWorkflowImage("frame-2", width=4, height=4)],
        model_id="model",
        class_agnostic_nms=None,
        class_filter=["car"],
        confidence=0.4,
        iou_threshold=None,
        max_detections=None,
        max_candidates=None,
        mask_decode_mode="accurate",
        tradeoff_factor=None,
        disable_active_learning=None,
        active_learning_target_dataset=None,
        enforce_dense_masks_in_inference_models=False,
    )

    assert second_result[0]["predictions"].result() == "frame-1:first-final"
    assert len(manager.stream_pipeline_context_ids) == 2
    assert (
        manager.stream_pipeline_context_ids[0] != manager.stream_pipeline_context_ids[1]
    )


def test_instance_segmentation_stream_pipeline_rejects_unknown_context_id() -> None:
    manager = _ContextAwareModelManager(mode="missing")
    block = RoboflowInstanceSegmentationModelBlockV3(
        model_manager=manager,
        api_key="api-key",
        step_execution_mode=StepExecutionMode.LOCAL,
    )
    block._get_stream_response_executor = lambda: _ImmediateExecutor()
    block._post_process_result = lambda images, predictions, class_filter, model_id: [
        {
            "predictions": f"{images[0].tag}:{predictions[0]['tag']}",
            "class_filter": class_filter,
            "model_id": model_id,
        }
    ]

    block.run_locally(
        images=[_FakeWorkflowImage("frame-1", width=8, height=8)],
        model_id="model",
        class_agnostic_nms=None,
        class_filter=["car"],
        confidence=0.4,
        iou_threshold=None,
        max_detections=None,
        max_candidates=None,
        mask_decode_mode="accurate",
        tradeoff_factor=None,
        disable_active_learning=None,
        active_learning_target_dataset=None,
        enforce_dense_masks_in_inference_models=False,
    )

    with pytest.raises(RuntimeError, match="context did not match"):
        block.run_locally(
            images=[_FakeWorkflowImage("frame-2", width=4, height=4)],
            model_id="model",
            class_agnostic_nms=None,
            class_filter=["car"],
            confidence=0.4,
            iou_threshold=None,
            max_detections=None,
            max_candidates=None,
            mask_decode_mode="accurate",
            tradeoff_factor=None,
            disable_active_learning=None,
            active_learning_target_dataset=None,
            enforce_dense_masks_in_inference_models=False,
        )


def test_instance_segmentation_stream_pipeline_rejects_image_metadata_mismatch() -> (
    None
):
    manager = _ContextAwareModelManager(mode="current-with-old-size")
    block = RoboflowInstanceSegmentationModelBlockV3(
        model_manager=manager,
        api_key="api-key",
        step_execution_mode=StepExecutionMode.LOCAL,
    )
    block._get_stream_response_executor = lambda: _ImmediateExecutor()
    block._post_process_result = lambda images, predictions, class_filter, model_id: [
        {
            "predictions": f"{images[0].tag}:{predictions[0]['tag']}",
            "class_filter": class_filter,
            "model_id": model_id,
        }
    ]

    block.run_locally(
        images=[_FakeWorkflowImage("frame-1", width=8, height=8)],
        model_id="model",
        class_agnostic_nms=None,
        class_filter=["car"],
        confidence=0.4,
        iou_threshold=None,
        max_detections=None,
        max_candidates=None,
        mask_decode_mode="accurate",
        tradeoff_factor=None,
        disable_active_learning=None,
        active_learning_target_dataset=None,
        enforce_dense_masks_in_inference_models=False,
    )
    second_result = block.run_locally(
        images=[_FakeWorkflowImage("frame-2", width=4, height=4)],
        model_id="model",
        class_agnostic_nms=None,
        class_filter=["car"],
        confidence=0.4,
        iou_threshold=None,
        max_detections=None,
        max_candidates=None,
        mask_decode_mode="accurate",
        tradeoff_factor=None,
        disable_active_learning=None,
        active_learning_target_dataset=None,
        enforce_dense_masks_in_inference_models=False,
    )

    with pytest.raises(RuntimeError, match="image metadata"):
        second_result[0]["predictions"].result()


def test_instance_segmentation_stream_context_id_includes_frame_metadata() -> None:
    block = RoboflowInstanceSegmentationModelBlockV3(
        model_manager=SimpleNamespace(__contains__=lambda *_args, **_kwargs: False),
        api_key="api-key",
        step_execution_mode=StepExecutionMode.LOCAL,
    )
    image = _FakeWorkflowImage("frame-7", width=8, height=8, frame_number=7)

    context_id = block._build_stream_context_id(
        images=Batch.init(content=[image], indices=[(0,)]),
    )

    assert "video-frame-7" in context_id
    assert ":7:" in context_id


def test_close_stream_pipeline_detaches_response_executor_finalizer() -> None:
    block = RoboflowInstanceSegmentationModelBlockV3(
        model_manager=SimpleNamespace(__contains__=lambda *_args, **_kwargs: False),
        api_key="api-key",
        step_execution_mode=StepExecutionMode.LOCAL,
    )
    executor = block._get_stream_response_executor()

    assert block._stream_response_executor_finalizer is not None
    assert block._stream_response_executor_finalizer.alive

    block.close_stream_pipeline()

    assert block._stream_response_executor is None
    assert block._stream_response_executor_finalizer is None
    assert executor._shutdown


class _FakeExecutionGraph:
    def __init__(self, edges: dict) -> None:
        self._edges = edges

    def successors(self, node):
        return iter(self._edges.get(node, []))


def _attach_compiled_workflow_graph(engine, steps: dict, edges: dict) -> None:
    engine._engine = SimpleNamespace(
        _compiled_workflow=SimpleNamespace(
            steps=steps,
            execution_graph=_FakeExecutionGraph(edges=edges),
        )
    )


def test_wrap_activates_downstream_deferral_for_linear_workflow() -> None:
    # given: every non-pipelined step is downstream of the deferring step
    engine = _FakeDeferringExecutionEngine(stream_buffer_depth=1)
    _attach_compiled_workflow_graph(
        engine,
        steps={
            "detection": SimpleNamespace(step=engine.step),
            "tracker": SimpleNamespace(step=SimpleNamespace()),
        },
        edges={
            "$steps.detection": ["$steps.tracker"],
            "$steps.tracker": ["$outputs.tracked"],
        },
    )
    workflow_runner = WorkflowRunner(
        workflows_parameters=None,
        execution_engine=engine,
        image_input_name="image",
        video_metadata_input_name="video_metadata",
    )

    # when
    runner = wrap_workflow_runner_for_stream_pipeline(
        workflow_runner=workflow_runner,
        execution_engine=engine,
    )

    # then
    assert isinstance(runner, FlushEmittingPipelinedWorkflowRunner)


def test_wrap_falls_back_to_sequential_for_workflow_with_side_branch() -> None:
    # given: "other" is not downstream of the deferring detection step, so its
    # outputs would be missing from flush-emitted results
    engine = _FakeDeferringExecutionEngine(stream_buffer_depth=1)
    _attach_compiled_workflow_graph(
        engine,
        steps={
            "detection": SimpleNamespace(step=engine.step),
            "tracker": SimpleNamespace(step=SimpleNamespace()),
            "other": SimpleNamespace(step=SimpleNamespace()),
        },
        edges={
            "$steps.detection": ["$steps.tracker"],
            "$steps.tracker": ["$outputs.tracked"],
            "$steps.other": ["$outputs.other"],
        },
    )
    workflow_runner = WorkflowRunner(
        workflows_parameters=None,
        execution_engine=engine,
        image_input_name="image",
        video_metadata_input_name="video_metadata",
    )

    # when
    runner = wrap_workflow_runner_for_stream_pipeline(
        workflow_runner=workflow_runner,
        execution_engine=engine,
    )

    # then
    assert runner is workflow_runner


def test_wrap_activates_downstream_deferral_for_two_independent_models() -> None:
    # given: two independent deferring model steps with every other step
    # downstream of their union
    engine = _FakeDeferringExecutionEngine(stream_buffer_depth=1)
    second_step = _FakeDeferringStep(stream_buffer_depth=1)
    _attach_compiled_workflow_graph(
        engine,
        steps={
            "model_a": SimpleNamespace(step=engine.step),
            "model_b": SimpleNamespace(step=second_step),
            "consensus": SimpleNamespace(step=SimpleNamespace()),
            "tracker": SimpleNamespace(step=SimpleNamespace()),
        },
        edges={
            "$steps.model_a": ["$steps.consensus"],
            "$steps.model_b": ["$steps.consensus"],
            "$steps.consensus": ["$steps.tracker"],
            "$steps.tracker": ["$outputs.tracked"],
        },
    )
    workflow_runner = WorkflowRunner(
        workflows_parameters=None,
        execution_engine=engine,
        image_input_name="image",
        video_metadata_input_name="video_metadata",
    )

    # when
    runner = wrap_workflow_runner_for_stream_pipeline(
        workflow_runner=workflow_runner,
        execution_engine=engine,
    )

    # then
    assert isinstance(runner, FlushEmittingPipelinedWorkflowRunner)


def test_wrap_falls_back_to_sequential_for_chained_deferring_models() -> None:
    # given: model_b consumes model_a's output, so it would block on its
    # upstream future in the deferred pass
    engine = _FakeDeferringExecutionEngine(stream_buffer_depth=1)
    second_step = _FakeDeferringStep(stream_buffer_depth=1)
    _attach_compiled_workflow_graph(
        engine,
        steps={
            "model_a": SimpleNamespace(step=engine.step),
            "model_b": SimpleNamespace(step=second_step),
            "tracker": SimpleNamespace(step=SimpleNamespace()),
        },
        edges={
            "$steps.model_a": ["$steps.model_b"],
            "$steps.model_b": ["$steps.tracker"],
            "$steps.tracker": ["$outputs.tracked"],
        },
    )
    workflow_runner = WorkflowRunner(
        workflows_parameters=None,
        execution_engine=engine,
        image_input_name="image",
        video_metadata_input_name="video_metadata",
    )

    # when
    runner = wrap_workflow_runner_for_stream_pipeline(
        workflow_runner=workflow_runner,
        execution_engine=engine,
    )

    # then
    assert runner is workflow_runner


def test_wrap_falls_back_to_sequential_for_two_models_with_side_branch() -> None:
    # given: "other" is downstream of neither deferring model step
    engine = _FakeDeferringExecutionEngine(stream_buffer_depth=1)
    second_step = _FakeDeferringStep(stream_buffer_depth=1)
    _attach_compiled_workflow_graph(
        engine,
        steps={
            "model_a": SimpleNamespace(step=engine.step),
            "model_b": SimpleNamespace(step=second_step),
            "consensus": SimpleNamespace(step=SimpleNamespace()),
            "other": SimpleNamespace(step=SimpleNamespace()),
        },
        edges={
            "$steps.model_a": ["$steps.consensus"],
            "$steps.model_b": ["$steps.consensus"],
            "$steps.consensus": ["$outputs.consensus"],
            "$steps.other": ["$outputs.other"],
        },
    )
    workflow_runner = WorkflowRunner(
        workflows_parameters=None,
        execution_engine=engine,
        image_input_name="image",
        video_metadata_input_name="video_metadata",
    )

    # when
    runner = wrap_workflow_runner_for_stream_pipeline(
        workflow_runner=workflow_runner,
        execution_engine=engine,
    )

    # then
    assert runner is workflow_runner


def test_flush_emitting_runner_flush_drains_frames_with_two_stream_steps() -> None:
    # given
    engine = _FakeDeferringExecutionEngine(stream_buffer_depth=1)
    second_step = _FakeDeferringStep(stream_buffer_depth=1)
    _attach_compiled_workflow_graph(
        engine,
        steps={
            "model_a": SimpleNamespace(step=engine.step),
            "model_b": SimpleNamespace(step=second_step),
            "consensus": SimpleNamespace(step=SimpleNamespace()),
        },
        edges={
            "$steps.model_a": ["$steps.consensus"],
            "$steps.model_b": ["$steps.consensus"],
            "$steps.consensus": ["$outputs.consensus"],
        },
    )
    workflow_runner = WorkflowRunner(
        workflows_parameters=None,
        execution_engine=engine,
        image_input_name="image",
        video_metadata_input_name="video_metadata",
    )
    runner = wrap_workflow_runner_for_stream_pipeline(
        workflow_runner=workflow_runner,
        execution_engine=engine,
    )
    assert isinstance(runner, FlushEmittingPipelinedWorkflowRunner)
    frame_1 = _make_frame(1)
    frame_2 = _make_frame(2)

    # when
    first_result = runner([frame_1])
    second_result = runner([frame_2])
    flushed_results = runner.flush()

    # then - flushing with two stream steps must not raise and drains the
    # pending frame via the engine flush path
    assert first_result is None
    assert second_result is not None
    assert second_result.predictions == [{"result": "flushed-frame-1"}]
    assert second_result.video_frames == [frame_1]
    assert flushed_results is not None
    assert len(flushed_results) == 1
    assert flushed_results[0].predictions == [{"result": "flushed-frame-2"}]
    assert flushed_results[0].video_frames == [frame_2]
    assert engine.flush_calls == [1, 2]
    runner.close()
    assert engine.step.close_calls == 1
    assert second_step.close_calls == 1


def test_base_pipelined_runner_flush_still_rejects_two_stream_steps() -> None:
    # given: two pipelined but non-deferring steps keep the base runner, whose
    # flush path supports exactly one pipelined step (regression guard)
    engine = _FakeDeferringExecutionEngine(stream_buffer_depth=1, defers=False)
    second_step = _FakeDeferringStep(stream_buffer_depth=1, defers=False)
    _attach_compiled_workflow_graph(
        engine,
        steps={
            "model_a": SimpleNamespace(step=engine.step),
            "model_b": SimpleNamespace(step=second_step),
        },
        edges={
            "$steps.model_a": ["$outputs.predictions_a"],
            "$steps.model_b": ["$outputs.predictions_b"],
        },
    )
    workflow_runner = WorkflowRunner(
        workflows_parameters=None,
        execution_engine=engine,
        image_input_name="image",
        video_metadata_input_name="video_metadata",
    )
    runner = wrap_workflow_runner_for_stream_pipeline(
        workflow_runner=workflow_runner,
        execution_engine=engine,
    )
    assert type(runner) is PipelinedWorkflowRunner

    # when
    first_result = runner([_make_frame(1)])

    # then
    assert first_result is None
    with pytest.raises(RuntimeError):
        runner.flush()

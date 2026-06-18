from concurrent.futures import Future
from datetime import datetime
from types import SimpleNamespace
from typing import Optional

import numpy as np
import pytest

from inference.core.interfaces.camera.entities import VideoFrame
from inference.core.interfaces.stream.model_handlers.workflows import (
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
        self.source_infos = []

    def infer_from_request_sync(self, model_id: str, request):
        self.infer_calls += 1
        self.source_infos.append(request.source_info)
        if self.infer_calls == 1:
            return [_FakeResponse("priming", width=8, height=8)]
        if self.mode == "previous":
            return [
                _make_async_placeholder(
                    "first-final",
                    context_id=self.source_infos[0],
                    response_width=8,
                    response_height=8,
                )
            ]
        if self.mode == "current-with-old-size":
            return [
                _make_async_placeholder(
                    "first-final",
                    context_id=request.source_info,
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
    assert len(manager.source_infos) == 2
    assert manager.source_infos[0] != manager.source_infos[1]


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

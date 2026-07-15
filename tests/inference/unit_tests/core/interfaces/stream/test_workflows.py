from concurrent.futures import Future
from datetime import datetime
from types import SimpleNamespace
from typing import List, Literal, Optional

import networkx as nx
import numpy as np
import pytest

from inference.core.interfaces.camera.entities import VideoFrame
from inference.core.interfaces.stream.model_handlers import (
    workflows as workflows_module,
)
from inference.core.interfaces.stream.model_handlers.workflows import (
    LookaheadPipelinedWorkflowRunner,
    PipelinedWorkflowRunner,
    StreamLookaheadDrainError,
    WorkflowRunner,
    wrap_workflow_runner_for_stream_pipeline,
)
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.models.roboflow.instance_segmentation.v3 import (
    RoboflowInstanceSegmentationModelBlockV3,
)
from inference.core.workflows.execution_engine.constants import (
    NODE_COMPILATION_OUTPUT_PROPERTY,
)
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    OutputDefinition,
    VideoMetadata,
)
from inference.core.workflows.execution_engine.v1.compiler.entities import NodeCategory
from inference.core.workflows.prototypes.block import WorkflowBlockManifest
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


# --- Stream lookahead (engine-offloaded async steps) ---


class _AsyncModelManifest(WorkflowBlockManifest):
    type: Literal["test/async_model@v1"]
    name: str

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="predictions")]

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["images"]

    @classmethod
    def is_stateful_for_video_processing(cls) -> bool:
        return False


class _StatefulStepManifest(WorkflowBlockManifest):
    type: Literal["test/stateful_step@v1"]
    name: str

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="consumed")]


class _FakeAsyncStep:
    def is_async_stream_step(self) -> bool:
        return True


class _FakeLookaheadExecutionEngine:
    def __init__(self) -> None:
        self._engine = SimpleNamespace(_compiled_workflow=SimpleNamespace(steps={}))
        self.lookahead_calls = []
        self.edms = []
        self.resume_calls = []
        self.fail_resume_for_frames = set()

    def run_stream_lookahead(
        self,
        runtime_parameters,
        frontier_step_selectors,
        async_step_selectors,
        lookahead_executor,
    ):
        frame_number = runtime_parameters["image"][0]["video_metadata"].frame_number
        self.lookahead_calls.append(
            {
                "frame_number": frame_number,
                "frontier_step_selectors": set(frontier_step_selectors),
                "async_step_selectors": set(async_step_selectors),
                "lookahead_executor": lookahead_executor,
            }
        )
        execution_data_manager = SimpleNamespace(frame_number=frame_number)
        self.edms.append(execution_data_manager)
        return execution_data_manager

    def resume_stream_lookahead(
        self,
        execution_data_manager,
        frontier_step_selectors,
        serialize_results,
        fps,
        _is_preview,
    ):
        self.resume_calls.append(execution_data_manager)
        if execution_data_manager.frame_number in self.fail_resume_for_frames:
            raise RuntimeError(
                f"resume failed for frame {execution_data_manager.frame_number}"
            )
        return [{"result": f"resumed-frame-{execution_data_manager.frame_number}"}]


def _initialised_step(step, manifest_class) -> SimpleNamespace:
    return SimpleNamespace(
        step=step,
        block_specification=SimpleNamespace(manifest_class=manifest_class),
    )


def _compiled_workflow_namespace(steps: dict, edges: dict) -> SimpleNamespace:
    graph = nx.DiGraph()
    for step_name in steps:
        graph.add_node(
            f"$steps.{step_name}",
            **{
                NODE_COMPILATION_OUTPUT_PROPERTY: SimpleNamespace(
                    node_category=NodeCategory.STEP_NODE
                )
            },
        )
    for edge_start, edge_ends in edges.items():
        for edge_end in edge_ends:
            if edge_end not in graph.nodes:
                graph.add_node(
                    edge_end,
                    **{
                        NODE_COMPILATION_OUTPUT_PROPERTY: SimpleNamespace(
                            node_category=NodeCategory.OUTPUT_NODE
                        )
                    },
                )
            graph.add_edge(edge_start, edge_end)
    return SimpleNamespace(steps=steps, execution_graph=graph)


def _make_wrap_case_engine(case: str) -> SimpleNamespace:
    if case == "async_linear":
        compiled_workflow = _compiled_workflow_namespace(
            steps={
                "model": _initialised_step(_FakeAsyncStep(), _AsyncModelManifest),
                "tracker": _initialised_step(SimpleNamespace(), _StatefulStepManifest),
            },
            edges={
                "$steps.model": ["$steps.tracker"],
                "$steps.tracker": ["$outputs.tracked"],
            },
        )
    elif case == "rfdetr_only":
        compiled_workflow = _compiled_workflow_namespace(
            steps={
                "segmentation": _initialised_step(
                    _FakePipelinedStep(stream_buffer_depth=1), _StatefulStepManifest
                ),
            },
            edges={"$steps.segmentation": ["$outputs.predictions"]},
        )
    elif case == "mixed":
        compiled_workflow = _compiled_workflow_namespace(
            steps={
                "model": _initialised_step(_FakeAsyncStep(), _AsyncModelManifest),
                "segmentation": _initialised_step(
                    _FakePipelinedStep(stream_buffer_depth=1), _StatefulStepManifest
                ),
            },
            edges={
                "$steps.model": ["$outputs.model"],
                "$steps.segmentation": ["$outputs.predictions"],
            },
        )
    elif case == "no_graph":
        compiled_workflow = SimpleNamespace(
            steps={"model": _initialised_step(_FakeAsyncStep(), _AsyncModelManifest)}
        )
    elif case == "stateful_fed_async":
        compiled_workflow = _compiled_workflow_namespace(
            steps={
                "stateful_pre": _initialised_step(
                    SimpleNamespace(), _StatefulStepManifest
                ),
                "model": _initialised_step(_FakeAsyncStep(), _AsyncModelManifest),
            },
            edges={
                "$steps.stateful_pre": ["$steps.model"],
                "$steps.model": ["$outputs.predictions"],
            },
        )
    else:
        raise ValueError(f"Unknown wrap case: {case}")
    return SimpleNamespace(
        _engine=SimpleNamespace(_compiled_workflow=compiled_workflow)
    )


@pytest.mark.parametrize(
    "case, lookahead_depth, allow_lookahead, expected",
    [
        ("async_linear", 4, True, "lookahead"),
        ("rfdetr_only", 4, True, "pipelined"),
        ("mixed", 4, True, "fallback"),
        ("async_linear", 1, True, "fallback"),
        ("async_linear", 4, False, "fallback"),
        ("no_graph", 4, True, "fallback"),
        ("stateful_fed_async", 4, True, "fallback"),
    ],
    ids=[
        "async_in_frontier_activates",
        "rfdetr_only_keeps_pipelined_runner",
        "mixed_async_and_pipelined_falls_back",
        "lookahead_depth_disabled_falls_back",
        "lookahead_disallowed_falls_back",
        "missing_execution_graph_falls_back",
        "async_fed_by_stateful_step_falls_back",
    ],
)
def test_wrap_selects_runner_for_workflow_shape_and_configuration(
    monkeypatch,
    case: str,
    lookahead_depth: int,
    allow_lookahead: bool,
    expected: str,
) -> None:
    # given
    monkeypatch.setattr(
        workflows_module, "WORKFLOWS_STREAM_LOOKAHEAD_DEPTH", lookahead_depth
    )
    engine = _make_wrap_case_engine(case)
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
        allow_lookahead=allow_lookahead,
    )

    # then
    if expected == "lookahead":
        assert isinstance(runner, LookaheadPipelinedWorkflowRunner)
        runner.close()
    elif expected == "pipelined":
        assert type(runner) is PipelinedWorkflowRunner
    else:
        assert runner is workflow_runner


def test_lookahead_runner_emits_frames_in_order_from_their_own_execution_state() -> (
    None
):
    # given
    engine = _FakeLookaheadExecutionEngine()
    workflow_runner = WorkflowRunner(
        workflows_parameters=None,
        execution_engine=engine,
        image_input_name="image",
        video_metadata_input_name="video_metadata",
    )
    runner = LookaheadPipelinedWorkflowRunner(
        workflow_runner=workflow_runner,
        frontier_step_selectors={"$steps.model"},
        async_step_selectors={"$steps.model"},
        buffer_depth=2,
    )
    frames = [_make_frame(frame_id) for frame_id in range(1, 5)]

    # when
    results = [runner([frame]) for frame in frames]
    flushed_results = runner.flush()

    # then - frames 1..4 at buffer depth 2 emit [1, 2], flush yields [3, 4],
    # each emission resuming ITS OWN buffered execution state, in frame order
    assert results[0] is None
    assert results[1] is None
    assert results[2].predictions == [{"result": "resumed-frame-1"}]
    assert results[2].video_frames == [frames[0]]
    assert results[3].predictions == [{"result": "resumed-frame-2"}]
    assert results[3].video_frames == [frames[1]]
    assert [result.video_frames for result in flushed_results] == [
        [frames[2]],
        [frames[3]],
    ]
    assert flushed_results[0].predictions == [{"result": "resumed-frame-3"}]
    assert flushed_results[1].predictions == [{"result": "resumed-frame-4"}]
    assert [edm.frame_number for edm in engine.resume_calls] == [1, 2, 3, 4]
    assert all(resumed is edm for resumed, edm in zip(engine.resume_calls, engine.edms))
    assert engine.lookahead_calls[0]["frontier_step_selectors"] == {"$steps.model"}
    assert engine.lookahead_calls[0]["async_step_selectors"] == {"$steps.model"}
    assert engine.lookahead_calls[0]["lookahead_executor"] is runner._lookahead_executor

    # then - close shuts down the runner-owned lookahead pool
    runner.close()
    assert runner._lookahead_executor._shutdown


def test_lookahead_runner_flush_skips_failing_frame_and_drains_the_rest() -> None:
    # given
    engine = _FakeLookaheadExecutionEngine()
    workflow_runner = WorkflowRunner(
        workflows_parameters=None,
        execution_engine=engine,
        image_input_name="image",
        video_metadata_input_name="video_metadata",
    )
    runner = LookaheadPipelinedWorkflowRunner(
        workflow_runner=workflow_runner,
        frontier_step_selectors={"$steps.model"},
        async_step_selectors={"$steps.model"},
        buffer_depth=2,
    )
    frame_1 = _make_frame(1)
    frame_2 = _make_frame(2)
    assert runner([frame_1]) is None
    assert runner([frame_2]) is None
    engine.fail_resume_for_frames.add(1)

    # when - the drain still emits the remaining frames, then re-raises so
    # a failed tail frame cannot disappear as a normal empty drain
    with pytest.raises(StreamLookaheadDrainError) as error:
        runner.flush()

    # then
    drained_results = error.value.drained_results
    assert len(drained_results) == 1
    assert drained_results[0].predictions == [{"result": "resumed-frame-2"}]
    assert drained_results[0].video_frames == [frame_2]
    assert isinstance(error.value.__cause__, RuntimeError)
    assert [edm.frame_number for edm in engine.resume_calls] == [1, 2]
    runner.close()


def test_lookahead_runner_pool_threads_are_daemon() -> None:
    # given - a hung remote request must not keep the interpreter alive
    # after close() (close() uses shutdown(wait=False))
    engine = _FakeLookaheadExecutionEngine()
    workflow_runner = WorkflowRunner(
        workflows_parameters=None,
        execution_engine=engine,
        image_input_name="image",
        video_metadata_input_name="video_metadata",
    )
    runner = LookaheadPipelinedWorkflowRunner(
        workflow_runner=workflow_runner,
        frontier_step_selectors={"$steps.model"},
        async_step_selectors={"$steps.model"},
        buffer_depth=2,
    )

    # when / then - every pool worker exists already and is a daemon thread
    pool_threads = runner._lookahead_executor._threads
    assert len(pool_threads) == 3  # (buffer_depth + 1) * len(async_steps)
    assert all(thread.daemon for thread in pool_threads)

    # after close(), the workers are detached from concurrent.futures'
    # interpreter-shutdown hook (which joins pool threads regardless of the
    # daemon flag) - both halves are needed for a hung request not to block
    # process exit
    runner.close()
    from concurrent.futures.thread import _threads_queues

    assert all(thread not in _threads_queues for thread in pool_threads)

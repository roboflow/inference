"""Unit tests for inference.core.models.inference_models_adapters."""

from collections import deque
from concurrent.futures import Future
from types import SimpleNamespace

import pytest
import torch

from inference.core.entities.responses.inference import (
    InstanceSegmentationInferenceResponse,
    InstanceSegmentationInferenceResponseDC,
)
from inference.core.exceptions import PostProcessingError
from inference.core.models.inference_models_adapters import (
    InferenceModelsDepthEstimationAdapter,
    InferenceModelsInstanceSegmentationAdapter,
    InferenceModelsObjectDetectionAdapter,
    _supports_independent_stage_execution,
    prepare_classification_response,
    prepare_multi_label_classification_response,
)
from inference_models import (
    ClassificationPrediction,
    InstanceDetections,
    MultiLabelClassificationPrediction,
)
from inference_models.models.base.async_handoff import attach_adapter_mapped_kwargs


class _ImmediateExecutor:
    def submit(self, fn, *args, **kwargs) -> Future:
        future = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except BaseException as error:  # pragma: no cover - defensive
            future.set_exception(error)
        return future


class _FakePipelineFuture:
    def __init__(self, name: str, ops: list[str]):
        self.name = name
        self.ops = ops
        self.submitted_meta = []
        self._adapter_gpu_work_submitted = False

    def submit_gpu_work(self, meta=None) -> None:
        self.ops.append(f"submit:{self.name}")
        self.submitted_meta.append(meta)

    def result(self):
        assert self.submitted_meta, f"result() called before submit for {self.name}"
        self.ops.append(f"result:{self.name}")
        return [SimpleNamespace(name=self.name)]

    def done(self) -> bool:
        return bool(self.submitted_meta)


class _FakePipelineModel:
    supported_mask_formats = {"rle"}

    def __init__(self, futures: list[_FakePipelineFuture], ops: list[str]):
        self._futures = deque(futures)
        self._ops = ops
        self.forward_calls = []
        self.post_process_calls = []

    def forward(self, img_in, **kwargs):
        self.forward_calls.append((img_in, kwargs))
        return "sync-raw"

    def forward_async(self, _img_in, _meta, **_kwargs):
        future = self._futures.popleft()
        self._ops.append(f"forward:{future.name}")
        return future

    def post_process(self, predictions, meta, **kwargs):
        self.post_process_calls.append((predictions, meta, kwargs))
        return ["sync-detections"]


class _FakeObjectDetectionModel:
    def __init__(self) -> None:
        self.pre_process_calls = []

    def pre_process(self, images, **kwargs):
        self.pre_process_calls.append((images, kwargs))
        return "preprocessed", "metadata"


class _FakeIndependentStageObjectDetectionModel(_FakeObjectDetectionModel):
    def pre_process(
        self,
        images,
        independent_stage_execution: bool = True,
        **kwargs,
    ):
        kwargs["independent_stage_execution"] = independent_stage_execution
        return super().pre_process(images, **kwargs)


def _make_meta(tag: str):
    return [
        SimpleNamespace(
            tag=tag,
            original_size=SimpleNamespace(width=10, height=20),
        )
    ]


@pytest.mark.parametrize(
    ("model", "expected_flag"),
    [
        (_FakeObjectDetectionModel(), None),
        (_FakeIndependentStageObjectDetectionModel(), False),
    ],
)
def test_object_detection_preprocess_only_disables_independent_execution_for_models_that_support_it(
    model,
    expected_flag,
) -> None:
    adapter = object.__new__(InferenceModelsObjectDetectionAdapter)
    adapter._model = model
    adapter._preprocess_supports_independent_stage_execution = (
        _supports_independent_stage_execution(model.pre_process)
    )

    result = adapter.preprocess(torch.zeros((8, 9, 3), dtype=torch.uint8).numpy())

    assert result == ("preprocessed", "metadata")
    _, call_kwargs = model.pre_process_calls[0]
    if expected_flag is None:
        assert "independent_stage_execution" not in call_kwargs
    else:
        assert call_kwargs["independent_stage_execution"] is expected_flag


def _make_pipeline_adapter(
    futures: list[_FakePipelineFuture],
    ops: list[str],
    pipeline_depth: int = 2,
) -> InferenceModelsInstanceSegmentationAdapter:
    adapter = object.__new__(InferenceModelsInstanceSegmentationAdapter)
    adapter._pipeline_depth = pipeline_depth
    adapter._response_delay = max(1, pipeline_depth - 1)
    adapter._pending_gpu_submissions = deque()
    adapter._pending_futures = deque()
    adapter._response_futures = deque()
    adapter._response_executor = None
    adapter._response_executor_finalizer = None
    adapter._model = _FakePipelineModel(futures=futures, ops=ops)
    adapter.class_names = []
    adapter.map_inference_kwargs = lambda kwargs: dict(kwargs)
    adapter._get_response_executor = lambda: _ImmediateExecutor()
    adapter._build_responses_from_detections = (
        lambda _detections, preprocess_return_metadata, **_kwargs: [
            preprocess_return_metadata[0].tag
        ]
    )

    return adapter


def _make_live_pipeline_adapter(
    pipeline_depth: int = 2,
) -> InferenceModelsInstanceSegmentationAdapter:
    adapter = object.__new__(InferenceModelsInstanceSegmentationAdapter)
    adapter._pipeline_depth = pipeline_depth
    adapter._response_delay = max(1, pipeline_depth - 1)
    adapter._pending_gpu_submissions = deque()
    adapter._pending_futures = deque()
    adapter._response_futures = deque()
    adapter._response_executor = None
    adapter._response_executor_finalizer = None
    adapter._gpu_submit_generation = 0
    adapter._model = _FakePipelineModel(futures=[], ops=[])
    adapter.class_names = []
    return adapter


def test_get_response_executor_registers_shutdown_finalizer() -> None:
    adapter = _make_live_pipeline_adapter()
    executor = InferenceModelsInstanceSegmentationAdapter._get_response_executor(
        adapter
    )

    assert adapter._response_executor is executor
    assert adapter._response_executor_finalizer is not None
    assert adapter._response_executor_finalizer.alive

    adapter.shutdown_pipeline()

    assert adapter._response_executor is None
    assert adapter._response_executor_finalizer is None
    assert executor._shutdown


def test_response_executor_finalizer_shuts_down_when_adapter_is_collected() -> None:
    import gc
    import weakref

    adapter = _make_live_pipeline_adapter()
    executor = InferenceModelsInstanceSegmentationAdapter._get_response_executor(
        adapter
    )
    assert not executor._shutdown

    adapter_ref = weakref.ref(adapter)
    del adapter
    gc.collect()

    assert adapter_ref() is None
    assert executor._shutdown


def test_pipeline_depth_falls_back_to_one_for_unsupported_models(monkeypatch) -> None:
    monkeypatch.setattr(
        "inference.core.models.inference_models_adapters.get_rfdetr_pipeline_depth",
        lambda: 2,
    )
    adapter = object.__new__(InferenceModelsInstanceSegmentationAdapter)
    adapter._model = SimpleNamespace(supports_stream_pipeline=False)

    assert adapter._resolve_pipeline_depth() == 1


def test_pipeline_depth_caps_requested_depth_for_supported_models(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "inference.core.models.inference_models_adapters.get_rfdetr_pipeline_depth",
        lambda: 3,
    )
    adapter = object.__new__(InferenceModelsInstanceSegmentationAdapter)
    adapter._model = SimpleNamespace(supports_stream_pipeline=True)

    assert adapter._resolve_pipeline_depth() == 2


def test_pipeline_uses_sync_forward_for_batched_requests() -> None:
    ops: list[str] = []
    future = _FakePipelineFuture(name="f1", ops=ops)
    adapter = _make_pipeline_adapter(
        futures=[future],
        ops=ops,
        pipeline_depth=2,
    )
    img_in = SimpleNamespace(_pre_processing_meta=[object(), object()])

    result = adapter.predict(img_in, response_mask_format="dense")

    assert result == "sync-raw"
    assert ops == []
    assert len(adapter._model.forward_calls) == 1


def test_pipeline_postprocess_uses_sync_path_for_non_future_predictions() -> None:
    ops: list[str] = []
    adapter = _make_pipeline_adapter(
        futures=[],
        ops=ops,
        pipeline_depth=2,
    )

    responses = adapter.postprocess(
        "sync-raw",
        _make_meta("meta-1"),
        response_mask_format="dense",
    )

    assert responses == ["meta-1"]
    assert len(adapter._model.post_process_calls) == 1
    assert adapter._model.post_process_calls[0][0] == "sync-raw"


def test_workflow_response_fast_dataclass_path_is_disabled_at_depth_one() -> None:
    adapter = object.__new__(InferenceModelsInstanceSegmentationAdapter)
    adapter._pipeline_depth = 1
    adapter.class_names = ["car"]
    metadata = [SimpleNamespace(original_size=SimpleNamespace(width=4, height=4))]
    detections = [
        InstanceDetections(
            xyxy=torch.tensor([[1, 1, 3, 3]], dtype=torch.int32),
            confidence=torch.tensor([0.9], dtype=torch.float32),
            class_id=torch.tensor([0], dtype=torch.int32),
            mask=torch.zeros((1, 4, 4), dtype=torch.uint8),
        )
    ]

    responses = adapter._build_responses_from_detections(
        detections,
        metadata,
        source="workflow-execution",
    )

    assert isinstance(responses[0], InstanceSegmentationInferenceResponse)


def test_workflow_response_fast_dataclass_path_is_enabled_above_depth_one() -> None:
    adapter = object.__new__(InferenceModelsInstanceSegmentationAdapter)
    adapter._pipeline_depth = 2
    adapter.class_names = ["car"]
    metadata = [SimpleNamespace(original_size=SimpleNamespace(width=4, height=4))]
    detections = [
        InstanceDetections(
            xyxy=torch.tensor([[1, 1, 3, 3]], dtype=torch.int32),
            confidence=torch.tensor([0.9], dtype=torch.float32),
            class_id=torch.tensor([0], dtype=torch.int32),
            mask=torch.zeros((1, 4, 4), dtype=torch.uint8),
        )
    ]

    responses = adapter._build_responses_from_detections(
        detections,
        metadata,
        source="workflow-execution",
    )

    assert isinstance(responses[0], InstanceSegmentationInferenceResponseDC)


def test_prepare_multi_label_response_uses_class_ids_for_predicted_classes() -> None:
    """The model's `post_process` is the source of truth for which classes
    are "predicted" (it owns the priority chain user → per-class → global
    → default). The response builder reads `prediction.class_ids` directly
    rather than re-thresholding the full confidence vector, so per-class
    refinement makes it through to the API response.

    The `confidence` field is the FULL per-class score vector — used to
    populate the per-class scores dict for UI display, but not as a filter.
    """
    class_names = ["a", "b", "c", "d"]
    confidence = torch.tensor([0.1, 0.2, 0.85, 0.9])
    # Note: only "c" is in class_ids even though "d" also has a high score.
    # This simulates the model's per-class filter dropping "d" because its
    # per-class threshold (e.g. 0.95) wasn't met. The response must respect
    # that decision and NOT add "d" to predicted_classes.
    class_ids = torch.tensor([2], dtype=torch.long)
    prediction = MultiLabelClassificationPrediction(
        class_ids=class_ids,
        confidence=confidence,
    )

    results = prepare_multi_label_classification_response(
        post_processed_predictions=[prediction],
        image_sizes=[(10, 20)],
        class_names=class_names,
    )

    assert len(results) == 1
    r = results[0]
    # All classes appear in the per-class scores dict regardless of threshold.
    assert r.predictions["a"].confidence == pytest.approx(0.1)
    assert r.predictions["b"].confidence == pytest.approx(0.2)
    assert r.predictions["c"].confidence == pytest.approx(0.85)
    assert r.predictions["d"].confidence == pytest.approx(0.9)
    # Only the model's filtered class_ids show up in predicted_classes.
    assert r.predicted_classes == ["c"]


def test_prepare_classification_response_flattens_singleton_output_dimensions() -> None:
    class_names = ["cat", "dog"]
    prediction = ClassificationPrediction(
        class_id=torch.tensor([[1]], dtype=torch.long),
        confidence=torch.tensor([[[0.1, 0.9]]]),
    )

    results = prepare_classification_response(
        post_processed_predictions=prediction,
        image_sizes=[(10, 20)],
        class_names=class_names,
        confidence_threshold=0.0,
    )

    assert len(results) == 1
    assert results[0].top == "dog"
    assert results[0].confidence == pytest.approx(0.9)
    assert [p.class_name for p in results[0].predictions] == ["dog", "cat"]


def test_prepare_classification_response_fails_on_class_count_mismatch() -> None:
    prediction = ClassificationPrediction(
        class_id=torch.tensor([0], dtype=torch.long),
        confidence=torch.tensor([[0.7]]),
    )

    with pytest.raises(PostProcessingError, match="class names metadata"):
        prepare_classification_response(
            post_processed_predictions=prediction,
            image_sizes=[(10, 20)],
            class_names=["cat", "dog"],
            confidence_threshold=0.0,
        )


def test_pipeline_submits_previous_future_before_next_forward() -> None:
    ops: list[str] = []
    future_1 = _FakePipelineFuture(name="f1", ops=ops)
    future_2 = _FakePipelineFuture(name="f2", ops=ops)
    adapter = _make_pipeline_adapter(
        futures=[future_1, future_2],
        ops=ops,
        pipeline_depth=2,
    )

    meta_1 = _make_meta("meta-1")
    prediction_1 = adapter.predict("frame-1", response_mask_format="dense")
    priming = adapter.postprocess(
        prediction_1,
        meta_1,
        response_mask_format="dense",
    )

    assert len(priming) == 1
    assert future_1.submitted_meta == []

    adapter.predict("frame-2", response_mask_format="dense")

    assert future_1.submitted_meta == [meta_1]
    assert ops == ["forward:f1", "submit:f1", "forward:f2"]


def test_pipeline_returns_previous_frame_response_using_previous_metadata() -> None:
    ops: list[str] = []
    future_1 = _FakePipelineFuture(name="f1", ops=ops)
    future_2 = _FakePipelineFuture(name="f2", ops=ops)
    adapter = _make_pipeline_adapter(
        futures=[future_1, future_2],
        ops=ops,
        pipeline_depth=2,
    )

    meta_1 = _make_meta("meta-1")
    meta_2 = _make_meta("meta-2")
    prediction_1 = adapter.predict("frame-1", response_mask_format="dense")
    adapter.postprocess(prediction_1, meta_1, response_mask_format="dense")

    prediction_2 = adapter.predict("frame-2", response_mask_format="dense")
    responses = adapter.postprocess(
        prediction_2,
        meta_2,
        response_mask_format="dense",
    )

    assert responses == ["meta-1"]
    assert future_1.submitted_meta == [meta_1]
    assert future_2.submitted_meta == [meta_2]
    assert ops == [
        "forward:f1",
        "submit:f1",
        "forward:f2",
        "submit:f2",
        "result:f1",
    ]


def test_pipeline_flush_submits_remaining_gpu_work_before_finalizing() -> None:
    ops: list[str] = []
    future_1 = _FakePipelineFuture(name="f1", ops=ops)
    adapter = _make_pipeline_adapter(
        futures=[future_1],
        ops=ops,
        pipeline_depth=2,
    )

    meta_1 = _make_meta("meta-1")
    prediction_1 = adapter.predict("frame-1", response_mask_format="dense")
    adapter.postprocess(prediction_1, meta_1, response_mask_format="dense")

    responses = adapter.flush()

    assert responses == ["meta-1"]
    assert future_1.submitted_meta == [meta_1]
    assert ops == ["forward:f1", "submit:f1", "result:f1"]


def test_pipeline_depth_three_submits_oldest_pending_before_forward() -> None:
    ops: list[str] = []
    future_1 = _FakePipelineFuture(name="f1", ops=ops)
    future_2 = _FakePipelineFuture(name="f2", ops=ops)
    future_3 = _FakePipelineFuture(name="f3", ops=ops)
    adapter = _make_pipeline_adapter(
        futures=[future_1, future_2, future_3],
        ops=ops,
        pipeline_depth=3,
    )

    meta_1 = _make_meta("meta-1")
    meta_2 = _make_meta("meta-2")
    meta_3 = _make_meta("meta-3")

    prediction_1 = adapter.predict("frame-1", response_mask_format="dense")
    adapter.postprocess(prediction_1, meta_1, response_mask_format="dense")

    prediction_2 = adapter.predict("frame-2", response_mask_format="dense")
    priming = adapter.postprocess(prediction_2, meta_2, response_mask_format="dense")

    prediction_3 = adapter.predict("frame-3", response_mask_format="dense")
    responses = adapter.postprocess(
        prediction_3,
        meta_3,
        response_mask_format="dense",
    )

    assert len(priming) == 1
    assert responses == ["meta-1"]
    assert future_1.submitted_meta == [meta_1]
    assert future_2.submitted_meta == [meta_2]
    assert future_3.submitted_meta == [meta_3]
    assert ops == [
        "forward:f1",
        "submit:f1",
        "forward:f2",
        "submit:f2",
        "forward:f3",
        "submit:f3",
        "result:f1",
    ]


def test_pipeline_flush_raises_on_response_future_timeout(monkeypatch) -> None:
    monkeypatch.setattr(
        "inference.core.models.inference_models_adapters.WORKFLOWS_ASYNC_FUTURE_RESULT_TIMEOUT",
        0.001,
    )
    ops: list[str] = []
    adapter = _make_pipeline_adapter(futures=[], ops=ops, pipeline_depth=2)
    hung_future = Future()
    adapter._response_futures.append((hung_future, None))

    with pytest.raises(RuntimeError, match="Timed out while waiting for"):
        adapter.flush()


def test_depth_estimation_adapter_normalization_matches_depth_anything_convention() -> None:
    """DepthAnything models serve disparity-style relative depth - larger means
    closer (see the "Flip to be consistent with V2" step in
    DepthAnythingV3Torch.forward). The metric-depth adapter must invert while
    normalizing so metric models are true drop-ins: nearest pixel -> 1.0,
    farthest -> 0.0."""
    import numpy as np

    adapter = InferenceModelsDepthEstimationAdapter.__new__(
        InferenceModelsDepthEstimationAdapter
    )
    adapter._model = lambda inputs: [torch.tensor([[1.0, 2.0], [3.0, 4.0]])]

    (result,) = adapter.predict(np.zeros((2, 2, 3), dtype=np.uint8))

    expected = np.array([[1.0, 2 / 3], [1 / 3, 0.0]], dtype=np.float32)
    assert np.allclose(result["normalized_depth"], expected, atol=1e-6)

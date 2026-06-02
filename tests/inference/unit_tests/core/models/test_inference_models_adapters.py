"""Unit tests for inference.core.models.inference_models_adapters."""

from collections import deque
from concurrent.futures import Future
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from inference.core.exceptions import PostProcessingError
from inference.core.models import inference_models_adapters as adapters_module
from inference.core.models.inference_models_adapters import (
    InferenceModelsInstanceSegmentationAdapter,
    prepare_classification_response,
    prepare_multi_label_classification_response,
)
from inference_models import (
    ClassificationPrediction,
    MultiLabelClassificationPrediction,
)


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

    def forward_async(self, _img_in, _meta, **_kwargs):
        future = self._futures.popleft()
        self._ops.append(f"forward:{future.name}")
        return future


def _make_meta(tag: str):
    return [
        SimpleNamespace(
            tag=tag,
            original_size=SimpleNamespace(width=10, height=20),
        )
    ]


def _make_pipeline_adapter(
    monkeypatch: pytest.MonkeyPatch,
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
    adapter._model = _FakePipelineModel(futures=futures, ops=ops)
    adapter.class_names = []
    adapter.map_inference_kwargs = lambda kwargs: dict(kwargs)
    adapter._get_response_executor = lambda: _ImmediateExecutor()
    adapter._build_responses_from_detections = (
        lambda _detections, preprocess_return_metadata, **_kwargs: [
            preprocess_return_metadata[0].tag
        ]
    )

    monkeypatch.setattr(adapters_module, "nsight_current_frame_id", lambda: 0)
    monkeypatch.setattr(adapters_module, "nsight_frame_label", lambda *_args: "trace")
    monkeypatch.setattr(adapters_module, "nsight_mark", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        adapters_module, "nsight_range", lambda *_args, **_kwargs: nullcontext()
    )
    return adapter


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


def test_pipeline_submits_previous_future_before_next_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ops: list[str] = []
    future_1 = _FakePipelineFuture(name="f1", ops=ops)
    future_2 = _FakePipelineFuture(name="f2", ops=ops)
    adapter = _make_pipeline_adapter(
        monkeypatch=monkeypatch,
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


def test_pipeline_returns_previous_frame_response_using_previous_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ops: list[str] = []
    future_1 = _FakePipelineFuture(name="f1", ops=ops)
    future_2 = _FakePipelineFuture(name="f2", ops=ops)
    adapter = _make_pipeline_adapter(
        monkeypatch=monkeypatch,
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


def test_pipeline_flush_submits_remaining_gpu_work_before_finalizing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ops: list[str] = []
    future_1 = _FakePipelineFuture(name="f1", ops=ops)
    adapter = _make_pipeline_adapter(
        monkeypatch=monkeypatch,
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


def test_pipeline_depth_three_submits_oldest_pending_before_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ops: list[str] = []
    future_1 = _FakePipelineFuture(name="f1", ops=ops)
    future_2 = _FakePipelineFuture(name="f2", ops=ops)
    future_3 = _FakePipelineFuture(name="f3", ops=ops)
    adapter = _make_pipeline_adapter(
        monkeypatch=monkeypatch,
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

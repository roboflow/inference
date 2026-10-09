"""Unit tests for inference.core.models.inference_models_adapters."""

from collections import deque
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import supervision as sv
import torch
from pycocotools import mask as mask_utils

from inference.core.entities.requests.embeddings import ImageEmbeddingRequest
from inference.core.entities.responses.embeddings import ImageEmbeddingResponse
from inference.core.entities.responses.inference import (
    InstanceSegmentationInferenceResponse,
    InstanceSegmentationInferenceResponseDC,
)
from inference.core.exceptions import InvalidMaskDecodeArgument, PostProcessingError
from inference.core.models.inference_models_adapters import (
    InferenceModelsClassificationAdapter,
    InferenceModelsDepthEstimationAdapter,
    InferenceModelsInstanceSegmentationAdapter,
    InferenceModelsObjectDetectionAdapter,
    _fixed_input_hw_from_backend,
    _supports_independent_stage_execution,
    prepare_classification_response,
    prepare_multi_label_classification_response,
    rle_masks2poly,
)
from inference.models.aliases import CLASSIFICATION_ALIASES
from inference_models import (
    ClassificationPrediction,
    InstanceDetections,
    MultiLabelClassificationPrediction,
)
from inference_models.models.auto_loaders.entities import PreProcessingOverrides
from inference_models.models.base.async_handoff import attach_adapter_mapped_kwargs
from inference_models.models.base.types import InstancesRLEMasks
from inference_models.utils.onnx_embeddings import embedding_definition


@pytest.mark.parametrize("output_type", ["feature_vector", "logits"])
@pytest.mark.parametrize("is_batch", [False, True])
def test_classifier_embedding_adapter_preserves_raw_vectors_and_preprocessing(
    monkeypatch, output_type, is_batch
) -> None:
    from inference.core.models import inference_models_adapters

    image_pre_processing = {"grayscale": {"enabled": True}}
    network_input = {"training_input_size": {"height": 16, "width": 16}}
    backend = MagicMock()
    backend.class_names = ["known", "other"]
    backend._inference_config = SimpleNamespace(
        image_pre_processing=SimpleNamespace(
            model_dump=MagicMock(return_value=image_pre_processing)
        ),
        network_input=SimpleNamespace(
            dynamic_spatial_size_supported=False,
            training_input_size=SimpleNamespace(height=16, width=16),
            model_dump=MagicMock(return_value=network_input),
        ),
    )
    images = [np.full((16, 16, 3), value, dtype=np.uint8) for value in (20, 50)]
    if not is_batch:
        images = images[:1]
    preprocessed = torch.stack([torch.from_numpy(image) for image in images])
    backend.pre_process.return_value = preprocessed
    vectors = (
        [[2.0, -3.0, 4.0], [5.0, -6.0, 7.0]]
        if output_type == "feature_vector"
        else [[-2.0, 3.0], [-4.0, 5.0]]
    )[: len(images)]
    backend.forward_embedding.return_value = torch.tensor(
        vectors, dtype=torch.float16, requires_grad=True
    )
    backend.get_embedding_info.return_value = {
        "feature_definition": embedding_definition(output_type),
        "output_type": output_type,
        "normalization": "none",
    }
    load_model = MagicMock(return_value=backend)
    monkeypatch.setattr(
        inference_models_adapters.AutoModel, "from_pretrained", load_model
    )
    adapter = InferenceModelsClassificationAdapter(
        model_id="resnet18",
        api_key="key",
        required_capabilities=["image_embeddings"],
        output_type=output_type,
    )
    request_images = [{"type": "numpy_object", "value": image} for image in images]
    request = ImageEmbeddingRequest(
        model_id="resnet18",
        output_type=output_type,
        image=request_images if is_batch else request_images[0],
        disable_preproc_auto_orient=True,
        disable_preproc_contrast=True,
        disable_preproc_grayscale=True,
        disable_preproc_static_crop=True,
    )

    response = adapter.infer_embeddings_from_request(request)

    assert isinstance(response, ImageEmbeddingResponse)
    assert response.embeddings == vectors
    assert response.inference_id == request.id
    assert response.time >= 0
    assert response.embedding_info.model_id == CLASSIFICATION_ALIASES["resnet18"]
    assert response.embedding_info.output_type == output_type
    assert response.embedding_info.feature_definition == embedding_definition(
        output_type
    )
    assert response.embedding_info.dimension == len(vectors[0])
    assert response.embedding_info.normalization == "none"
    assert response.embedding_info.precision == "torch.float16"
    assert response.embedding_info.preprocessing == {
        "image_pre_processing": image_pre_processing,
        "network_input": network_input,
        "overrides": {
            "disable_preproc_auto_orient": True,
            "disable_preproc_contrast": True,
            "disable_preproc_grayscale": True,
            "disable_preproc_static_crop": True,
        },
    }
    for actual, expected in zip(backend.pre_process.call_args.args[0], images):
        np.testing.assert_array_equal(actual, expected)
    preprocessing_kwargs = backend.pre_process.call_args.kwargs
    assert preprocessing_kwargs["input_color_format"] == "bgr"
    assert preprocessing_kwargs["pre_processing_overrides"] == PreProcessingOverrides(
        disable_contrast_enhancement=True,
        disable_grayscale=True,
        disable_static_crop=True,
    )
    backend.forward_embedding.assert_called_once_with(
        preprocessed, output_type=output_type
    )
    backend.get_embedding_info.assert_called_once_with(output_type)
    assert (
        load_model.call_args.kwargs["model_id_or_path"]
        == CLASSIFICATION_ALIASES["resnet18"]
    )
    assert load_model.call_args.kwargs["required_capabilities"] == ["image_embeddings"]
    assert load_model.call_args.kwargs["output_type"] == output_type

    # Native execution must retain the caller's tensors and the model's storage,
    # without entering the HTTP/JSON conversion path tested above.
    native_images = [torch.from_numpy(image).permute(2, 0, 1) for image in images]
    backend.pre_process.reset_mock()
    monkeypatch.setattr(
        adapter,
        "infer_embeddings_from_request",
        MagicMock(side_effect=AssertionError("HTTP path")),
    )

    def reject_conversion(*args, **kwargs):
        raise AssertionError("CPU/Python conversion")

    with monkeypatch.context() as guard:
        guard.setattr(torch.Tensor, "cpu", reject_conversion)
        guard.setattr(torch.Tensor, "tolist", reject_conversion)
        native = adapter.run_tensor_native_embeddings(
            images=native_images,
            input_color_format="rgb",
            output_type=output_type,
            **response.embedding_info.preprocessing["overrides"],
        )

    assert (
        native["embeddings"].data_ptr()
        == backend.forward_embedding.return_value.data_ptr()
    )
    assert native["embeddings"].dtype == torch.float16
    assert not native["embeddings"].requires_grad
    assert native["embedding_info"] == response.embedding_info.model_dump(
        exclude_none=True
    )
    assert backend.pre_process.call_args.args[0] is native_images
    assert backend.pre_process.call_args.kwargs["input_color_format"] == "rgb"
    adapter.infer_embeddings_from_request.assert_not_called()


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
    metadata = [
        SimpleNamespace(
            original_size=SimpleNamespace(width=4, height=4),
            size_after_pre_processing=SimpleNamespace(width=4, height=4),
        )
    ]
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
    metadata = [
        SimpleNamespace(
            original_size=SimpleNamespace(width=4, height=4),
            size_after_pre_processing=SimpleNamespace(width=4, height=4),
        )
    ]
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


def test_depth_estimation_adapter_normalization_matches_depth_anything_convention() -> (
    None
):
    """The endpoint's ordinal proximity convention uses larger values for nearer
    predictions (see the "Flip to be consistent with V2" step in
    DepthAnythingV3Torch.forward). The metric-depth adapter must reverse depth
    while normalizing: nearest pixel -> 1.0, farthest -> 0.0."""
    import numpy as np

    adapter = InferenceModelsDepthEstimationAdapter.__new__(
        InferenceModelsDepthEstimationAdapter
    )
    adapter._model = lambda inputs: [torch.tensor([[1.0, 2.0], [3.0, 4.0]])]

    (result,) = adapter.predict(np.zeros((2, 2, 3), dtype=np.uint8))

    expected = np.array([[1.0, 2 / 3], [1 / 3, 0.0]], dtype=np.float32)
    assert np.allclose(result["normalized_depth"], expected, atol=1e-6)


def _make_depth_estimation_adapter(model) -> InferenceModelsDepthEstimationAdapter:
    adapter = object.__new__(InferenceModelsDepthEstimationAdapter)
    adapter._model = model
    return adapter


def test_depth_estimation_adapter_tensor_native_negates_metric_depth() -> None:
    """The tensor-native depth contract mirrors the DepthAnything adapters:
    raw per-image maps in which larger means closer, normalized by the caller.
    YOLO26-depth emits metric depth (larger == farther), so the adapter must
    negate it while keeping kwargs mapping consistent with the other five
    tensor-native adapter overrides."""
    captured = {}

    def fake_model(images, **kwargs):
        captured["images"] = images
        captured["kwargs"] = kwargs
        return [torch.tensor([[1.0, 2.0], [3.0, 4.0]])]

    adapter = _make_depth_estimation_adapter(fake_model)
    images = [torch.zeros((3, 2, 2), dtype=torch.uint8)]

    result = adapter.run_tensor_native_inference(images, input_color_format="rgb")

    assert captured["images"] is images
    assert captured["kwargs"]["input_color_format"] == "rgb"
    assert isinstance(
        captured["kwargs"]["pre_processing_overrides"], PreProcessingOverrides
    )
    assert len(result) == 1
    assert torch.equal(result[0], torch.tensor([[-1.0, -2.0], [-3.0, -4.0]]))


def test_depth_estimation_adapter_tensor_native_passes_missing_color_format_as_none() -> (
    None
):
    captured = {}

    def fake_model(images, **kwargs):
        captured["kwargs"] = kwargs
        return [torch.tensor([[0.0, 1.0]])]

    adapter = _make_depth_estimation_adapter(fake_model)

    adapter.run_tensor_native_inference([torch.zeros((3, 1, 2), dtype=torch.uint8)])

    assert captured["kwargs"]["input_color_format"] is None


def test_depth_estimation_adapter_tensor_native_composes_to_numpy_normalization() -> (
    None
):
    """min-max normalization of the tensor-native output (what the flag-on
    depth-estimation block computes) must reproduce the numpy path's
    `(max - map) / (max - min)` proximity map exactly."""
    adapter = _make_depth_estimation_adapter(
        lambda images, **kwargs: [torch.tensor([[1.0, 2.0], [3.0, 4.0]])]
    )

    (depth_map,) = adapter.run_tensor_native_inference(
        [torch.zeros((3, 2, 2), dtype=torch.uint8)], input_color_format="bgr"
    )
    normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())

    expected = np.array([[1.0, 2 / 3], [1 / 3, 0.0]], dtype=np.float32)
    assert np.allclose(normalized.numpy(), expected, atol=1e-6)


def test_semantic_segmentation_adapter_postprocess_populates_present_class_ids():
    import base64 as _base64
    import io as _io

    import numpy as _np
    from PIL import Image as _Image

    from inference.core.models.inference_models_adapters import (
        InferenceModelsSemanticSegmentationAdapter,
    )

    # given: an adapter shell around a fake underlying model
    adapter = InferenceModelsSemanticSegmentationAdapter.__new__(
        InferenceModelsSemanticSegmentationAdapter
    )
    adapter.class_names = ["background", "cat", "dog"]
    seg = torch.zeros((30, 40), dtype=torch.int64)
    seg[5:10, 5:15] = 1
    seg[20:25, 20:30] = 2
    segmentation = SimpleNamespace(
        segmentation_map=seg, confidence=torch.full((30, 40), 0.5)
    )
    adapter._model = SimpleNamespace(
        post_process=lambda predictions, metadata, **kwargs: [segmentation]
    )
    metadata = SimpleNamespace(original_size=SimpleNamespace(height=30, width=40))

    # when
    responses = adapter.postprocess(None, [metadata])

    # then
    prediction = responses[0].predictions
    assert prediction.present_class_ids == [0, 1, 2]
    decoded = _np.asarray(
        _Image.open(_io.BytesIO(_base64.b64decode(prediction.segmentation_mask)))
    )
    assert _np.array_equal(decoded, seg.numpy().astype(_np.uint8))


def test_semantic_segmentation_adapter_postprocess_numpy_mask_format():
    import numpy as _np

    from inference.core.models.inference_models_adapters import (
        InferenceModelsSemanticSegmentationAdapter,
    )

    # given: an adapter shell around a fake underlying model
    adapter = InferenceModelsSemanticSegmentationAdapter.__new__(
        InferenceModelsSemanticSegmentationAdapter
    )
    adapter.class_names = ["background", "cat"]
    seg = torch.zeros((20, 30), dtype=torch.int64)
    seg[4:9, 6:16] = 1
    confidence = torch.full((20, 30), 0.5)
    segmentation = SimpleNamespace(segmentation_map=seg, confidence=confidence)
    adapter._model = SimpleNamespace(
        post_process=lambda predictions, metadata, **kwargs: [segmentation]
    )
    metadata = SimpleNamespace(original_size=SimpleNamespace(height=20, width=30))

    # when
    responses = adapter.postprocess(None, [metadata], response_mask_format="numpy")

    # then: raw arrays, no PNG/base64 encode happened
    prediction = responses[0].predictions
    assert isinstance(prediction.segmentation_mask, _np.ndarray)
    assert isinstance(prediction.confidence_mask, _np.ndarray)
    assert _np.array_equal(prediction.segmentation_mask, seg.numpy().astype(_np.uint8))
    assert _np.array_equal(
        prediction.confidence_mask,
        (confidence * 255).to(torch.uint8).numpy(),
    )
    assert prediction.present_class_ids == [0, 1]


def _backend_with_network_input(
    *,
    height: int,
    width: int,
    dynamic_spatial_size_supported: bool = False,
):
    return SimpleNamespace(
        _inference_config=SimpleNamespace(
            network_input=SimpleNamespace(
                dynamic_spatial_size_supported=dynamic_spatial_size_supported,
                training_input_size=SimpleNamespace(height=height, width=width),
            )
        )
    )


def test_fixed_input_hw_from_backend():
    assert _fixed_input_hw_from_backend(
        _backend_with_network_input(height=518, width=640)
    ) == (518, 640)


def test_fixed_input_hw_from_backend_skips_dynamic_spatial():
    assert (
        _fixed_input_hw_from_backend(
            _backend_with_network_input(
                height=640,
                width=640,
                dynamic_spatial_size_supported=True,
            )
        )
        is None
    )


def test_fixed_input_hw_from_backend_skips_missing_or_invalid_network_input():
    assert _fixed_input_hw_from_backend(SimpleNamespace(_inference_config=None)) is None
    assert (
        _fixed_input_hw_from_backend(_backend_with_network_input(height=0, width=640))
        is None
    )


def _seg_adapter() -> InferenceModelsInstanceSegmentationAdapter:
    adapter = object.__new__(InferenceModelsInstanceSegmentationAdapter)
    adapter._pipeline_depth = 1
    adapter.class_names = ["car"]
    return adapter


def test_polygon_points_are_scaled_from_mask_space_into_image_space() -> None:
    # given
    # a 4x4 mask describing a 16x16 image: contours come out in mask
    # coordinates and must be lifted by 4x before they are reported
    adapter = _seg_adapter()
    metadata = [
        SimpleNamespace(
            original_size=SimpleNamespace(width=16, height=16),
            size_after_pre_processing=SimpleNamespace(width=16, height=16),
        )
    ]
    detections = [
        InstanceDetections(
            xyxy=torch.tensor([[0, 0, 16, 16]], dtype=torch.int32),
            confidence=torch.tensor([0.9], dtype=torch.float32),
            class_id=torch.tensor([0], dtype=torch.int32),
            mask=torch.ones((1, 4, 4), dtype=torch.uint8),
        )
    ]

    # when
    responses = adapter._build_responses_from_detections(detections, metadata)

    # then
    xs = [p.x for p in responses[0].predictions[0].points]
    ys = [p.y for p in responses[0].predictions[0].points]
    assert max(xs) > 4, f"x still in mask space: {xs}"
    assert max(ys) > 4, f"y still in mask space: {ys}"


def test_polygon_points_unchanged_when_mask_matches_the_image() -> None:
    # given
    # today's situation: the scaling must be an exact no-op
    adapter = _seg_adapter()
    metadata = [
        SimpleNamespace(
            original_size=SimpleNamespace(width=8, height=8),
            size_after_pre_processing=SimpleNamespace(width=8, height=8),
        )
    ]
    mask = torch.zeros((1, 8, 8), dtype=torch.uint8)
    mask[0, 2:6, 2:6] = 1
    detections = [
        InstanceDetections(
            xyxy=torch.tensor([[2, 2, 6, 6]], dtype=torch.int32),
            confidence=torch.tensor([0.9], dtype=torch.float32),
            class_id=torch.tensor([0], dtype=torch.int32),
            mask=mask,
        )
    ]

    # when
    responses = adapter._build_responses_from_detections(detections, metadata)

    # then
    xs = [p.x for p in responses[0].predictions[0].points]
    ys = [p.y for p in responses[0].predictions[0].points]
    assert min(xs) == 2 and max(xs) == 5
    assert min(ys) == 2 and max(ys) == 5


class TestMaskDecodeModeMapping:
    """The adapter collapses the published enum onto one float.

    `mask_decode_mode` and `tradeoff_factor` are the documented request
    surface; `masks_resolution_factor` is what post-processing consumes. The
    translation belongs here, and the source keys must not reach the model.
    """

    @staticmethod
    def _map(**kwargs) -> dict:
        adapter = object.__new__(InferenceModelsInstanceSegmentationAdapter)
        adapter._model = SimpleNamespace(supported_mask_formats=[])
        return adapter.map_inference_kwargs(kwargs)

    def test_accurate_maps_to_full_resolution(self) -> None:
        # given / when
        mapped = self._map(mask_decode_mode="accurate")

        # then
        assert mapped["masks_resolution_factor"] == 1.0

    def test_fast_maps_to_mask_resolution(self) -> None:
        # given / when
        mapped = self._map(mask_decode_mode="fast", allow_reduced_mask_resolution=True)

        # then
        assert mapped["masks_resolution_factor"] == 0.0

    def test_tradeoff_reads_the_factor(self) -> None:
        # given / when
        mapped = self._map(
            mask_decode_mode="tradeoff",
            tradeoff_factor=0.25,
            allow_reduced_mask_resolution=True,
        )

        # then
        assert mapped["masks_resolution_factor"] == 0.25

    def test_default_is_full_resolution_when_nothing_is_passed(self) -> None:
        # given / when
        # an unset mode must reproduce today's behaviour exactly
        mapped = self._map()

        # then
        assert mapped["masks_resolution_factor"] == 1.0

    def test_source_keys_do_not_reach_the_model(self) -> None:
        # given / when
        # deeper pre/post stages do not accept arbitrary kwargs
        mapped = self._map(
            mask_decode_mode="tradeoff",
            tradeoff_factor=0.5,
            allow_reduced_mask_resolution=True,
        )

        # then
        assert "mask_decode_mode" not in mapped
        assert "tradeoff_factor" not in mapped
        assert "allow_reduced_mask_resolution" not in mapped

    @pytest.mark.parametrize("bad_factor", [-0.1, 1.1])
    def test_out_of_range_factor_is_rejected(self, bad_factor: float) -> None:
        # given / when / then
        with pytest.raises(InvalidMaskDecodeArgument):
            self._map(mask_decode_mode="tradeoff", tradeoff_factor=bad_factor)

    def test_unknown_mode_is_rejected(self) -> None:
        # given / when / then
        # the legacy path raises for this; the two must agree
        with pytest.raises(InvalidMaskDecodeArgument):
            self._map(mask_decode_mode="nonsense")

    @pytest.mark.parametrize("mode", ["fast", "tradeoff"])
    @pytest.mark.parametrize("opt_in", [None, False])
    def test_existing_requests_keep_image_resolution(self, mode, opt_in) -> None:
        kwargs = {"mask_decode_mode": mode, "tradeoff_factor": 0.25}
        if opt_in is not None:
            kwargs["allow_reduced_mask_resolution"] = opt_in

        mapped = self._map(**kwargs)

        assert mapped["masks_resolution_factor"] == 1.0
        assert "allow_reduced_mask_resolution" not in mapped

    def test_internal_resolution_kwarg_does_not_bypass_opt_in(self) -> None:
        mapped = self._map(masks_resolution_factor=0.25)

        assert mapped["masks_resolution_factor"] == 1.0


class TestRLEBackedPolygons:
    """Polygons derived from RLE carriers at a reduced resolution.

    The adapter asks for RLE whenever the model supports it, so this is the
    default route for a plain polygon response. The counts are encoded on the
    mask grid; decoding them on the image grid makes pycocotools reinterpret
    the runs without raising and produces unusable contours.
    """

    @staticmethod
    def _carrier(mask_h: int, mask_w: int, image_h: int, image_w: int):
        dense = np.zeros((mask_h, mask_w), dtype=np.uint8)
        dense[mask_h // 4 : mask_h // 2, mask_w // 4 : mask_w // 2] = 1
        counts = mask_utils.encode(np.asfortranarray(dense))["counts"]

        return (
            InstancesRLEMasks(
                image_size=(image_h, image_w),
                masks=[counts],
                mask_size=(mask_h, mask_w),
            ),
            dense,
        )

    def test_polygon_follows_the_encoded_grid_not_the_image(self) -> None:
        # given
        # a 160x160 mask describing a 640x640 image: the contour must come back
        # in mask coordinates, which the adapter then scales into image space
        carrier, dense = self._carrier(160, 160, 640, 640)

        # when
        polygon = rle_masks2poly(carrier)[0]

        # then
        ys, xs = np.where(dense)
        assert polygon[:, 0].max() <= 160
        assert abs(polygon[:, 0].min() - xs.min()) <= 1
        assert abs(polygon[:, 1].max() - ys.max()) <= 1

    def test_unchanged_when_the_grid_is_the_image(self) -> None:
        # given
        carrier, dense = self._carrier(64, 64, 64, 64)

        # when
        polygon = rle_masks2poly(carrier)[0]

        # then
        ys, xs = np.where(dense)
        assert abs(polygon[:, 0].min() - xs.min()) <= 1
        assert abs(polygon[:, 1].max() - ys.max()) <= 1

    def test_non_square_grid(self) -> None:
        # given
        # a transposed decode is invisible on a square grid
        carrier, dense = self._carrier(80, 160, 400, 800)

        # when
        polygon = rle_masks2poly(carrier)[0]

        # then
        assert polygon[:, 0].max() <= 160
        assert polygon[:, 1].max() <= 80


@pytest.mark.parametrize("opt_in", [False, True])
@pytest.mark.parametrize("mask_format", ["dense", "rle"])
@pytest.mark.parametrize("factor", [1.0, 0.5, 0.0])
@pytest.mark.parametrize("offset", [(0, 0), (60, 40)])
@pytest.mark.parametrize("response_format", ["polygon", "rle"])
def test_crop_response_uses_the_frame_represented_by_mask(
    mask_format: str, factor: float, offset: tuple, response_format: str, opt_in: bool
) -> None:
    from inference_models.entities import ImageDimensions
    from inference_models.models.common.roboflow.model_packages import (
        PreProcessingMetadata,
        StaticCropOffset,
    )
    from inference_models.models.common.roboflow.post_processing import (
        align_instance_segmentation_results,
        align_instance_segmentation_results_to_rle_masks,
    )

    offset_x, offset_y = offset
    original_size = ImageDimensions(height=200, width=300)
    crop_size = ImageDimensions(height=100, width=150)
    crop_offset = StaticCropOffset(
        offset_x=offset_x, offset_y=offset_y, crop_width=150, crop_height=100
    )
    metadata = PreProcessingMetadata(
        pad_left=0,
        pad_top=0,
        pad_right=0,
        pad_bottom=0,
        original_size=original_size,
        size_after_pre_processing=crop_size,
        inference_size=crop_size,
        scale_width=1.0,
        scale_height=1.0,
        static_crop_offset=crop_offset,
    )
    masks = torch.full((1, 20, 30), -1.0)
    masks[:, 2:10, 4:16] = 1.0
    kwargs = dict(
        image_bboxes=torch.tensor([[20, 10, 80, 50]], dtype=torch.float32),
        masks=masks,
        padding=(0, 0, 0, 0),
        scale_width=1.0,
        scale_height=1.0,
        original_size=original_size,
        size_after_pre_processing=crop_size,
        inference_size=crop_size,
        static_crop_offset=crop_offset,
        masks_resolution_factor=factor,
    )
    if mask_format == "dense":
        boxes, aligned_masks = align_instance_segmentation_results(**kwargs)
    else:
        box, rle = next(align_instance_segmentation_results_to_rle_masks(**kwargs))
        boxes = box.unsqueeze(0)
        aligned_masks = InstancesRLEMasks(
            image_size=original_size,
            masks=[rle["counts"]],
            mask_size=tuple(rle["size"]),
        )
    detections = InstanceDetections(
        xyxy=boxes,
        confidence=torch.tensor([0.9]),
        class_id=torch.tensor([0]),
        mask=aligned_masks,
    )

    response = _seg_adapter()._build_responses_from_detections(
        [detections],
        [metadata],
        response_mask_format=response_format,
        allow_reduced_mask_resolution=opt_in,
    )[0]

    coordinate_metadata = response.mask_metadata
    if opt_in:
        assert coordinate_metadata.height == detections.mask_size[0]
        assert coordinate_metadata.width == detections.mask_size[1]
        assert coordinate_metadata.coordinate_system == "mask_grid"
    else:
        assert coordinate_metadata is None

    if response_format == "rle":
        serialized = response.model_dump(by_alias=True)
        if opt_in:
            assert serialized["image"] == {
                "width": detections.mask_size[1],
                "height": detections.mask_size[0],
            }
            assert serialized["original_image"] == {"width": 300, "height": 200}
            serialized["image"] = serialized["original_image"]
        converted = sv.Detections.from_inference(serialized)
        expected = np.zeros((1, 200, 300), dtype=bool)
        expected[:, 10 + offset_y : 50 + offset_y, 20 + offset_x : 80 + offset_x] = True
        assert converted.mask.shape == expected.shape
        np.testing.assert_allclose(
            sv.mask_to_xyxy(converted.mask), sv.mask_to_xyxy(expected), atol=1
        )
        assert (
            np.logical_and(converted.mask, expected).sum()
            / np.logical_or(converted.mask, expected).sum()
            > 0.95
        )
        sv.MaskAnnotator().annotate(np.zeros((200, 300, 3), dtype=np.uint8), converted)
        return

    prediction = response.predictions[0]
    scale_x = coordinate_metadata.scale_x if coordinate_metadata else 1.0
    scale_y = coordinate_metadata.scale_y if coordinate_metadata else 1.0
    xs = [point.x * scale_x for point in prediction.points]
    ys = [point.y * scale_y for point in prediction.points]
    assert min(xs) == pytest.approx(20 + offset_x)
    assert min(ys) == pytest.approx(10 + offset_y)
    pixel_size = 150 / round(30 * (1 - factor) + 150 * factor)
    assert max(xs) == pytest.approx(80 + offset_x - pixel_size)
    assert max(ys) == pytest.approx(50 + offset_y - pixel_size)
    assert prediction.x * scale_x == pytest.approx(50 + offset_x)
    assert prediction.y * scale_y == pytest.approx(30 + offset_y)


@pytest.mark.parametrize("use_dc", [False, True])
@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("response_format", ["polygon", "rle"])
def test_deferred_response_preserves_coordinate_contract(
    use_dc, empty, response_format
):
    adapter = _seg_adapter()
    adapter._pipeline_depth = 2
    count = 0 if empty else 1
    detections = InstanceDetections(
        xyxy=torch.tensor([[0, 0, 10, 20]], dtype=torch.int32)[:count],
        confidence=torch.tensor([0.9])[:count],
        class_id=torch.tensor([0])[:count],
        mask=torch.ones((count, 5, 2), dtype=torch.bool),
    )
    future = SimpleNamespace(result=lambda: [detections])
    response = adapter._finalize_future(
        future,
        _make_meta("mask-grid"),
        {
            "masks_resolution_factor": 0.25,
            "response_mask_format": response_format,
            "source": "workflow-execution" if use_dc else "api",
        },
    )[0]

    serialized = (
        response.to_dict()
        if use_dc and response_format == "polygon"
        else response.model_dump(exclude_none=True)
    )
    assert serialized["image"] == {"width": 2, "height": 5}
    assert serialized["original_image"] == {"width": 10, "height": 20}
    assert serialized["mask_metadata"] == {
        "coordinate_system": "mask_grid",
        "width": 2,
        "height": 5,
        "scale_x": 5.0,
        "scale_y": 4.0,
    }
    assert len(serialized["predictions"]) == count
    if not empty and response_format == "polygon":
        assert max(p["x"] for p in serialized["predictions"][0]["points"]) == 1
        assert max(p["y"] for p in serialized["predictions"][0]["points"]) == 4


@pytest.mark.parametrize("mask_format", ["dense", "rle"])
def test_opted_in_polygons_share_box_coordinates_on_non_square_image(mask_format):
    masks = torch.zeros((1, 160, 160), dtype=torch.bool)
    masks[:, 48:72, 48:72] = True
    mask = masks
    if mask_format == "rle":
        from inference_models.entities import ImageDimensions

        rle = mask_utils.encode(np.asfortranarray(masks[0].numpy().astype(np.uint8)))
        mask = InstancesRLEMasks(
            image_size=ImageDimensions(height=800, width=1000),
            masks=[rle["counts"]],
            mask_size=(160, 160),
        )
    detections = InstanceDetections(
        xyxy=torch.tensor([[300, 240, 450, 360]], dtype=torch.float32),
        confidence=torch.tensor([0.9]),
        class_id=torch.tensor([0]),
        mask=mask,
    )
    metadata = SimpleNamespace(original_size=SimpleNamespace(width=1000, height=800))

    response = _seg_adapter()._build_responses_from_detections(
        [detections], [metadata], allow_reduced_mask_resolution=True
    )[0]

    prediction = response.predictions[0]
    xs = [point.x for point in prediction.points]
    ys = [point.y for point in prediction.points]
    assert response.mask_metadata.scale_x == 6.25
    assert response.mask_metadata.scale_y == 5.0
    assert response.image.width == response.image.height == 160
    assert (min(xs), min(ys), max(xs), max(ys)) == (48, 48, 71, 71)
    converted = sv.Detections.from_inference(response.model_dump(by_alias=True))
    np.testing.assert_array_equal(converted.xyxy, [[48, 48, 72, 72]])
    np.testing.assert_allclose(
        sv.mask_to_xyxy(converted.mask), [[48, 48, 71, 71]], atol=1
    )
    sv.MaskAnnotator().annotate(np.zeros((160, 160, 3), dtype=np.uint8), converted)


@pytest.mark.parametrize("opt_in", [False, True])
@pytest.mark.parametrize("response_format", ["polygon", "rle"])
@pytest.mark.parametrize("factor", [0.0, 0.5, 1.0])
def test_response_geometry_uses_one_grid(opt_in, response_format, factor):
    from inference_models.entities import ImageDimensions
    from inference_models.models.common.roboflow.model_packages import StaticCropOffset
    from inference_models.models.common.roboflow.post_processing import (
        align_instance_segmentation_results,
    )

    original_size = ImageDimensions(height=800, width=1000)
    input_boxes = torch.tensor([[300, 240, 450, 360]], dtype=torch.float32)
    input_masks = torch.full((1, 160, 160), -1.0)
    input_masks[:, 48:72, 48:72] = 1.0
    adapter = _seg_adapter()

    def post_process(predictions, metadata, **kwargs):
        assert "allow_reduced_mask_resolution" not in kwargs
        boxes, masks = align_instance_segmentation_results(
            image_bboxes=input_boxes.clone(),
            masks=input_masks.clone(),
            padding=(0, 0, 0, 0),
            scale_width=1.0,
            scale_height=1.0,
            original_size=original_size,
            size_after_pre_processing=original_size,
            inference_size=original_size,
            static_crop_offset=StaticCropOffset(
                offset_x=0, offset_y=0, crop_width=1000, crop_height=800
            ),
            masks_resolution_factor=kwargs["masks_resolution_factor"],
        )
        return [
            InstanceDetections(
                xyxy=boxes,
                confidence=torch.tensor([0.9]),
                class_id=torch.tensor([0]),
                mask=masks,
            )
        ]

    adapter._model = SimpleNamespace(
        supported_mask_formats=[], post_process=post_process
    )
    response = adapter._postprocess_sync(
        [],
        [SimpleNamespace(original_size=original_size)],
        allow_reduced_mask_resolution=opt_in,
        mask_decode_mode="tradeoff",
        tradeoff_factor=factor,
        response_mask_format=response_format,
    )[0]
    width = round(160 * (1 - factor) + 1000 * factor) if opt_in else 1000
    height = round(160 * (1 - factor) + 800 * factor) if opt_in else 800
    assert response.image.width == width
    assert response.image.height == height
    prediction = response.predictions[0]
    np.testing.assert_allclose(
        [prediction.x, prediction.y, prediction.width, prediction.height],
        np.array([375, 300, 150, 120])
        * [width / 1000, height / 800, width / 1000, height / 800],
    )
    if opt_in:
        assert response.original_image.model_dump() == {"width": 1000, "height": 800}
        assert response.mask_metadata.scale_x == pytest.approx(1000 / width)
        assert response.mask_metadata.scale_y == pytest.approx(800 / height)
    else:
        assert response.original_image is None and response.mask_metadata is None
    if response_format == "rle":
        assert prediction.rle["size"] == [height, width]
    converted = sv.Detections.from_inference(response.model_dump(by_alias=True))
    assert converted.mask.shape == (1, height, width)
    np.testing.assert_allclose(
        sv.mask_to_xyxy(converted.mask), converted.xyxy, atol=1.1
    )
    sv.MaskAnnotator().annotate(np.zeros((height, width, 3), dtype=np.uint8), converted)


@pytest.mark.parametrize("response_format", ["polygon", "rle"])
def test_async_opt_in_survives_factor_one(response_format):
    adapter = _seg_adapter()
    adapter._pipeline_depth = 2
    detections = InstanceDetections(
        xyxy=torch.tensor([[0, 0, 10, 20]], dtype=torch.float32),
        confidence=torch.tensor([0.9]),
        class_id=torch.tensor([0]),
        mask=torch.ones((1, 20, 10), dtype=torch.bool),
    )
    future = SimpleNamespace(result=lambda: [detections])
    adapter._model = SimpleNamespace(
        supported_mask_formats=[], forward_async=lambda *args, **kwargs: future
    )
    adapter._submit_next_pending_gpu_work = lambda: None
    adapter._submit_ready_responses = lambda: None
    returned_future = adapter.predict(
        torch.zeros((1, 3, 20, 10)),
        allow_reduced_mask_resolution=True,
        mask_decode_mode="accurate",
        response_mask_format=response_format,
    )
    from inference_models.models.base.async_handoff import get_adapter_mapped_kwargs

    mapped_kwargs = get_adapter_mapped_kwargs(returned_future)
    assert "allow_reduced_mask_resolution" not in mapped_kwargs
    response = adapter._finalize_future(
        returned_future, _make_meta("accurate"), mapped_kwargs
    )[0]
    assert response.original_image.model_dump() == {"width": 10, "height": 20}
    assert response.mask_metadata.scale_x == response.mask_metadata.scale_y == 1.0


def test_native_grid_batch_does_not_mutate_model_boxes():
    adapter = _seg_adapter()
    boxes = torch.tensor([[20, 10, 60, 30]], dtype=torch.float32)
    det = InstanceDetections(
        xyxy=boxes,
        confidence=torch.tensor([0.9]),
        class_id=torch.tensor([0]),
        mask=torch.ones((1, 10, 20), dtype=torch.bool),
    )
    metadata = [
        SimpleNamespace(original_size=SimpleNamespace(width=100, height=50)),
        SimpleNamespace(original_size=SimpleNamespace(width=200, height=200)),
    ]
    responses = adapter._build_responses_from_detections(
        [det, det], metadata, allow_reduced_mask_resolution=True
    )
    assert [r.predictions[0].x for r in responses] == [8, 4]
    assert [r.predictions[0].y for r in responses] == [4, 1]
    assert [r.mask_metadata.scale_x for r in responses] == [5, 10]
    torch.testing.assert_close(
        boxes, torch.tensor([[20, 10, 60, 30]], dtype=torch.float32)
    )

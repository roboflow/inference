"""Unit tests for inference.core.models.inference_models_adapters."""

import pytest
import torch

import inference.core.models.inference_models_adapters as adapters
from inference.core.exceptions import PostProcessingError
from inference.core.models.inference_models_adapters import (
    PINNED_HOST_BUFFERS,
    clear_pinned_buffers,
    get_pinned_buffer,
    prepare_classification_response,
    prepare_multi_label_classification_response,
)
from inference_models import (
    ClassificationPrediction,
    MultiLabelClassificationPrediction,
)


@pytest.fixture(autouse=True)
def clear_pinned_host_buffers() -> None:
    clear_pinned_buffers()
    yield
    clear_pinned_buffers()


def _fake_pinned_empty(shape, dtype, pin_memory: bool = False) -> torch.Tensor:
    assert pin_memory is True
    return torch.zeros(shape, dtype=dtype)


def test_get_pinned_buffer_reuses_cached_storage_for_smaller_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(adapters.torch, "empty", _fake_pinned_empty)

    first = get_pinned_buffer("mask", (16, 4), torch.float32)
    second = get_pinned_buffer("mask", (8, 4), torch.float32)

    assert first.shape == (16, 4)
    assert second.shape == (8, 4)
    assert second.data_ptr() == first.data_ptr()
    assert tuple(PINNED_HOST_BUFFERS[("mask", torch.float32)].shape) == (16, 4)


def test_get_pinned_buffer_shrinks_massively_oversized_cached_buffer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(adapters.torch, "empty", _fake_pinned_empty)

    first = get_pinned_buffer("mask", (64, 4), torch.float32)
    second = get_pinned_buffer("mask", (4, 4), torch.float32)

    assert first.shape == (64, 4)
    assert second.shape == (4, 4)
    assert second.data_ptr() != first.data_ptr()
    assert tuple(PINNED_HOST_BUFFERS[("mask", torch.float32)].shape) == (4, 4)


def test_clear_pinned_buffers_clears_all_or_single_named_buffer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(adapters.torch, "empty", _fake_pinned_empty)

    get_pinned_buffer("mask", (8, 8), torch.float32)
    get_pinned_buffer("xyxy", (8, 4), torch.int32)

    clear_pinned_buffers(name="mask")
    assert ("mask", torch.float32) not in PINNED_HOST_BUFFERS
    assert ("xyxy", torch.int32) in PINNED_HOST_BUFFERS

    clear_pinned_buffers()
    assert PINNED_HOST_BUFFERS == {}


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

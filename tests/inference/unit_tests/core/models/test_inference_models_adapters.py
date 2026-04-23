"""Unit tests for inference.core.models.inference_models_adapters."""

import pytest
import torch

from inference.core.models.inference_models_adapters import (
    prepare_classification_response,
    prepare_multi_label_classification_response,
)
from inference.core.exceptions import PostProcessingError
from inference_models import (
    ClassificationPrediction,
    MultiLabelClassificationPrediction,
)


def test_prepare_multi_label_response_maps_scores_by_class_index() -> None:
    """Per-class scores come from the full confidence vector, not zip(class_ids, ...).

    Models attach a thresholded ``class_ids`` list and a full ``confidence`` vector;
    indices in ``class_ids`` are not aligned with positions in ``confidence``.
    """
    class_names = ["a", "b", "c", "d"]
    confidence = torch.tensor([0.1, 0.2, 0.85, 0.9])
    class_ids = torch.tensor([2, 3], dtype=torch.long)
    prediction = MultiLabelClassificationPrediction(
        class_ids=class_ids,
        confidence=confidence,
    )

    results = prepare_multi_label_classification_response(
        post_processed_predictions=[prediction],
        image_sizes=[(10, 20)],
        class_names=class_names,
        confidence_threshold=0.5,
    )

    assert len(results) == 1
    r = results[0]
    assert r.predictions["a"].confidence == pytest.approx(0.1)
    assert r.predictions["b"].confidence == pytest.approx(0.2)
    assert r.predictions["c"].confidence == pytest.approx(0.85)
    assert r.predictions["d"].confidence == pytest.approx(0.9)
    assert set(r.predicted_classes) == {"c", "d"}


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
        class_id=torch.tensor([2], dtype=torch.long),
        confidence=torch.tensor([[0.1, 0.2, 0.7]]),
    )

    with pytest.raises(PostProcessingError, match="class names metadata"):
        prepare_classification_response(
            post_processed_predictions=prediction,
            image_sizes=[(10, 20)],
            class_names=["cat", "dog"],
            confidence_threshold=0.0,
        )

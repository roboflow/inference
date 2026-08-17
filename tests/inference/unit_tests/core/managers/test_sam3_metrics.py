import pytest
from prometheus_client import REGISTRY

from inference.core.managers.sam3_metrics import (
    SAM3_EMBEDDING_CACHE_OUTCOMES,
    record_sam3_visual_segment_embedding_cache_outcome,
)


def test_sam3_embedding_cache_metric_has_fixed_outcome_series() -> None:
    metric = next(
        metric
        for metric in REGISTRY.collect()
        if metric.name == "inference_sam3_visual_segment_embedding_cache"
    )
    total_samples = [
        sample
        for sample in metric.samples
        if sample.name == "inference_sam3_visual_segment_embedding_cache_total"
    ]

    assert {sample.labels["outcome"] for sample in total_samples} == set(
        SAM3_EMBEDDING_CACHE_OUTCOMES
    )
    assert all(set(sample.labels) == {"outcome"} for sample in total_samples)


def test_sam3_embedding_cache_metric_rejects_unbounded_outcomes() -> None:
    with pytest.raises(ValueError, match="Unsupported SAM3 embedding cache outcome"):
        record_sam3_visual_segment_embedding_cache_outcome("image-id")  # type: ignore[arg-type]

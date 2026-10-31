from typing import Literal

from prometheus_client import Counter

Sam3EmbeddingCacheOutcome = Literal["hit", "miss", "not_attempted"]
SAM3_EMBEDDING_CACHE_OUTCOMES = ("hit", "miss", "not_attempted")

SAM3_VISUAL_SEGMENT_EMBEDDING_CACHE = Counter(
    "inference_sam3_visual_segment_embedding_cache",
    "SAM3 visual_segment requests by embedding cache lookup outcome.",
    labelnames=("outcome",),
)

# Export zero-valued series before the first request so dashboards have a stable
# three-series shape. The label values are intentionally fixed and low-cardinality.
for _outcome in SAM3_EMBEDDING_CACHE_OUTCOMES:
    SAM3_VISUAL_SEGMENT_EMBEDDING_CACHE.labels(outcome=_outcome)


def record_sam3_visual_segment_embedding_cache_outcome(
    outcome: Sam3EmbeddingCacheOutcome,
) -> None:
    if outcome not in SAM3_EMBEDDING_CACHE_OUTCOMES:
        raise ValueError(f"Unsupported SAM3 embedding cache outcome: {outcome}")
    SAM3_VISUAL_SEGMENT_EMBEDDING_CACHE.labels(outcome=outcome).inc()

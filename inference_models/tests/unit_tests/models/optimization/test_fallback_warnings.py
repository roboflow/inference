from concurrent.futures import ThreadPoolExecutor

from inference_models.models.optimization.contracts import OptimizationStage
from inference_models.models.optimization.fallback_warnings import (
    FallbackWarningTracker,
)


def test_fallback_warning_is_claimed_once_per_distinct_key() -> None:
    tracker = FallbackWarningTracker()
    warning = {
        "stage": OptimizationStage.PREPROCESS,
        "requested_id": "optimized",
        "effective_id": "base",
        "reason": "unsupported request",
    }

    assert tracker.claim(**warning)
    assert not tracker.claim(**warning)
    assert tracker.claim(**{**warning, "reason": "different reason"})
    assert tracker.claim(**{**warning, "stage": OptimizationStage.POSTPROCESS})


def test_fallback_warning_claim_is_thread_safe() -> None:
    tracker = FallbackWarningTracker()

    def claim() -> bool:
        return tracker.claim(
            stage=OptimizationStage.PREPROCESS,
            requested_id="optimized",
            effective_id="base",
            reason="unsupported request",
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        claims = list(executor.map(lambda _: claim(), range(32)))

    assert claims.count(True) == 1

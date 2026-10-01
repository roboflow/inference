import threading
from collections import OrderedDict
from contextlib import contextmanager

import numpy as np
import pytest
from roboflow_workflows.core_steps.sinks.roboflow.dataset_upload.active_learning import (
    ImageDimensions,
    StrategyLimit,
    StrategyLimitType,
    generate_cache_key_for_active_learning_usage,
    prepare_image_to_registration_with_metadata,
    return_strategy_credit,
    use_credit_of_matching_strategy,
)
from roboflow_workflows.utils.in_memory_cache import InMemoryWorkflowsCache


def _limits(minutely: int, hourly: int = 100, daily: int = 1000):
    return [
        StrategyLimit(limit_type=StrategyLimitType.MINUTELY, value=minutely),
        StrategyLimit(limit_type=StrategyLimitType.HOURLY, value=hourly),
        StrategyLimit(limit_type=StrategyLimitType.DAILY, value=daily),
    ]


def _usage(cache, strategy: str, limit_type: StrategyLimitType):
    value = cache.get(
        generate_cache_key_for_active_learning_usage(
            limit_type=limit_type,
            workspace="ws",
            project="proj",
            strategy_name=strategy,
        )
    )
    return None if value is None else value["usage"]


class _LockingCache(InMemoryWorkflowsCache):
    def __init__(self) -> None:
        super().__init__()
        self.locked_keys = []

    @contextmanager
    def lock(self, key: str, expire: float):
        self.locked_keys.append((key, expire))
        yield threading.Lock()


def test_credit_is_consumed_until_the_limit_is_reached() -> None:
    cache = InMemoryWorkflowsCache()
    limits = OrderedDict({"quota": _limits(minutely=2)})

    results = [
        use_credit_of_matching_strategy(
            cache=cache,
            workspace="ws",
            project="proj",
            matching_strategies_limits=limits,
        )
        for _ in range(3)
    ]

    assert results == ["quota", "quota", None]
    for limit_type in StrategyLimitType:
        assert _usage(cache, "quota", limit_type) == 2


def test_the_next_strategy_is_used_when_the_first_is_exhausted() -> None:
    cache = InMemoryWorkflowsCache()
    limits = OrderedDict({"first": _limits(minutely=0), "second": _limits(minutely=1)})

    result = use_credit_of_matching_strategy(
        cache=cache, workspace="ws", project="proj", matching_strategies_limits=limits
    )

    assert result == "second"
    assert _usage(cache, "first", StrategyLimitType.MINUTELY) is None


def test_returned_credit_is_decremented_and_never_goes_negative() -> None:
    cache = InMemoryWorkflowsCache()
    limits = OrderedDict({"quota": _limits(minutely=5)})
    use_credit_of_matching_strategy(
        cache=cache, workspace="ws", project="proj", matching_strategies_limits=limits
    )

    return_strategy_credit(
        cache=cache, workspace="ws", project="proj", strategy_name="quota"
    )
    return_strategy_credit(
        cache=cache, workspace="ws", project="proj", strategy_name="quota"
    )

    for limit_type in StrategyLimitType:
        assert _usage(cache, "quota", limit_type) == 0


def test_returning_unused_credit_writes_nothing() -> None:
    cache = InMemoryWorkflowsCache()

    return_strategy_credit(
        cache=cache, workspace="ws", project="proj", strategy_name="quota"
    )

    for limit_type in StrategyLimitType:
        assert _usage(cache, "quota", limit_type) is None


def test_a_cache_with_its_own_lock_is_locked_as_before() -> None:
    cache = _LockingCache()

    use_credit_of_matching_strategy(
        cache=cache,
        workspace="ws",
        project="proj",
        matching_strategies_limits=OrderedDict({"quota": _limits(minutely=1)}),
    )

    assert cache.locked_keys == [("active_learning:usage:ws:proj:usage:lock", 5)]


def test_concurrent_consumers_never_exceed_the_limit_without_a_cache_lock() -> None:
    cache = InMemoryWorkflowsCache()
    limits = OrderedDict({"quota": _limits(minutely=10)})
    results = []

    def consume() -> None:
        results.append(
            use_credit_of_matching_strategy(
                cache=cache,
                workspace="ws",
                project="proj",
                matching_strategies_limits=limits,
            )
        )

    threads = [threading.Thread(target=consume) for _ in range(30)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert results.count("quota") == 10
    assert _usage(cache, "quota", StrategyLimitType.MINUTELY) == 10


def test_prepare_image_downscales_and_reports_the_transform() -> None:
    image = np.zeros((200, 400, 3), dtype=np.uint8)

    prepared = prepare_image_to_registration_with_metadata(
        image=image,
        desired_size=ImageDimensions(height=100, width=100),
        jpeg_compression_level=75,
    )

    assert prepared.original_size_wh == (400, 200)
    assert prepared.final_size_wh == (100, 50)
    assert prepared.scale_x == pytest.approx(0.25)
    assert prepared.scale_y == pytest.approx(0.25)
    assert prepared.scaling_factor == pytest.approx(0.25)
    assert prepared.encoded_image[:2] == b"\xff\xd8"


def test_prepare_image_keeps_a_small_image() -> None:
    image = np.zeros((20, 30, 3), dtype=np.uint8)

    prepared = prepare_image_to_registration_with_metadata(
        image=image, desired_size=None, jpeg_compression_level=90
    )

    assert prepared.final_size_wh == (30, 20)
    assert prepared.scaling_factor == 1.0


def test_prepare_image_rejects_an_empty_image() -> None:
    with pytest.raises(ValueError):
        prepare_image_to_registration_with_metadata(
            image=np.zeros((0, 10, 3), dtype=np.uint8),
            desired_size=None,
            jpeg_compression_level=90,
        )

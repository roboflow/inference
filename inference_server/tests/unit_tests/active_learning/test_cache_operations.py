import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from datetime import datetime
from typing import Any, List, Optional
from unittest import mock
from unittest.mock import MagicMock

import pytest
from roboflow_workflows.core_steps.sinks.roboflow.dataset_upload import (
    active_learning as accounting_source,
)
from roboflow_workflows.utils.in_memory_cache import InMemoryWorkflowsCache

from inference_server.active_learning.cache_operations import (
    LIMIT_TYPE2KEY_EXPIRATION,
    find_strategy_with_spare_usage_credit,
    generate_cache_key_for_active_learning_usage,
    get_current_strategy_limit_usage,
    lock_limits,
    return_strategy_credit,
    use_credit_of_matching_strategy,
)
from inference_server.active_learning.entities import StrategyLimit, StrategyLimitType


class _RecordingCache(InMemoryWorkflowsCache):
    def __init__(self) -> None:
        super().__init__()
        self.expirations: dict = {}
        self.locked_keys: List[str] = []

    def set(self, key: str, value: Any, expire: Optional[float] = None) -> None:
        self.expirations[key] = expire
        super().set(key=key, value=value, expire=expire)

    @contextmanager
    def lock(self, key: str, expire: Optional[float] = None):
        self.locked_keys.append(key)
        yield None


class _SlowReadCache(InMemoryWorkflowsCache):
    def get(self, key: str) -> Any:
        value = super().get(key)
        time.sleep(0.002)
        return value


@pytest.fixture(autouse=True)
def frozen_clock():
    with mock.patch.object(accounting_source, "datetime") as datetime_mock:
        datetime_mock.utcnow.return_value = datetime(
            year=2023, month=10, day=26, hour=21, minute=37
        )
        yield datetime_mock


def _set_usage(
    cache: InMemoryWorkflowsCache,
    strategy_name: str,
    limit_type: StrategyLimitType,
    value: int,
) -> None:
    cache.set(
        key=generate_cache_key_for_active_learning_usage(
            limit_type=limit_type,
            workspace="some",
            project="other",
            strategy_name=strategy_name,
        ),
        value={"usage": value},
        expire=LIMIT_TYPE2KEY_EXPIRATION[limit_type],
    )


def _usage(
    cache: InMemoryWorkflowsCache,
    strategy_name: str,
    limit_type: StrategyLimitType,
) -> Optional[int]:
    return get_current_strategy_limit_usage(
        cache=cache,
        workspace="some",
        project="other",
        strategy_name=strategy_name,
        limit_type=limit_type,
    )


@pytest.mark.parametrize(
    "limit_type, expected_result",
    [
        (
            StrategyLimitType.MINUTELY,
            "active_learning:usage:some:other:my_strategy:minute_37",
        ),
        (
            StrategyLimitType.HOURLY,
            "active_learning:usage:some:other:my_strategy:hour_21",
        ),
        (
            StrategyLimitType.DAILY,
            "active_learning:usage:some:other:my_strategy:day_2023_10_26",
        ),
    ],
)
def test_generate_cache_key_for_active_learning_usage(
    limit_type: StrategyLimitType,
    expected_result: str,
) -> None:
    result = generate_cache_key_for_active_learning_usage(
        limit_type=limit_type,
        workspace="some",
        project="other",
        strategy_name="my_strategy",
    )

    assert result == expected_result


def test_get_current_strategy_limit_usage_when_usage_not_set() -> None:
    cache = InMemoryWorkflowsCache()

    result = _usage(cache, "my-strategy", StrategyLimitType.DAILY)

    assert result is None


def test_get_current_strategy_limit_usage_when_usage_previously_set() -> None:
    cache = InMemoryWorkflowsCache()

    _set_usage(cache, "my-strategy", StrategyLimitType.DAILY, 39)
    result = _usage(cache, "my-strategy", StrategyLimitType.DAILY)

    assert result == 39


def test_return_strategy_credit_when_usage_value_is_not_set() -> None:
    cache = _RecordingCache()

    return_strategy_credit(
        cache=cache, workspace="some", project="other", strategy_name="my-strategy"
    )

    assert cache.expirations == {}
    for limit_type in StrategyLimitType:
        assert _usage(cache, "my-strategy", limit_type) is None


def test_return_strategy_credit_when_usage_value_is_zero() -> None:
    cache = InMemoryWorkflowsCache()
    for limit_type in StrategyLimitType:
        _set_usage(cache, "my-strategy", limit_type, 0)

    return_strategy_credit(
        cache=cache, workspace="some", project="other", strategy_name="my-strategy"
    )

    for limit_type in StrategyLimitType:
        assert _usage(cache, "my-strategy", limit_type) == 0


def test_return_strategy_credit_when_usage_value_is_greater_than_zero() -> None:
    cache = InMemoryWorkflowsCache()
    for limit_type in StrategyLimitType:
        _set_usage(cache, "my-strategy", limit_type, 10)

    return_strategy_credit(
        cache=cache, workspace="some", project="other", strategy_name="my-strategy"
    )

    for limit_type in StrategyLimitType:
        assert _usage(cache, "my-strategy", limit_type) == 9


def test_return_strategy_credit_touches_only_the_periods_that_hold_usage() -> None:
    cache = InMemoryWorkflowsCache()
    _set_usage(cache, "my-strategy", StrategyLimitType.HOURLY, 4)

    return_strategy_credit(
        cache=cache, workspace="some", project="other", strategy_name="my-strategy"
    )

    assert _usage(cache, "my-strategy", StrategyLimitType.MINUTELY) is None
    assert _usage(cache, "my-strategy", StrategyLimitType.HOURLY) == 3
    assert _usage(cache, "my-strategy", StrategyLimitType.DAILY) is None


def test_consume_strategy_credit_when_usage_value_is_not_set() -> None:
    cache = _RecordingCache()

    result = use_credit_of_matching_strategy(
        cache=cache,
        workspace="some",
        project="other",
        matching_strategies_limits=OrderedDict({"my-strategy": []}),
    )

    assert result == "my-strategy"
    for limit_type in StrategyLimitType:
        assert _usage(cache, "my-strategy", limit_type) == 1
    assert sorted(cache.expirations.values()) == [120, 7200, 90000]


def test_consume_strategy_credit_when_usage_value_is_set() -> None:
    cache = InMemoryWorkflowsCache()
    for limit_type in StrategyLimitType:
        _set_usage(cache, "my-strategy", limit_type, 10)

    result = use_credit_of_matching_strategy(
        cache=cache,
        workspace="some",
        project="other",
        matching_strategies_limits=OrderedDict({"my-strategy": []}),
    )

    assert result == "my-strategy"
    for limit_type in StrategyLimitType:
        assert _usage(cache, "my-strategy", limit_type) == 11


def test_datapoint_is_rejected_when_limit_usage_is_reached() -> None:
    cache = InMemoryWorkflowsCache()
    _set_usage(cache, "my-strategy", StrategyLimitType.MINUTELY, 30)

    result = find_strategy_with_spare_usage_credit(
        cache=cache,
        workspace="some",
        project="other",
        matching_strategies_limits=OrderedDict(
            {
                "my-strategy": [
                    StrategyLimit(limit_type=StrategyLimitType.MINUTELY, value=30)
                ]
            }
        ),
    )

    assert result is None


def test_datapoint_is_not_rejected_when_limit_usage_is_below_the_limit() -> None:
    cache = InMemoryWorkflowsCache()
    _set_usage(cache, "my-strategy", StrategyLimitType.MINUTELY, 29)

    result = find_strategy_with_spare_usage_credit(
        cache=cache,
        workspace="some",
        project="other",
        matching_strategies_limits=OrderedDict(
            {
                "my-strategy": [
                    StrategyLimit(limit_type=StrategyLimitType.MINUTELY, value=30)
                ]
            }
        ),
    )

    assert result == "my-strategy"


def test_datapoint_is_not_rejected_when_no_limits_assigned() -> None:
    cache = InMemoryWorkflowsCache()
    for limit_type in StrategyLimitType:
        _set_usage(cache, "my-strategy", limit_type, 1_000_000)

    result = find_strategy_with_spare_usage_credit(
        cache=cache,
        workspace="some",
        project="other",
        matching_strategies_limits=OrderedDict({"my-strategy": []}),
    )

    assert result == "my-strategy"


@pytest.mark.parametrize(
    "usages, expected_result",
    [
        ((9, 100, 109), None),
        ((9, 99, 999), "my-strategy"),
    ],
)
def test_datapoint_rejection_based_on_strategy_usage_limits(
    usages: tuple, expected_result: Optional[str]
) -> None:
    cache = InMemoryWorkflowsCache()
    for limit_type, usage in zip(StrategyLimitType, usages):
        _set_usage(cache, "my-strategy", limit_type, usage)
    strategy_limits = [
        StrategyLimit(limit_type=StrategyLimitType.MINUTELY, value=10),
        StrategyLimit(limit_type=StrategyLimitType.HOURLY, value=100),
        StrategyLimit(limit_type=StrategyLimitType.DAILY, value=1000),
    ]

    result = find_strategy_with_spare_usage_credit(
        cache=cache,
        workspace="some",
        project="other",
        matching_strategies_limits=OrderedDict({"my-strategy": strategy_limits}),
    )

    assert result == expected_result


def test_find_strategy_with_spare_limit_when_strategy_exists() -> None:
    cache = InMemoryWorkflowsCache()
    _set_usage(cache, "strategy_c", StrategyLimitType.MINUTELY, 10)
    _set_usage(cache, "strategy_a", StrategyLimitType.DAILY, 1002)
    matching_strategies_limits = OrderedDict(
        [
            (
                "strategy_c",
                [
                    StrategyLimit(limit_type=StrategyLimitType.MINUTELY, value=10),
                    StrategyLimit(limit_type=StrategyLimitType.DAILY, value=1000),
                ],
            ),
            (
                "strategy_b",
                [
                    StrategyLimit(limit_type=StrategyLimitType.MINUTELY, value=11),
                    StrategyLimit(limit_type=StrategyLimitType.DAILY, value=1001),
                ],
            ),
            (
                "strategy_a",
                [
                    StrategyLimit(limit_type=StrategyLimitType.MINUTELY, value=12),
                    StrategyLimit(limit_type=StrategyLimitType.DAILY, value=1002),
                ],
            ),
        ]
    )

    result = find_strategy_with_spare_usage_credit(
        cache=cache,
        workspace="some",
        project="other",
        matching_strategies_limits=matching_strategies_limits,
    )

    assert result == "strategy_b"


def test_find_strategy_with_spare_limit_when_strategy_does_not_exist() -> None:
    cache = InMemoryWorkflowsCache()
    _set_usage(cache, "strategy_c", StrategyLimitType.MINUTELY, 10)
    _set_usage(cache, "strategy_b", StrategyLimitType.DAILY, 1001)
    matching_strategies_limits = OrderedDict(
        [
            (
                "strategy_c",
                [
                    StrategyLimit(limit_type=StrategyLimitType.MINUTELY, value=10),
                    StrategyLimit(limit_type=StrategyLimitType.DAILY, value=1000),
                ],
            ),
            (
                "strategy_b",
                [
                    StrategyLimit(limit_type=StrategyLimitType.MINUTELY, value=11),
                    StrategyLimit(limit_type=StrategyLimitType.DAILY, value=1001),
                ],
            ),
        ]
    )

    result = find_strategy_with_spare_usage_credit(
        cache=cache,
        workspace="some",
        project="other",
        matching_strategies_limits=matching_strategies_limits,
    )

    assert result is None


def test_lock_limits() -> None:
    cache = MagicMock()

    with lock_limits(cache=cache, workspace="some", project="other"):
        pass

    cache.lock.assert_called_once_with(
        key="active_learning:usage:some:other:usage:lock",
        expire=5,
    )


def test_lock_limits_serialises_callers_when_the_cache_has_no_lock() -> None:
    cache = InMemoryWorkflowsCache()

    with lock_limits(cache=cache, workspace="some", project="other") as first:
        assert first.locked()
    with lock_limits(cache=cache, workspace="some", project="other") as second:
        assert second is first
    with lock_limits(cache=cache, workspace="some", project="another") as third:
        assert third is not first


def test_use_credit_of_matching_strategy_when_spare_strategy_not_found() -> None:
    cache = _RecordingCache()
    _set_usage(cache, "some", StrategyLimitType.HOURLY, 5)
    cache.expirations.clear()

    result = use_credit_of_matching_strategy(
        cache=cache,
        workspace="some",
        project="other",
        matching_strategies_limits=OrderedDict(
            {"some": [StrategyLimit(limit_type=StrategyLimitType.HOURLY, value=5)]}
        ),
    )

    assert result is None
    assert cache.expirations == {}
    assert cache.locked_keys == ["active_learning:usage:some:other:usage:lock"]


def test_use_credit_of_matching_strategy_when_spare_strategy_found() -> None:
    cache = _RecordingCache()
    _set_usage(cache, "b", StrategyLimitType.HOURLY, 5)

    result = use_credit_of_matching_strategy(
        cache=cache,
        workspace="some",
        project="other",
        matching_strategies_limits=OrderedDict(
            {
                "b": [StrategyLimit(limit_type=StrategyLimitType.HOURLY, value=5)],
                "a": [],
            }
        ),
    )

    assert result == "a"
    for limit_type in StrategyLimitType:
        assert _usage(cache, "a", limit_type) == 1
    assert _usage(cache, "b", StrategyLimitType.HOURLY) == 5
    assert _usage(cache, "b", StrategyLimitType.MINUTELY) is None
    assert cache.locked_keys == ["active_learning:usage:some:other:usage:lock"]


@pytest.mark.timeout(60)
def test_concurrent_threads_consume_exactly_the_limit() -> None:
    threads_number, limit = 24, 5
    cache = _SlowReadCache()
    matching_strategies_limits = OrderedDict(
        {
            "my-strategy": [
                StrategyLimit(limit_type=StrategyLimitType.MINUTELY, value=limit)
            ]
        }
    )
    start = threading.Barrier(threads_number)
    results: List[Optional[str]] = []

    def consume() -> None:
        start.wait()
        results.append(
            use_credit_of_matching_strategy(
                cache=cache,
                workspace="some",
                project="other",
                matching_strategies_limits=matching_strategies_limits,
            )
        )

    threads = [threading.Thread(target=consume) for _ in range(threads_number)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(results) == threads_number
    assert len([result for result in results if result is not None]) == limit
    for limit_type in StrategyLimitType:
        assert _usage(cache, "my-strategy", limit_type) == limit


@pytest.mark.timeout(60)
def test_concurrent_threads_return_every_consumed_credit() -> None:
    threads_number = 16
    cache = _SlowReadCache()
    for limit_type in StrategyLimitType:
        _set_usage(cache, "my-strategy", limit_type, threads_number)
    start = threading.Barrier(threads_number)

    def give_back() -> None:
        start.wait()
        return_strategy_credit(
            cache=cache,
            workspace="some",
            project="other",
            strategy_name="my-strategy",
        )

    threads = [threading.Thread(target=give_back) for _ in range(threads_number)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    for limit_type in StrategyLimitType:
        assert _usage(cache, "my-strategy", limit_type) == 0

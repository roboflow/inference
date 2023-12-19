from collections import OrderedDict
from datetime import datetime
from unittest import mock
from unittest.mock import MagicMock, call

import pytest

from inference.core.active_learning import cache_operations
from inference.core.active_learning.cache_operations import (
    consume_strategy_limit_usage_credit,
    datapoint_should_be_rejected_based_on_limit_usage,
    datapoint_should_be_rejected_based_on_strategy_usage_limits,
    find_strategy_with_spare_usage_credit,
    generate_cache_key_for_active_learning_usage,
    generate_cache_key_for_active_learning_usage_lock,
    get_current_strategy_limit_usage,
    lock_limits,
    return_strategy_limit_usage_credit,
    set_current_strategy_limit_usage,
    use_credit_of_matching_strategy,
)
from inference.core.active_learning.entities import StrategyLimit, StrategyLimitType
from inference.core.cache import MemoryCache


@mock.patch.object(cache_operations, "datetime")
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
    datetime_mock: MagicMock,
    limit_type: StrategyLimitType,
    expected_result: str,
) -> None:
    # given
    datetime_mock.utcnow.return_value = datetime(
        year=2023, month=10, day=26, hour=21, minute=37
    )
    # when
    result = generate_cache_key_for_active_learning_usage(
        limit_type=limit_type,
        workspace="some",
        project="other",
        strategy_name="my_strategy",
    )

    # then
    assert result == expected_result


def test_generate_cache_key_for_active_learning_usage_lock() -> None:
    # when
    result = generate_cache_key_for_active_learning_usage_lock(
        workspace="some",
        project="other",
    )

    # then
    assert result == "active_learning:usage:some:other:usage:lock"


def test_get_current_strategy_limit_usage_when_usage_not_set() -> None:
    # given
    cache = MemoryCache()

    # when
    result = get_current_strategy_limit_usage(
        cache=cache,
        workspace="some",
        project="other",
        strategy_name="my-strategy",
        limit_type=StrategyLimitType.DAILY,
    )

    # then
    assert result is None


def test_get_current_strategy_limit_usage_when_usage_previously_set() -> None:
    # given
    cache = MemoryCache()

    # when
    set_current_strategy_limit_usage(
        current_value=39,
        cache=cache,
        workspace="some",
        project="other",
        strategy_name="my-strategy",
        limit_type=StrategyLimitType.DAILY,
    )
    result = get_current_strategy_limit_usage(
        cache=cache,
        workspace="some",
        project="other",
        strategy_name="my-strategy",
        limit_type=StrategyLimitType.DAILY,
    )

    # then
    assert result == 39


@mock.patch.object(cache_operations, "set_current_strategy_limit_usage")
@mock.patch.object(cache_operations, "get_current_strategy_limit_usage")
def test_return_strategy_limit_usage_credit_when_usage_value_is_not_set(
    get_current_strategy_limit_usage_mock: MagicMock,
    set_current_strategy_limit_usage_mock: MagicMock,
) -> None:
    # given
    cache = MagicMock()
    get_current_strategy_limit_usage_mock.return_value = None

    # when
    return_strategy_limit_usage_credit(
        cache=cache,
        workspace="some",
        project="other",
        strategy_name="my-strategy",
        limit_type=StrategyLimitType.MINUTELY,
    )

    # then
    set_current_strategy_limit_usage_mock.assert_not_called()
    get_current_strategy_limit_usage_mock.assert_called_once_with(
        cache=cache,
        workspace="some",
        project="other",
        strategy_name="my-strategy",
        limit_type=StrategyLimitType.MINUTELY,
    )


@mock.patch.object(cache_operations, "set_current_strategy_limit_usage")
@mock.patch.object(cache_operations, "get_current_strategy_limit_usage")
def test_return_strategy_limit_usage_credit_when_usage_value_is_zero(
    get_current_strategy_limit_usage_mock: MagicMock,
    set_current_strategy_limit_usage_mock: MagicMock,
) -> None:
    # given
    cache = MagicMock()
    get_current_strategy_limit_usage_mock.return_value = 0

    # when
    return_strategy_limit_usage_credit(
        cache=cache,
        workspace="some",
        project="other",
        strategy_name="my-strategy",
        limit_type=StrategyLimitType.MINUTELY,
    )

    # then
    set_current_strategy_limit_usage_mock.assert_called_once_with(
        current_value=0,
        cache=cache,
        workspace="some",
        project="other",
        strategy_name="my-strategy",
        limit_type=StrategyLimitType.MINUTELY,
    )
    get_current_strategy_limit_usage_mock.assert_called_once_with(
        cache=cache,
        workspace="some",
        project="other",
        strategy_name="my-strategy",
        limit_type=StrategyLimitType.MINUTELY,
    )


@mock.patch.object(cache_operations, "set_current_strategy_limit_usage")
@mock.patch.object(cache_operations, "get_current_strategy_limit_usage")
def test_return_strategy_limit_usage_credit_when_usage_value_is_greater_than_zero(
    get_current_strategy_limit_usage_mock: MagicMock,
    set_current_strategy_limit_usage_mock: MagicMock,
) -> None:
    # given
    cache = MagicMock()
    get_current_strategy_limit_usage_mock.return_value = 10

    # when
    return_strategy_limit_usage_credit(
        cache=cache,
        workspace="some",
        project="other",
        strategy_name="my-strategy",
        limit_type=StrategyLimitType.MINUTELY,
    )

    # then
    set_current_strategy_limit_usage_mock.assert_called_once_with(
        current_value=9,
        cache=cache,
        workspace="some",
        project="other",
        strategy_name="my-strategy",
        limit_type=StrategyLimitType.MINUTELY,
    )
    get_current_strategy_limit_usage_mock.assert_called_once_with(
        cache=cache,
        workspace="some",
        project="other",
        strategy_name="my-strategy",
        limit_type=StrategyLimitType.MINUTELY,
    )


@mock.patch.object(cache_operations, "set_current_strategy_limit_usage")
@mock.patch.object(cache_operations, "get_current_strategy_limit_usage")
def test_consume_strategy_limit_usage_credit_when_usage_value_is_not_set(
    get_current_strategy_limit_usage_mock: MagicMock,
    set_current_strategy_limit_usage_mock: MagicMock,
) -> None:
    # given
    cache = MagicMock()
    get_current_strategy_limit_usage_mock.return_value = None

    # when
    consume_strategy_limit_usage_credit(
        cache=cache,
        workspace="some",
        project="other",
        strategy_name="my-strategy",
        limit_type=StrategyLimitType.MINUTELY,
    )

    # then
    set_current_strategy_limit_usage_mock.assert_called_once_with(
        current_value=1,
        cache=cache,
        workspace="some",
        project="other",
        strategy_name="my-strategy",
        limit_type=StrategyLimitType.MINUTELY,
    )
    get_current_strategy_limit_usage_mock.assert_called_once_with(
        cache=cache,
        workspace="some",
        project="other",
        strategy_name="my-strategy",
        limit_type=StrategyLimitType.MINUTELY,
    )


@mock.patch.object(cache_operations, "set_current_strategy_limit_usage")
@mock.patch.object(cache_operations, "get_current_strategy_limit_usage")
def test_consume_strategy_limit_usage_credit_when_usage_value_is_set(
    get_current_strategy_limit_usage_mock: MagicMock,
    set_current_strategy_limit_usage_mock: MagicMock,
) -> None:
    # given
    cache = MagicMock()
    get_current_strategy_limit_usage_mock.return_value = 10

    # when
    consume_strategy_limit_usage_credit(
        cache=cache,
        workspace="some",
        project="other",
        strategy_name="my-strategy",
        limit_type=StrategyLimitType.MINUTELY,
    )

    # then
    set_current_strategy_limit_usage_mock.assert_called_once_with(
        current_value=11,
        cache=cache,
        workspace="some",
        project="other",
        strategy_name="my-strategy",
        limit_type=StrategyLimitType.MINUTELY,
    )
    get_current_strategy_limit_usage_mock.assert_called_once_with(
        cache=cache,
        workspace="some",
        project="other",
        strategy_name="my-strategy",
        limit_type=StrategyLimitType.MINUTELY,
    )


@mock.patch.object(cache_operations, "get_current_strategy_limit_usage")
def test_datapoint_should_be_rejected_based_on_limit_usage_when_rejection_should_happen(
    get_current_strategy_limit_usage_mock: MagicMock,
) -> None:
    # given
    cache = MagicMock()
    get_current_strategy_limit_usage_mock.return_value = 30

    # when
    result = datapoint_should_be_rejected_based_on_limit_usage(
        cache=cache,
        workspace="some",
        project="other",
        strategy_name="my-strategy",
        strategy_limit=StrategyLimit(limit_type=StrategyLimitType.MINUTELY, value=30),
    )

    # then
    assert result is True


@mock.patch.object(cache_operations, "get_current_strategy_limit_usage")
def test_datapoint_should_be_rejected_based_on_limit_usage_when_rejection_should_not_happen(
    get_current_strategy_limit_usage_mock: MagicMock,
) -> None:
    # given
    cache = MagicMock()
    get_current_strategy_limit_usage_mock.return_value = 29

    # when
    result = datapoint_should_be_rejected_based_on_limit_usage(
        cache=cache,
        workspace="some",
        project="other",
        strategy_name="my-strategy",
        strategy_limit=StrategyLimit(limit_type=StrategyLimitType.MINUTELY, value=30),
    )

    # then
    assert result is False


def test_datapoint_should_be_rejected_based_on_strategy_usage_limits_when_no_limits_assigned() -> (
    None
):
    # when
    result = datapoint_should_be_rejected_based_on_strategy_usage_limits(
        cache=MagicMock(),
        workspace="some",
        project="other",
        strategy_name="my-strategy",
        strategy_limits=[],
    )

    # then
    assert result is False


@mock.patch.object(cache_operations, "get_current_strategy_limit_usage")
def test_datapoint_should_be_rejected_based_on_strategy_usage_limits_when_rejection_should_happen(
    get_current_strategy_limit_usage_mock: MagicMock,
) -> None:
    # given
    get_current_strategy_limit_usage_mock.side_effect = [9, 100, 109]
    strategy_limits = [
        StrategyLimit(limit_type=StrategyLimitType.MINUTELY, value=10),
        StrategyLimit(limit_type=StrategyLimitType.HOURLY, value=100),
        StrategyLimit(limit_type=StrategyLimitType.DAILY, value=1000),
    ]

    # when
    result = datapoint_should_be_rejected_based_on_strategy_usage_limits(
        cache=MagicMock(),
        workspace="some",
        project="other",
        strategy_name="my-strategy",
        strategy_limits=strategy_limits,
    )

    # then
    assert result is True


@mock.patch.object(cache_operations, "get_current_strategy_limit_usage")
def test_datapoint_should_be_rejected_based_on_strategy_usage_limits_when_rejection_should_not_happen(
    get_current_strategy_limit_usage_mock: MagicMock,
) -> None:
    # given
    get_current_strategy_limit_usage_mock.side_effect = [9, 99, 999]
    strategy_limits = [
        StrategyLimit(limit_type=StrategyLimitType.MINUTELY, value=10),
        StrategyLimit(limit_type=StrategyLimitType.HOURLY, value=100),
        StrategyLimit(limit_type=StrategyLimitType.DAILY, value=1000),
    ]

    # when
    result = datapoint_should_be_rejected_based_on_strategy_usage_limits(
        cache=MagicMock(),
        workspace="some",
        project="other",
        strategy_name="my-strategy",
        strategy_limits=strategy_limits,
    )

    # then
    assert result is False


@mock.patch.object(
    cache_operations, "datapoint_should_be_rejected_based_on_strategy_usage_limits"
)
def test_find_strategy_with_spare_limit_when_strategy_exists(
    datapoint_should_be_rejected_based_on_strategy_usage_limits_mock: MagicMock,
) -> None:
    # given
    cache = MagicMock()
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
    datapoint_should_be_rejected_based_on_strategy_usage_limits_mock.side_effect = [
        True,
        False,
        True,
    ]

    # when
    result = find_strategy_with_spare_usage_credit(
        cache=cache,
        workspace="a",
        project="b",
        matching_strategies_limits=matching_strategies_limits,
    )

    # then
    assert result == "strategy_b"
    datapoint_should_be_rejected_based_on_strategy_usage_limits_mock.assert_has_calls(
        [
            call(
                cache=cache,
                workspace="a",
                project="b",
                strategy_name="strategy_c",
                strategy_limits=matching_strategies_limits["strategy_c"],
            ),
            call(
                cache=cache,
                workspace="a",
                project="b",
                strategy_name="strategy_b",
                strategy_limits=matching_strategies_limits["strategy_b"],
            ),
        ]
    )


@mock.patch.object(
    cache_operations, "datapoint_should_be_rejected_based_on_strategy_usage_limits"
)
def test_find_strategy_with_spare_limit_when_strategy_does_not_exist(
    datapoint_should_be_rejected_based_on_strategy_usage_limits_mock: MagicMock,
) -> None:
    # given
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
    datapoint_should_be_rejected_based_on_strategy_usage_limits_mock.return_value = True

    # when
    result = find_strategy_with_spare_usage_credit(
        cache=MagicMock(),
        workspace="a",
        project="b",
        matching_strategies_limits=matching_strategies_limits,
    )

    # then
    assert result is None


def test_lock_limits() -> None:
    # given
    cache = MagicMock()

    # when
    with lock_limits(cache=cache, workspace="some", project="other"):
        pass

    # then
    cache.lock.assert_called_once_with(
        key="active_learning:usage:some:other:usage:lock",
        expire=5,
    )


@mock.patch.object(cache_operations, "consume_strategy_limits_usage_credit")
@mock.patch.object(cache_operations, "find_strategy_with_spare_usage_credit")
def test_use_credit_of_matching_strategy_when_spare_strategy_not_found(
    find_strategy_with_spare_usage_credit_mock: MagicMock,
    consume_strategy_limits_usage_credit_mock: MagicMock,
) -> None:
    # given
    cache = MagicMock()
    find_strategy_with_spare_usage_credit_mock.return_value = None

    # when
    result = use_credit_of_matching_strategy(
        cache=cache,
        workspace="some",
        project="other",
        matching_strategies_limits=OrderedDict({"some": []}),
    )

    # then
    assert result is None
    consume_strategy_limits_usage_credit_mock.assert_not_called()
    cache.lock.assert_called_once()


@mock.patch.object(cache_operations, "consume_strategy_limits_usage_credit")
@mock.patch.object(cache_operations, "find_strategy_with_spare_usage_credit")
def test_use_credit_of_matching_strategy_when_spare_strategy_found(
    find_strategy_with_spare_usage_credit_mock: MagicMock,
    consume_strategy_limits_usage_credit_mock: MagicMock,
) -> None:
    # given
    cache = MagicMock()
    find_strategy_with_spare_usage_credit_mock.return_value = "a"

    # when
    result = use_credit_of_matching_strategy(
        cache=cache,
        workspace="some",
        project="other",
        matching_strategies_limits=OrderedDict({"b": [], "a": []}),
    )

    # then
    assert result == "a"
    consume_strategy_limits_usage_credit_mock.assert_called_once_with(
        cache=cache,
        workspace="some",
        project="other",
        strategy_name="a",
    )
    cache.lock.assert_called_once()

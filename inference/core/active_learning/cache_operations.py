import threading
from contextlib import contextmanager
from datetime import datetime
from typing import Generator, List, Optional, OrderedDict, Union

import redis.lock

from inference.core import logger
from inference.core.active_learning.entities import StrategyLimit, StrategyLimitType
from inference.core.active_learning.utils import TIMESTAMP_FORMAT
from inference.core.cache.base import BaseCache

MAX_LOCK_TIME = 5
SECONDS_IN_HOUR = 60 * 60
USAGE_KEY = "usage"

LIMIT_TYPE2KEY_INFIX_GENERATOR = {
    StrategyLimitType.MINUTELY: lambda: f"minute_{datetime.utcnow().minute}",
    StrategyLimitType.HOURLY: lambda: f"hour_{datetime.utcnow().hour}",
    StrategyLimitType.DAILY: lambda: f"day_{datetime.utcnow().strftime(TIMESTAMP_FORMAT)}",
}
LIMIT_TYPE2KEY_EXPIRATION = {
    StrategyLimitType.MINUTELY: 120,
    StrategyLimitType.HOURLY: 2 * SECONDS_IN_HOUR,
    StrategyLimitType.DAILY: 25 * SECONDS_IN_HOUR,
}


def use_credit_of_matching_strategy(
    cache: BaseCache,
    workspace: str,
    project: str,
    matching_strategies_limits: OrderedDict[str, List[StrategyLimit]],
) -> Optional[str]:
    # In scope of this function, cache keys updates regarding usage limits for
    # specific :workspace and :project are locked - to ensure increment to be done atomically
    # Limits are accounted at the moment of registration - which may introduce inaccuracy
    # given that registration is postponed from prediction
    # Returns: strategy with spare credit if found - else None
    with lock_limits(cache=cache, workspace=workspace, project=project):
        strategy_with_spare_credit = find_strategy_with_spare_usage_credit(
            cache=cache,
            workspace=workspace,
            project=project,
            matching_strategies_limits=matching_strategies_limits,
        )
        if strategy_with_spare_credit is None:
            return None
        consume_strategy_limits_usage_credit(
            cache=cache,
            workspace=workspace,
            project=project,
            strategy_name=strategy_with_spare_credit,
        )
        return strategy_with_spare_credit


def return_strategy_credit(
    cache: BaseCache,
    workspace: str,
    project: str,
    strategy_name: str,
) -> None:
    # In scope of this function, cache keys updates regarding usage limits for
    # specific :workspace and :project are locked - to ensure decrement to be done atomically
    # Returning strategy is a bit naive (we may add to a pool of credits from the next period - but only
    # if we have previously taken from the previous one and some credits are used in the new pool) -
    # in favour of easier implementation.
    with lock_limits(cache=cache, workspace=workspace, project=project):
        return_strategy_limits_usage_credit(
            cache=cache,
            workspace=workspace,
            project=project,
            strategy_name=strategy_name,
        )


@contextmanager
def lock_limits(
    cache: BaseCache,
    workspace: str,
    project: str,
) -> Generator[Union[threading.Lock, redis.lock.Lock], None, None]:
    limits_lock_key = generate_cache_key_for_active_learning_usage_lock(
        workspace=workspace,
        project=project,
    )
    with cache.lock(key=limits_lock_key, expire=MAX_LOCK_TIME) as lock:
        yield lock


def find_strategy_with_spare_usage_credit(
    cache: BaseCache,
    workspace: str,
    project: str,
    matching_strategies_limits: OrderedDict[str, List[StrategyLimit]],
) -> Optional[str]:
    for strategy_name, strategy_limits in matching_strategies_limits.items():
        rejected_by_strategy = (
            datapoint_should_be_rejected_based_on_strategy_usage_limits(
                cache=cache,
                workspace=workspace,
                project=project,
                strategy_name=strategy_name,
                strategy_limits=strategy_limits,
            )
        )
        if not rejected_by_strategy:
            return strategy_name
    return None


def datapoint_should_be_rejected_based_on_strategy_usage_limits(
    cache: BaseCache,
    workspace: str,
    project: str,
    strategy_name: str,
    strategy_limits: List[StrategyLimit],
) -> bool:
    for strategy_limit in strategy_limits:
        limit_reached = datapoint_should_be_rejected_based_on_limit_usage(
            cache=cache,
            workspace=workspace,
            project=project,
            strategy_name=strategy_name,
            strategy_limit=strategy_limit,
        )
        if limit_reached:
            logger.debug(
                f"Violated Active Learning strategy limit: {strategy_limit.limit_type.name} "
                f"with value {strategy_limit.value} for sampling strategy: {strategy_name}."
            )
            return True
    return False


def datapoint_should_be_rejected_based_on_limit_usage(
    cache: BaseCache,
    workspace: str,
    project: str,
    strategy_name: str,
    strategy_limit: StrategyLimit,
) -> bool:
    current_usage = get_current_strategy_limit_usage(
        cache=cache,
        workspace=workspace,
        project=project,
        strategy_name=strategy_name,
        limit_type=strategy_limit.limit_type,
    )
    if current_usage is None:
        current_usage = 0
    return current_usage >= strategy_limit.value


def consume_strategy_limits_usage_credit(
    cache: BaseCache,
    workspace: str,
    project: str,
    strategy_name: str,
) -> None:
    for limit_type in StrategyLimitType:
        consume_strategy_limit_usage_credit(
            cache=cache,
            workspace=workspace,
            project=project,
            strategy_name=strategy_name,
            limit_type=limit_type,
        )


def consume_strategy_limit_usage_credit(
    cache: BaseCache,
    workspace: str,
    project: str,
    strategy_name: str,
    limit_type: StrategyLimitType,
) -> None:
    current_value = get_current_strategy_limit_usage(
        cache=cache,
        limit_type=limit_type,
        workspace=workspace,
        project=project,
        strategy_name=strategy_name,
    )
    if current_value is None:
        current_value = 0
    current_value += 1
    set_current_strategy_limit_usage(
        current_value=current_value,
        cache=cache,
        limit_type=limit_type,
        workspace=workspace,
        project=project,
        strategy_name=strategy_name,
    )


def return_strategy_limits_usage_credit(
    cache: BaseCache,
    workspace: str,
    project: str,
    strategy_name: str,
) -> None:
    for limit_type in StrategyLimitType:
        return_strategy_limit_usage_credit(
            cache=cache,
            workspace=workspace,
            project=project,
            strategy_name=strategy_name,
            limit_type=limit_type,
        )


def return_strategy_limit_usage_credit(
    cache: BaseCache,
    workspace: str,
    project: str,
    strategy_name: str,
    limit_type: StrategyLimitType,
) -> None:
    current_value = get_current_strategy_limit_usage(
        cache=cache,
        limit_type=limit_type,
        workspace=workspace,
        project=project,
        strategy_name=strategy_name,
    )
    if current_value is None:
        return None
    current_value = max(current_value - 1, 0)
    set_current_strategy_limit_usage(
        current_value=current_value,
        cache=cache,
        limit_type=limit_type,
        workspace=workspace,
        project=project,
        strategy_name=strategy_name,
    )


def get_current_strategy_limit_usage(
    cache: BaseCache,
    workspace: str,
    project: str,
    strategy_name: str,
    limit_type: StrategyLimitType,
) -> Optional[int]:
    usage_key = generate_cache_key_for_active_learning_usage(
        limit_type=limit_type,
        workspace=workspace,
        project=project,
        strategy_name=strategy_name,
    )
    value = cache.get(usage_key)
    if value is None:
        return value
    return value[USAGE_KEY]


def set_current_strategy_limit_usage(
    current_value: int,
    cache: BaseCache,
    workspace: str,
    project: str,
    strategy_name: str,
    limit_type: StrategyLimitType,
) -> None:
    usage_key = generate_cache_key_for_active_learning_usage(
        limit_type=limit_type,
        workspace=workspace,
        project=project,
        strategy_name=strategy_name,
    )
    expire = LIMIT_TYPE2KEY_EXPIRATION[limit_type]
    cache.set(key=usage_key, value={USAGE_KEY: current_value}, expire=expire)  # type: ignore


def generate_cache_key_for_active_learning_usage_lock(
    workspace: str,
    project: str,
) -> str:
    return f"active_learning:usage:{workspace}:{project}:usage:lock"


def generate_cache_key_for_active_learning_usage(
    limit_type: StrategyLimitType,
    workspace: str,
    project: str,
    strategy_name: str,
) -> str:
    time_infix = LIMIT_TYPE2KEY_INFIX_GENERATOR[limit_type]()
    return f"active_learning:usage:{workspace}:{project}:{strategy_name}:{time_infix}"

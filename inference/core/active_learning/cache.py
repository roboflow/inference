from datetime import datetime
from typing import List, Dict, Optional

from inference.core.active_learning.entities import StrategyLimit, StrategyLimitType
from inference.core.active_learning.utils import TIMESTAMP_FORMAT
from inference.core.cache.base import BaseCache

MAX_LOCK_TIME = 5
SECONDS_IN_HOUR = 60 * 60
LIMIT_TYPE2KEY_INFIX_GENERATOR = {
    StrategyLimitType.HOURLY: lambda: f"hour_{datetime.now().hour}",
    StrategyLimitType.DAILY: lambda: f"day_{datetime.now().strftime(TIMESTAMP_FORMAT)}",
}
LIMIT_TYPE2KEY_EXPIRATION = {
    StrategyLimitType.HOURLY: 2 * SECONDS_IN_HOUR,
    StrategyLimitType.DAILY: 25 * SECONDS_IN_HOUR,
}


def find_strategy_with_spare_usage_limit(
    cache: BaseCache,
    workspace: str,
    project: str,
    matching_strategies_limits: Dict[str, List[StrategyLimit]],
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
        limit_reached = datapoint_should_be_rejected_based_on_usage_limit(
            cache=cache,
            workspace=workspace,
            project=project,
            strategy_name=strategy_name,
            strategy_limit=strategy_limit,
        )
        if limit_reached:
            return True
    return False


def datapoint_should_be_rejected_based_on_usage_limit(
    cache: BaseCache,
    workspace: str,
    project: str,
    strategy_name: str,
    strategy_limit: StrategyLimit,
) -> bool:
    current_usage = get_strategy_usage(
        cache=cache,
        workspace=workspace,
        project=project,
        strategy_name=strategy_name,
        limit_type=strategy_limit.limit_type,
    )
    if current_usage is None:
        return False
    return current_usage >= strategy_limit.value


def get_strategy_usage(
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
    print(f"getting: {usage_key} - {cache.get(usage_key)}")
    return cache.get(usage_key)


def increment_strategies_usage(
    cache: BaseCache,
    workspace: str,
    project: str,
    strategy_name: str,
) -> None:
    for limit_type in StrategyLimitType:
        increment_strategy_usage(
            cache=cache,
            workspace=workspace,
            project=project,
            strategy_name=strategy_name,
            limit_type=limit_type,
        )


def increment_strategy_usage(
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
    current_value = cache.get(usage_key)
    if current_value is None:
        current_value = 0
    current_value += 1
    expire = LIMIT_TYPE2KEY_EXPIRATION[limit_type]
    cache.set(key=usage_key, value=current_value, expire=expire)


def generate_cache_key_for_active_learning_usage_lock(
    workspace: str,
    project: str,
) -> str:
    return f"active_learning:usage:{workspace}:{project}:lock"


def generate_cache_key_for_active_learning_usage(
    limit_type: StrategyLimitType,
    workspace: str,
    project: str,
    strategy_name: Optional[str] = None,
) -> str:
    time_infix = LIMIT_TYPE2KEY_INFIX_GENERATOR[limit_type]()
    key = f"active_learning:usage:{time_infix}:{workspace}:{project}"
    if strategy_name is not None:
        key = f"{key}:{strategy_name}"
    return key

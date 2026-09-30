"""Usage-quota accounting and image preparation for the dataset-upload blocks.

Ported from `inference.core.active_learning.{entities,cache_operations,core}` so
the dataset-upload blocks run on a host that does not ship the `inference`
package. Cache keys, expiry times and the image-preparation contract are
unchanged, so quota state kept in a shared cache by an older server is read
and written the same way.

The one difference: `WorkflowsCache` declares only `get`/`set`. When the
injected cache also offers `lock(key, expire)` (the `inference` server's
Redis/in-memory caches do) it is used, exactly as before; otherwise a
process-local lock per key serialises the read-modify-write instead.
"""

import threading
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Generator, List, NamedTuple, Optional, Tuple

import numpy as np
from roboflow_workflows._compat_names import get_logger
from roboflow_workflows.prototypes.cache import WorkflowsCache
from roboflow_workflows.utils.images import (
    downscale_image_keeping_aspect_ratio,
    encode_image_to_jpeg_bytes,
)

logger = get_logger(__name__)

TIMESTAMP_FORMAT = "%Y_%m_%d"
MAX_LOCK_TIME = 5
SECONDS_IN_HOUR = 60 * 60
USAGE_KEY = "usage"


@dataclass(frozen=True)
class ImageDimensions:
    height: int
    width: int

    def to_hw(self) -> Tuple[int, int]:
        return self.height, self.width

    def to_wh(self) -> Tuple[int, int]:
        return self.width, self.height


class StrategyLimitType(Enum):
    MINUTELY = "minutely"
    HOURLY = "hourly"
    DAILY = "daily"


@dataclass(frozen=True)
class StrategyLimit:
    limit_type: StrategyLimitType
    value: int

    @classmethod
    def from_dict(cls, specification: dict) -> "StrategyLimit":
        return cls(
            limit_type=StrategyLimitType(specification["type"]),
            value=specification["value"],
        )


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

_LOCAL_LOCKS: Dict[str, threading.Lock] = {}
_LOCAL_LOCKS_GUARD = threading.Lock()


def use_credit_of_matching_strategy(
    cache: WorkflowsCache,
    workspace: str,
    project: str,
    matching_strategies_limits: "OrderedDict[str, List[StrategyLimit]]",
) -> Optional[str]:
    """Consume one credit of the first strategy under all its limits.

    Returns that strategy's name, or None when every strategy is exhausted.
    """
    with lock_limits(cache=cache, workspace=workspace, project=project):
        strategy_with_spare_credit = find_strategy_with_spare_usage_credit(
            cache=cache,
            workspace=workspace,
            project=project,
            matching_strategies_limits=matching_strategies_limits,
        )
        if strategy_with_spare_credit is None:
            return None
        _update_strategy_limits_usage(
            cache=cache,
            workspace=workspace,
            project=project,
            strategy_name=strategy_with_spare_credit,
            delta=1,
        )
        return strategy_with_spare_credit


def return_strategy_credit(
    cache: WorkflowsCache,
    workspace: str,
    project: str,
    strategy_name: str,
) -> None:
    with lock_limits(cache=cache, workspace=workspace, project=project):
        _update_strategy_limits_usage(
            cache=cache,
            workspace=workspace,
            project=project,
            strategy_name=strategy_name,
            delta=-1,
        )


@contextmanager
def lock_limits(
    cache: WorkflowsCache,
    workspace: str,
    project: str,
) -> Generator[Any, None, None]:
    limits_lock_key = f"active_learning:usage:{workspace}:{project}:usage:lock"
    cache_lock = getattr(cache, "lock", None)
    if callable(cache_lock):
        with cache_lock(key=limits_lock_key, expire=MAX_LOCK_TIME) as lock:
            yield lock
        return
    with _LOCAL_LOCKS_GUARD:
        lock = _LOCAL_LOCKS.setdefault(limits_lock_key, threading.Lock())
    with lock:
        yield lock


def find_strategy_with_spare_usage_credit(
    cache: WorkflowsCache,
    workspace: str,
    project: str,
    matching_strategies_limits: "OrderedDict[str, List[StrategyLimit]]",
) -> Optional[str]:
    for strategy_name, strategy_limits in matching_strategies_limits.items():
        rejected = any(
            _limit_reached(
                cache=cache,
                workspace=workspace,
                project=project,
                strategy_name=strategy_name,
                strategy_limit=strategy_limit,
            )
            for strategy_limit in strategy_limits
        )
        if not rejected:
            return strategy_name
    return None


def _limit_reached(
    cache: WorkflowsCache,
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
    if current_usage >= strategy_limit.value:
        logger.debug(
            f"Violated Active Learning strategy limit: {strategy_limit.limit_type.name} "
            f"with value {strategy_limit.value} for sampling strategy: {strategy_name}."
        )
        return True
    return False


def _update_strategy_limits_usage(
    cache: WorkflowsCache,
    workspace: str,
    project: str,
    strategy_name: str,
    delta: int,
) -> None:
    for limit_type in StrategyLimitType:
        current_value = get_current_strategy_limit_usage(
            cache=cache,
            workspace=workspace,
            project=project,
            strategy_name=strategy_name,
            limit_type=limit_type,
        )
        if current_value is None:
            if delta < 0:
                # Nothing was consumed in this period - nothing to return.
                continue
            current_value = 0
        cache.set(
            key=generate_cache_key_for_active_learning_usage(
                limit_type=limit_type,
                workspace=workspace,
                project=project,
                strategy_name=strategy_name,
            ),
            value={USAGE_KEY: max(current_value + delta, 0)},
            expire=LIMIT_TYPE2KEY_EXPIRATION[limit_type],
        )


def get_current_strategy_limit_usage(
    cache: WorkflowsCache,
    workspace: str,
    project: str,
    strategy_name: str,
    limit_type: StrategyLimitType,
) -> Optional[int]:
    value = cache.get(
        generate_cache_key_for_active_learning_usage(
            limit_type=limit_type,
            workspace=workspace,
            project=project,
            strategy_name=strategy_name,
        )
    )
    if value is None:
        return value
    return value[USAGE_KEY]


def generate_cache_key_for_active_learning_usage(
    limit_type: StrategyLimitType,
    workspace: str,
    project: str,
    strategy_name: str,
) -> str:
    time_infix = LIMIT_TYPE2KEY_INFIX_GENERATOR[limit_type]()
    return f"active_learning:usage:{workspace}:{project}:{strategy_name}:{time_infix}"


class PreparedRegistrationImage(NamedTuple):
    """JPEG bytes plus the exact size transform applied before upload."""

    encoded_image: bytes
    scaling_factor: float
    original_size_wh: Tuple[int, int]
    final_size_wh: Tuple[int, int]

    @property
    def scale_x(self) -> float:
        return self.final_size_wh[0] / self.original_size_wh[0]

    @property
    def scale_y(self) -> float:
        return self.final_size_wh[1] / self.original_size_wh[1]


def prepare_image_to_registration_with_metadata(
    image: np.ndarray,
    desired_size: Optional[ImageDimensions],
    jpeg_compression_level: int,
) -> PreparedRegistrationImage:
    original_size_wh = (int(image.shape[1]), int(image.shape[0]))
    if original_size_wh[0] <= 0 or original_size_wh[1] <= 0:
        width, height = original_size_wh
        raise ValueError(
            "Could not prepare image for registration because its dimensions "
            f"are invalid: width={width}, height={height}."
        )
    if desired_size is not None:
        image = downscale_image_keeping_aspect_ratio(
            image=image,
            desired_size=desired_size.to_wh(),
        )
    final_size_wh = (int(image.shape[1]), int(image.shape[0]))
    # Height ratio kept for callers that still expect a single factor.
    scaling_factor = final_size_wh[1] / original_size_wh[1]
    return PreparedRegistrationImage(
        encoded_image=encode_image_to_jpeg_bytes(
            image=image, jpeg_quality=jpeg_compression_level
        ),
        scaling_factor=scaling_factor,
        original_size_wh=original_size_wh,
        final_size_wh=final_size_wh,
    )

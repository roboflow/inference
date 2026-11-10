"""Host-owned upload quota accounting shared with the existing V1 uploader."""

from collections import OrderedDict
from typing import Optional

from inference.core.active_learning.cache_operations import (
    return_strategy_credit,
    use_credit_of_matching_strategy,
)
from inference.core.active_learning.entities import StrategyLimit, StrategyLimitType
from inference.core.cache.base import BaseCache


class SharedUploadQuota:
    """Reserve upload credit using a caller-owned host cache.

    Every session and V1 uploader that should share a quota must use the same
    host cache. This bridge never creates or closes that resource. Synchronization,
    key namespaces, expiration and window accounting belong to the existing V1
    credit functions and the injected cache.

    The future host upload operation owns reserve, upload and conditional refund
    together: dispatch nothing when denied, retain credit on success, and return
    it exactly once on a duplicate or failure. This helper does not track upload
    outcomes or make refunds idempotent. V1 refunds use the current time windows;
    a refund after rollover can therefore release credit from the new window.

    Args:
        cache (BaseCache): Shared host cache with working quota locks.

    Raises:
        ValueError: If no host cache is supplied.
    """

    def __init__(self, cache: BaseCache) -> None:
        if cache is None:
            raise ValueError("Upload quotas require an explicitly supplied host cache.")

        self._cache = cache

    def reserve(
        self,
        *,
        workspace: str,
        project: str,
        usage_quota_name: str,
        minutely_usage_limit: int,
        hourly_usage_limit: int,
        daily_usage_limit: int,
    ) -> Optional[str]:
        """Reserve one credit using the V1 uploader's three limits.

        Args:
            workspace (str): Host workspace name used by V1 accounting.
            project (str): Target project used by V1 accounting.
            usage_quota_name (str): Shared quota strategy identifier.
            minutely_usage_limit (int): Maximum uploads in the current minute.
            hourly_usage_limit (int): Maximum uploads in the current hour.
            daily_usage_limit (int): Maximum uploads in the current day.

        Returns:
            Optional[str]: Selected strategy identifier, or None when denied.
                Zero in any limit denies admission, matching V1.
        """
        matching_strategies_limits = OrderedDict(
            {
                usage_quota_name: [
                    StrategyLimit(
                        limit_type=StrategyLimitType.MINUTELY,
                        value=minutely_usage_limit,
                    ),
                    StrategyLimit(
                        limit_type=StrategyLimitType.HOURLY,
                        value=hourly_usage_limit,
                    ),
                    StrategyLimit(
                        limit_type=StrategyLimitType.DAILY,
                        value=daily_usage_limit,
                    ),
                ]
            }
        )
        strategy = use_credit_of_matching_strategy(
            cache=self._cache,
            workspace=workspace,
            project=project,
            matching_strategies_limits=matching_strategies_limits,
        )

        return strategy

    def return_credit(
        self,
        *,
        workspace: str,
        project: str,
        strategy_name: str,
    ) -> None:
        """Return a granted credit through the unchanged V1 refund function.

        Call once for a duplicate or failed upload and never for a denial.
        The caller must supply the workspace/project from the reservation and
        its returned strategy. Refunds retain V1's time-rollover limitation.

        Args:
            workspace (str): Workspace from the successful reservation.
            project (str): Project from the successful reservation.
            strategy_name (str): Strategy identifier returned by reserve.
        """
        return_strategy_credit(
            cache=self._cache,
            workspace=workspace,
            project=project,
            strategy_name=strategy_name,
        )

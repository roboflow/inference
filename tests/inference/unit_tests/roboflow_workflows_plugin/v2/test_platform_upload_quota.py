"""Exercise real host credit functions with shared, locked, expiring storage."""

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from threading import Barrier, Event, Lock

import pytest

from inference.core.active_learning import cache_operations
from inference.core.active_learning.entities import StrategyLimit, StrategyLimitType
from inference.core.cache.base import BaseCache
from inference.roboflow_workflows_plugin.v2.platform_upload_quota import (
    SharedUploadQuota,
)


class _Clock:
    def __init__(self):
        self.seconds = 0
        self.start = datetime(2026, 10, 13, 12, 34, 45)

    def advance(self, seconds):
        self.seconds += seconds

    def utcnow(self):
        return self.start + timedelta(seconds=self.seconds)


class _ThreadSafeCache(BaseCache):
    """Test-only storage; the inherited BaseCache.lock uses real thread locks."""

    def __init__(self, clock):
        self.clock = clock
        self.data = {}
        self.mutex = Lock()
        self.locks = {}
        self.lock_requests = []

    def get(self, key):
        with self.mutex:
            item = self.data.get(key)
            if item is None:
                return None

            value, expires_at = item
            if self.clock.seconds >= expires_at:
                del self.data[key]
                return None

            return dict(value)

    def set(self, key, value, expire=None):
        with self.mutex:
            self.data[key] = (dict(value), self.clock.seconds + expire)

    def acquire_lock(self, key, expire=None):
        with self.mutex:
            lock = self.locks.setdefault(key, Lock())
            self.lock_requests.append((key, expire))

        if not lock.acquire(timeout=5):
            pytest.fail("Quota lock acquisition timed out")

        return lock


class _ContendedCache(_ThreadSafeCache):
    """Hold the first real quota lock until the second caller tries to acquire it."""

    def __init__(self, clock):
        super().__init__(clock)
        self.first_lock_acquired = Event()
        self.second_attempt = Event()
        self.attempts = 0

    def acquire_lock(self, key, expire=None):
        with self.mutex:
            self.attempts += 1
            first = self.attempts == 1

        if not first:
            if not self.first_lock_acquired.wait(timeout=5):
                pytest.fail("First caller never acquired the quota lock")

            self.second_attempt.set()

        lock = super().acquire_lock(key, expire=expire)
        if first:
            self.first_lock_acquired.set()
            if not self.second_attempt.wait(timeout=5):
                lock.release()
                pytest.fail("Second caller never attempted the held quota lock")

        return lock


@pytest.fixture
def clock(monkeypatch):
    clock = _Clock()
    monkeypatch.setattr(cache_operations, "datetime", clock)

    return clock


@pytest.fixture
def cache(clock):
    cache = _ThreadSafeCache(clock)

    return cache


def _reserve(quota, **overrides):
    args = dict(
        workspace="test-workspace",
        project="test-project",
        usage_quota_name="shared-uploads",
        minutely_usage_limit=1,
        hourly_usage_limit=10,
        daily_usage_limit=100,
    )
    args.update(overrides)
    strategy = quota.reserve(**args)

    return strategy


def _v1_reserve(cache):
    strategy = cache_operations.use_credit_of_matching_strategy(
        cache=cache,
        workspace="test-workspace",
        project="test-project",
        matching_strategies_limits=OrderedDict(
            {
                "shared-uploads": [
                    StrategyLimit(StrategyLimitType.MINUTELY, 1),
                    StrategyLimit(StrategyLimitType.HOURLY, 10),
                    StrategyLimit(StrategyLimitType.DAILY, 100),
                ]
            }
        ),
    )

    return strategy


def _refund(quota, strategy):
    quota.return_credit(
        workspace="test-workspace",
        project="test-project",
        strategy_name=strategy,
    )


def _usage(cache, limit_type):
    value = cache_operations.get_current_strategy_limit_usage(
        cache=cache,
        workspace="test-workspace",
        project="test-project",
        strategy_name="shared-uploads",
        limit_type=limit_type,
    )

    return value


def test_host_cache_is_required():
    with pytest.raises(TypeError):
        SharedUploadQuota()

    with pytest.raises(ValueError, match="explicitly supplied host cache"):
        SharedUploadQuota(None)


def test_two_sessions_share_quota_one(cache):
    first_session = SharedUploadQuota(cache)
    second_session = SharedUploadQuota(cache)

    assert [_reserve(first_session), _reserve(second_session)] == [
        "shared-uploads",
        None,
    ]
    assert cache.lock_requests == [
        ("active_learning:usage:test-workspace:test-project:usage:lock", 5),
        ("active_learning:usage:test-workspace:test-project:usage:lock", 5),
    ]


def test_isolated_session_caches_reproduce_double_admission(clock):
    sessions = [SharedUploadQuota(_ThreadSafeCache(clock)) for _ in range(2)]

    assert [_reserve(session) for session in sessions] == [
        "shared-uploads",
        "shared-uploads",
    ]


@pytest.mark.parametrize("v1_first", [True, False])
def test_v1_and_v2_share_accounting(cache, v1_first):
    quota = SharedUploadQuota(cache)
    callers = [lambda: _v1_reserve(cache), lambda: _reserve(quota)]
    if not v1_first:
        callers.reverse()

    assert [caller() for caller in callers] == ["shared-uploads", None]


@pytest.mark.parametrize("mixed_v1", [False, True])
def test_barrier_started_sessions_contend_for_one_host_lock(clock, mixed_v1):
    cache = _ContendedCache(clock)
    start = Barrier(2)
    sessions = [SharedUploadQuota(cache) for _ in range(2)]

    def upload(session):
        start.wait(timeout=5)
        if mixed_v1 and session == 0:
            return _v1_reserve(cache)

        strategy = _reserve(sessions[session])

        return strategy

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(upload, range(2)))

    assert results.count("shared-uploads") == 1
    assert results.count(None) == 1
    assert cache.attempts == 2
    assert cache.second_attempt.is_set()
    assert all(not lock.locked() for lock in cache.locks.values())
    assert [_usage(cache, limit_type) for limit_type in StrategyLimitType] == [1, 1, 1]


@pytest.mark.parametrize(
    "different_scope",
    [
        {"workspace": "another-workspace"},
        {"project": "another-project"},
        {"usage_quota_name": "another-quota"},
    ],
)
def test_workspace_project_and_quota_are_independent(cache, different_scope):
    quota = SharedUploadQuota(cache)

    assert _reserve(quota) == "shared-uploads"
    assert _reserve(quota) is None
    assert _reserve(quota, **different_scope) == different_scope.get(
        "usage_quota_name", "shared-uploads"
    )


@pytest.mark.parametrize(
    "limits",
    [
        dict(minutely_usage_limit=1, hourly_usage_limit=10, daily_usage_limit=100),
        dict(minutely_usage_limit=10, hourly_usage_limit=1, daily_usage_limit=100),
        dict(minutely_usage_limit=10, hourly_usage_limit=100, daily_usage_limit=1),
    ],
)
def test_each_limit_can_deny_and_denial_consumes_nothing(cache, limits):
    quota = SharedUploadQuota(cache)

    assert _reserve(quota, **limits) == "shared-uploads"
    assert _reserve(quota, **limits) is None
    assert [_usage(cache, limit_type) for limit_type in StrategyLimitType] == [1, 1, 1]


@pytest.mark.parametrize(
    "zero_limit", ["minutely_usage_limit", "hourly_usage_limit", "daily_usage_limit"]
)
def test_zero_limit_denies_without_creating_credit(cache, zero_limit):
    assert _reserve(SharedUploadQuota(cache), **{zero_limit: 0}) is None
    assert cache.data == {}


def test_explicit_return_releases_credit_for_another_session(cache):
    first = SharedUploadQuota(cache)
    second = SharedUploadQuota(cache)
    strategy = _reserve(first)
    assert _reserve(second) is None

    _refund(first, strategy)

    assert [_usage(cache, limit_type) for limit_type in StrategyLimitType] == [0, 0, 0]
    assert _reserve(second) == "shared-uploads"


@pytest.mark.parametrize("outcome", ["success", "duplicate", "failure", "denied"])
def test_future_upload_operation_owns_dispatch_and_conditional_refund(cache, outcome):
    quota = SharedUploadQuota(cache)
    dispatched = []
    returned = []

    def simulated_upload():
        limits = {"minutely_usage_limit": 0} if outcome == "denied" else {}
        strategy = _reserve(quota, **limits)
        if strategy is None:
            return

        refund = False
        try:
            dispatched.append("upload")
            if outcome == "failure":
                raise RuntimeError("Invented upload failure")

            refund = outcome == "duplicate"
        except RuntimeError:
            refund = True
        finally:
            if refund:
                _refund(quota, strategy)
                returned.append(strategy)

    simulated_upload()

    assert dispatched == ([] if outcome == "denied" else ["upload"])
    assert returned == (
        ["shared-uploads"] if outcome in ("duplicate", "failure") else []
    )
    assert _reserve(SharedUploadQuota(cache)) == (
        None if outcome == "success" else "shared-uploads"
    )


@pytest.mark.parametrize(
    "limits,until_rollover",
    [
        ({"minutely_usage_limit": 1}, 15),
        ({"minutely_usage_limit": 100, "hourly_usage_limit": 1}, 25 * 60 + 15),
        (
            {
                "minutely_usage_limit": 100,
                "hourly_usage_limit": 100,
                "daily_usage_limit": 1,
            },
            11 * 3600 + 25 * 60 + 15,
        ),
    ],
)
def test_host_time_windows_reset_at_their_boundaries(
    cache, clock, limits, until_rollover
):
    quota = SharedUploadQuota(cache)
    assert _reserve(quota, **limits) == "shared-uploads"
    clock.advance(until_rollover - 1)
    assert _reserve(quota, **limits) is None

    clock.advance(1)

    assert _reserve(quota, **limits) == "shared-uploads"


@pytest.mark.parametrize(
    "limit_type,expected_ttl",
    [
        (StrategyLimitType.MINUTELY, 120),
        (StrategyLimitType.HOURLY, 7200),
        (StrategyLimitType.DAILY, 90000),
    ],
)
def test_existing_host_functions_set_and_refresh_expiry(
    cache, clock, limit_type, expected_ttl
):
    quota = SharedUploadQuota(cache)
    assert _reserve(quota, minutely_usage_limit=10) == "shared-uploads"
    key = cache_operations.generate_cache_key_for_active_learning_usage(
        limit_type=limit_type,
        workspace="test-workspace",
        project="test-project",
        strategy_name="shared-uploads",
    )
    assert cache.data[key][1] == expected_ttl
    clock.advance(1)
    assert _reserve(quota, minutely_usage_limit=10) == "shared-uploads"
    assert cache.data[key][1] == expected_ttl + 1

    clock.advance(expected_ttl - 1)
    assert cache.get(key) == {"usage": 2}
    clock.advance(1)
    assert cache.get(key) is None


def test_refund_after_rollover_preserves_v1_current_window_limitation(cache, clock):
    quota = SharedUploadQuota(cache)
    previous_window_strategy = _reserve(quota)
    clock.advance(15)
    assert _reserve(quota) == "shared-uploads"
    assert _reserve(quota) is None

    _refund(quota, previous_window_strategy)

    # V1 refunds the current window rather than the window of the reservation.
    assert _reserve(quota) == "shared-uploads"


def test_host_cache_is_never_closed_by_bridge(cache):
    quota = SharedUploadQuota(cache)
    _refund(quota, _reserve(quota))
    del quota

    assert _reserve(SharedUploadQuota(cache)) == "shared-uploads"

import logging
import sys
import types

import pytest
from roboflow_workflows.utils.in_memory_cache import InMemoryWorkflowsCache

from inference_server import configuration
from inference_server.workflows.redis_cache import (
    RedisWorkflowsCache,
    build_workflows_cache,
)


class LockNotOwned(Exception):
    pass


class FakeLock:
    def __init__(self, acquired=True, release_error=None):
        self._acquired = acquired
        self._release_error = release_error
        self.calls = []

    def acquire(self, blocking_timeout=None):
        self.calls.append(("acquire", blocking_timeout))
        return self._acquired

    def extend(self, additional_time):
        self.calls.append(("extend", additional_time))

    def release(self):
        self.calls.append(("release",))
        if self._release_error is not None:
            raise self._release_error


class FakeRedis:
    def __init__(self, lock=None, ping_error=None, **kwargs):
        self.kwargs = kwargs
        self.store = {}
        self.set_calls = []
        self.lock_calls = []
        self._lock = lock or FakeLock()
        self._ping_error = ping_error

    def ping(self):
        if self._ping_error is not None:
            raise self._ping_error
        return True

    def get(self, key):
        return self.store.get(key)

    def set(self, key, value, ex=None):
        self.set_calls.append((key, value, ex))
        self.store[key] = value if isinstance(value, bytes) else value.encode()

    def lock(self, key, blocking=True, timeout=None):
        self.lock_calls.append((key, blocking, timeout))
        return self._lock


def _cache(client=None):
    return RedisWorkflowsCache(client or FakeRedis(), lock_not_owned_error=LockNotOwned)


def test_round_trip_encodes_values_as_json_like_legacy():
    client = FakeRedis()
    cache = _cache(client)

    cache.set("workflow:a", {"specification": {"version": "1.0"}, "n": [1, 2]})

    assert client.set_calls == [
        ("workflow:a", '{"specification": {"version": "1.0"}, "n": [1, 2]}', None)
    ]
    assert cache.get("workflow:a") == {"specification": {"version": "1.0"}, "n": [1, 2]}


def test_bytes_values_are_stored_raw_and_returned_raw_when_not_json():
    client = FakeRedis()
    cache = _cache(client)

    cache.set("blob", b"\x80\x81")

    assert client.set_calls == [("blob", b"\x80\x81", None)]
    assert cache.get("blob") == b"\x80\x81"


def test_missing_key_is_none():
    assert _cache().get("absent") is None


def test_expiry_is_passed_through_as_ex():
    client = FakeRedis()

    _cache(client).set("k", "v", expire=900)

    assert client.set_calls == [("k", '"v"', 900)]


def test_lock_acquires_extends_and_releases_like_legacy():
    lock = FakeLock()
    client = FakeRedis(lock=lock)

    with _cache(client).lock("al:lock", expire=5) as held:
        assert held is lock
        assert lock.calls == [("acquire", 5), ("extend", 5)]

    assert client.lock_calls == [("al:lock", True, 5)]
    assert lock.calls[-1] == ("release",)


def test_lock_without_expiry_is_not_extended():
    lock = FakeLock()

    with _cache(FakeRedis(lock=lock)).lock("k"):
        assert lock.calls == [("acquire", None)]


def test_lock_timeout_raises_timeout_error():
    client = FakeRedis(lock=FakeLock(acquired=False))

    with pytest.raises(TimeoutError, match="Couldn't get lock"):
        with _cache(client).lock("k", expire=1):
            pass


def test_lock_expired_before_release_is_only_a_warning(caplog):
    client = FakeRedis(lock=FakeLock(release_error=LockNotOwned()))

    with caplog.at_level(logging.WARNING):
        with _cache(client).lock("k", expire=1):
            pass

    assert "Lock at cache key k expired before release" in caplog.text


def test_lock_release_errors_other_than_expiry_propagate():
    client = FakeRedis(lock=FakeLock(release_error=RuntimeError("boom")))

    with pytest.raises(RuntimeError):
        with _cache(client).lock("k", expire=1):
            pass


def _fake_redis_module(monkeypatch, **client_kwargs):
    module = types.ModuleType("redis")
    created = []

    def _redis(**kwargs):
        client = FakeRedis(**client_kwargs, **kwargs)
        created.append(client)
        return client

    module.Redis = _redis
    module.exceptions = types.SimpleNamespace(
        TimeoutError=type("TimeoutError", (Exception,), {}),
        ConnectionError=type("ConnectionError", (Exception,), {}),
        LockNotOwnedError=LockNotOwned,
    )
    monkeypatch.setitem(sys.modules, "redis", module)
    return module, created


@pytest.fixture
def redis_settings(monkeypatch):
    monkeypatch.setattr(configuration, "REDIS_HOST", "cache.internal")
    monkeypatch.setattr(configuration, "REDIS_PORT", 6380)
    monkeypatch.setattr(configuration, "REDIS_SSL", True)
    monkeypatch.setattr(configuration, "REDIS_TIMEOUT", 1.5)


def test_unset_redis_host_selects_the_in_memory_cache(monkeypatch):
    monkeypatch.setattr(configuration, "REDIS_HOST", None)

    assert isinstance(build_workflows_cache(), InMemoryWorkflowsCache)


def test_redis_host_selects_the_redis_cache_with_the_legacy_client_settings(
    redis_settings, monkeypatch
):
    _, created = _fake_redis_module(monkeypatch)

    cache = build_workflows_cache()

    assert isinstance(cache, RedisWorkflowsCache)
    assert cache.client is created[0]
    assert created[0].kwargs == {
        "host": "cache.internal",
        "port": 6380,
        "db": 0,
        "decode_responses": False,
        "ssl": True,
        "socket_timeout": 1.5,
        "socket_connect_timeout": 1.5,
    }


def test_redis_cache_releases_with_the_redis_lock_error(redis_settings, monkeypatch):
    module, created = _fake_redis_module(monkeypatch)
    cache = build_workflows_cache()
    created[0]._lock = FakeLock(release_error=module.exceptions.LockNotOwnedError())

    with cache.lock("k", expire=1):
        pass


def test_unreachable_redis_falls_back_to_memory_with_one_error_line(
    redis_settings, monkeypatch, caplog
):
    module, _ = _fake_redis_module(monkeypatch)
    monkeypatch.setattr(
        module,
        "Redis",
        lambda **kwargs: FakeRedis(ping_error=module.exceptions.ConnectionError()),
    )

    with caplog.at_level(logging.ERROR):
        cache = build_workflows_cache()

    assert isinstance(cache, InMemoryWorkflowsCache)
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1
    assert "Could not connect to Redis under cache.internal:6380" in errors[0].message


def test_missing_redis_package_falls_back_to_memory_with_one_error_line(
    redis_settings, monkeypatch, caplog
):
    monkeypatch.setitem(sys.modules, "redis", None)

    with caplog.at_level(logging.ERROR):
        cache = build_workflows_cache()

    assert isinstance(cache, InMemoryWorkflowsCache)
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1
    assert "redis" in errors[0].message

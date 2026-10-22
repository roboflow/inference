from unittest.mock import MagicMock, patch

import pytest
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import TimeoutError as RedisTimeoutError

from inference.core.cache.redis import RedisCache


def _build_cache_with_mock_client():
    """Construct a RedisCache without touching a real Redis or starting the
    background ``_expire`` thread. Returns (cache, mock_client)."""
    with patch("inference.core.cache.redis.redis.Redis") as mock_redis_cls, patch(
        "inference.core.cache.redis.threading.Thread"
    ):
        mock_client = MagicMock()
        mock_redis_cls.return_value = mock_client
        cache = RedisCache()
    return cache, mock_client


def test_zadd_with_expire_sets_server_side_ttl():
    # given
    cache, mock_client = _build_cache_with_mock_client()
    pipe = MagicMock()
    mock_client.pipeline.return_value.__enter__.return_value = pipe

    # when
    cache.zadd(
        "inference:srv-0:model/1", value={"foo": "bar"}, score=123.0, expire=120.0
    )

    # then - ZADD and a real server-side EXPIRE are issued in one pipeline
    pipe.zadd.assert_called_once()
    assert pipe.zadd.call_args.args[0] == "inference:srv-0:model/1"
    pipe.expire.assert_called_once_with("inference:srv-0:model/1", 120)
    pipe.execute.assert_called_once()
    # and the in-process bookkeeping is still recorded for fine-grained trimming
    assert ("inference:srv-0:model/1", 123.0) in cache.zexpires


def test_zadd_expire_is_floored_to_at_least_one_second():
    # given - a sub-second expire must never become EXPIRE 0 (immediate delete)
    cache, mock_client = _build_cache_with_mock_client()
    pipe = MagicMock()
    mock_client.pipeline.return_value.__enter__.return_value = pipe

    # when
    cache.zadd("k", value={"a": 1}, score=1.0, expire=0.4)

    # then
    pipe.expire.assert_called_once_with("k", 1)


def test_zadd_without_expire_does_not_set_ttl_or_bookkeeping():
    # given
    cache, mock_client = _build_cache_with_mock_client()

    # when
    cache.zadd("k", value={"a": 1}, score=1.0)

    # then - plain ZADD, no pipeline / EXPIRE, no zexpires entry
    mock_client.zadd.assert_called_once()
    mock_client.pipeline.assert_not_called()
    assert cache.zexpires == {}


@pytest.mark.parametrize("error", [RedisConnectionError, RedisTimeoutError])
def test_expiration_retries_after_a_transient_redis_error(error):
    cache, mock_client = _build_cache_with_mock_client()
    cache.zexpires[("stream", 1.0)] = 90.0
    mock_client.zremrangebyscore.side_effect = [error("unavailable"), 1]

    with patch("inference.core.cache.redis.time.time", return_value=100.0), patch(
        "inference.core.cache.redis.time.sleep", side_effect=[None, StopIteration]
    ):
        with pytest.raises(StopIteration):
            cache._expire()

    assert mock_client.zremrangebyscore.call_count == 2
    assert cache.zexpires == {}

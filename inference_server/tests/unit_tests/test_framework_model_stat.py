from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock, patch

import pytest

from inference_models.errors import (
    ModelNotFoundError,
    ModelRetrievalError,
    RetryError,
    UnauthorizedModelAccessError,
)
from inference_server.framework.entities import CommonRequestParams
from inference_server.framework.model_stat import (
    _TtlLruCache,
    _reset_cache_for_tests,
    stat_model_while_checking_auth,
)


def _meta(task_type: str = "object-detection", architecture: str = "yolov8"):
    return MagicMock(task_type=task_type, model_architecture=architecture)


@pytest.fixture(autouse=True)
def _reset():
    _reset_cache_for_tests()
    yield
    _reset_cache_for_tests()


@pytest.mark.asyncio
async def test_cache_hits_skip_upstream_after_first_call():
    with patch(
        "inference_server.framework.model_stat.get_one_page_of_model_metadata",
        return_value=_meta(),
    ) as m:
        cp = CommonRequestParams(model_id="acme/1", api_key="k")
        r1 = await stat_model_while_checking_auth(cp)
        r2 = await stat_model_while_checking_auth(cp)
        assert r1 == ("object-detection", "infer")
        assert r2 == r1
        assert m.call_count == 1


@pytest.mark.asyncio
async def test_concurrent_first_load_dedupes_to_one_upstream_call():
    calls = 0

    def slow(**_):
        nonlocal calls
        calls += 1
        time.sleep(0.05)
        return _meta()

    with patch(
        "inference_server.framework.model_stat.get_one_page_of_model_metadata",
        side_effect=slow,
    ):
        cp = CommonRequestParams(model_id="acme/2", api_key="k")
        results = await asyncio.gather(
            *(stat_model_while_checking_auth(cp) for _ in range(5))
        )
        assert all(r == ("object-detection", "infer") for r in results)
        assert calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raised, expected",
    [
        (UnauthorizedModelAccessError(message="no", help_url=""), PermissionError),
        (ModelNotFoundError(message="nope", help_url=""), LookupError),
        (RetryError(message="down", help_url=""), RuntimeError),
        (ModelRetrievalError(message="boom", help_url=""), RuntimeError),
        (OSError("conn refused"), RuntimeError),
    ],
)
async def test_upstream_errors_map_to_dispatcher_codes(raised, expected):
    with patch(
        "inference_server.framework.model_stat.get_one_page_of_model_metadata",
        side_effect=raised,
    ):
        cp = CommonRequestParams(model_id="m", api_key="k")
        with pytest.raises(expected):
            await stat_model_while_checking_auth(cp)


@pytest.mark.asyncio
async def test_empty_task_type_from_upstream_raises_runtime_error():
    with patch(
        "inference_server.framework.model_stat.get_one_page_of_model_metadata",
        return_value=_meta(task_type=""),
    ):
        cp = CommonRequestParams(model_id="m", api_key="k")
        with pytest.raises(RuntimeError, match="empty taskType"):
            await stat_model_while_checking_auth(cp)


@pytest.mark.asyncio
async def test_failed_calls_are_not_cached():
    seq = iter([RetryError(message="1", help_url="")])
    calls = 0

    def side(**_):
        nonlocal calls
        calls += 1
        try:
            raise next(seq)
        except StopIteration:
            return _meta()

    with patch(
        "inference_server.framework.model_stat.get_one_page_of_model_metadata",
        side_effect=side,
    ):
        cp = CommonRequestParams(model_id="m", api_key="k")
        with pytest.raises(RuntimeError):
            await stat_model_while_checking_auth(cp)
        r = await stat_model_while_checking_auth(cp)
        assert r == ("object-detection", "infer")
        assert calls == 2


@pytest.mark.asyncio
async def test_cache_key_includes_api_key():
    with patch(
        "inference_server.framework.model_stat.get_one_page_of_model_metadata",
        return_value=_meta(),
    ) as m:
        await stat_model_while_checking_auth(
            CommonRequestParams(model_id="acme/1", api_key="k1")
        )
        await stat_model_while_checking_auth(
            CommonRequestParams(model_id="acme/1", api_key="k2")
        )
        assert m.call_count == 2


def test_ttl_lru_expires_entries_past_ttl():
    cache = _TtlLruCache(maxsize=4, ttl_s=0.01)
    cache.set(("a", ""), ("object-detection", "infer"))
    assert cache.get(("a", "")) == ("object-detection", "infer")
    time.sleep(0.02)
    assert cache.get(("a", "")) is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "task_type, expected_action",
    [
        ("object-detection", "infer"),
        ("classification", "infer"),
        ("vlm", "prompt"),
        ("embedding", "embed_images"),
        ("interactive-instance-segmentation", "embed"),
        ("structured-ocr", "infer"),
    ],
)
async def test_default_action_per_task_type(task_type, expected_action):
    with patch(
        "inference_server.framework.model_stat.get_one_page_of_model_metadata",
        return_value=_meta(task_type=task_type),
    ):
        cp = CommonRequestParams(model_id="m", api_key="k")
        result = await stat_model_while_checking_auth(cp)
        assert result == (task_type, expected_action)


def test_ttl_lru_evicts_oldest_when_full():
    cache = _TtlLruCache(maxsize=2, ttl_s=60)
    cache.set(("a", ""), ("t", "infer"))
    cache.set(("b", ""), ("t", "infer"))
    cache.set(("c", ""), ("t", "infer"))
    assert cache.get(("a", "")) is None
    assert cache.get(("b", "")) is not None
    assert cache.get(("c", "")) is not None

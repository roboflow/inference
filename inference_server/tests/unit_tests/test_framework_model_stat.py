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
    _reset_cache_for_tests,
    _TtlLruCache,
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


_DET_SMALL = "pp-ocrv6-det/small"
_REC_SMALL = "pp-ocrv6-rec/small"
_REC_MEDIUM = "pp-ocrv6-rec/medium"


def _recording_registry(table: dict, calls: list):
    def _metadata(model_id: str, api_key=None):
        calls.append((model_id, api_key))
        outcome = table.get(model_id)
        if outcome is None:
            raise ModelNotFoundError(message=model_id, help_url="")
        if isinstance(outcome, Exception):
            raise outcome
        return _meta(task_type=outcome)

    return patch(
        "inference_server.framework.model_stat.get_one_page_of_model_metadata",
        side_effect=_metadata,
    )


@pytest.mark.asyncio
async def test_pipeline_id_authorizes_every_enabled_stage_with_the_callers_key():
    calls: list = []
    table = {_DET_SMALL: "object-detection", _REC_MEDIUM: "text-only-ocr"}
    with _recording_registry(table, calls):
        result = await stat_model_while_checking_auth(
            CommonRequestParams(model_id="pp_ocr/small-medium", api_key="key-1")
        )
    assert result == ("structured-ocr", "infer")
    assert sorted(calls) == [(_DET_SMALL, "key-1"), (_REC_MEDIUM, "key-1")]


@pytest.mark.asyncio
async def test_pipeline_id_is_never_sent_to_the_registry():
    calls: list = []
    table = {_DET_SMALL: "object-detection", _REC_MEDIUM: "text-only-ocr"}
    with _recording_registry(table, calls):
        await stat_model_while_checking_auth(
            CommonRequestParams(model_id="pp_ocr/small-medium", api_key="key-1")
        )
    assert all(not model_id.startswith("pp_ocr") for model_id, _ in calls)


@pytest.mark.asyncio
async def test_pipeline_id_skips_a_disabled_stage():
    calls: list = []
    with _recording_registry({_DET_SMALL: "object-detection"}, calls):
        result = await stat_model_while_checking_auth(
            CommonRequestParams(model_id="pp_ocr/small-none", api_key="k")
        )
    assert result == ("structured-ocr", "infer")
    assert calls == [(_DET_SMALL, "k")]


@pytest.mark.asyncio
async def test_pipeline_id_with_a_missing_stage_is_lookup_error():
    calls: list = []
    with _recording_registry({_DET_SMALL: "object-detection"}, calls):
        with pytest.raises(LookupError):
            await stat_model_while_checking_auth(
                CommonRequestParams(model_id="pp_ocr/small-small", api_key="k")
            )
    assert (_REC_SMALL, "k") in calls


@pytest.mark.asyncio
async def test_pipeline_id_with_a_denied_stage_is_permission_error():
    calls: list = []
    table = {
        _DET_SMALL: "object-detection",
        _REC_SMALL: UnauthorizedModelAccessError(message="denied", help_url=""),
    }
    with _recording_registry(table, calls):
        with pytest.raises(PermissionError):
            await stat_model_while_checking_auth(
                CommonRequestParams(model_id="pp_ocr/small-small", api_key="k")
            )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_id", ["pp_ocr/bogus-bogus", "pp_ocr/small-medium-tiny", "pp_ocr/none-none"]
)
async def test_unusable_pipeline_id_is_lookup_error_without_touching_registry(model_id):
    calls: list = []
    with _recording_registry({}, calls):
        with pytest.raises(LookupError):
            await stat_model_while_checking_auth(
                CommonRequestParams(model_id=model_id, api_key="k")
            )
    assert calls == []


@pytest.mark.asyncio
async def test_non_pipeline_id_is_statted_as_itself():
    calls: list = []
    with _recording_registry({"acme/1": "object-detection"}, calls):
        result = await stat_model_while_checking_auth(
            CommonRequestParams(model_id="acme/1", api_key="k")
        )
    assert result == ("object-detection", "infer")
    assert calls == [("acme/1", "k")]


@pytest.mark.asyncio
async def test_pipeline_stage_authorization_is_cached_per_api_key():
    calls: list = []
    table = {_DET_SMALL: "object-detection", _REC_MEDIUM: "text-only-ocr"}
    with _recording_registry(table, calls):
        await stat_model_while_checking_auth(
            CommonRequestParams(model_id="pp_ocr/small-medium", api_key="k1")
        )
        await stat_model_while_checking_auth(
            CommonRequestParams(model_id="pp_ocr/small-medium", api_key="k1")
        )
        assert sorted(calls) == [(_DET_SMALL, "k1"), (_REC_MEDIUM, "k1")]
        await stat_model_while_checking_auth(
            CommonRequestParams(model_id="pp_ocr/small-medium", api_key="k2")
        )
    assert sorted(calls) == [
        (_DET_SMALL, "k1"),
        (_DET_SMALL, "k2"),
        (_REC_MEDIUM, "k1"),
        (_REC_MEDIUM, "k2"),
    ]


@pytest.mark.asyncio
async def test_pipeline_result_is_not_cached_under_the_synthetic_id():
    from inference_server.framework import model_stat

    calls: list = []
    table = {_DET_SMALL: "object-detection", _REC_MEDIUM: "text-only-ocr"}
    with _recording_registry(table, calls):
        await stat_model_while_checking_auth(
            CommonRequestParams(model_id="pp_ocr/small-medium", api_key="k")
        )
        assert model_stat._cache.get(("pp_ocr/small-medium", "k")) is None
        model_stat._cache._data.pop((_DET_SMALL, "k"))
        model_stat._cache._data.pop((_REC_MEDIUM, "k"))
        table[_REC_MEDIUM] = UnauthorizedModelAccessError(
            message=_REC_MEDIUM, help_url=""
        )
        with pytest.raises(PermissionError):
            await stat_model_while_checking_auth(
                CommonRequestParams(model_id="pp_ocr/small-medium", api_key="k")
            )


@pytest.mark.asyncio
async def test_broken_pipeline_defaults_are_not_a_client_lookup_error():
    calls: list = []
    with _recording_registry({}, calls):
        with patch(
            "inference_models.model_pipelines.auto_loaders.pipelines_registry"
            ".get_default_pipeline_parameters",
            return_value=None,
        ):
            with pytest.raises(ValueError) as exc_info:
                await stat_model_while_checking_auth(
                    CommonRequestParams(model_id="pp_ocr", api_key="k")
                )
    assert not isinstance(exc_info.value, LookupError)
    assert calls == []

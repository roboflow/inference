from __future__ import annotations

import asyncio
import os
import time
from collections import OrderedDict
from typing import Optional

from inference_models.errors import (
    ModelNotFoundError,
    ModelRetrievalError,
    RetryError,
    UnauthorizedModelAccessError,
)
from inference_models.weights_providers.roboflow import (
    get_one_page_of_model_metadata,
)

from inference_server.framework.entities import CommonRequestParams


_CACHE_MAXSIZE = int(os.environ.get("INFERENCE_MODEL_STAT_CACHE_SIZE", "1024"))
_CACHE_TTL_S = float(os.environ.get("INFERENCE_MODEL_STAT_CACHE_TTL_S", "300"))


class _TtlLruCache:
    def __init__(self, maxsize: int, ttl_s: float):
        self._maxsize = maxsize
        self._ttl = ttl_s
        self._data: "OrderedDict[tuple[str, str], tuple[float, tuple[str, str]]]" = (
            OrderedDict()
        )

    def get(self, key: tuple[str, str]) -> Optional[tuple[str, str]]:
        entry = self._data.get(key)
        if entry is None:
            return None
        ts, value = entry
        if time.monotonic() - ts > self._ttl:
            del self._data[key]
            return None
        self._data.move_to_end(key)
        return value

    def set(self, key: tuple[str, str], value: tuple[str, str]) -> None:
        if key in self._data:
            self._data.move_to_end(key)
        self._data[key] = (time.monotonic(), value)
        while len(self._data) > self._maxsize:
            self._data.popitem(last=False)

    def clear(self) -> None:
        self._data.clear()


_DEFAULT_ACTION_BY_TASK_TYPE: dict[str, str] = {
    "vlm": "prompt",
    "embedding": "embed_images",
    "interactive-instance-segmentation": "embed",
}


_cache: _TtlLruCache = _TtlLruCache(_CACHE_MAXSIZE, _CACHE_TTL_S)
_inflight: dict[tuple[str, str], asyncio.Task] = {}


async def stat_model_while_checking_auth(
    common_params: CommonRequestParams,
) -> tuple[str, str]:
    key = (common_params.model_id, common_params.api_key)

    cached = _cache.get(key)
    if cached is not None:
        return cached

    task = _inflight.get(key)
    if task is None:
        task = asyncio.create_task(_fetch_cache_and_map(common_params, key))
        _inflight[key] = task
        task.add_done_callback(lambda _t: _inflight.pop(key, None))

    return await task


async def _fetch_cache_and_map(
    common_params: CommonRequestParams, key: tuple[str, str]
) -> tuple[str, str]:
    result = await _fetch_and_map(common_params)
    _cache.set(key, result)
    return result


async def _fetch_and_map(common_params: CommonRequestParams) -> tuple[str, str]:
    try:
        meta = await asyncio.to_thread(
            get_one_page_of_model_metadata,
            model_id=common_params.model_id,
            api_key=common_params.api_key or None,
        )
    except UnauthorizedModelAccessError as exc:
        raise PermissionError(str(exc)) from exc
    except ModelNotFoundError as exc:
        raise LookupError(str(exc)) from exc
    except (RetryError, ModelRetrievalError, OSError) as exc:
        raise RuntimeError(str(exc) or "Roboflow registry unreachable") from exc

    task_type = (meta.task_type or "").strip()
    if not task_type:
        raise RuntimeError(
            f"Roboflow registry returned empty taskType for model_id={common_params.model_id!r}"
        )

    default_action = _DEFAULT_ACTION_BY_TASK_TYPE.get(task_type, "infer")
    return (task_type, default_action)


def _reset_cache_for_tests() -> None:
    _cache.clear()
    _inflight.clear()

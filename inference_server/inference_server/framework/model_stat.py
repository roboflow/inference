from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from typing import NamedTuple, Optional

from inference_model_manager.pipelines import (
    InvalidPipelineIdError,
    PipelineRequest,
    is_pipeline_model_id,
    resolve_pipeline_request,
)
from inference_models.errors import (
    ModelNotFoundError,
    ModelRetrievalError,
    RetryError,
    UnauthorizedModelAccessError,
)
from inference_models.weights_providers.roboflow import (
    get_one_page_of_model_metadata,
    roboflow_secure_gateway_proxy_url_builder,
)
from inference_server import configuration
from inference_server.framework.entities import CommonRequestParams
from inference_server.framework.fanout import gather_bounded
from inference_server.hosted.assume_identity import (
    ENFORCE_CREDITS_VERIFICATION_HEADER,
    add_assume_identity_headers,
    enforce_credits_verification,
)

_CACHE_MAXSIZE = configuration.MODEL_STAT_CACHE_SIZE
_CACHE_TTL_S = configuration.MODEL_STAT_CACHE_TTL_S


class ModelStat(NamedTuple):
    """What the registry reports about a model before it is loaded.

    Attributes:
        task_type: Registry task type, such as ``object-detection``.
        default_action: Action run when a request names none.
        model_architecture: Registry ``modelArchitecture``; None when unknown.
        model_variant: Registry ``modelVariant``; None when unknown.
    """

    task_type: str
    default_action: str
    model_architecture: Optional[str] = None
    model_variant: Optional[str] = None


class _TtlLruCache:
    def __init__(self, maxsize: int, ttl_s: float):
        self._maxsize = maxsize
        self._ttl = ttl_s
        self._data: "OrderedDict[tuple, tuple[float, ModelStat]]" = OrderedDict()

    def get(self, key: tuple) -> Optional[ModelStat]:
        entry = self._data.get(key)
        if entry is None:
            return None
        ts, value = entry
        if time.monotonic() - ts > self._ttl:
            del self._data[key]
            return None
        self._data.move_to_end(key)
        return value

    def set(self, key: tuple, value: ModelStat) -> None:
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
_inflight: dict[tuple, asyncio.Task] = {}


async def stat_model_while_checking_auth(
    common_params: CommonRequestParams,
) -> tuple[str, str]:
    stat = await stat_model_details_while_checking_auth(common_params)

    return (stat.task_type, stat.default_action)


async def stat_model_details_while_checking_auth(
    common_params: CommonRequestParams,
) -> ModelStat:
    """Describe a model through the registry while checking the caller's access.

    Args:
        common_params: Model id and API key of the request.

    Returns:
        Task type, default action and, when the registry reports them, the
        model architecture and variant.

    Raises:
        PermissionError: When the key may not access the model.
        LookupError: When the model is unknown.
        RuntimeError: When the registry cannot be reached or answers nothing.
    """
    if common_params.model_id == "passthrough" or common_params.model_id.startswith(
        "passthrough/"
    ):
        return ModelStat("passthrough", "infer")

    extra_headers = _platform_stat_headers()
    key = (
        common_params.model_id,
        common_params.api_key,
        tuple(sorted(extra_headers.items())),
    )

    cached = _cache.get(key)
    if cached is not None:
        return cached

    task = _inflight.get(key)
    if task is None:
        task = asyncio.create_task(
            _fetch_cache_and_map(common_params, key, extra_headers)
        )
        _inflight[key] = task
        task.add_done_callback(lambda _t: _inflight.pop(key, None))

    return await task


async def _fetch_cache_and_map(
    common_params: CommonRequestParams,
    key: tuple[str, str, tuple[tuple[str, str], ...]],
    extra_headers: dict[str, str],
) -> ModelStat:
    result = await _fetch_and_map(common_params, extra_headers)
    if not is_pipeline_model_id(common_params.model_id):
        _cache.set(key, result)
    return result


def _pipeline_request(model_id: str) -> Optional[PipelineRequest]:
    try:
        return resolve_pipeline_request(model_id)
    except InvalidPipelineIdError as exc:
        raise LookupError(str(exc)) from exc


async def _authorize_pipeline_stages(request: PipelineRequest, api_key: str) -> None:
    await gather_bounded(
        *(
            stat_model_while_checking_auth(
                CommonRequestParams(model_id=stage_model_id, api_key=api_key)
            )
            for stage_model_id in request.stage_model_ids
            if stage_model_id is not None
        )
    )


def _platform_stat_headers() -> dict[str, str]:
    headers: dict[str, str] = {}
    if (
        configuration.ENFORCE_CREDITS_VERIFICATION
        and enforce_credits_verification.get()
    ):
        headers[ENFORCE_CREDITS_VERIFICATION_HEADER] = "true"
    add_assume_identity_headers(headers)
    return headers


def _reported_label(meta: object, name: str) -> Optional[str]:
    label = getattr(meta, name, None)
    if not isinstance(label, str) or not label:
        return None

    return label


async def _fetch_and_map(
    common_params: CommonRequestParams, extra_headers: dict[str, str]
) -> ModelStat:
    pipeline_request = _pipeline_request(common_params.model_id)
    if pipeline_request is not None:
        await _authorize_pipeline_stages(pipeline_request, common_params.api_key)
        family, _, stages = common_params.model_id.partition("/")
        return ModelStat(
            pipeline_request.family.task_type,
            pipeline_request.family.default_action,
            model_architecture=family,
            model_variant=stages or None,
        )

    fetch_kwargs: dict = {
        "model_id": common_params.model_id,
        "api_key": common_params.api_key or None,
        "proxy_url_builder": roboflow_secure_gateway_proxy_url_builder,
    }
    if extra_headers:
        fetch_kwargs["extra_headers"] = extra_headers
    try:
        meta = await asyncio.to_thread(get_one_page_of_model_metadata, **fetch_kwargs)
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
    return ModelStat(
        task_type,
        default_action,
        model_architecture=_reported_label(meta, "model_architecture"),
        model_variant=_reported_label(meta, "model_variant"),
    )


def _reset_cache_for_tests() -> None:
    _cache.clear()
    _inflight.clear()

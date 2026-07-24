import asyncio
import base64
import binascii
import contextvars
import hashlib
import hmac
import json
import os
import re
import stat
import tempfile
import time
import urllib.parse
from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import aiohttp
import backoff
import requests
from cachetools.func import ttl_cache
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import TimeoutError as RedisTimeoutError
from requests import Response, Timeout
from requests_toolbelt import MultipartEncoder
from yarl import URL

from inference.core import logger
from inference.core.cache import cache
from inference.core.cache.base import BaseCache
from inference.core.cache.model_artifacts import get_cache_dir, initialise_cache
from inference.core.entities.types import (
    DatasetID,
    ModelID,
    ModelType,
    TaskType,
    VersionID,
    WorkspaceID,
)
from inference.core.env import (
    API_BASE_URL,
    API_PROXY_BASE_URL,
    ENFORCE_CREDITS_VERIFICATION,
    GCP_SERVERLESS,
    INTERNAL_WEIGHTS_URL_SUFFIX,
    MD5_VERIFICATION_ENABLED,
    MODEL_CACHE_DIR,
    MODELS_CACHE_AUTH_CACHE_MAX_SIZE,
    MODELS_CACHE_AUTH_CACHE_TTL,
    MODELS_CACHE_AUTH_ENABLED,
    OFFLINE_MODE,
    RETRY_CONNECTION_ERRORS_TO_ROBOFLOW_API,
    ROBOFLOW_API_EXTRA_HEADERS,
    ROBOFLOW_API_REQUEST_TIMEOUT,
    ROBOFLOW_API_VERIFY_SSL,
    ROBOFLOW_ASSUME_IDENTITY_SERVICE_ACCESS_TOKEN,
    ROBOFLOW_INTERNAL_SERVICE_SECRET,
    ROBOFLOW_SERVICE_SECRET,
    SINGLE_TENANT_WORKFLOW_CACHE,
    TRANSIENT_ROBOFLOW_API_ERRORS,
    TRANSIENT_ROBOFLOW_API_ERRORS_RETRIES,
    TRANSIENT_ROBOFLOW_API_ERRORS_RETRY_INTERVAL,
    USE_FILE_CACHE_FOR_WORKFLOWS_DEFINITIONS,
    WORKFLOWS_DEFINITION_CACHE_EXPIRY,
)
from inference.core.exceptions import (
    CacheUnavailableError,
    MalformedRoboflowAPIResponseError,
    MalformedWorkflowResponseError,
    MissingDefaultModelError,
    PaymentRequiredError,
    RetryRequestError,
    RoboflowAPIConnectionError,
    RoboflowAPIForbiddenError,
    RoboflowAPIIAlreadyAnnotatedError,
    RoboflowAPIIAnnotationRejectionError,
    RoboflowAPIImageUploadRejectionError,
    RoboflowAPINotAuthorizedError,
    RoboflowAPINotNotFoundError,
    RoboflowAPITimeoutError,
    RoboflowAPIUnsuccessfulRequestError,
    RoboflowAPIUsagePausedError,
    WorkspaceLoadError,
)
from inference.core.telemetry import record_api_call, record_error, start_span
from inference.core.utils.file_system import sanitize_path_segment
from inference.core.utils.requests import (
    api_key_safe_raise_for_status,
    api_key_safe_raise_for_status_aiohttp,
)
from inference.core.utils.url_utils import wrap_url
from inference.core.version import __version__

LOCAL_API_KEY = "local"

_WINDOWS_RESERVED_PATH_SEGMENTS = frozenset(
    {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{index}" for index in range(1, 10)),
        *(f"LPT{index}" for index in range(1, 10)),
    }
)
_WORKFLOW_CACHE_AMBIGUOUS_LEGACY_FILENAME_SUFFIX = re.compile(
    r"(?:_[0-9a-f]{64}|_v.+)$"
)
_WORKFLOW_LEGACY_CANONICAL_SEGMENT = re.compile(r"[a-z0-9-]+")
_WORKFLOW_CANONICAL_CACHE_NAMESPACE = ".canonical-v2"
_WORKFLOW_TENANT_CACHE_NAMESPACE = ".tenanted-v2"

_EPHEMERAL_CACHE_UNAVAILABLE_EXCEPTIONS = (
    RedisConnectionError,
    RedisTimeoutError,
)

ENFORCE_CREDITS_VERIFICATION_HEADER = "x-enforce-credits-verification"
ENFORCE_INTERNAL_ARTIFACTS_URLS_HEADER = "x-enforce-internal-artefacts-urls"
ASSUME_IDENTITY_ACCESS_TOKEN_HEADER = "x-assume-identity-access-token"
ASSUME_IDENTITY_AUTHORISED_WORKSPACE_HEADER = "x-assume-identity-authorised-workspace"

assume_identity_authorised_workspace_db_id: contextvars.ContextVar[
    Optional[WorkspaceID]
] = contextvars.ContextVar("assume_identity_authorised_workspace_db_id", default=None)

MODEL_TYPE_DEFAULTS = {
    "object-detection": "yolov5v2s",
    "instance-segmentation": "yolact",
    "classification": "vit",
    "keypoint-detection": "yolov8n",
}
PROJECT_TASK_TYPE_KEY = "project_task_type"
MODEL_TYPE_KEY = "model_type"

NOT_FOUND_ERROR_MESSAGE = (
    "Could not find requested Roboflow resource. Check that the provided dataset and "
    "version are correct, and check that the provided Roboflow API key has the correct permissions."
)

ROBOFLOW_INFERENCE_VERSION_HEADER = "X-Roboflow-Inference-Version"
ALLOW_CHUNKED_RESPONSE_HEADER = "X-Allow-Chunked"
API_PROXY_ENDPOINT_PREFIXES = ("apiproxy", "api-proxy")


@dataclass(frozen=True)
class ServerlessUsageCheckResponse:
    status_code: int
    workspace_id: Optional[WorkspaceID] = None
    workspace_db_id: Optional[WorkspaceID] = None
    under_cap: Optional[bool] = None
    error: Optional[str] = None


def raise_from_lambda(
    inner_error: Exception, exception_type: Type[Exception], message: str
) -> None:
    raise exception_type(message) from inner_error


DEFAULT_ERROR_HANDLERS = {
    401: lambda e: raise_from_lambda(
        e,
        RoboflowAPINotAuthorizedError,
        "Unauthorized access to roboflow API - check API key. Visit "
        "https://docs.roboflow.com/api-reference/authentication#retrieve-an-api-key to learn how to retrieve one.",
    ),
    402: lambda e: raise_from_lambda(
        e,
        PaymentRequiredError,
        "Not enough credits to perform this request. Verify your workspace billing page.",
    ),
    403: lambda e: raise_from_lambda(
        e,
        RoboflowAPIForbiddenError,
        "Unauthorized access to roboflow API - check API key regarding correctness and required scopes. Visit "
        "https://docs.roboflow.com/api-reference/authentication#retrieve-an-api-key to learn how to retrieve one.",
    ),
    404: lambda e: raise_from_lambda(
        e, RoboflowAPINotNotFoundError, NOT_FOUND_ERROR_MESSAGE
    ),
    423: lambda e: raise_from_lambda(
        e,
        RoboflowAPIUsagePausedError,
        "Roboflow API usage is paused. Please contact your workspace administrator to re-enable api keys.",
    ),
}


def wrap_roboflow_api_errors(
    http_errors_handlers: Optional[
        Dict[int, Callable[[Union[requests.exceptions.HTTPError]], None]]
    ] = None,
) -> callable:
    def decorator(function: callable) -> callable:
        def wrapper(*args, **kwargs) -> Any:
            t_start = time.perf_counter()
            with start_span(
                "roboflow_api.call",
                {"roboflow_api.function": function.__name__},
            ):
                try:
                    try:
                        return function(*args, **kwargs)
                    except RetryRequestError as error:
                        raise error.inner_error
                except Timeout as error:
                    record_error(error)
                    raise RoboflowAPITimeoutError(
                        "Timeout when attempting to connect to Roboflow API."
                    ) from error
                except (
                    requests.exceptions.ConnectionError,
                    ConnectionError,
                ) as error:
                    record_error(error)
                    raise RoboflowAPIConnectionError(
                        "Could not connect to Roboflow API."
                    ) from error
                except requests.exceptions.HTTPError as error:
                    record_error(error)
                    user_handler_override = (
                        http_errors_handlers if http_errors_handlers is not None else {}
                    )
                    status_code = error.response.status_code
                    default_handler = DEFAULT_ERROR_HANDLERS.get(status_code)
                    error_handler = user_handler_override.get(
                        status_code, default_handler
                    )
                    if error_handler is not None:
                        error_handler(error)
                    raise RoboflowAPIUnsuccessfulRequestError(
                        f"Unsuccessful request to Roboflow API with response code: {status_code}"
                    ) from error
                except requests.exceptions.InvalidJSONError as error:
                    record_error(error)
                    raise MalformedRoboflowAPIResponseError(
                        "Could not decode JSON response from Roboflow API."
                    ) from error
                finally:
                    record_api_call(function.__name__, time.perf_counter() - t_start)

        return wrapper

    return decorator


def wrap_roboflow_api_errors_async(
    http_errors_handlers: Optional[
        Dict[int, Callable[[Union[aiohttp.ClientError]], None]]
    ] = None,
) -> callable:
    def decorator(function: callable) -> callable:
        async def wrapper(*args, **kwargs) -> Any:
            t_start = time.perf_counter()
            with start_span(
                "roboflow_api.call",
                {"roboflow_api.function": function.__name__},
            ):
                try:
                    try:
                        return await function(*args, **kwargs)
                    except RetryRequestError as error:
                        raise error.inner_error
                except asyncio.TimeoutError as error:
                    record_error(error)
                    raise RoboflowAPITimeoutError(
                        "Timeout when attempting to connect to Roboflow API."
                    ) from error
                except (aiohttp.ClientConnectionError, ConnectionError) as error:
                    record_error(error)
                    raise RoboflowAPIConnectionError(
                        "Could not connect to Roboflow API."
                    ) from error
                except (aiohttp.ContentTypeError, JSONDecodeError) as error:
                    record_error(error)
                    raise MalformedRoboflowAPIResponseError(
                        "Could not decode JSON response from Roboflow API."
                    ) from error
                except aiohttp.ClientResponseError as error:
                    record_error(error)
                    user_handler_override = (
                        http_errors_handlers if http_errors_handlers is not None else {}
                    )
                    status_code = error.status
                    default_handler = DEFAULT_ERROR_HANDLERS.get(status_code)
                    error_handler = user_handler_override.get(
                        status_code, default_handler
                    )
                    if error_handler is not None:
                        error_handler(error)
                    raise RoboflowAPIUnsuccessfulRequestError(
                        f"Unsuccessful request to Roboflow API with response code: {status_code}"
                    ) from error
                # remaining aiohttp.ClientError seems to qualify to simply pass-through raise
                finally:
                    record_api_call(function.__name__, time.perf_counter() - t_start)

        return wrapper

    return decorator


@ttl_cache(ttl=MODELS_CACHE_AUTH_CACHE_TTL, maxsize=MODELS_CACHE_AUTH_CACHE_MAX_SIZE)
@wrap_roboflow_api_errors()
def get_roboflow_workspace(api_key: str) -> WorkspaceID:
    if not api_key:
        raise WorkspaceLoadError("Empty workspace encountered, check your API key.")

    api_url = _add_params_to_url(
        url=f"{API_BASE_URL}/",
        params=[("api_key", api_key), ("nocache", "true")],
    )
    api_key_info = _get_from_url(url=api_url)
    workspace_id = api_key_info.get("workspace")
    if workspace_id is None:
        raise WorkspaceLoadError("Empty workspace encountered, check your API key.")
    return workspace_id


@wrap_roboflow_api_errors_async()
@backoff.on_exception(
    backoff.constant,
    exception=RetryRequestError,
    max_tries=TRANSIENT_ROBOFLOW_API_ERRORS_RETRIES,
    interval=TRANSIENT_ROBOFLOW_API_ERRORS_RETRY_INTERVAL,
)
async def get_roboflow_workspace_async(api_key: str) -> WorkspaceID:
    if OFFLINE_MODE:
        raise RoboflowAPIConnectionError(
            "Cannot fetch workspace - OFFLINE_MODE is enabled."
        )
    try:
        headers = build_roboflow_api_headers()
        full_url = wrap_url(
            _add_params_to_url(
                url=f"{API_BASE_URL}/",
                params=[("api_key", api_key), ("nocache", "true")],
            )
        )
        async with aiohttp.ClientSession() as session:
            async with session.get(
                URL(full_url, encoded=True),
                headers=headers,
                timeout=ROBOFLOW_API_REQUEST_TIMEOUT,
            ) as response:
                try:
                    api_key_safe_raise_for_status_aiohttp(response=response)
                except Exception as error:
                    if response.status in TRANSIENT_ROBOFLOW_API_ERRORS:
                        raise RetryRequestError(
                            message=str(error), inner_error=error
                        ) from error
                    raise error
                response_payload = await response.json()
                workspace_id = response_payload.get("workspace")
                if workspace_id is None:
                    raise WorkspaceLoadError(
                        "Empty workspace encountered, check your API key."
                    )
                return workspace_id
    except (aiohttp.ClientConnectionError, ConnectionError) as error:
        if RETRY_CONNECTION_ERRORS_TO_ROBOFLOW_API:
            raise RetryRequestError(
                message="Connectivity error", inner_error=error
            ) from error
        raise error


@wrap_roboflow_api_errors_async()
@backoff.on_exception(
    backoff.constant,
    exception=RetryRequestError,
    max_tries=TRANSIENT_ROBOFLOW_API_ERRORS_RETRIES,
    interval=TRANSIENT_ROBOFLOW_API_ERRORS_RETRY_INTERVAL,
)
async def get_serverless_usage_check_async(
    api_key: str,
) -> ServerlessUsageCheckResponse:
    if OFFLINE_MODE:
        raise RoboflowAPIConnectionError(
            "Cannot run serverless usage check - OFFLINE_MODE is enabled."
        )
    try:
        headers = build_roboflow_api_headers()
        full_url = wrap_url(
            _add_params_to_url(
                url=f"{API_BASE_URL}/serverless/usage-check",
                params=[("api_key", api_key), ("nocache", "true")],
            )
        )
        async with aiohttp.ClientSession() as session:
            async with session.get(
                URL(full_url, encoded=True),
                headers=headers,
                timeout=ROBOFLOW_API_REQUEST_TIMEOUT,
            ) as response:
                if response.status == 401:
                    return ServerlessUsageCheckResponse(status_code=401)
                if response.status == 402:
                    response_payload = await response.json()
                    workspace = response_payload.get(
                        "workspace"
                    ) or response_payload.get("workspaceId")
                    return ServerlessUsageCheckResponse(
                        status_code=402,
                        workspace_id=workspace,
                        workspace_db_id=response_payload.get("workspaceId"),
                        under_cap=response_payload.get("underCap"),
                        error=response_payload.get("error"),
                    )
                try:
                    api_key_safe_raise_for_status_aiohttp(response=response)
                except Exception as error:
                    if response.status in TRANSIENT_ROBOFLOW_API_ERRORS:
                        raise RetryRequestError(
                            message=str(error), inner_error=error
                        ) from error
                    raise error
                response_payload = await response.json()
                workspace_id = response_payload.get(
                    "workspace"
                ) or response_payload.get("workspaceId")
                if workspace_id is None or response_payload.get("underCap") is not True:
                    raise WorkspaceLoadError(
                        "Unexpected serverless usage-check response received from Roboflow API."
                    )
                return ServerlessUsageCheckResponse(
                    status_code=200,
                    workspace_id=workspace_id,
                    workspace_db_id=response_payload.get("workspaceId"),
                    under_cap=True,
                )
    except (aiohttp.ClientConnectionError, ConnectionError) as error:
        if RETRY_CONNECTION_ERRORS_TO_ROBOFLOW_API:
            raise RetryRequestError(
                message="Connectivity error", inner_error=error
            ) from error
        raise error


@wrap_roboflow_api_errors()
def add_custom_metadata(
    api_key: str,
    workspace_id: WorkspaceID,
    inference_ids: List[str],
    field_name: str,
    field_value: str,
):
    api_url = wrap_url(
        _add_params_to_url(
            url=f"{API_BASE_URL}/{workspace_id}/inference-stats/metadata",
            params=[("api_key", api_key), ("nocache", "true")],
        )
    )
    if OFFLINE_MODE:
        return
    response = requests.post(
        url=api_url,
        json={
            "data": [
                {
                    "inference_ids": inference_ids,
                    "field_name": field_name,
                    "field_value": field_value,
                }
            ]
        },
        headers=build_roboflow_api_headers(),
        timeout=ROBOFLOW_API_REQUEST_TIMEOUT,
        verify=ROBOFLOW_API_VERIFY_SSL,
    )
    api_key_safe_raise_for_status(response=response)


@wrap_roboflow_api_errors()
def get_roboflow_dataset_type(
    api_key: str, workspace_id: WorkspaceID, dataset_id: DatasetID
) -> TaskType:
    api_url = _add_params_to_url(
        url=f"{API_BASE_URL}/{workspace_id}/{dataset_id}",
        params=[("api_key", api_key), ("nocache", "true")],
    )
    dataset_info = _get_from_url(url=api_url)
    project_task_type = dataset_info.get("project", {})
    if "type" not in project_task_type:
        logger.warning(
            f"Project task type not defined for workspace={workspace_id} and dataset={dataset_id}, defaulting "
            f"to object-detection."
        )
    return project_task_type.get("type", "object-detection")


@wrap_roboflow_api_errors(
    http_errors_handlers={
        500: lambda e: raise_from_lambda(
            e, RoboflowAPINotNotFoundError, NOT_FOUND_ERROR_MESSAGE
        )
        # this is temporary solution, empirically checked that backend API responds HTTP 500 on incorrect version.
        # TO BE FIXED at backend, otherwise this error handling may overshadow existing backend problems.
    }
)
def get_roboflow_model_type(
    api_key: str,
    workspace_id: WorkspaceID,
    dataset_id: DatasetID,
    version_id: VersionID,
    project_task_type: ModelType,
) -> ModelType:
    api_url = _add_params_to_url(
        url=f"{API_BASE_URL}/{workspace_id}/{dataset_id}/{version_id}",
        params=[("api_key", api_key), ("nocache", "true")],
    )
    version_info = _get_from_url(url=api_url)
    model_type = version_info["version"]
    if "modelType" not in model_type:
        if project_task_type not in MODEL_TYPE_DEFAULTS:
            raise MissingDefaultModelError(
                f"Could not set default model for {project_task_type}"
            )
        logger.warning(
            f"Model type not defined - using default for {project_task_type} task."
        )
    return model_type.get("modelType", MODEL_TYPE_DEFAULTS[project_task_type])


class ModelEndpointType(Enum):
    ORT = "ort"
    CORE_MODEL = "core_model"
    OWLV2 = "owlv2"


@wrap_roboflow_api_errors()
def get_roboflow_model_data(
    api_key: str,
    model_id: str,
    endpoint_type: ModelEndpointType,
    device_id: str,
    countinference: Optional[bool] = None,
    service_secret: Optional[str] = None,
) -> dict:
    api_data_cache_key = f"roboflow_api_data:{endpoint_type.value}:{model_id}"
    api_data = None
    if not MODELS_CACHE_AUTH_ENABLED:
        api_data = cache.get(api_data_cache_key)
    if api_data is not None:
        logger.debug(f"Loaded model data from cache with key: {api_data_cache_key}.")
        return api_data
    else:
        params = [
            ("nocache", "true"),
            ("device", device_id),
            ("dynamic", "true"),
        ]
        if api_key is not None and api_key != LOCAL_API_KEY:
            params.append(("api_key", api_key))

        if (
            INTERNAL_WEIGHTS_URL_SUFFIX == "serverless"
            and countinference is False
            and service_secret == ROBOFLOW_SERVICE_SECRET
        ):
            params.append(("countinference", str(countinference).lower()))
            params.append(("service_secret", service_secret))

        api_base_url = urllib.parse.urljoin(API_BASE_URL, INTERNAL_WEIGHTS_URL_SUFFIX)
        api_url = _add_params_to_url(
            url=f"{api_base_url}/{endpoint_type.value}/{model_id}",
            params=params,
        )
        api_data = _get_from_url(url=api_url)
        cache.set(
            api_data_cache_key,
            api_data,
            expire=10,
        )
        logger.debug(
            f"Loaded model data from Roboflow API and saved to cache with key: {api_data_cache_key}."
        )
        return api_data


@wrap_roboflow_api_errors()
def get_roboflow_instant_model_data(
    api_key: str,
    model_id: ModelID,
    cache_prefix: str = "roboflow_api_data",
    countinference: Optional[bool] = None,
    service_secret: Optional[str] = None,
) -> dict:
    api_data_cache_key = f"{cache_prefix}:{model_id}"
    api_data = None
    if not MODELS_CACHE_AUTH_ENABLED:
        api_data = cache.get(api_data_cache_key)
    if api_data is not None:
        logger.debug(f"Loaded model data from cache with key: {api_data_cache_key}.")
        return api_data
    else:
        params = [
            ("model", model_id),
        ]
        if api_key is not None and api_key != LOCAL_API_KEY:
            params.append(("api_key", api_key))

        if (
            INTERNAL_WEIGHTS_URL_SUFFIX == "serverless"
            and countinference is False
            and service_secret == ROBOFLOW_SERVICE_SECRET
        ):
            params.append(("countinference", str(countinference).lower()))
            params.append(("service_secret", service_secret))

        api_base_url = urllib.parse.urljoin(API_BASE_URL, INTERNAL_WEIGHTS_URL_SUFFIX)
        api_url = _add_params_to_url(
            url=f"{api_base_url}/getWeights",
            params=params,
        )
        api_data = _get_from_url(url=api_url)
        cache.set(
            api_data_cache_key,
            api_data,
            expire=10,
        )
        logger.debug(
            f"Loaded model data from Roboflow API and saved to cache with key: {api_data_cache_key}."
        )
        return api_data


@wrap_roboflow_api_errors()
def get_model_metadata_from_inference_models_registry(
    api_key: Optional[str],
    model_id: ModelID,
    cache_prefix: str = "roboflow_api_data:inference_models_registry",
    countinference: Optional[bool] = None,
    service_secret: Optional[str] = None,
) -> dict:
    # This endpoint is only used for auth and model/task type lookup. Actual
    # artifact downloads still go through the inference-models weights provider.
    api_data_cache_key = f"{cache_prefix}:{model_id}"
    api_data = None
    if not MODELS_CACHE_AUTH_ENABLED:
        api_data = cache.get(api_data_cache_key)
    if api_data is not None:
        logger.debug(f"Loaded model data from cache with key: {api_data_cache_key}.")
        return api_data
    query = [("modelId", model_id)]
    headers = {}
    if api_key is not None and api_key != LOCAL_API_KEY:
        headers["Authorization"] = f"Bearer {api_key}"
    if GCP_SERVERLESS:
        headers[ENFORCE_INTERNAL_ARTIFACTS_URLS_HEADER] = "true"
    if ENFORCE_CREDITS_VERIFICATION:
        skip = (
            countinference is False
            and service_secret is not None
            and service_secret == ROBOFLOW_SERVICE_SECRET
        )
        if not skip:
            headers[ENFORCE_CREDITS_VERIFICATION_HEADER] = "true"
    if ROBOFLOW_INTERNAL_SERVICE_SECRET:
        headers["X-Roboflow-Internal-Service-Secret"] = ROBOFLOW_INTERNAL_SERVICE_SECRET
    _add_assume_identity_headers(headers=headers)
    api_url = _add_params_to_url(
        url=f"{API_BASE_URL}/models/v1/external/stat",
        params=query,
    )
    raw_api_data = _get_from_url(url=api_url, headers=headers)
    model_metadata = raw_api_data["modelMetadata"]
    api_data = {
        "modelType": model_metadata["modelArchitecture"],
        "taskType": model_metadata["taskType"],
    }
    cache.set(
        api_data_cache_key,
        api_data,
        expire=10,
    )
    logger.debug(
        f"Loaded model data from Roboflow API (inference-models registry) "
        f"and saved to cache with key: {api_data_cache_key}."
    )
    return api_data


def _add_assume_identity_headers(headers: Dict[str, str]) -> None:
    if not ROBOFLOW_ASSUME_IDENTITY_SERVICE_ACCESS_TOKEN:
        return
    authorised_workspace = assume_identity_authorised_workspace_db_id.get()
    if not authorised_workspace:
        return
    headers[ASSUME_IDENTITY_ACCESS_TOKEN_HEADER] = (
        ROBOFLOW_ASSUME_IDENTITY_SERVICE_ACCESS_TOKEN
    )
    headers[ASSUME_IDENTITY_AUTHORISED_WORKSPACE_HEADER] = authorised_workspace


@wrap_roboflow_api_errors()
def get_roboflow_base_lora(
    api_key: str, repo: str, revision: str, device_id: str
) -> dict:
    full_path = f"{repo.strip('/')}/{revision.strip('/')}"
    api_data_cache_key = f"roboflow_api_data:lora-bases:{full_path}"
    api_data = None
    if not MODELS_CACHE_AUTH_ENABLED:
        api_data = cache.get(api_data_cache_key)
    if api_data is not None:
        logger.debug(f"Loaded model data from cache with key: {api_data_cache_key}.")
        return api_data
    else:
        params = [
            ("nocache", "true"),
            ("device", device_id),
            ("repoAndRevision", full_path),
        ]
        if api_key is not None and api_key != LOCAL_API_KEY:
            params.append(("api_key", api_key))
        api_url = _add_params_to_url(
            url=f"{API_BASE_URL}/lora_bases",
            params=params,
        )
        api_data = _get_from_url(url=api_url)
        cache.set(
            api_data_cache_key,
            api_data,
            expire=10,
        )
        logger.debug(
            f"Loaded lora base model data from Roboflow API and saved to cache with key: {api_data_cache_key}."
        )
        return api_data


@wrap_roboflow_api_errors()
def get_roboflow_active_learning_configuration(
    api_key: str,
    workspace_id: WorkspaceID,
    dataset_id: DatasetID,
) -> dict:
    api_url = _add_params_to_url(
        url=f"{API_BASE_URL}/{workspace_id}/{dataset_id}/active_learning",
        params=[("api_key", api_key)],
    )
    return _get_from_url(url=api_url)


@wrap_roboflow_api_errors()
def register_image_at_roboflow(
    api_key: str,
    dataset_id: DatasetID,
    local_image_id: str,
    image_bytes: bytes,
    batch_name: str,
    tags: Optional[List[str]] = None,
    inference_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> dict:
    if OFFLINE_MODE:
        # callers consume the response (e.g. `["id"]`) - raise so existing
        # error handling reports a clear cause instead of a KeyError
        raise RoboflowAPIConnectionError(
            "Cannot register image at Roboflow - OFFLINE_MODE is enabled."
        )
    url = f"{API_BASE_URL}/dataset/{dataset_id}/upload"
    params = [
        ("api_key", api_key),
        ("batch", batch_name),
    ]
    if inference_id is not None:
        params.append(("inference_id", inference_id))
    tags = tags if tags is not None else []
    for tag in tags:
        params.append(("tag", tag))
    wrapped_url = wrap_url(_add_params_to_url(url=url, params=params))
    fields = {
        "name": f"{local_image_id}.jpg",
        "file": ("imageToUpload", image_bytes, "image/jpeg"),
    }
    if metadata is not None:
        fields["metadata"] = json.dumps(metadata)
    m = MultipartEncoder(fields=fields)
    headers = build_roboflow_api_headers(
        explicit_headers={"Content-Type": m.content_type},
    )
    response = requests.post(
        url=wrapped_url,
        data=m,
        headers=headers,
        timeout=ROBOFLOW_API_REQUEST_TIMEOUT,
        verify=ROBOFLOW_API_VERIFY_SSL,
    )
    api_key_safe_raise_for_status(response=response)
    parsed_response = response.json()
    if not parsed_response.get("duplicate") and not parsed_response.get("success"):
        raise RoboflowAPIImageUploadRejectionError(
            f"Server rejected image: {parsed_response}"
        )
    return parsed_response


@wrap_roboflow_api_errors(
    http_errors_handlers={
        409: lambda e: raise_from_lambda(
            e,
            RoboflowAPIIAlreadyAnnotatedError,
            "Given datapoint already has annotation.",
        )
    }
)
def annotate_image_at_roboflow(
    api_key: str,
    dataset_id: DatasetID,
    local_image_id: str,
    roboflow_image_id: str,
    annotation_content: str,
    annotation_file_type: str,
    is_prediction: bool = True,
) -> dict:
    if OFFLINE_MODE:
        raise RoboflowAPIConnectionError(
            "Cannot annotate image at Roboflow - OFFLINE_MODE is enabled."
        )
    url = f"{API_BASE_URL}/dataset/{dataset_id}/annotate/{roboflow_image_id}"
    params = [
        ("api_key", api_key),
        ("name", f"{local_image_id}.{annotation_file_type}"),
        ("prediction", str(is_prediction).lower()),
    ]
    wrapped_url = wrap_url(_add_params_to_url(url=url, params=params))
    headers = build_roboflow_api_headers(
        explicit_headers={"Content-Type": "text/plain"},
    )
    response = requests.post(
        wrapped_url,
        data=annotation_content,
        headers=headers,
        timeout=ROBOFLOW_API_REQUEST_TIMEOUT,
        verify=ROBOFLOW_API_VERIFY_SSL,
    )
    api_key_safe_raise_for_status(response=response)
    parsed_response = response.json()
    if "error" in parsed_response or not parsed_response.get("success"):
        raise RoboflowAPIIAnnotationRejectionError(
            f"Failed to save annotation for {roboflow_image_id}. API response: {parsed_response}"
        )
    return parsed_response


@wrap_roboflow_api_errors()
def update_image_metadata_at_roboflow(
    api_key: str,
    workspace_id: WorkspaceID,
    image_id: str,
    metadata: Optional[Dict[str, Any]] = None,
    add_tags: Optional[List[str]] = None,
) -> Dict[str, Any]:
    if OFFLINE_MODE:
        raise RoboflowAPIConnectionError(
            "Cannot update image metadata at Roboflow - OFFLINE_MODE is enabled."
        )
    payload = {}
    if metadata is not None:
        payload["metadata"] = metadata
    if add_tags is not None:
        payload["addTags"] = add_tags

    encoded_image_id = urllib.parse.quote(image_id, safe="")
    api_url = wrap_url(
        _add_params_to_url(
            url=f"{API_BASE_URL}/{workspace_id}/images/{encoded_image_id}/metadata",
            params=[("api_key", api_key)],
        )
    )
    response = requests.post(
        url=api_url,
        json=payload,
        headers=build_roboflow_api_headers(),
        timeout=ROBOFLOW_API_REQUEST_TIMEOUT,
        verify=ROBOFLOW_API_VERIFY_SSL,
    )
    api_key_safe_raise_for_status(response=response)
    return response.json()


@wrap_roboflow_api_errors()
def batch_update_image_metadata_at_roboflow(
    api_key: str,
    workspace_id: WorkspaceID,
    updates: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if OFFLINE_MODE:
        raise RoboflowAPIConnectionError(
            "Cannot update image metadata at Roboflow - OFFLINE_MODE is enabled."
        )
    api_url = wrap_url(
        _add_params_to_url(
            url=f"{API_BASE_URL}/{workspace_id}/images/metadata",
            params=[("api_key", api_key)],
        )
    )
    response = requests.post(
        url=api_url,
        json={"updates": updates},
        headers=build_roboflow_api_headers(),
        timeout=ROBOFLOW_API_REQUEST_TIMEOUT,
        verify=ROBOFLOW_API_VERIFY_SSL,
    )
    api_key_safe_raise_for_status(response=response)
    return response.json()


def search_project_images_at_roboflow(
    api_key: str,
    workspace: str,
    project: str,
    image_base64: str,
    limit: int,
    fields: Optional[List[str]] = None,
) -> Dict[str, Any]:
    payload = {
        "image_base64": image_base64,
        "limit": limit,
        "fields": fields
        or [
            "id",
            "name",
            "filename",
            "url",
            "user_metadata",
            "tags",
            "width",
            "height",
            "aspectRatio",
        ],
    }
    return post_to_roboflow_api(
        endpoint=f"{workspace}/{project}/search",
        api_key=api_key,
        payload=payload,
    )


@wrap_roboflow_api_errors()
def get_roboflow_labeling_batches(
    api_key: str, workspace_id: WorkspaceID, dataset_id: str
) -> dict:
    api_url = _add_params_to_url(
        url=f"{API_BASE_URL}/{workspace_id}/{dataset_id}/batches",
        params=[("api_key", api_key)],
    )
    return _get_from_url(url=api_url)


@wrap_roboflow_api_errors()
def get_roboflow_labeling_jobs(
    api_key: str, workspace_id: WorkspaceID, dataset_id: str
) -> dict:
    api_url = _add_params_to_url(
        url=f"{API_BASE_URL}/{workspace_id}/{dataset_id}/jobs",
        params=[("api_key", api_key)],
    )
    return _get_from_url(url=api_url)


def _workflow_cache_identity_fingerprint(value: str) -> str:
    """Return a stable fingerprint for non-secret Workflow identities."""

    # This only hashes public identifiers, not credentials.
    return hashlib.sha256(value.encode("utf-8"), usedforsecurity=False).hexdigest()


def _workflow_cache_tenant_fingerprint(
    workspace_id: WorkspaceID,
    api_key: Optional[str],
) -> str:
    """Return a domain-separated tenant identity without exposing the API key."""

    message = json.dumps(
        ["inference-workflow-tenant-v2", workspace_id],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hmac.new(
        key=(api_key or "").encode("utf-8"),
        msg=message,
        digestmod=hashlib.sha256,
    ).hexdigest()


def _workflow_cache_path_segment(
    value: str,
    *,
    reject_legacy_filename_shape: bool = False,
) -> str:
    """Create a collision-resistant readable path segment."""

    sanitized_value = sanitize_path_segment(value)
    # Leave enough room for the version identity, tenant fingerprint, and
    # extension while staying below the common 255-byte filename limit.
    # Uppercase and Windows device names get a fingerprint too, preventing
    # aliases on case-insensitive filesystems and unusable reserved paths. The
    # tilde prefix cannot be emitted by the legacy sanitizer, keeping every
    # transformed path disjoint from old cache names.
    if (
        sanitized_value == value
        and sanitized_value
        and len(sanitized_value) <= 96
        and value == value.lower()
        and value.upper() not in _WINDOWS_RESERVED_PATH_SEGMENTS
        and (
            not reject_legacy_filename_shape
            or _WORKFLOW_CACHE_AMBIGUOUS_LEGACY_FILENAME_SUFFIX.search(value) is None
        )
    ):
        return sanitized_value
    value_fingerprint = _workflow_cache_identity_fingerprint(value)
    readable_prefix = sanitized_value[:48] or "empty"
    return f"~{readable_prefix}_{value_fingerprint}"


def _versioned_workflow_cache_stem(
    workflow_id: str,
    workflow_version_id: str,
) -> str:
    identity = json.dumps(
        [workflow_id, workflow_version_id],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    identity_fingerprint = _workflow_cache_identity_fingerprint(identity)
    return (
        f"{_workflow_cache_path_segment(workflow_id, reject_legacy_filename_shape=True)}_"
        f"{identity_fingerprint}"
    )


def get_workflow_cache_file(
    workspace_id: WorkspaceID,
    workflow_id: str,
    api_key: Optional[str],
    workflow_version_id: Optional[str] = None,
) -> str:
    sanitized_workspace_id = _workflow_cache_path_segment(workspace_id)
    # Preserve the public API's historical truthiness semantics: an empty
    # optional query value means "latest", just like ``None``. This also keeps
    # file-cache attribution aligned with the ephemeral-cache key.
    if not workflow_version_id:
        cache_subdirectory = ""
        cache_stem = _workflow_cache_path_segment(
            workflow_id,
            reject_legacy_filename_shape=True,
        )
    else:
        if not isinstance(workflow_version_id, str):
            # Truthy non-string values failed in sanitize_path_segment before
            # versioned cache paths were introduced. Keep that source behavior
            # instead of silently creating a new cache identity for bad input.
            raise TypeError("workflow_version_id must be a string or None")
        # Keep pinned entries in a distinct namespace so an unversioned
        # workflow ID can never collide with a workflow/version tuple.
        cache_subdirectory = "versions"
        cache_stem = _versioned_workflow_cache_stem(
            workflow_id=workflow_id,
            workflow_version_id=workflow_version_id,
        )

    if SINGLE_TENANT_WORKFLOW_CACHE:
        filename = f"{cache_stem}.json"
        cache_subdirectory = os.path.join(
            _WORKFLOW_CANONICAL_CACHE_NAMESPACE,
            cache_subdirectory,
        )
    else:
        cache_fingerprint = _workflow_cache_tenant_fingerprint(
            workspace_id=workspace_id,
            api_key=api_key,
        )
        filename = f"{cache_stem}_{cache_fingerprint}.json"
        # Flat hashed filenames are indistinguishable from legacy canonical
        # Workflow IDs that happen to end in ``_<64 hex>``. Keep newly warmed
        # tenant-attributed entries in a reserved namespace so credential-free
        # offline lookup never has to guess which Workflow a file belongs to.
        cache_subdirectory = os.path.join(
            _WORKFLOW_TENANT_CACHE_NAMESPACE,
            cache_subdirectory,
        )
    prefix = os.path.abspath(os.path.join(MODEL_CACHE_DIR, "workflow"))
    result = os.path.abspath(
        os.path.join(
            prefix,
            sanitized_workspace_id,
            cache_subdirectory,
            filename,
        )
    )
    prefix_with_separator = prefix.rstrip(os.sep) + os.sep
    if result.startswith(prefix_with_separator):
        try:
            if os.path.commonpath([prefix, result]) == prefix:
                return result
        except ValueError:
            pass
    raise ValueError(
        "Detected attempt to save workflow definition in insecure location"
    )


def _find_offline_hashed_workflow_cache_file(
    workspace_id: WorkspaceID,
    workflow_id: str,
    api_key: Optional[str],
    workflow_version_id: Optional[str] = None,
) -> Optional[str]:
    """Find a safely attributable cache written before offline mode was enabled.

    Current multi-tenant cache filenames live in a reserved namespace and
    contain a workspace/API-key fingerprint. Versioned entries also contain the
    exact Workflow version. An exact fingerprint match is safe when the API key
    is still configured. Without an API key, a sole entry for the requested
    version is usable in an offline single-tenant deployment; multiple entries
    remain ambiguous. Legacy flat hashed entries are deliberately ignored:
    their names cannot be distinguished from canonical Workflow IDs ending in
    ``_<64 hex>``.
    """

    if not OFFLINE_MODE or not SINGLE_TENANT_WORKFLOW_CACHE:
        return None

    cache_file = Path(
        get_workflow_cache_file(
            workspace_id=workspace_id,
            workflow_id=workflow_id,
            api_key=api_key,
            workflow_version_id=workflow_version_id,
        )
    )
    validated_cache_file = _validated_workflow_cache_path(str(cache_file))
    if validated_cache_file is None:
        return None
    cache_file = Path(validated_cache_file)

    if workflow_version_id:
        workspace_cache_directory = cache_file.parent.parent.parent
        tenanted_cache_directory = (
            workspace_cache_directory / _WORKFLOW_TENANT_CACHE_NAMESPACE / "versions"
        )
    else:
        workspace_cache_directory = cache_file.parent.parent
        tenanted_cache_directory = (
            workspace_cache_directory / _WORKFLOW_TENANT_CACHE_NAMESPACE
        )
    validated_tenanted_probe = _validated_workflow_cache_path(
        str(tenanted_cache_directory / cache_file.name)
    )
    if validated_tenanted_probe is None:
        return None
    tenanted_cache_directory = Path(validated_tenanted_probe).parent
    if not tenanted_cache_directory.is_dir():
        return None

    filename_pattern = re.compile(rf"{re.escape(cache_file.stem)}_[0-9a-f]{{64}}\.json")
    candidates = []
    try:
        for candidate in tenanted_cache_directory.iterdir():
            if not filename_pattern.fullmatch(candidate.name):
                continue
            validated_candidate = _validated_workflow_cache_path(str(candidate))
            if validated_candidate is None:
                continue
            candidate = Path(validated_candidate)
            if candidate.is_file():
                candidates.append(candidate)
    except OSError:
        return None
    if api_key is not None:
        cache_fingerprint = _workflow_cache_tenant_fingerprint(
            workspace_id=workspace_id,
            api_key=api_key,
        )
        exact_filename = f"{cache_file.stem}_{cache_fingerprint}.json"
        return next(
            (
                str(candidate)
                for candidate in candidates
                if candidate.name == exact_filename
            ),
            None,
        )
    if len(candidates) == 1:
        return str(candidates[0])
    if len(candidates) > 1:
        logger.warning(
            "Cannot choose among %d hashed offline Workflow cache entries",
            len(candidates),
        )
    return None


def _case_swapped_sibling(path: Path) -> Optional[Path]:
    """Return one case-only alias probe for an ASCII cache path."""

    for index, character in enumerate(path.name):
        if "a" <= character <= "z":
            swapped_name = (
                path.name[:index] + character.upper() + path.name[index + 1 :]
            )
            return path.with_name(swapped_name)
        if "A" <= character <= "Z":
            swapped_name = (
                path.name[:index] + character.lower() + path.name[index + 1 :]
            )
            return path.with_name(swapped_name)
    return None


def _legacy_cache_path_has_exact_case(path: Path) -> bool:
    """Reject missing, case-aliased, or case-insensitive legacy entries."""

    try:
        if path.name not in os.listdir(path.parent):
            return False
    except OSError:
        return False
    case_alias = _case_swapped_sibling(path)
    return case_alias is None or not os.path.lexists(case_alias)


def _find_legacy_canonical_workflow_cache_file(
    workspace_id: WorkspaceID,
    workflow_id: str,
    workflow_version_id: Optional[str] = None,
) -> Optional[str]:
    """Find a flat pre-v2 cache only when its identity is injective."""

    if (
        not SINGLE_TENANT_WORKFLOW_CACHE
        or workflow_version_id
        or _WORKFLOW_LEGACY_CANONICAL_SEGMENT.fullmatch(workspace_id) is None
        or _WORKFLOW_LEGACY_CANONICAL_SEGMENT.fullmatch(workflow_id) is None
    ):
        return None

    cache_root = os.path.abspath(os.path.join(MODEL_CACHE_DIR, "workflow"))
    legacy_cache_file = Path(
        os.path.abspath(
            os.path.join(
                cache_root,
                workspace_id,
                f"{workflow_id}.json",
            )
        )
    )
    validated_legacy_cache_file = _validated_workflow_cache_path(str(legacy_cache_file))
    if validated_legacy_cache_file is None:
        return None
    legacy_cache_file = Path(validated_legacy_cache_file)
    workspace_cache_directory = legacy_cache_file.parent
    if not _legacy_cache_path_has_exact_case(
        workspace_cache_directory
    ) or not _legacy_cache_path_has_exact_case(legacy_cache_file):
        return None
    return str(legacy_cache_file)


def _validated_workflow_cache_path(workflow_cache_file: str) -> Optional[str]:
    """Return a normalized cache path only when it is safe to access."""

    model_cache_root = os.path.abspath(MODEL_CACHE_DIR)
    cache_root = os.path.abspath(os.path.join(model_cache_root, "workflow"))
    candidate = os.path.abspath(workflow_cache_file)
    cache_prefix = cache_root.rstrip(os.sep) + os.sep

    # The explicit normalized-prefix check is both a fast rejection and the
    # analyzer-visible path traversal barrier. commonpath below remains the
    # platform-aware authority.
    if not candidate.startswith(cache_prefix):
        return None
    try:
        if os.path.commonpath([cache_root, candidate]) != cache_root:
            return None
    except ValueError:
        return None
    if Path(candidate).suffix != ".json" or os.path.islink(cache_root):
        return None
    is_junction = getattr(os.path, "isjunction", lambda _path: False)
    if is_junction(cache_root):
        return None

    resolved_root = os.path.realpath(cache_root)
    expected_resolved_root = os.path.normpath(
        os.path.join(os.path.realpath(model_cache_root), "workflow")
    )
    if os.path.normcase(resolved_root) != os.path.normcase(expected_resolved_root):
        return None

    relative_path = os.path.relpath(candidate, cache_root)
    if relative_path in {"", os.curdir}:
        return None

    # MODEL_CACHE_DIR itself may intentionally be a mounted/symlinked volume.
    # Child symlinks are rejected because they can cross workspace, API-key, or
    # filesystem attribution boundaries.
    current_path = cache_root
    for path_part in relative_path.split(os.sep):
        current_path = os.path.join(current_path, path_part)
        if os.path.islink(current_path):
            return None

    expected_resolved_path = os.path.normpath(
        os.path.join(resolved_root, relative_path)
    )
    if os.path.realpath(candidate) != expected_resolved_path:
        return None
    return candidate


def _workflow_cache_response_is_valid(response: object) -> bool:
    if not isinstance(response, dict):
        return False
    workflow = response.get("workflow")
    if not isinstance(workflow, dict) or not isinstance(workflow.get("config"), str):
        return False
    try:
        workflow_config = json.loads(workflow["config"])
    except (TypeError, ValueError):
        return False
    return isinstance(workflow_config, dict) and isinstance(
        workflow_config.get("specification"), dict
    )


def _read_json_regular_file_no_follow(path: str) -> Any:
    """Read JSON from a regular file without following a final symlink."""

    path_status = os.lstat(path)
    if not stat.S_ISREG(path_status.st_mode):
        raise OSError(f"Refusing to read non-regular JSON file: {path}")
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0),
    )
    try:
        descriptor_status = os.fstat(descriptor)
        if not stat.S_ISREG(descriptor_status.st_mode):
            raise OSError(f"Refusing to read non-regular JSON file: {path}")
        if (path_status.st_dev, path_status.st_ino) != (
            descriptor_status.st_dev,
            descriptor_status.st_ino,
        ):
            raise OSError(f"JSON file changed while it was being opened: {path}")
        file_handle = os.fdopen(descriptor, "r", encoding="utf-8")
        descriptor = -1
        with file_handle:
            return json.load(file_handle)
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _load_workflow_response_file(workflow_cache_file: str) -> Optional[dict]:
    """Load a Workflow cache file without following unsafe filesystem objects."""

    validated_cache_file = _validated_workflow_cache_path(workflow_cache_file)
    if validated_cache_file is None:
        logger.warning("Refusing to read an unsafe Workflow cache file")
        return None
    try:
        response = _read_json_regular_file_no_follow(validated_cache_file)
        if not _workflow_cache_response_is_valid(response):
            raise ValueError("Malformed Workflow cache response")
        return response
    except (OSError, TypeError, ValueError):
        # Do not unlink a malformed path here. A concurrent atomic writer can
        # replace the inode after this reader opens it; deleting by pathname
        # would then remove the writer's valid replacement.
        return None


def cache_workflow_response(
    workspace_id: WorkspaceID,
    workflow_id: str,
    api_key: Optional[str],
    response: dict,
    workflow_version_id: Optional[str] = None,
):
    if not _workflow_cache_response_is_valid(response):
        logger.warning("Refusing to cache a malformed Workflow response")
        return None
    workflow_cache_file = get_workflow_cache_file(
        workspace_id=workspace_id,
        workflow_id=workflow_id,
        api_key=api_key,
        workflow_version_id=workflow_version_id,
    )
    validated_cache_file = _validated_workflow_cache_path(workflow_cache_file)
    if validated_cache_file is None:
        raise ValueError("Refusing to write an unsafe Workflow cache path")
    workflow_cache_dir = os.path.dirname(validated_cache_file)
    os.makedirs(workflow_cache_dir, exist_ok=True)
    validated_cache_file = _validated_workflow_cache_path(validated_cache_file)
    if validated_cache_file is None:
        raise ValueError("Refusing to write an unsafe Workflow cache path")

    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=workflow_cache_dir,
            # The final filename can already be close to NAME_MAX after adding
            # version and tenant fingerprints. Keep the temporary component
            # short so atomic writes still work for maximum-length IDs.
            prefix=".workflow.",
            suffix=".tmp",
            delete=False,
        ) as file_handle:
            temporary_path = file_handle.name
            json.dump(response, file_handle)
            file_handle.flush()
            os.fsync(file_handle.fileno())
        validated_cache_file = _validated_workflow_cache_path(validated_cache_file)
        if validated_cache_file is None:
            raise ValueError("Refusing to replace an unsafe Workflow cache path")
        # os.replace replaces a final symlink rather than following it and
        # prevents readers from observing partially-written JSON.
        os.replace(temporary_path, validated_cache_file)
        temporary_path = None
    finally:
        if temporary_path is not None:
            try:
                os.unlink(temporary_path)
            except OSError:
                pass


def delete_cached_workflow_response_if_exists(
    workspace_id: WorkspaceID,
    workflow_id: str,
    api_key: Optional[str],
    workflow_version_id: Optional[str] = None,
) -> None:
    """Delete the entry in the currently configured Workflow cache namespace."""

    workflow_cache_file = get_workflow_cache_file(
        workspace_id=workspace_id,
        workflow_id=workflow_id,
        api_key=api_key,
        workflow_version_id=workflow_version_id,
    )
    if not os.path.lexists(workflow_cache_file):
        return
    validated_cache_file = _validated_workflow_cache_path(workflow_cache_file)
    if validated_cache_file is None:
        logger.warning("Refusing to delete an unsafe Workflow cache file")
        return
    try:
        os.remove(validated_cache_file)
    except FileNotFoundError:
        pass


def load_cached_workflow_response(
    workspace_id: WorkspaceID,
    workflow_id: str,
    api_key: Optional[str],
    workflow_version_id: Optional[str] = None,
) -> Optional[dict]:
    workflow_cache_file = get_workflow_cache_file(
        workspace_id=workspace_id,
        workflow_id=workflow_id,
        api_key=api_key,
        workflow_version_id=workflow_version_id,
    )
    validated_cache_file = _validated_workflow_cache_path(workflow_cache_file)
    if validated_cache_file is None:
        return None
    if os.path.lexists(validated_cache_file):
        cached_response = _load_workflow_response_file(validated_cache_file)
        if cached_response is not None:
            return cached_response

    fallback_cache_file = _find_offline_hashed_workflow_cache_file(
        workspace_id=workspace_id,
        workflow_id=workflow_id,
        api_key=api_key,
        workflow_version_id=workflow_version_id,
    )
    if fallback_cache_file is not None:
        cached_response = _load_workflow_response_file(fallback_cache_file)
        if cached_response is not None:
            return cached_response

    legacy_cache_file = _find_legacy_canonical_workflow_cache_file(
        workspace_id=workspace_id,
        workflow_id=workflow_id,
        workflow_version_id=workflow_version_id,
    )
    if legacy_cache_file is None:
        return None
    return _load_workflow_response_file(legacy_cache_file)


@wrap_roboflow_api_errors()
def get_workflow_specification(
    api_key: Optional[str],
    workspace_id: WorkspaceID,
    workflow_id: str,
    use_cache: bool = True,
    ephemeral_cache: Optional[BaseCache] = None,
    workflow_version_id: Optional[str] = None,
) -> dict:
    """Fetch a workflow specification from cache or the Roboflow API.

    When ephemeral cache (Redis/Dragonfly) is enabled but unreachable, falls back
    to the Roboflow API instead of failing the request.

    Args:
        api_key: Roboflow API key, or None for unauthenticated fetches.
        workspace_id: Workspace slug, or ``local`` for filesystem-backed workflows.
        workflow_id: Workflow identifier within the workspace.
        use_cache: If True, read and write the ephemeral workflow-definition cache.
        ephemeral_cache: Cache backend; defaults to the process-global cache.
        workflow_version_id: Optional pinned workflow version.

    Returns:
        Parsed workflow specification dict.

    Raises:
        MalformedWorkflowResponseError: API response lacks a valid specification.
        RoboflowAPIRequestError: API request failed and no file-cache fallback applies.
        FileNotFoundError: Local workspace workflow file is missing.
        ValueError: Invalid local workflow id.
    """
    ephemeral_cache = ephemeral_cache or cache
    if use_cache:
        cached_entry = _try_retrieve_workflow_specification_from_ephemeral_cache(
            api_key=api_key,
            workspace_id=workspace_id,
            workflow_id=workflow_id,
            workflow_version_id=workflow_version_id,
            ephemeral_cache=ephemeral_cache,
        )
        if cached_entry:
            return cached_entry

    if workspace_id == "local":
        if not re.match(r"^[\w\-]+$", workflow_id):
            raise ValueError("Invalid workflow id")

        workflow_hash = sha256(workflow_id.encode()).hexdigest()
        local_file_path = (
            Path(MODEL_CACHE_DIR) / "workflow" / "local" / f"{workflow_hash}.json"
        )
        validated_local_file_path = _validated_workflow_cache_path(str(local_file_path))
        if validated_local_file_path is None:
            raise FileNotFoundError(f"Local workflow file not found: {local_file_path}")
        local_file_path = Path(validated_local_file_path)
        if not local_file_path.exists():
            raise FileNotFoundError(f"Local workflow file not found: {local_file_path}")

        try:
            local_config = _read_json_regular_file_no_follow(str(local_file_path))
        except OSError as error:
            raise FileNotFoundError(
                f"Local workflow file not found: {local_file_path}"
            ) from error

        # Mimic the same shape as the cloud response:
        response = {"workflow": local_config}
    else:
        params = []
        if api_key is not None and api_key != LOCAL_API_KEY:
            params.append(("api_key", api_key))
        if workflow_version_id is not None:
            params.append(("workflow_version", workflow_version_id))
        api_url = _add_params_to_url(
            url=f"{API_BASE_URL}/{workspace_id}/workflows/{workflow_id}",
            params=params,
        )
        try:
            response = _get_from_url(url=api_url)
            if USE_FILE_CACHE_FOR_WORKFLOWS_DEFINITIONS:
                cache_workflow_response(
                    workspace_id=workspace_id,
                    workflow_id=workflow_id,
                    api_key=api_key,
                    response=response,
                    workflow_version_id=workflow_version_id,
                )
        except (
            requests.exceptions.ConnectionError,
            ConnectionError,
            requests.exceptions.Timeout,
        ) as error:
            if not USE_FILE_CACHE_FOR_WORKFLOWS_DEFINITIONS:
                raise error
            response = load_cached_workflow_response(
                workspace_id=workspace_id,
                workflow_id=workflow_id,
                api_key=api_key,
                workflow_version_id=workflow_version_id,
            )
            if response is None:
                raise error

    if "workflow" not in response or "config" not in response["workflow"]:
        raise MalformedWorkflowResponseError(
            "Could not find workflow specification in API response"
        )
    try:
        workflow_config = json.loads(response["workflow"]["config"])
        specification = workflow_config["specification"]
        if not isinstance(specification, dict):
            raise TypeError("Workflow specification must be a dictionary")
        specification["id"] = response["workflow"].get("id")
        if use_cache:
            _try_cache_workflow_specification_in_ephemeral_cache(
                api_key=api_key,
                workspace_id=workspace_id,
                workflow_id=workflow_id,
                workflow_version_id=workflow_version_id,
                specification=specification,
                ephemeral_cache=ephemeral_cache,
            )

        return specification
    except KeyError as error:
        raise MalformedWorkflowResponseError(
            "Workflow specification not found in Roboflow API response"
        ) from error
    except (ValueError, TypeError) as error:
        raise MalformedWorkflowResponseError(
            "Could not decode workflow specification in Roboflow API response"
        ) from error


def _try_retrieve_workflow_specification_from_ephemeral_cache(
    api_key: Optional[str],
    workspace_id: WorkspaceID,
    workflow_id: str,
    ephemeral_cache: BaseCache,
    workflow_version_id: Optional[str] = None,
) -> Optional[dict]:
    """Return a cached specification, or None when the cache is down or misses."""
    try:
        cached_entry = _retrieve_workflow_specification_from_ephemeral_cache(
            api_key=api_key,
            workspace_id=workspace_id,
            workflow_id=workflow_id,
            workflow_version_id=workflow_version_id,
            ephemeral_cache=ephemeral_cache,
        )
    except CacheUnavailableError as error:
        logger.warning(
            "Ephemeral workflow specification cache unavailable, fetching from Roboflow API: %s",
            error,
        )
        return None

    return cached_entry


def _try_cache_workflow_specification_in_ephemeral_cache(
    api_key: Optional[str],
    workspace_id: WorkspaceID,
    workflow_id: str,
    specification: dict,
    ephemeral_cache: BaseCache,
    workflow_version_id: Optional[str] = None,
) -> None:
    """Best-effort write of a specification to ephemeral cache."""
    try:
        _cache_workflow_specification_in_ephemeral_cache(
            api_key=api_key,
            workspace_id=workspace_id,
            workflow_id=workflow_id,
            workflow_version_id=workflow_version_id,
            specification=specification,
            ephemeral_cache=ephemeral_cache,
        )
    except CacheUnavailableError as error:
        logger.warning(
            "Failed to cache workflow specification in ephemeral cache: %s",
            error,
        )


def _retrieve_workflow_specification_from_ephemeral_cache(
    api_key: Optional[str],
    workspace_id: WorkspaceID,
    workflow_id: str,
    ephemeral_cache: BaseCache,
    workflow_version_id: Optional[str] = None,
) -> Optional[dict]:
    cache_key = _prepare_workflow_response_cache_key(
        api_key=api_key,
        workspace_id=workspace_id,
        workflow_id=workflow_id,
        workflow_version_id=workflow_version_id,
    )
    try:
        cached_entry = ephemeral_cache.get(key=cache_key)
    except _EPHEMERAL_CACHE_UNAVAILABLE_EXCEPTIONS as error:
        _raise_cache_unavailable_error(operation="read", error=error)

    return cached_entry


def _cache_workflow_specification_in_ephemeral_cache(
    api_key: Optional[str],
    workspace_id: WorkspaceID,
    workflow_id: str,
    specification: dict,
    ephemeral_cache: BaseCache,
    workflow_version_id: Optional[str] = None,
) -> None:
    cache_key = _prepare_workflow_response_cache_key(
        api_key=api_key,
        workspace_id=workspace_id,
        workflow_id=workflow_id,
        workflow_version_id=workflow_version_id,
    )
    try:
        ephemeral_cache.set(
            key=cache_key,
            value=specification,
            expire=WORKFLOWS_DEFINITION_CACHE_EXPIRY,
        )
    except _EPHEMERAL_CACHE_UNAVAILABLE_EXCEPTIONS as error:
        _raise_cache_unavailable_error(operation="write", error=error)


def _raise_cache_unavailable_error(operation: str, error: Exception) -> None:
    if operation == "read":
        message = "Could not read workflow specification from ephemeral cache"
    else:
        message = "Could not write workflow specification to ephemeral cache"

    raise CacheUnavailableError(message) from error


def _prepare_workflow_response_cache_key(
    api_key: Optional[str],
    workspace_id: WorkspaceID,
    workflow_id: str,
    workflow_version_id: Optional[str] = None,
) -> str:
    cache_identity = json.dumps(
        [
            "inference-workflow-definition-v2",
            workspace_id,
            workflow_id,
            workflow_version_id or None,
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    cache_fingerprint = _workflow_cache_identity_fingerprint(cache_identity)
    if SINGLE_TENANT_WORKFLOW_CACHE:
        return f"workflow_definition:v2:{cache_fingerprint}"
    tenant_fingerprint = _workflow_cache_tenant_fingerprint(
        workspace_id=workspace_id,
        api_key=api_key,
    )
    return f"workflow_definition:v2:{cache_fingerprint}:{tenant_fingerprint}"


@wrap_roboflow_api_errors()
def get_from_url(
    url: str,
    json_response: bool = True,
) -> Union[Response, dict]:
    return _get_from_url(
        url=url,
        json_response=json_response,
    )


@backoff.on_exception(
    backoff.constant,
    exception=RetryRequestError,
    max_tries=TRANSIENT_ROBOFLOW_API_ERRORS_RETRIES,
    interval=TRANSIENT_ROBOFLOW_API_ERRORS_RETRY_INTERVAL,
)
def _get_from_url(
    url: str,
    json_response: bool = True,
    headers: Optional[dict] = None,
) -> Union[Response, dict]:
    if OFFLINE_MODE:
        raise ConnectionError("OFFLINE_MODE is enabled - cannot make API requests.")
    full_url = wrap_url(url)
    try:
        response = requests.get(
            full_url,
            headers=build_roboflow_api_headers(explicit_headers=headers),
            timeout=ROBOFLOW_API_REQUEST_TIMEOUT,
            verify=ROBOFLOW_API_VERIFY_SSL,
        )

    except (ConnectionError, Timeout, requests.exceptions.ConnectionError) as error:
        if RETRY_CONNECTION_ERRORS_TO_ROBOFLOW_API:
            raise RetryRequestError(
                message="Connectivity error", inner_error=error
            ) from error
        raise error
    try:
        api_key_safe_raise_for_status(response=response)
    except Exception as error:
        if response.status_code in TRANSIENT_ROBOFLOW_API_ERRORS:
            raise RetryRequestError(message=str(error), inner_error=error) from error
        raise error

    if MD5_VERIFICATION_ENABLED:
        x_goog_hash = response.headers.get("x-goog-hash")
        if x_goog_hash is None:
            logger.warning(
                f"MD5 verification enabled but response missing x-goog-hash header. "
                f"Request url: {_url_for_safe_logging(full_url)}"
            )
        else:
            md5_part = None
            for part in x_goog_hash.split(","):
                if part.strip().startswith("md5="):
                    md5_part = part.strip()[4:]
                    break
            if md5_part is not None:
                try:
                    md5_from_header = base64.b64decode(md5_part)
                except binascii.Error as decode_error:
                    raise RoboflowAPIUnsuccessfulRequestError(
                        "Invalid MD5 value in x-goog-hash header: not valid base64"
                    ) from decode_error
                if md5_from_header != hashlib.md5(response.content).digest():
                    raise RoboflowAPIUnsuccessfulRequestError(
                        "MD5 hash does not match MD5 received from x-goog-hash header"
                    )
            else:
                logger.warning(
                    f"MD5 verification enabled but x-goog-hash header has no md5= part. "
                    f"Request url: {_url_for_safe_logging(full_url)}"
                )

    if json_response:
        return response.json()
    return response


def _test_range_request(url: str, timeout: int = 10) -> bool:
    """Test if server actually honors range requests by making a real range GET request.

    Note: We can't rely on Accept-Ranges header alone because some servers/CDNs
    advertise range support but return 200 (full file) instead of 206 (partial).
    """
    if OFFLINE_MODE:
        return False
    try:
        headers = {"Range": "bytes=0-0"}
        response = requests.get(
            wrap_url(url), headers=headers, stream=True, timeout=timeout
        )
        response.close()
        if response.status_code == 206:
            return True

        return False
    except Exception as e:
        logger.warning(
            f"Failed to test range request support: {e}. Falling back to single-threaded download."
        )
        return False


def stream_url_to_cache(
    url: str,
    filename: str,
    model_id: str,
) -> None:
    if OFFLINE_MODE:
        raise RoboflowAPIConnectionError(
            "Cannot download model artifacts - OFFLINE_MODE is enabled."
        )
    from inference_models.utils.download import download_files_to_directory

    with start_span(
        "roboflow_api.download_artifact",
        {"http.url": _url_for_safe_logging(url), "model.id": model_id},
    ):
        initialise_cache(model_id=model_id)
        cache_dir = get_cache_dir(model_id=model_id)
        md5_hash = None

        max_threads = 8 if _test_range_request(url) else 1

        try:
            download_files_to_directory(
                target_dir=cache_dir,
                files_specs=[(filename, wrap_url(url), md5_hash)],
                verbose=True,
                download_files_without_hash=True,
                verify_hash_while_download=False,
                max_threads_per_download=max_threads,
            )
        except Exception as e:
            raise RoboflowAPIUnsuccessfulRequestError(
                f"Failed to download {filename}: {str(e)}"
            ) from e


def _url_for_safe_logging(url: str) -> str:
    """Return a URL safe to log by stripping the query string (and params/fragment).

    Expects the full URL as used for the request (e.g. already wrapped).
    Use this when logging request URLs so that sensitive query parameters
    (e.g. api_key, tokens) are never written to logs.
    """
    parsed = urllib.parse.urlparse(url)
    return urllib.parse.urlunparse(
        (parsed.scheme, parsed.netloc, parsed.path, "", "", "")
    )


def _add_params_to_url(url: str, params: List[Tuple[str, str]]) -> str:
    if len(params) == 0:
        return url
    params_chunks = [
        f"{name}={urllib.parse.quote_plus(value)}" for name, value in params
    ]
    parameters_string = "&".join(params_chunks)
    return f"{url}?{parameters_string}"


def _api_base_url_for_endpoint(endpoint: str) -> str:
    endpoint_path = endpoint.strip("/")
    if any(
        endpoint_path == prefix or endpoint_path.startswith(f"{prefix}/")
        for prefix in API_PROXY_ENDPOINT_PREFIXES
    ):
        return API_PROXY_BASE_URL
    return API_BASE_URL


@wrap_roboflow_api_errors()
def send_inference_results_to_model_monitoring(
    api_key: str,
    workspace_id: WorkspaceID,
    inference_data: dict,
):
    if OFFLINE_MODE:
        return
    api_url = wrap_url(
        _add_params_to_url(
            url=f"{API_BASE_URL}/{workspace_id}/inference-stats",
            params=[("api_key", api_key)],
        )
    )
    response = requests.post(
        url=api_url,
        json=inference_data,
        headers=build_roboflow_api_headers(),
        timeout=ROBOFLOW_API_REQUEST_TIMEOUT,
        verify=ROBOFLOW_API_VERIFY_SSL,
    )
    api_key_safe_raise_for_status(response=response)


def get_extra_weights_provider_headers(
    countinference: Optional[bool] = None,
    service_secret: Optional[str] = None,
) -> Optional[Dict[str, str]]:
    headers = {}
    if GCP_SERVERLESS:
        headers[ENFORCE_INTERNAL_ARTIFACTS_URLS_HEADER] = "true"
    if ENFORCE_CREDITS_VERIFICATION:
        skip = (
            countinference is False
            and service_secret is not None
            and service_secret == ROBOFLOW_SERVICE_SECRET
        )
        if not skip:
            headers[ENFORCE_CREDITS_VERIFICATION_HEADER] = "true"
    if ROBOFLOW_INTERNAL_SERVICE_SECRET:
        headers["X-Roboflow-Internal-Service-Secret"] = ROBOFLOW_INTERNAL_SERVICE_SECRET
    return build_roboflow_api_headers(explicit_headers=headers)


def build_roboflow_api_headers(
    explicit_headers: Optional[Dict[str, Union[str, List[str]]]] = None,
) -> Dict[str, Union[str, List[str]]]:
    if explicit_headers is None:
        explicit_headers = {}
    explicit_headers = {
        **explicit_headers,
        ROBOFLOW_INFERENCE_VERSION_HEADER: __version__,
        ALLOW_CHUNKED_RESPONSE_HEADER: "true",
    }
    if not ROBOFLOW_API_EXTRA_HEADERS:
        return explicit_headers
    try:
        extra_headers: dict = json.loads(ROBOFLOW_API_EXTRA_HEADERS)
        if explicit_headers:
            extra_headers.update(explicit_headers)
        return extra_headers
    except ValueError:
        logger.warning("Could not decode ROBOFLOW_API_EXTRA_HEADERS")
        return explicit_headers


def post_to_roboflow_api(
    endpoint: str,
    api_key: Optional[str],
    payload: Optional[dict] = None,
    params: Optional[List[Tuple[str, str]]] = None,
    http_errors_handlers: Optional[
        Dict[int, Callable[[Union[requests.exceptions.HTTPError]], None]]
    ] = None,
) -> dict:
    """Generic function to make a POST request to the Roboflow API.

    Args:
        endpoint: API endpoint path
        api_key: Roboflow API key
        payload: JSON payload
        params: Additional URL parameters
        http_errors_handlers: Optional custom HTTP error handlers by status code
    """

    if OFFLINE_MODE:
        raise RoboflowAPIConnectionError(
            "Cannot make API requests - OFFLINE_MODE is enabled."
        )

    @wrap_roboflow_api_errors(http_errors_handlers=http_errors_handlers)
    def _make_request():
        url_params = []
        if api_key and api_key != LOCAL_API_KEY:
            url_params.append(("api_key", api_key))
        if params:
            url_params.extend(params)

        api_base_url = _api_base_url_for_endpoint(endpoint=endpoint).rstrip("/")
        full_url = _add_params_to_url(
            url=f"{api_base_url}/{endpoint.strip('/')}", params=url_params
        )
        wrapped_url = wrap_url(full_url)

        headers = build_roboflow_api_headers()

        response = requests.post(
            url=wrapped_url,
            json=payload,
            headers=headers,
            timeout=ROBOFLOW_API_REQUEST_TIMEOUT,
            verify=ROBOFLOW_API_VERIFY_SSL,
        )
        api_key_safe_raise_for_status(response=response)
        return response.json()

    return _make_request()

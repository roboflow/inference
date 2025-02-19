import hashlib
import json
import os
import re
import urllib.parse
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import backoff
import requests
from requests import Response, Timeout
from requests_toolbelt import MultipartEncoder

from inference.core import logger
from inference.core.cache import cache
from inference.core.cache.base import BaseCache
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
    MODEL_CACHE_DIR,
    RETRY_CONNECTION_ERRORS_TO_ROBOFLOW_API,
    ROBOFLOW_API_EXTRA_HEADERS,
    ROBOFLOW_API_REQUEST_TIMEOUT,
    TRANSIENT_ROBOFLOW_API_ERRORS,
    TRANSIENT_ROBOFLOW_API_ERRORS_RETRIES,
    TRANSIENT_ROBOFLOW_API_ERRORS_RETRY_INTERVAL,
    USE_FILE_CACHE_FOR_WORKFLOWS_DEFINITIONS,
    WORKFLOWS_DEFINITION_CACHE_EXPIRY,
)
from inference.core.exceptions import (
    MalformedRoboflowAPIResponseError,
    MalformedWorkflowResponseError,
    MissingDefaultModelError,
    RetryRequestError,
    RoboflowAPIConnectionError,
    RoboflowAPIIAlreadyAnnotatedError,
    RoboflowAPIIAnnotationRejectionError,
    RoboflowAPIImageUploadRejectionError,
    RoboflowAPINotAuthorizedError,
    RoboflowAPINotNotFoundError,
    RoboflowAPITimeoutError,
    RoboflowAPIUnsuccessfulRequestError,
    WorkspaceLoadError,
)
from inference.core.utils.file_system import sanitize_path_segment
from inference.core.utils.requests import api_key_safe_raise_for_status
from inference.core.utils.url_utils import wrap_url

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
    404: lambda e: raise_from_lambda(
        e, RoboflowAPINotNotFoundError, NOT_FOUND_ERROR_MESSAGE
    ),
}


def wrap_roboflow_api_errors(
    http_errors_handlers: Optional[
        Dict[int, Callable[[Union[requests.exceptions.HTTPError]], None]]
    ] = None,
) -> callable:
    def decorator(function: callable) -> callable:
        def wrapper(*args, **kwargs) -> Any:
            try:
                try:
                    return function(*args, **kwargs)
                except RetryRequestError as error:
                    raise error.inner_error
            except Timeout as error:
                raise RoboflowAPITimeoutError(
                    "Timeout when attempting to connect to Roboflow API."
                ) from error
            except (requests.exceptions.ConnectionError, ConnectionError) as error:
                raise RoboflowAPIConnectionError(
                    "Could not connect to Roboflow API."
                ) from error
            except requests.exceptions.HTTPError as error:
                user_handler_override = (
                    http_errors_handlers if http_errors_handlers is not None else {}
                )
                status_code = error.response.status_code
                default_handler = DEFAULT_ERROR_HANDLERS.get(status_code)
                error_handler = user_handler_override.get(status_code, default_handler)
                if error_handler is not None:
                    error_handler(error)
                raise RoboflowAPIUnsuccessfulRequestError(
                    f"Unsuccessful request to Roboflow API with response code: {status_code}"
                ) from error
            except requests.exceptions.InvalidJSONError as error:
                raise MalformedRoboflowAPIResponseError(
                    "Could not decode JSON response from Roboflow API."
                ) from error

        return wrapper

    return decorator


@wrap_roboflow_api_errors()
def get_roboflow_workspace(api_key: str) -> WorkspaceID:
    api_url = _add_params_to_url(
        url=f"{API_BASE_URL}/",
        params=[("api_key", api_key), ("nocache", "true")],
    )
    api_key_info = _get_from_url(url=api_url)
    workspace_id = api_key_info.get("workspace")
    if workspace_id is None:
        raise WorkspaceLoadError(f"Empty workspace encountered, check your API key.")
    return workspace_id


@wrap_roboflow_api_errors()
def add_custom_metadata(
    api_key: str,
    workspace_id: WorkspaceID,
    inference_ids: List[str],
    field_name: str,
    field_value: str,
):
    api_url = _add_params_to_url(
        url=f"{API_BASE_URL}/{workspace_id}/inference-stats/metadata",
        params=[("api_key", api_key), ("nocache", "true")],
    )
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
) -> dict:
    api_data_cache_key = f"roboflow_api_data:{endpoint_type.value}:{model_id}"
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
        if api_key is not None:
            params.append(("api_key", api_key))
        api_url = _add_params_to_url(
            url=f"{API_BASE_URL}/{endpoint_type.value}/{model_id}",
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
) -> dict:
    api_data_cache_key = f"{cache_prefix}:{model_id}"
    api_data = cache.get(api_data_cache_key)
    if api_data is not None:
        logger.debug(f"Loaded model data from cache with key: {api_data_cache_key}.")
        return api_data
    else:
        params = [
            ("model", model_id),
        ]
        if api_key is not None:
            params.append(("api_key", api_key))
        api_url = _add_params_to_url(
            url=f"{API_BASE_URL}/getWeights",
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
def get_roboflow_base_lora(
    api_key: str, repo: str, revision: str, device_id: str
) -> dict:
    full_path = os.path.join(repo, revision)
    api_data_cache_key = f"roboflow_api_data:lora-bases:{full_path}"
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
        if api_key is not None:
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
) -> dict:
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
    m = MultipartEncoder(
        fields={
            "name": f"{local_image_id}.jpg",
            "file": ("imageToUpload", image_bytes, "image/jpeg"),
        }
    )
    headers = build_roboflow_api_headers(
        explicit_headers={"Content-Type": m.content_type},
    )
    response = requests.post(
        url=wrapped_url,
        data=m,
        headers=headers,
        timeout=ROBOFLOW_API_REQUEST_TIMEOUT,
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
    )
    api_key_safe_raise_for_status(response=response)
    parsed_response = response.json()
    if "error" in parsed_response or not parsed_response.get("success"):
        raise RoboflowAPIIAnnotationRejectionError(
            f"Failed to save annotation for {roboflow_image_id}. API response: {parsed_response}"
        )
    return parsed_response


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


def get_workflow_cache_file(
    workspace_id: WorkspaceID, workflow_id: str, api_key: Optional[str]
) -> str:
    sanitized_workspace_id = sanitize_path_segment(workspace_id)
    sanitized_workflow_id = sanitize_path_segment(workflow_id)
    api_key_hash = (
        hashlib.md5(api_key.encode("utf-8")).hexdigest()
        if api_key is not None
        else "None"
    )
    prefix = os.path.abspath(os.path.join(MODEL_CACHE_DIR, "workflow"))
    result = os.path.abspath(
        os.path.join(
            prefix,
            sanitized_workspace_id,
            f"{sanitized_workflow_id}_{api_key_hash}.json",
        )
    )
    if not result.startswith(prefix):
        raise ValueError(
            "Detected attempt to save workflow definition in insecure location"
        )
    return result


def cache_workflow_response(
    workspace_id: WorkspaceID, workflow_id: str, api_key: Optional[str], response: dict
):
    workflow_cache_file = get_workflow_cache_file(
        workspace_id=workspace_id,
        workflow_id=workflow_id,
        api_key=api_key,
    )
    workflow_cache_dir = os.path.dirname(workflow_cache_file)
    if not os.path.exists(workflow_cache_dir):
        os.makedirs(workflow_cache_dir, exist_ok=True)
    with open(workflow_cache_file, "w") as f:
        json.dump(response, f)


def delete_cached_workflow_response_if_exists(
    workspace_id: WorkspaceID,
    workflow_id: str,
    api_key: Optional[str],
) -> None:
    workflow_cache_file = get_workflow_cache_file(
        workspace_id=workspace_id,
        workflow_id=workflow_id,
        api_key=api_key,
    )
    if os.path.exists(workflow_cache_file):
        os.remove(workflow_cache_file)


def load_cached_workflow_response(
    workspace_id: WorkspaceID,
    workflow_id: str,
    api_key: Optional[str],
) -> Optional[dict]:
    workflow_cache_file = get_workflow_cache_file(
        workspace_id=workspace_id,
        workflow_id=workflow_id,
        api_key=api_key,
    )
    if not os.path.exists(workflow_cache_file):
        return None
    try:
        with open(workflow_cache_file, "r") as f:
            return json.load(f)
    except:
        delete_cached_workflow_response_if_exists(
            workspace_id=workspace_id,
            workflow_id=workflow_id,
            api_key=api_key,
        )


@wrap_roboflow_api_errors()
def get_workflow_specification(
    api_key: Optional[str],
    workspace_id: WorkspaceID,
    workflow_id: str,
    use_cache: bool = True,
    ephemeral_cache: Optional[BaseCache] = None,
) -> dict:
    ephemeral_cache = ephemeral_cache or cache
    if use_cache:
        cached_entry = _retrieve_workflow_specification_from_ephemeral_cache(
            api_key=api_key,
            workspace_id=workspace_id,
            workflow_id=workflow_id,
            ephemeral_cache=ephemeral_cache,
        )
        if cached_entry:
            return cached_entry

    if workspace_id == "local":
        if not re.match(r"^[\w\-]+$", workflow_id):
            raise ValueError("Invalid workflow id")

        local_file_path = (
            Path(MODEL_CACHE_DIR) / "workflow" / "local" / f"{workflow_id}.json"
        )
        if not local_file_path.exists():
            raise FileNotFoundError(f"Local workflow file not found: {local_file_path}")

        with local_file_path.open("r", encoding="utf-8") as f:
            local_config = json.load(f)

        # Mimic the same shape as the cloud response:
        response = {"workflow": local_config}
    else:
        params = []
        if api_key is not None:
            params.append(("api_key", api_key))
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
                )
        except (requests.exceptions.ConnectionError, ConnectionError) as error:
            if not USE_FILE_CACHE_FOR_WORKFLOWS_DEFINITIONS:
                raise error
            response = load_cached_workflow_response(
                workspace_id=workspace_id,
                workflow_id=workflow_id,
                api_key=api_key,
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
        if isinstance(specification, dict):
            specification["id"] = response["workflow"].get("id")
        if use_cache:
            _cache_workflow_specification_in_ephemeral_cache(
                api_key=api_key,
                workspace_id=workspace_id,
                workflow_id=workflow_id,
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


def _retrieve_workflow_specification_from_ephemeral_cache(
    api_key: Optional[str],
    workspace_id: WorkspaceID,
    workflow_id: str,
    ephemeral_cache: BaseCache,
) -> Optional[dict]:
    cache_key = _prepare_workflow_response_cache_key(
        api_key=api_key,
        workspace_id=workspace_id,
        workflow_id=workflow_id,
    )
    return ephemeral_cache.get(key=cache_key)


def _cache_workflow_specification_in_ephemeral_cache(
    api_key: Optional[str],
    workspace_id: WorkspaceID,
    workflow_id: str,
    specification: dict,
    ephemeral_cache: BaseCache,
) -> None:
    cache_key = _prepare_workflow_response_cache_key(
        api_key=api_key,
        workspace_id=workspace_id,
        workflow_id=workflow_id,
    )
    ephemeral_cache.set(
        key=cache_key,
        value=specification,
        expire=WORKFLOWS_DEFINITION_CACHE_EXPIRY,
    )


def _prepare_workflow_response_cache_key(
    api_key: Optional[str],
    workspace_id: WorkspaceID,
    workflow_id: str,
) -> str:
    api_key_hash = (
        hashlib.md5(api_key.encode("utf-8")).hexdigest()
        if api_key is not None
        else "None"
    )
    return f"workflow_definition:{workspace_id}:{workflow_id}:{api_key_hash}"


@wrap_roboflow_api_errors()
def get_from_url(
    url: str,
    json_response: bool = True,
) -> Union[Response, dict]:
    return _get_from_url(url=url, json_response=json_response)


@backoff.on_exception(
    backoff.constant,
    exception=RetryRequestError,
    max_tries=TRANSIENT_ROBOFLOW_API_ERRORS_RETRIES,
    interval=TRANSIENT_ROBOFLOW_API_ERRORS_RETRY_INTERVAL,
)
def _get_from_url(url: str, json_response: bool = True) -> Union[Response, dict]:
    try:
        response = requests.get(
            wrap_url(url),
            headers=build_roboflow_api_headers(),
            timeout=ROBOFLOW_API_REQUEST_TIMEOUT,
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
    if json_response:
        return response.json()
    return response


def _add_params_to_url(url: str, params: List[Tuple[str, str]]) -> str:
    if len(params) == 0:
        return url
    params_chunks = [
        f"{name}={urllib.parse.quote_plus(value)}" for name, value in params
    ]
    parameters_string = "&".join(params_chunks)
    return f"{url}?{parameters_string}"


@wrap_roboflow_api_errors()
def send_inference_results_to_model_monitoring(
    api_key: str,
    workspace_id: WorkspaceID,
    inference_data: dict,
):
    api_url = _add_params_to_url(
        url=f"{API_BASE_URL}/{workspace_id}/inference-stats",
        params=[("api_key", api_key)],
    )
    response = requests.post(
        url=api_url,
        json=inference_data,
        headers=build_roboflow_api_headers(),
        timeout=ROBOFLOW_API_REQUEST_TIMEOUT,
    )
    api_key_safe_raise_for_status(response=response)


def build_roboflow_api_headers(
    explicit_headers: Optional[Dict[str, Union[str, List[str]]]] = None,
) -> Optional[Dict[str, Union[List[str]]]]:
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

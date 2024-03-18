import json
import urllib.parse
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import requests
from requests import Response
from requests_toolbelt import MultipartEncoder

from inference.core import logger
from inference.core.cache import cache
from inference.core.entities.types import (
    DatasetID,
    ModelType,
    TaskType,
    VersionID,
    WorkspaceID,
)
from inference.core.env import API_BASE_URL
from inference.core.exceptions import (
    MalformedRoboflowAPIResponseError,
    MalformedWorkflowResponseError,
    MissingDefaultModelError,
    RoboflowAPIConnectionError,
    RoboflowAPIIAlreadyAnnotatedError,
    RoboflowAPIIAnnotationRejectionError,
    RoboflowAPIImageUploadRejectionError,
    RoboflowAPINotAuthorizedError,
    RoboflowAPINotNotFoundError,
    RoboflowAPIUnsuccessfulRequestError,
    WorkspaceLoadError,
)
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
                return function(*args, **kwargs)
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
    inference_id=None,
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
    response = requests.post(
        url=wrapped_url,
        data=m,
        headers={"Content-Type": m.content_type},
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
    response = requests.post(
        wrapped_url,
        data=annotation_content,
        headers={"Content-Type": "text/plain"},
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


@wrap_roboflow_api_errors()
def get_workflow_specification(
    api_key: str,
    workspace_id: WorkspaceID,
    workflow_name: str,
) -> dict:
    api_url = _add_params_to_url(
        url=f"{API_BASE_URL}/{workspace_id}/workflows/{workflow_name}",
        params=[("api_key", api_key)],
    )
    response = _get_from_url(url=api_url)
    if "workflow" not in response or "config" not in response["workflow"]:
        raise MalformedWorkflowResponseError(
            f"Could not found workflow specification in API response"
        )
    try:
        return json.loads(response["workflow"]["config"])
    except (ValueError, TypeError) as error:
        raise MalformedWorkflowResponseError(
            "Could not decode workflow specification in Roboflow API response"
        ) from error


@wrap_roboflow_api_errors()
def get_from_url(
    url: str,
    json_response: bool = True,
) -> Union[Response, dict]:
    return _get_from_url(url=url, json_response=json_response)


def _get_from_url(url: str, json_response: bool = True) -> Union[Response, dict]:
    response = requests.get(wrap_url(url))
    api_key_safe_raise_for_status(response=response)
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

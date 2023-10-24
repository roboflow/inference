from enum import Enum
from typing import Any, Callable, Dict, Optional, Type, Union, List

import numpy as np
import requests
from requests import Response
from requests_toolbelt import MultipartEncoder

from inference.core import logger
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
    MissingDefaultModelError,
    RoboflowAPIConnectionError,
    RoboflowAPINotAuthorizedError,
    RoboflowAPINotNotFoundError,
    RoboflowAPIUnsuccessfulRequestError,
    WorkspaceLoadError,
)
from inference.core.utils.url_utils import wrap_url

MODEL_TYPE_DEFAULTS = {
    "object-detection": "yolov5v2s",
    "instance-segmentation": "yolact",
    "classification": "vit",
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
        "Unauthorized access to roboflow API - check API key.",
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
                logger.error(f"Could not connect to Roboflow API. Error: {error}")
                raise RoboflowAPIConnectionError(
                    "Could not connect to Roboflow API."
                ) from error
            except requests.exceptions.HTTPError as error:
                logger.error(
                    f"HTTP error encountered while requesting Roboflow API response: {error}"
                )
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
                logger.error(
                    f"Could not decode JSON response from Roboflow API. Error: {error}."
                )
                raise MalformedRoboflowAPIResponseError(
                    "Could not decode JSON response from Roboflow API."
                ) from error

        return wrapper

    return decorator


@wrap_roboflow_api_errors()
def get_roboflow_workspace(api_key: str) -> WorkspaceID:
    api_url = wrap_url("/".join([API_BASE_URL, f"?api_key={api_key}"]))
    api_key_info = requests.get(api_url)
    api_key_info.raise_for_status()
    workspace_id = api_key_info.json().get("workspace")
    if workspace_id is None:
        raise WorkspaceLoadError(f"Empty workspace encountered, check your API key.")
    return workspace_id


@wrap_roboflow_api_errors()
def get_roboflow_dataset_type(
    api_key: str, workspace_id: WorkspaceID, dataset_id: DatasetID
) -> TaskType:
    api_url = wrap_url(
        "/".join(
            [API_BASE_URL, workspace_id, dataset_id, f"?api_key={api_key}&nocache=true"]
        )
    )
    dataset_info = requests.get(api_url)
    dataset_info.raise_for_status()
    project_task_type = dataset_info.json().get("project", {})
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
    api_url = wrap_url(
        "/".join(
            [
                API_BASE_URL,
                workspace_id,
                dataset_id,
                version_id,
                f"?api_key={api_key}&nocache=true",
            ]
        )
    )
    version_info = requests.get(api_url)
    version_info.raise_for_status()
    model_type = version_info.json()["version"]
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
    api_url = wrap_url(
        f"{API_BASE_URL}/{endpoint_type.value}/{model_id}?api_key={api_key}&device={device_id}&nocache=true&dynamic=true"
    )
    model_data = requests.get(api_url)
    model_data.raise_for_status()
    return model_data.json()


@wrap_roboflow_api_errors()
def get_active_learning_configuration(
    api_key: str,
    workspace_id: WorkspaceID,
    dataset_id: DatasetID,
) -> dict:
    url = f"{API_BASE_URL}/dataset/{workspace_id}/{dataset_id}?api_key={api_key}"
    # return _get_from_roboflow_api(url=url)
    return {
        "enabled": True,
        "sampling_strategies": [
            {
                "type": "random_sampling",
                "name": "default_strategy",
                "persist_predictions": True,
                "traffic_percentage": 0.1,   # float 0-1
                "dataset_splits": {   # how much of sampled traffic should go to which split. Must sum to one.
                    "train": 0.8,
                    "val": 0.1,
                    "test": 0.1,
                },
                "max_image_size": (1200, 1200),  # (h, w)
                "jpeg_compression_level": 75  # int 0-100
            }
        ],
        "batching_strategies": {
            "creation_strategy": "time_based",  # "time_based" | "num_images_based"
            "creation_interval": "daily",  # "never" | "daily" | "weekly" | "monthly" | None
            "max_batch_images": None,  # Optional[int]
        }
    }


@wrap_roboflow_api_errors()
def register_image(
    api_key: str,
    dataset_id: DatasetID,
    local_image_id: str,
    image_bytes: bytes,
    split: str,
    batch_name: str,
    tags: Optional[List[str]] = None,
) -> str:
    url = f"{API_BASE_URL}/dataset/{dataset_id}/upload"
    params = {
        "api_key": api_key,
        "batch": batch_name,
    }
    if tags is not None:
        params["tag"] = tags
    m = MultipartEncoder(
        fields={
            "name": f"{local_image_id}.jpg",
            "split": split,
            "file": ("imageToUpload", image_bytes, "image/jpeg"),
        }
    )
    response = requests.post(
        url,
        data=m, headers={"Content-Type": m.content_type},
        params=params
    )
    response.raise_for_status()
    return response.json()


@wrap_roboflow_api_errors()
def get_from_roboflow_api(
    url: str, json_response: bool = True
) -> Union[Response, dict]:
    return _get_from_roboflow_api(url=url, json_response=json_response)


def _get_from_roboflow_api(url: str, json_response: bool = True) -> Union[Response, dict]:
    response = requests.get(wrap_url(url))
    response.raise_for_status()
    if json_response:
        return response.json()
    return response

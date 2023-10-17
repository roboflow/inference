import json
import os
from typing import Any, Callable, Optional, Tuple, Type, Union

import requests

from inference.core.entities.types import (
    DatasetID,
    ModelType,
    TaskType,
    VersionID,
    WorkspaceID,
)
from inference.core.env import API_BASE_URL, MODEL_CACHE_DIR
from inference.core.exceptions import (
    DatasetLoadError,
    InvalidModelIDError,
    MalformedRoboflowAPIResponseError,
    MissingDefaultModelError,
    ModelNotRecognisedError,
    WorkspaceLoadError,
)
from inference.core.logger import logger
from inference.core.models.base import Model
from inference.core.registries.base import ModelRegistry
from inference.core.utils.file_system import dump_json, read_json
from inference.core.utils.url_utils import wrap_url

MODEL_TYPE_DEFAULTS = {
    "object-detection": "yolov5v2s",
    "instance-segmentation": "yolact",
    "classification": "vit",
}

GENERIC_MODELS = {
    "clip": ("embed", "clip"),
    "sam": ("embed", "sam"),
    "gaze": ("gaze", "l2cs"),
}
PROJECT_TASK_TYPE_KEY = "project_task_type"
MODEL_TYPE_KEY = "model_type"


class RoboflowModelRegistry(ModelRegistry):
    """A Roboflow-specific model registry which gets the model type using the model id,
    then returns a model class based on the model type.
    """

    def get_model(self, model_id: str, api_key: str) -> Model:
        """Returns the model class based on the given model id and API key.

        Args:
            model_id (str): The ID of the model to be retrieved.
            api_key (str): The API key used to authenticate.

        Returns:
            Model: The model class corresponding to the given model ID and type.

        Raises:
            DatasetLoadError: If the model type is not supported or found.
        """
        model_type = get_model_type(model_id, api_key)
        if model_type not in self.registry_dict:
            raise ModelNotRecognisedError(f"Model type not supported: {model_type}")
        return self.registry_dict[model_type]


def get_model_type(model_id: str, api_key: str) -> Tuple[TaskType, ModelType]:
    """Retrieves the model type based on the given model ID and API key.

    Args:
        model_id (str): The ID of the model.
        api_key (str): The API key used to authenticate.

    Returns:
        tuple: The project task type and the model type.

    Raises:
        WorkspaceLoadError: If the workspace could not be loaded or if the API key is invalid.
        DatasetLoadError: If the dataset could not be loaded due to invalid ID, workspace ID or version ID.
        MissingDefaultModelError: If default model is not configured and API does not provide this info
        MalformedRoboflowAPIResponseError: Roboflow API responds in invalid format.
    """
    dataset_id, version_id = get_model_id_chunks(model_id=model_id)
    if dataset_id in GENERIC_MODELS:
        logger.info(f"Loading generic model: {dataset_id}.")
        return GENERIC_MODELS[dataset_id]
    cached_metadata = get_model_metadata_from_cache(
        dataset_id=dataset_id, version_id=version_id
    )
    if cached_metadata is not None:
        return cached_metadata[0], cached_metadata[1]
    workspace_id = get_roboflow_workspace(api_key=api_key)
    project_task_type = get_roboflow_dataset_type(
        api_key=api_key, workspace_id=workspace_id, dataset_id=dataset_id
    )
    model_type = get_roboflow_model_type(
        api_key=api_key,
        workspace_id=workspace_id,
        dataset_id=dataset_id,
        version_id=version_id,
        project_task_type=project_task_type,
    )
    save_model_metadata_in_cache(
        dataset_id=dataset_id,
        version_id=version_id,
        project_task_type=project_task_type,
        model_type=model_type,
    )
    return project_task_type, model_type


def get_model_id_chunks(model_id: str) -> Tuple[DatasetID, VersionID]:
    model_id_chunks = model_id.split("/")
    if len(model_id_chunks) != 2:
        raise InvalidModelIDError(f"Model ID: {model_id} is invalid.")
    return model_id_chunks[0], model_id_chunks[1]


def get_model_metadata_from_cache(
    dataset_id: str, version_id: str
) -> Optional[Tuple[TaskType, ModelType]]:
    model_type_cache_path = construct_model_type_cache_path(
        dataset_id=dataset_id, version_id=version_id
    )
    if not os.path.isfile(model_type_cache_path):
        return None
    try:
        model_metadata = read_json(path=model_type_cache_path)
        if model_metadata_content_is_invalid(content=model_metadata):
            return None
        return model_metadata[PROJECT_TASK_TYPE_KEY], model_metadata[MODEL_TYPE_KEY]
    except ValueError as e:
        logger.warning(
            f"Could not load model description from cache under path: {model_type_cache_path} - decoding issue: {e}."
        )
        return None


def model_metadata_content_is_invalid(content: Optional[Union[list, dict]]) -> bool:
    if content is None:
        logger.warning("Empty model metadata file encountered in cache.")
        return True
    if not issubclass(type(content), dict):
        logger.warning("Malformed file encountered in cache.")
        return True
    if PROJECT_TASK_TYPE_KEY not in content or MODEL_TYPE_KEY not in content:
        logger.warning(
            f"Could not find one of required keys {PROJECT_TASK_TYPE_KEY} or {MODEL_TYPE_KEY} in cache."
        )
        return True
    return False


def save_model_metadata_in_cache(
    dataset_id: DatasetID,
    version_id: VersionID,
    project_task_type: TaskType,
    model_type: ModelType,
) -> None:
    model_type_cache_path = construct_model_type_cache_path(
        dataset_id=dataset_id, version_id=version_id
    )
    metadata = {PROJECT_TASK_TYPE_KEY: project_task_type, MODEL_TYPE_KEY: model_type}
    dump_json(
        path=model_type_cache_path, content=metadata, allow_override=True, indent=4
    )


def construct_model_type_cache_path(dataset_id: str, version_id: str) -> str:
    cache_dir = os.path.join(MODEL_CACHE_DIR, dataset_id, version_id)
    return os.path.join(cache_dir, "model_type.json")


def wrap_roboflow_api_errors(
    on_connection_error: Callable[
        [Union[requests.exceptions.ConnectionError, ConnectionError]], None
    ],
    on_http_error: Callable[[Union[requests.exceptions.HTTPError]], None],
) -> callable:
    def decorator(function: callable) -> callable:
        def wrapper(*args, **kwargs) -> Any:
            try:
                return function(*args, **kwargs)
            except (requests.exceptions.ConnectionError, ConnectionError) as error:
                logger.error(f"Could not connect to Roboflow API. Error: {error}")
                on_connection_error(error)
            except requests.exceptions.HTTPError as error:
                logger.error(
                    f"HTTP error encountered while requesting Roboflow API response: {error}"
                )
                on_http_error(error)
            except requests.exceptions.InvalidJSONError as error:
                logger.error(
                    f"Could not decode JSON response from Roboflow API. Error: {error}."
                )
                raise MalformedRoboflowAPIResponseError(
                    "Could not decode JSON response from Roboflow API."
                ) from error

        return wrapper

    return decorator


def raise_from_lambda(
    inner_error: Exception, exception_type: Type[Exception], message: str
) -> None:
    raise exception_type(message) from inner_error


@wrap_roboflow_api_errors(
    on_connection_error=lambda e: raise_from_lambda(
        e, WorkspaceLoadError, "Could not connect to Roboflow API."
    ),
    on_http_error=lambda e: raise_from_lambda(
        e, WorkspaceLoadError, "Could not load workspace, check your API key"
    ),
)
def get_roboflow_workspace(api_key: str) -> WorkspaceID:
    api_url = wrap_url("/".join([API_BASE_URL, f"?api_key={api_key}"]))
    api_key_info = requests.get(api_url)
    api_key_info.raise_for_status()
    workspace_id = api_key_info.json().get("workspace")
    if workspace_id is None:
        raise WorkspaceLoadError(f"Empty workspace encountered, check your API key.")
    return workspace_id


@wrap_roboflow_api_errors(
    on_connection_error=lambda e: raise_from_lambda(
        e, DatasetLoadError, "Could not connect to Roboflow API."
    ),
    on_http_error=lambda e: raise_from_lambda(
        e,
        DatasetLoadError,
        "Could not load dataset info, check your API key and workspace.",
    ),
)
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
    on_connection_error=lambda e: raise_from_lambda(
        e, DatasetLoadError, "Could not connect to Roboflow API."
    ),
    on_http_error=lambda e: raise_from_lambda(
        e,
        DatasetLoadError,
        "Could not load version info, check your API key and workspace.",
    ),
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

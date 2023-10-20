import os
from typing import Optional, Tuple, Union

from inference.core.entities.types import DatasetID, ModelType, TaskType, VersionID
from inference.core.env import MODEL_CACHE_DIR
from inference.core.exceptions import InvalidModelIDError, ModelNotRecognisedError
from inference.core.logger import logger
from inference.core.models.base import Model
from inference.core.registries.base import ModelRegistry
from inference.core.roboflow_api import (
    MODEL_TYPE_KEY,
    PROJECT_TASK_TYPE_KEY,
    get_roboflow_dataset_type,
    get_roboflow_model_type,
    get_roboflow_workspace,
)
from inference.core.utils.file_system import dump_json, read_json

GENERIC_MODELS = {
    "clip": ("embed", "clip"),
    "sam": ("embed", "sam"),
    "gaze": ("gaze", "l2cs"),
    "doctr": ("ocr", "doctr"),
}


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
            ModelNotRecognisedError: If the model type is not supported or found.
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
        logger.debug(f"Loading generic model: {dataset_id}.")
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
        raise InvalidModelIDError(f"Model ID: `{model_id}` is invalid.")
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

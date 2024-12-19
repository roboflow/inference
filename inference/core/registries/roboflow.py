import os
from typing import Optional, Tuple, Union

from inference.core.cache import cache
from inference.core.devices.utils import GLOBAL_DEVICE_ID
from inference.core.entities.types import ModelType, TaskType
from inference.core.env import LAMBDA, MODEL_CACHE_DIR
from inference.core.exceptions import (
    MissingApiKeyError,
    ModelArtefactError,
    ModelNotRecognisedError,
)
from inference.core.logger import logger
from inference.core.models.base import Model
from inference.core.registries.base import ModelRegistry
from inference.core.roboflow_api import (
    MODEL_TYPE_DEFAULTS,
    MODEL_TYPE_KEY,
    PROJECT_TASK_TYPE_KEY,
    ModelEndpointType,
    get_roboflow_dataset_type,
    get_roboflow_model_data,
    get_roboflow_workspace,
)
from inference.core.utils.file_system import dump_json, read_json
from inference.core.utils.roboflow import get_model_id_chunks
from inference.models.aliases import resolve_roboflow_model_alias

GENERIC_MODELS = {
    "clip": ("embed", "clip"),
    "sam": ("embed", "sam"),
    "sam2": ("embed", "sam2"),
    "gaze": ("gaze", "l2cs"),
    "doctr": ("ocr", "doctr"),
    "trocr": ("ocr", "trocr"),
    "grounding_dino": ("object-detection", "grounding-dino"),
    "cogvlm": ("llm", "cogvlm"),
    "paligemma": ("llm", "paligemma"),
    "yolo_world": ("object-detection", "yolo-world"),
    "owlv2": ("object-detection", "owlv2"),
}

STUB_VERSION_ID = "0"
CACHE_METADATA_LOCK_TIMEOUT = 1.0


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
        logger.debug(f"Model type: {model_type}")
        if model_type not in self.registry_dict:
            raise ModelNotRecognisedError(f"Model type not supported: {model_type}")
        return self.registry_dict[model_type]


def get_model_type(
    model_id: str,
    api_key: Optional[str] = None,
) -> Tuple[TaskType, ModelType]:
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
    model_id = resolve_roboflow_model_alias(model_id=model_id)
    dataset_id, version_id = get_model_id_chunks(model_id=model_id)
    lock_key, cache_path = determine_cache_paths(dataset_or_model_id=dataset_id, version_id=version_id)
    
    if dataset_id in GENERIC_MODELS:
        logger.debug(f"Loading generic model: {dataset_id}.")
        return GENERIC_MODELS[dataset_id]
    
    cached_metadata = get_model_metadata_from_cache(
        cache_path=cache_path, lock_key=lock_key
    )
    if cached_metadata is not None:
        return cached_metadata[0], cached_metadata[1]
   
    
    # THis path will never be executed for a model ID
    if version_id == STUB_VERSION_ID:
        if api_key is None:
            raise MissingApiKeyError(
                "Stub model version provided but no API key was provided. API key is required to load stub models."
            )
        workspace_id = get_roboflow_workspace(api_key=api_key)
        
        project_task_type = get_roboflow_dataset_type(
            api_key=api_key, workspace_id=workspace_id, dataset_id=dataset_id
        )
        model_type = "stub"
        save_model_metadata_in_cache(
            cache_path=cache_path,
            lock_key=lock_key,
            project_task_type=project_task_type,
            model_type=model_type,
            # TODO: do we need to save the workspace_id here/for the cache path to be unique?
        )
        return project_task_type, model_type

    api_data = get_roboflow_model_data(
        api_key=api_key,
        model_id=model_id,
        endpoint_type=ModelEndpointType.ORT,
        device_id=GLOBAL_DEVICE_ID,
    ).get("ort")

    if api_data is None:
        raise ModelArtefactError("Error loading model artifacts from Roboflow API.")
    # some older projects do not have type field - hence defaulting
    project_task_type = api_data.get("taskType", "object-detection")
    model_type = api_data.get("modelType")
    if model_type is None or model_type == "ort":
        # some very old model versions do not have modelType reported - and API respond in a generic way -
        # then we shall attempt using default model for given task type
        model_type = MODEL_TYPE_DEFAULTS.get(project_task_type)
    if model_type is None or project_task_type is None:
        raise ModelArtefactError("Error loading model artifacts from Roboflow API.")
    save_model_metadata_in_cache(
        cache_path=cache_path,
        lock_key=lock_key,
        project_task_type=project_task_type,
        model_type=model_type,
    )
    return project_task_type, model_type

def determine_cache_paths(dataset_or_model_id: str, version_id: Optional[str]) -> Tuple[str, str]:    
    if dataset_or_model_id and version_id:
        # It's a dataset/version ID
        lock_key = f"lock:metadata:dataset:{dataset_or_model_id}:{version_id}"
        cache_path = construct_dataset_version_cache_path(dataset_or_model_id, version_id)
    else:
        # It's a model ID
        lock_key = f"lock:metadata:model:{dataset_or_model_id}"
        cache_path = construct_model_id_cache_path(dataset_or_model_id)
    
    return lock_key, cache_path

def get_model_metadata_from_cache(
    cache_path: str,
    lock_key: str
) -> Optional[Tuple[TaskType, ModelType]]:
    if LAMBDA:
        return _get_model_metadata_from_cache(cache_path=cache_path)
    
    with cache.lock(lock_key, expire=CACHE_METADATA_LOCK_TIMEOUT):
        return _get_model_metadata_from_cache(cache_path=cache_path)

def _get_model_metadata_from_cache(cache_path: str) -> Optional[Tuple[TaskType, ModelType]]:
    if not os.path.isfile(cache_path):
        return None
    try:
        model_metadata = read_json(path=cache_path)
        if model_metadata_content_is_invalid(content=model_metadata):
            return None
        return model_metadata[PROJECT_TASK_TYPE_KEY], model_metadata[MODEL_TYPE_KEY]
    except ValueError as e:
        logger.warning(
            f"Could not load model description from cache under path: {cache_path} - decoding issue: {e}."
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
    cache_path: str,
    lock_key: str,
    project_task_type: TaskType,
    model_type: ModelType,
) -> None:
    if LAMBDA:
        _save_model_metadata_in_cache(
            cache_path=cache_path,
            project_task_type=project_task_type,
            model_type=model_type,
        )
        return None
    
    with cache.lock(lock_key, expire=CACHE_METADATA_LOCK_TIMEOUT):
        _save_model_metadata_in_cache(
            cache_path=cache_path,
            project_task_type=project_task_type,
            model_type=model_type,
        )
        return None

def _save_model_metadata_in_cache(
    cache_path: str,
    project_task_type: TaskType,
    model_type: ModelType,
) -> None:
    metadata = {
        PROJECT_TASK_TYPE_KEY: project_task_type,
        MODEL_TYPE_KEY: model_type,
    }
    dump_json(
        path=cache_path, content=metadata, allow_override=True, indent=4
    )

def construct_model_id_cache_path(model_id: str) -> str:
    """Constructs the cache path for a given model ID."""
    return os.path.join(MODEL_CACHE_DIR, "models", model_id, "model_type.json")

def construct_dataset_version_cache_path(dataset_id: str, version_id: str) -> str:
    """Constructs the cache path for a given dataset ID and version ID."""
    return os.path.join(MODEL_CACHE_DIR, dataset_id, version_id, "model_type.json")
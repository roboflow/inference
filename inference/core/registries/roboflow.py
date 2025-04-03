import os
from typing import Any, Dict, Optional, Tuple, Union

from cachetools.func import ttl_cache

from inference.core.cache import cache
from inference.core.devices.utils import GLOBAL_DEVICE_ID
from inference.core.entities.types import (
    DatasetID,
    ModelID,
    ModelType,
    TaskType,
    VersionID,
)
from inference.core.env import (
    LAMBDA,
    MODEL_CACHE_DIR,
    MODELS_CACHE_AUTH_CACHE_MAX_SIZE,
    MODELS_CACHE_AUTH_CACHE_TTL,
    MODELS_CACHE_AUTH_ENABLED,
)
from inference.core.exceptions import (
    MissingApiKeyError,
    ModelArtefactError,
    ModelNotRecognisedError,
    RoboflowAPINotAuthorizedError,
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
    get_roboflow_instant_model_data,
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

    def get_model(self, model_id: ModelID, api_key: str) -> Model:
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


@ttl_cache(ttl=MODELS_CACHE_AUTH_CACHE_TTL, maxsize=MODELS_CACHE_AUTH_CACHE_MAX_SIZE)
def _check_if_api_key_has_access_to_model(
    api_key: str,
    model_id: str,
) -> bool:
    _, version_id = get_model_id_chunks(model_id=model_id)
    try:
        if version_id is not None:
            get_roboflow_model_data(
                api_key=api_key,
                model_id=model_id,
                endpoint_type=ModelEndpointType.ORT,
                device_id=GLOBAL_DEVICE_ID,
            ).get("ort")
        else:
            get_roboflow_instant_model_data(
                api_key=api_key,
                model_id=model_id,
            )
    except RoboflowAPINotAuthorizedError:
        return False
    return True


def get_model_type(
    model_id: ModelID,
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
    if dataset_id in GENERIC_MODELS:
        logger.debug(f"Loading generic model: {dataset_id}.")
        return GENERIC_MODELS[dataset_id]

    if MODELS_CACHE_AUTH_ENABLED:
        if not _check_if_api_key_has_access_to_model(
            api_key=api_key, model_id=model_id
        ):
            raise RoboflowAPINotAuthorizedError(
                f"API key {api_key} does not have access to model {model_id}"
            )

    cached_metadata = get_model_metadata_from_cache(
        dataset_id=dataset_id, version_id=version_id
    )
    if cached_metadata is not None:
        return cached_metadata[0], cached_metadata[1]
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
            dataset_id=dataset_id,
            version_id=version_id,
            project_task_type=project_task_type,
            model_type=model_type,
        )
        return project_task_type, model_type

    if version_id is not None:
        api_data = get_roboflow_model_data(
            api_key=api_key,
            model_id=model_id,
            endpoint_type=ModelEndpointType.ORT,
            device_id=GLOBAL_DEVICE_ID,
        ).get("ort")
        project_task_type = api_data.get("type", "object-detection")
    else:
        api_data = get_roboflow_instant_model_data(
            api_key=api_key,
            model_id=model_id,
        )
        project_task_type = api_data.get("taskType", "object-detection")
    if api_data is None:
        raise ModelArtefactError("Error loading model artifacts from Roboflow API.")
    # some older projects do not have type field - hence defaulting
    model_type = api_data.get("modelType")
    if model_type is None or model_type == "ort":
        # some very old model versions do not have modelType reported - and API respond in a generic way -
        # then we shall attempt using default model for given task type
        model_type = MODEL_TYPE_DEFAULTS.get(project_task_type)
    if model_type is None or project_task_type is None:
        raise ModelArtefactError("Error loading model artifacts from Roboflow API.")
    save_model_metadata_in_cache(
        dataset_id=dataset_id,
        version_id=version_id,
        project_task_type=project_task_type,
        model_type=model_type,
    )

    return project_task_type, model_type


def get_model_metadata_from_cache(
    dataset_id: Union[DatasetID, ModelID],
    version_id: Optional[VersionID],
) -> Optional[Tuple[TaskType, ModelType]]:
    if LAMBDA:
        return _get_model_metadata_from_cache(
            dataset_id=dataset_id, version_id=version_id
        )
    with cache.lock(
        f"lock:metadata:{dataset_id}:{version_id}", expire=CACHE_METADATA_LOCK_TIMEOUT
    ):
        return _get_model_metadata_from_cache(
            dataset_id=dataset_id, version_id=version_id
        )


def _get_model_metadata_from_cache(
    dataset_id: Union[DatasetID, ModelID], version_id: Optional[VersionID]
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
    dataset_id: Union[DatasetID, ModelID],
    version_id: Optional[VersionID],
    project_task_type: TaskType,
    model_type: ModelType,
) -> None:
    if LAMBDA:
        _save_model_metadata_in_cache(
            dataset_id=dataset_id,
            version_id=version_id,
            project_task_type=project_task_type,
            model_type=model_type,
        )
        return None
    with cache.lock(
        f"lock:metadata:{dataset_id}:{version_id}", expire=CACHE_METADATA_LOCK_TIMEOUT
    ):
        _save_model_metadata_in_cache(
            dataset_id=dataset_id,
            version_id=version_id,
            project_task_type=project_task_type,
            model_type=model_type,
        )
        return None


def _save_model_metadata_in_cache(
    dataset_id: Union[DatasetID, ModelID],
    version_id: Optional[VersionID],
    project_task_type: TaskType,
    model_type: ModelType,
) -> None:
    model_type_cache_path = construct_model_type_cache_path(
        dataset_id=dataset_id, version_id=version_id
    )
    metadata = {
        PROJECT_TASK_TYPE_KEY: project_task_type,
        MODEL_TYPE_KEY: model_type,
    }
    dump_json(
        path=model_type_cache_path, content=metadata, allow_override=True, indent=4
    )


def construct_model_type_cache_path(
    dataset_id: Union[DatasetID, ModelID], version_id: Optional[VersionID]
) -> str:
    cache_dir = os.path.join(
        MODEL_CACHE_DIR, dataset_id, version_id if version_id else ""
    )
    return os.path.join(cache_dir, "model_type.json")

import os
import time
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Dict, Optional, Tuple, Union

from inference.core.cache import cache
from inference.core.cache.lru_cache import LRUCache
from inference.core.cache.redis import RedisCache
from inference.core.devices.utils import GLOBAL_DEVICE_ID
from inference.core.entities.types import (
    DatasetID,
    ModelID,
    ModelType,
    TaskType,
    VersionID,
)
from inference.core.env import (
    CACHE_METADATA_LOCK_TIMEOUT,
    INTERNAL_WEIGHTS_URL_SUFFIX,
    LAMBDA,
    MODEL_CACHE_DIR,
    MODELS_CACHE_AUTH_CACHE_MAX_SIZE,
    MODELS_CACHE_AUTH_CACHE_TTL,
    MODELS_CACHE_AUTH_ENABLED,
    MODELS_CACHE_AUTH_FAILURE_CACHE_TTL,
    MODELS_CACHE_AUTH_SHARED_CACHE_ENABLED,
    MODELS_CACHE_AUTH_SHARED_CACHE_TTL,
    ROBOFLOW_SERVICE_SECRET,
    USE_INFERENCE_MODELS,
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
    get_model_metadata_from_inference_models_registry,
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
    "sam3": ("embed", "sam3"),
    "sam3/sam3_interactive": ("interactive-segmentation", "sam3"),
    "sam3-3d-objects": ("3d-reconstruction", "sam3-3d-objects"),
    "gaze": ("gaze", "l2cs"),
    "doctr": ("ocr", "doctr"),
    "easy_ocr": ("ocr", "easy_ocr"),
    "trocr": ("ocr", "trocr"),
    "grounding_dino": ("object-detection", "grounding-dino"),
    "paligemma": ("llm", "paligemma"),
    "yolo_world": ("object-detection", "yolo-world"),
    "owlv2": ("object-detection", "owlv2"),
    "smolvlm2": ("lmm", "smolvlm-2.2b-instruct"),
    "depth-anything-v2": ("depth-estimation", "depth-anything-v2"),
    "depth-anything-v3": ("depth-estimation", "depth-anything-v3"),
    "moondream2": ("lmm", "moondream2"),
    "perception_encoder": ("embed", "perception_encoder"),
    "qwen3_5-0.8b": ("lmm", "qwen3_5-0.8b"),
    "qwen3_5-2b": ("lmm", "qwen3_5-2b"),
}

STUB_VERSION_ID = "0"

# In-process cache for model metadata to avoid Redis lock contention on every request.
_in_process_metadata_cache = LRUCache(capacity=1000)
_in_process_model_auth_cache = LRUCache(capacity=MODELS_CACHE_AUTH_CACHE_MAX_SIZE)
_shared_model_auth_cache_available = (
    MODELS_CACHE_AUTH_SHARED_CACHE_ENABLED and isinstance(cache, RedisCache)
)

if MODELS_CACHE_AUTH_SHARED_CACHE_ENABLED and not _shared_model_auth_cache_available:
    logger.warning(
        "MODELS_CACHE_AUTH_SHARED_CACHE_ENABLED is set but Redis cache is unavailable. "
        "Using in-process model auth cache only."
    )


@dataclass(frozen=True)
class ModelAuthCacheEntry:
    authorized: bool
    expires_at: float


class RoboflowModelRegistry(ModelRegistry):
    """A Roboflow-specific model registry which gets the model type using the model id,
    then returns a model class based on the model type.
    """

    def get_model(
        self,
        model_id: ModelID,
        api_key: str,
        countinference: Optional[bool] = None,
        service_secret: Optional[str] = None,
    ) -> Model:
        """Returns the model class based on the given model id and API key.

        Args:
            model_id (str): The ID of the model to be retrieved.
            api_key (str): The API key used to authenticate.

        Returns:
            Model: The model class corresponding to the given model ID and type.

        Raises:
            ModelNotRecognisedError: If the model type is not supported or found.
        """
        model_type = get_model_type(
            model_id,
            api_key,
            countinference=countinference,
            service_secret=service_secret,
        )
        logger.debug(f"Model type: {model_type}")

        if model_type not in self.registry_dict:
            raise ModelNotRecognisedError(
                f"Model type not supported, you may want to try a different inference server configuration or endpoint: {model_type}"
            )
        return self.registry_dict[model_type]


def _check_if_api_key_has_access_to_model(
    api_key: str,
    model_id: str,
    endpoint_type: ModelEndpointType = ModelEndpointType.ORT,
    countinference: Optional[bool] = None,
    service_secret: Optional[str] = None,
) -> bool:
    model_id = resolve_roboflow_model_alias(model_id=model_id)
    cache_key = _construct_model_auth_cache_key(
        api_key=api_key,
        model_id=model_id,
        endpoint_type=endpoint_type,
        countinference=countinference,
        service_secret=service_secret,
    )
    cached_result = _get_model_auth_result_from_local_cache(cache_key=cache_key)
    if cached_result is not None:
        return cached_result
    if _shared_model_auth_cache_enabled():
        cached_result = _get_model_auth_result_from_shared_cache(cache_key=cache_key)
        if cached_result is not None:
            _save_model_auth_result_to_local_cache(
                cache_key=cache_key, authorized=cached_result
            )
            return cached_result
    auth_result = _resolve_model_auth_result(
        api_key=api_key,
        model_id=model_id,
        endpoint_type=endpoint_type,
        countinference=countinference,
        service_secret=service_secret,
    )
    _save_model_auth_result_to_local_cache(cache_key=cache_key, authorized=auth_result)
    if _shared_model_auth_cache_enabled():
        _save_model_auth_result_to_shared_cache(
            cache_key=cache_key, authorized=auth_result
        )
    return auth_result


def _construct_model_auth_cache_key(
    api_key: Optional[str],
    model_id: str,
    endpoint_type: ModelEndpointType,
    countinference: Optional[bool],
    service_secret: Optional[str],
) -> str:
    _, version_id = get_model_id_chunks(model_id=model_id)
    if version_id is not None:
        model_source = "versioned"
    elif USE_INFERENCE_MODELS:
        model_source = "inference-models-registry"
    else:
        model_source = "instant"
    api_key_fingerprint = sha256(
        (api_key or "").encode("utf-8"), usedforsecurity=False
    ).hexdigest()
    credits_bypass_enabled = (
        INTERNAL_WEIGHTS_URL_SUFFIX == "serverless"
        and countinference is False
        and service_secret is not None
        and service_secret == ROBOFLOW_SERVICE_SECRET
    )
    return (
        "model_auth:"
        f"{model_source}:{endpoint_type.value}:{model_id}:"
        f"countinference={countinference}:credits_bypass={int(credits_bypass_enabled)}:"
        f"api_key={api_key_fingerprint}"
    )


def _resolve_model_auth_result(
    api_key: str,
    model_id: str,
    endpoint_type: ModelEndpointType,
    countinference: Optional[bool],
    service_secret: Optional[str],
) -> bool:
    _, version_id = get_model_id_chunks(model_id=model_id)
    try:
        if version_id is not None:
            get_roboflow_model_data(
                api_key=api_key,
                model_id=model_id,
                endpoint_type=endpoint_type,
                device_id=GLOBAL_DEVICE_ID,
                countinference=countinference,
                service_secret=service_secret,
            )
        elif not USE_INFERENCE_MODELS:
            get_roboflow_instant_model_data(
                api_key=api_key,
                model_id=model_id,
                countinference=countinference,
                service_secret=service_secret,
            )
        else:
            get_model_metadata_from_inference_models_registry(
                api_key=api_key,
                model_id=model_id,
                countinference=countinference,
                service_secret=service_secret,
            )
    except RoboflowAPINotAuthorizedError:
        return False
    return True


def _shared_model_auth_cache_enabled() -> bool:
    return _shared_model_auth_cache_available


def _get_model_auth_result_from_local_cache(cache_key: str) -> Optional[bool]:
    cache_entry = _in_process_model_auth_cache.get(cache_key)
    if cache_entry is None:
        return None
    if cache_entry.expires_at < time.time():
        _in_process_model_auth_cache.cache.pop(cache_key, None)
        return None
    return cache_entry.authorized


def _save_model_auth_result_to_local_cache(cache_key: str, authorized: bool) -> None:
    ttl = _get_model_auth_cache_ttl(authorized=authorized, shared=False)
    if ttl <= 0:
        return None
    _in_process_model_auth_cache.set(
        cache_key,
        ModelAuthCacheEntry(authorized=authorized, expires_at=time.time() + ttl),
    )


def _get_model_auth_result_from_shared_cache(cache_key: str) -> Optional[bool]:
    try:
        cached_result = cache.get(cache_key)
    except Exception as error:
        logger.warning(
            "Failed to read shared model auth cache for key=%s. Cause: %s",
            "<redacted>",
            error,
        )
        return None
    if cached_result is True or cached_result is False:
        return cached_result
    return None


def _save_model_auth_result_to_shared_cache(cache_key: str, authorized: bool) -> None:
    ttl = _get_model_auth_cache_ttl(authorized=authorized, shared=True)
    if ttl <= 0:
        return None
    try:
        cache.set(key=cache_key, value=authorized, expire=ttl)
    except Exception as error:
        logger.warning(
            "Failed to write shared model auth cache for key=%s. Cause: %s",
            "<redacted>",
            error,
        )
        return None


def _get_model_auth_cache_ttl(authorized: bool, shared: bool) -> int:
    if authorized:
        return (
            MODELS_CACHE_AUTH_SHARED_CACHE_TTL
            if shared
            else MODELS_CACHE_AUTH_CACHE_TTL
        )
    return MODELS_CACHE_AUTH_FAILURE_CACHE_TTL


def get_model_type(
    model_id: ModelID,
    api_key: Optional[str] = None,
    countinference: Optional[bool] = None,
    service_secret: Optional[str] = None,
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
    # first check if the model id as a whole is in the GENERIC_MODELS dictionary
    if model_id in GENERIC_MODELS:
        logger.debug(f"Loading generic model: {model_id}.")
        return GENERIC_MODELS[model_id]

    # then check if the dataset id is in the GENERIC_MODELS dictionary
    if dataset_id in GENERIC_MODELS:
        logger.debug(f"Loading generic model: {dataset_id}.")
        return GENERIC_MODELS[dataset_id]

    if MODELS_CACHE_AUTH_ENABLED:
        if not _check_if_api_key_has_access_to_model(
            api_key=api_key,
            model_id=model_id,
            countinference=countinference,
            service_secret=service_secret,
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
            countinference=countinference,
            service_secret=service_secret,
            endpoint_type=ModelEndpointType.ORT,
            device_id=GLOBAL_DEVICE_ID,
        ).get("ort")
        project_task_type = api_data.get("type", "object-detection")
    elif not USE_INFERENCE_MODELS:
        api_data = get_roboflow_instant_model_data(
            api_key=api_key,
            model_id=model_id,
            countinference=countinference,
            service_secret=service_secret,
        )
        project_task_type = api_data.get("taskType", "object-detection")
    else:
        api_data = get_model_metadata_from_inference_models_registry(
            api_key=api_key,
            model_id=model_id,
            countinference=countinference,
            service_secret=service_secret,
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
    cache_key = (dataset_id, version_id)
    cached = _in_process_metadata_cache.get(cache_key)
    if cached is not None:
        return cached
    if LAMBDA:
        result = _get_model_metadata_from_cache(
            dataset_id=dataset_id, version_id=version_id
        )
    else:
        with cache.lock(
            f"lock:metadata:{dataset_id}:{version_id}",
            expire=CACHE_METADATA_LOCK_TIMEOUT,
        ):
            result = _get_model_metadata_from_cache(
                dataset_id=dataset_id, version_id=version_id
            )
    if result is not None:
        _in_process_metadata_cache.set(cache_key, result)
    return result


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
    else:
        with cache.lock(
            f"lock:metadata:{dataset_id}:{version_id}",
            expire=CACHE_METADATA_LOCK_TIMEOUT,
        ):
            _save_model_metadata_in_cache(
                dataset_id=dataset_id,
                version_id=version_id,
                project_task_type=project_task_type,
                model_type=model_type,
            )
    _in_process_metadata_cache.set(
        (dataset_id, version_id), (project_task_type, model_type)
    )


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

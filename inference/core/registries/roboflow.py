import json
import os
import re
import stat
from typing import Any, Dict, Optional, Tuple, Union

from cachetools.func import ttl_cache

from inference.core.cache import cache
from inference.core.cache.lru_cache import LRUCache
from inference.core.cache.model_artifacts import get_cache_dir
from inference.core.devices.utils import GLOBAL_DEVICE_ID
from inference.core.entities.types import (
    DatasetID,
    ModelID,
    ModelType,
    TaskType,
    VersionID,
)
from inference.core.env import (
    ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES,
    CACHE_METADATA_LOCK_TIMEOUT,
    LAMBDA,
    MODELS_CACHE_AUTH_CACHE_MAX_SIZE,
    MODELS_CACHE_AUTH_CACHE_TTL,
    MODELS_CACHE_AUTH_ENABLED,
    OFFLINE_MODE,
    SAM3_FINE_TUNED_MODELS_ENABLED,
    USE_INFERENCE_MODELS,
)
from inference.core.exceptions import (
    MissingApiKeyError,
    ModelArtefactError,
    ModelDeploymentNotSupportedError,
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
from inference.core.utils.file_system import dump_json_atomic, read_json
from inference.core.utils.roboflow import get_model_id_chunks
from inference.models.aliases import resolve_roboflow_model_alias
from inference_models.models.auto_loaders import core as inference_models_auto_loaders
from inference_models.models.auto_loaders.core import parse_model_config
from inference_models.models.auto_loaders.entities import MODEL_CONFIG_FILE_NAME
from inference_models.models.auto_loaders.model_cache_paths import (
    generate_model_cache_root_for_model_id,
    generate_models_cache_dir,
)

# fallback model_type for local `inference_models` packages that do not declare
# model_architecture in model_config.json.
LOCAL_INFERENCE_MODELS_MODEL_TYPE = "inference-models-local"

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
    "pp_ocr": ("ocr", "pp_ocr"),
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
    "qwen3_5-4b": ("lmm", "qwen3_5-4b"),
}

STUB_VERSION_ID = "0"

# In-process cache for model metadata to avoid Redis lock contention on every request.
_in_process_metadata_cache = LRUCache(capacity=1000)

FINE_TUNED_SAM3_DEPLOYMENT_ERROR = (
    "Fine-tuned SAM3 models are not supported on this deployment. "
    "Please use a workflow or self-host the server."
)


def _find_cached_model_package_dir_compat(model_id: str) -> Optional[str]:
    """Find a cached package when inference-models predates its public helper."""
    models_cache_root = os.path.realpath(generate_models_cache_dir())
    try:
        lexical_model_cache_root = generate_model_cache_root_for_model_id(
            model_id=model_id
        )
    except Exception:
        return None
    if os.path.islink(lexical_model_cache_root):
        return None
    model_cache_root = os.path.realpath(lexical_model_cache_root)
    if not model_cache_root.startswith(models_cache_root + os.sep):
        return None
    if not os.path.isdir(model_cache_root):
        return None
    try:
        entries = sorted(os.listdir(model_cache_root))
    except OSError:
        return None
    for entry in entries:
        if entry.startswith(".") or re.fullmatch(r"[A-Za-z0-9]+", entry) is None:
            continue
        lexical_package_dir = os.path.join(model_cache_root, entry)
        if os.path.islink(lexical_package_dir):
            continue
        package_dir = os.path.realpath(lexical_package_dir)
        if not package_dir.startswith(model_cache_root + os.sep):
            continue
        config_path = os.path.join(package_dir, MODEL_CONFIG_FILE_NAME)
        if (
            not os.path.isdir(package_dir)
            or os.path.islink(config_path)
            or not os.path.isfile(config_path)
        ):
            continue
        try:
            config = read_json(path=config_path)
        except (OSError, ValueError):
            continue
        if not isinstance(config, dict):
            continue
        cached_model_id = config.get("model_id")
        if cached_model_id is not None and cached_model_id != model_id:
            continue
        task_type = config.get("task_type")
        if not isinstance(task_type, str) or not task_type:
            continue
        model_architecture = config.get("model_architecture")
        has_library_model = isinstance(model_architecture, str) and bool(
            model_architecture
        )
        has_custom_model = (
            isinstance(config.get("model_module"), str)
            and bool(config.get("model_module"))
            and isinstance(config.get("model_class"), str)
            and bool(config.get("model_class"))
        )
        if not has_library_model and not has_custom_model:
            continue
        if not isinstance(config.get("backend_type"), str):
            # A package without a backend cannot be resolved by the library
            # model registry. Custom-code packages are handled by their module
            # and class metadata instead.
            if not has_custom_model:
                continue
        return package_dir
    return None


# Runtime images install the latest released inference-models before inference.
# Remove this fallback once that release includes find_cached_model_package_dir.
find_cached_model_package_dir = getattr(
    inference_models_auto_loaders,
    "find_cached_model_package_dir",
    _find_cached_model_package_dir_compat,
)


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


@ttl_cache(ttl=MODELS_CACHE_AUTH_CACHE_TTL, maxsize=MODELS_CACHE_AUTH_CACHE_MAX_SIZE)
def _check_if_api_key_has_access_to_model(
    api_key: str,
    model_id: str,
    endpoint_type: ModelEndpointType = ModelEndpointType.ORT,
    countinference: Optional[bool] = None,
    service_secret: Optional[str] = None,
) -> bool:
    model_id = resolve_roboflow_model_alias(model_id=model_id)
    if _get_local_model_type(model_id=model_id) is not None:
        return True
    dataset_id, version_id = get_model_id_chunks(model_id=model_id)
    use_legacy_core_model_auth = (
        endpoint_type == ModelEndpointType.CORE_MODEL and dataset_id == "yolo_world"
    )
    try:
        if USE_INFERENCE_MODELS and not use_legacy_core_model_auth:
            get_model_metadata_from_inference_models_registry(
                api_key=api_key,
                model_id=model_id,
                countinference=countinference,
                service_secret=service_secret,
            )
        elif version_id is not None or use_legacy_core_model_auth:
            get_roboflow_model_data(
                api_key=api_key,
                model_id=model_id,
                endpoint_type=endpoint_type,
                device_id=GLOBAL_DEVICE_ID,
                countinference=countinference,
                service_secret=service_secret,
            )
        else:
            get_roboflow_instant_model_data(
                api_key=api_key,
                model_id=model_id,
                countinference=countinference,
                service_secret=service_secret,
            )
    except RoboflowAPINotAuthorizedError:
        return False
    return True


def _get_local_model_type(model_id: str) -> Optional[Tuple[TaskType, ModelType]]:
    """Returns model metadata read from a local `inference_models` package directory.

    Returns None when `model_id` is not a local directory or local loading is disabled,
    in which case the regular Roboflow model id resolution applies.
    """
    if not (
        USE_INFERENCE_MODELS
        and ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES
        and isinstance(model_id, str)
        and os.path.isdir(model_id)
    ):
        return None

    model_config = parse_model_config(
        config_path=os.path.join(model_id, MODEL_CONFIG_FILE_NAME)
    )
    if model_config.task_type is None:
        return None
    return (
        model_config.task_type,
        model_config.model_architecture or LOCAL_INFERENCE_MODELS_MODEL_TYPE,
    )


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
    local_model_type = _get_local_model_type(model_id=model_id)
    if local_model_type is not None:
        return local_model_type
    dataset_id, version_id = get_model_id_chunks(model_id=model_id)
    # first check if the model id as a whole is in the GENERIC_MODELS dictionary
    if model_id in GENERIC_MODELS:
        logger.debug(f"Loading generic model: {model_id}.")
        return GENERIC_MODELS[model_id]

    # then check if the dataset id is in the GENERIC_MODELS dictionary
    if dataset_id in GENERIC_MODELS:
        logger.debug(f"Loading generic model: {dataset_id}.")
        return GENERIC_MODELS[dataset_id]

    if MODELS_CACHE_AUTH_ENABLED and not OFFLINE_MODE:
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
        _ensure_model_supported_on_this_deployment(
            model_id=model_id,
            project_task_type=cached_metadata[0],
            model_type=cached_metadata[1],
        )
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

    if USE_INFERENCE_MODELS:
        api_data = get_model_metadata_from_inference_models_registry(
            api_key=api_key,
            model_id=model_id,
            countinference=countinference,
            service_secret=service_secret,
        )
        project_task_type = api_data.get("taskType", "object-detection")
    elif version_id is not None:
        api_data = get_roboflow_model_data(
            api_key=api_key,
            model_id=model_id,
            countinference=countinference,
            service_secret=service_secret,
            endpoint_type=ModelEndpointType.ORT,
            device_id=GLOBAL_DEVICE_ID,
        ).get("ort")
        project_task_type = api_data.get("type", "object-detection")
    else:
        api_data = get_roboflow_instant_model_data(
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
    _ensure_model_supported_on_this_deployment(
        model_id=model_id,
        project_task_type=project_task_type,
        model_type=model_type,
    )
    save_model_metadata_in_cache(
        dataset_id=dataset_id,
        version_id=version_id,
        project_task_type=project_task_type,
        model_type=model_type,
    )

    return project_task_type, model_type


def _ensure_model_supported_on_this_deployment(
    model_id: ModelID,
    project_task_type: TaskType,
    model_type: ModelType,
) -> None:
    if SAM3_FINE_TUNED_MODELS_ENABLED:
        return None
    if model_type not in {"sam3", "sam3-large"}:
        return None
    if project_task_type != "instance-segmentation":
        return None
    if isinstance(model_id, str) and model_id.startswith("sam3/"):
        return None
    raise ModelDeploymentNotSupportedError(FINE_TUNED_SAM3_DEPLOYMENT_ERROR)


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
    # Layout 1: traditional model_type.json
    try:
        model_type_cache_path = construct_model_type_cache_path(
            dataset_id=dataset_id, version_id=version_id
        )
    except ValueError as error:
        logger.warning(
            "Could not load model description from an unsafe cache path for "
            "%s/%s: %s",
            dataset_id,
            version_id,
            error,
        )
    else:
        if os.path.isfile(model_type_cache_path):
            try:
                model_metadata = _read_model_metadata_json(path=model_type_cache_path)
            except (OSError, ValueError) as error:
                logger.warning(
                    "Could not load model description from cache under path: "
                    "%s - read or decoding issue: %s.",
                    model_type_cache_path,
                    error,
                )
            else:
                if not model_metadata_content_is_invalid(content=model_metadata):
                    return (
                        model_metadata[PROJECT_TASK_TYPE_KEY],
                        model_metadata[MODEL_TYPE_KEY],
                    )

    # Layout 2: `inference-models` model_config.json
    model_id = f"{dataset_id}/{version_id}" if version_id else dataset_id
    return _get_model_metadata_from_inference_models_cache(model_id=model_id)


def _get_model_metadata_from_inference_models_cache(
    model_id: str,
) -> Optional[Tuple[TaskType, ModelType]]:
    """Check the `inference-models` cache layout for model metadata.

    Best-effort fallback used when the traditional ``model_type.json`` is
    absent (e.g. cache warmed through `inference-models` directly). The
    ``model_architecture`` stored in ``model_config.json`` is used as the
    model type - architecture-level keys are registered in
    ``ROBOFLOW_MODEL_TYPES``.
    """
    if not USE_INFERENCE_MODELS:
        return None
    cached_dir = find_cached_model_package_dir(model_id=model_id)
    if cached_dir is None:
        return None
    config_path = os.path.join(cached_dir, "model_config.json")
    try:
        metadata = read_json(path=config_path)
    except (OSError, ValueError):
        return None
    if not isinstance(metadata, dict):
        return None
    task_type = metadata.get("task_type", "")
    model_architecture = metadata.get("model_architecture", "")
    if (
        isinstance(task_type, str)
        and task_type
        and isinstance(model_architecture, str)
        and model_architecture
    ):
        return task_type, model_architecture
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
    dump_json_atomic(
        path=model_type_cache_path, content=metadata, allow_override=True, indent=4
    )


def _read_model_metadata_json(path: str) -> Optional[Union[dict, list]]:
    """Read regular-file metadata without following a final symlink."""

    # The sole caller supplies construct_model_type_cache_path's validated result.
    path_status = os.lstat(path)
    if not stat.S_ISREG(path_status.st_mode):
        raise OSError(f"Refusing to read non-regular metadata file: {path}")
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0),
    )
    try:
        descriptor_status = os.fstat(descriptor)
        if not stat.S_ISREG(descriptor_status.st_mode):
            raise OSError(f"Refusing to read non-regular metadata file: {path}")
        if (path_status.st_dev, path_status.st_ino) != (
            descriptor_status.st_dev,
            descriptor_status.st_ino,
        ):
            raise OSError(f"Metadata file changed while it was being opened: {path}")
        file_handle = os.fdopen(descriptor, "r", encoding="utf-8")
        descriptor = -1
        with file_handle:
            return json.load(file_handle)
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def construct_model_type_cache_path(
    dataset_id: Union[DatasetID, ModelID], version_id: Optional[VersionID]
) -> str:
    model_id = dataset_id if version_id is None else f"{dataset_id}/{version_id}"
    cache_dir = get_cache_dir(model_id=model_id)
    model_type_cache_path = os.path.join(cache_dir, "model_type.json")

    # MODEL_CACHE_DIR itself may be a mounted symlink. Every component below
    # that boundary must remain lexical so one model cannot alias another
    # model's metadata (or a file outside the cache).
    absolute_cache_root = os.path.abspath(get_cache_dir())
    absolute_metadata_path = os.path.abspath(model_type_cache_path)
    cache_prefix = absolute_cache_root.rstrip(os.sep) + os.sep
    if not absolute_metadata_path.startswith(cache_prefix):
        raise ValueError(
            f"Model metadata cache path for model {model_id} escapes the model cache directory."
        )
    try:
        if (
            os.path.commonpath([absolute_cache_root, absolute_metadata_path])
            != absolute_cache_root
        ):
            raise ValueError
        relative_metadata_path = os.path.relpath(
            absolute_metadata_path, absolute_cache_root
        )
    except ValueError as error:
        raise ValueError(
            f"Model metadata cache path for model {model_id} escapes the model cache directory."
        ) from error

    current_path = absolute_cache_root
    for path_part in relative_metadata_path.split(os.sep):
        current_path = os.path.join(current_path, path_part)
        if os.path.islink(current_path):
            raise ValueError(
                f"Model metadata cache path for model {model_id} traverses a symbolic link."
            )

    expected_resolved_path = os.path.normpath(
        os.path.join(os.path.realpath(absolute_cache_root), relative_metadata_path)
    )
    if os.path.realpath(absolute_metadata_path) != expected_resolved_path:
        # Covers Windows junctions and other path aliases that are not reported
        # by os.path.islink on every supported Python version.
        raise ValueError(
            f"Model metadata cache path for model {model_id} traverses a symbolic link."
        )
    return absolute_metadata_path

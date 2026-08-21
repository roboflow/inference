import hashlib
import inspect
import json
import os
import re
import stat
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

from cachetools.func import ttl_cache

from inference.core.cache import cache
from inference.core.cache.lru_cache import LRUCache
from inference.core.cache.model_artifacts import (
    get_cache_dir,
    get_legacy_model_id_cache_path,
    get_model_id_cache_path,
    validate_model_id_for_cache,
)
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
    FINE_TUNED_SAM3_DEPLOYMENT_ERROR,
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
    MODEL_VARIANT_KEY,
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
from inference.usage_tracking.model_types import record_model_type
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


@dataclass(frozen=True)
class ModelPipelineDefinition:
    """A synthetic model ID composed of concrete `inference_models` stage models.

    Pipeline IDs do not exist in remote model registries - only their downstream
    stage models do. Recognition maps the pipeline ID to a (task_type, model_type)
    pair served by a pipeline adapter; authorization is delegated to every
    downstream stage model ID.
    """

    task_type: TaskType
    model_type: ModelType
    downstream_model_ids: Tuple[str, ...]


PP_OCR_STAGE_VARIANTS = ("none", "tiny", "small", "medium")
PP_OCR_DEFAULT_STAGE_VARIANT = "small"


def _pp_ocr_pipeline_definition(
    text_detection: str, text_recognition: str
) -> ModelPipelineDefinition:
    downstream_model_ids = []
    if text_detection != "none":
        downstream_model_ids.append(f"pp-ocrv6-det/{text_detection}")
    if text_recognition != "none":
        downstream_model_ids.append(f"pp-ocrv6-rec/{text_recognition}")
    return ModelPipelineDefinition(
        task_type="ocr",
        model_type="pp_ocr",
        downstream_model_ids=tuple(downstream_model_ids),
    )


def _build_pp_ocr_pipelines() -> Dict[str, ModelPipelineDefinition]:
    # The recognized IDs mirror what InferenceModelsPPOCRAdapter._parse_det_rec
    # accepts: `pp_ocr/{det}-{rec}` for every valid combination (at least one
    # stage enabled), the single-token alias `pp_ocr/{variant}` (applies the
    # variant to both stages), and bare `pp_ocr` (defaults both stages).
    pipelines: Dict[str, ModelPipelineDefinition] = {}
    for text_detection in PP_OCR_STAGE_VARIANTS:
        for text_recognition in PP_OCR_STAGE_VARIANTS:
            if (text_detection, text_recognition) == ("none", "none"):
                continue
            pipelines[f"pp_ocr/{text_detection}-{text_recognition}"] = (
                _pp_ocr_pipeline_definition(text_detection, text_recognition)
            )
    for variant in PP_OCR_STAGE_VARIANTS:
        if variant == "none":
            continue
        pipelines[f"pp_ocr/{variant}"] = _pp_ocr_pipeline_definition(variant, variant)
    pipelines["pp_ocr"] = _pp_ocr_pipeline_definition(
        PP_OCR_DEFAULT_STAGE_VARIANT, PP_OCR_DEFAULT_STAGE_VARIANT
    )
    return pipelines


MODEL_PIPELINES: Dict[str, ModelPipelineDefinition] = _build_pp_ocr_pipelines()


def _get_model_pipeline_definition(model_id: str) -> Optional[ModelPipelineDefinition]:
    """Returns the pipeline definition for `model_id`, or None.

    Pipeline adapters are backed by `inference_models`, so pipeline IDs are only
    recognized when USE_INFERENCE_MODELS is enabled - otherwise they fall through
    to the regular Roboflow model resolution (and fail there).
    """
    if not USE_INFERENCE_MODELS:
        return None
    return MODEL_PIPELINES.get(model_id)


STUB_VERSION_ID = "0"
MODEL_ID_METADATA_KEY = "model_id"

# In-process cache for model metadata to avoid Redis lock contention on every request.
_in_process_metadata_cache = LRUCache(capacity=1000)


def _get_in_process_metadata_cache_key(
    dataset_id: Union[DatasetID, ModelID],
    version_id: Optional[VersionID],
    api_key: Optional[str],
) -> Tuple[Union[DatasetID, ModelID], Optional[VersionID], str]:
    credential_cache_key = hashlib.sha256(
        f"{api_key is None}:{api_key or ''}".encode("utf-8")
    ).hexdigest()
    return dataset_id, version_id, credential_cache_key


def _find_cached_model_package_dir_compat(
    model_id: str,
    api_key: Optional[str] = None,
) -> Optional[str]:
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
_find_cached_model_package_dir_impl = getattr(
    inference_models_auto_loaders,
    "find_cached_model_package_dir",
    _find_cached_model_package_dir_compat,
)


def find_cached_model_package_dir(
    model_id: str,
    api_key: Optional[str] = None,
) -> Optional[str]:
    """Call both current and pre-credential inference-models cache helpers."""

    try:
        finder_parameters = inspect.signature(
            _find_cached_model_package_dir_impl
        ).parameters.values()
    except (TypeError, ValueError):
        finder_parameters = ()
    supports_api_key = any(
        parameter.name == "api_key" or parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in finder_parameters
    )
    if supports_api_key:
        return _find_cached_model_package_dir_impl(
            model_id=model_id,
            api_key=api_key,
        )
    return _find_cached_model_package_dir_impl(model_id=model_id)


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
    pipeline_definition = _get_model_pipeline_definition(model_id=model_id)
    if pipeline_definition is not None:
        # Pipeline IDs are synthetic - they do not exist in remote model
        # registries, only their downstream stage models do. Authorization is
        # therefore delegated to every stage model the pipeline is composed of.
        return all(
            _check_if_api_key_has_access_to_model(
                api_key=api_key,
                model_id=downstream_model_id,
                endpoint_type=endpoint_type,
                countinference=countinference,
                service_secret=service_secret,
            )
            for downstream_model_id in pipeline_definition.downstream_model_ids
        )
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
    task_type, model_type, model_variant = _resolve_model_type(
        model_id=model_id,
        api_key=api_key,
        countinference=countinference,
        service_secret=service_secret,
    )
    # Usage tracking labels rows from this map so the registry (and its API
    # calls) stay off the inference hot path. Prefer the platform variant
    # (size / task suffix, e.g. yolov8-n) when present; class lookup still
    # uses the architecture returned below. Both id spellings are recorded
    # because callers may pass an alias while the loaded model reports its
    # resolved id.
    recorded_model_type = model_variant or model_type
    record_model_type(model_id=model_id, model_type=recorded_model_type)
    record_model_type(
        model_id=resolve_roboflow_model_alias(model_id=model_id),
        model_type=recorded_model_type,
    )
    return task_type, model_type


def _resolve_model_type(
    model_id: ModelID,
    api_key: Optional[str] = None,
    countinference: Optional[bool] = None,
    service_secret: Optional[str] = None,
) -> Tuple[TaskType, ModelType, Optional[str]]:
    model_id = resolve_roboflow_model_alias(model_id=model_id)
    local_model_type = _get_local_model_type(model_id=model_id)
    if local_model_type is not None:
        return local_model_type[0], local_model_type[1], None
    pipeline_definition = _get_model_pipeline_definition(model_id=model_id)
    if pipeline_definition is not None:
        logger.debug(f"Loading model pipeline: {model_id}.")
        return pipeline_definition.task_type, pipeline_definition.model_type, None
    validate_model_id_for_cache(model_id=model_id)
    dataset_id, version_id = get_model_id_chunks(model_id=model_id)
    # first check if the model id as a whole is in the GENERIC_MODELS dictionary
    if model_id in GENERIC_MODELS:
        logger.debug(f"Loading generic model: {model_id}.")
        task_type, model_type = GENERIC_MODELS[model_id]
        return task_type, model_type, None

    # then check if the dataset id is in the GENERIC_MODELS dictionary
    if dataset_id in GENERIC_MODELS:
        logger.debug(f"Loading generic model: {dataset_id}.")
        task_type, model_type = GENERIC_MODELS[dataset_id]
        return task_type, model_type, None

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

    cached_metadata = _get_cached_model_metadata(
        dataset_id=dataset_id,
        version_id=version_id,
        api_key=api_key,
    )

    if cached_metadata is not None:
        _ensure_model_supported_on_this_deployment(
            model_id=model_id,
            project_task_type=cached_metadata[0],
            model_type=cached_metadata[1],
        )
        return cached_metadata[0], cached_metadata[1], cached_metadata[2]
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
            api_key=api_key,
        )
        return project_task_type, model_type, None

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
    model_variant = api_data.get("modelVariant") or None
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
        model_variant=model_variant,
        api_key=api_key,
    )

    return project_task_type, model_type, model_variant


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
    api_key: Optional[str] = None,
) -> Optional[Tuple[TaskType, ModelType]]:
    cached_metadata = _get_cached_model_metadata(
        dataset_id=dataset_id,
        version_id=version_id,
        api_key=api_key,
    )
    if cached_metadata is None:
        return None
    return cached_metadata[0], cached_metadata[1]


def _get_cached_model_metadata(
    dataset_id: Union[DatasetID, ModelID],
    version_id: Optional[VersionID],
    api_key: Optional[str] = None,
) -> Optional[Tuple[TaskType, ModelType, Optional[str]]]:
    model_id = _combine_model_id(dataset_id=dataset_id, version_id=version_id)
    validate_model_id_for_cache(model_id=model_id)
    cache_key = _get_in_process_metadata_cache_key(
        dataset_id=dataset_id,
        version_id=version_id,
        api_key=api_key,
    )
    cached = _in_process_metadata_cache.get(cache_key)
    if cached is not None:
        return _normalize_cached_model_metadata(cached)
    if LAMBDA:
        result = _get_model_metadata_from_cache(
            dataset_id=dataset_id,
            version_id=version_id,
            api_key=api_key,
        )
    else:
        with cache.lock(
            f"lock:metadata:{dataset_id}:{version_id}",
            expire=CACHE_METADATA_LOCK_TIMEOUT,
        ):
            result = _get_model_metadata_from_cache(
                dataset_id=dataset_id,
                version_id=version_id,
                api_key=api_key,
            )
    normalized = _normalize_cached_model_metadata(result)
    if normalized is not None:
        _in_process_metadata_cache.set(cache_key, normalized)
    return normalized


def _normalize_cached_model_metadata(
    cached: Optional[Tuple[Any, ...]],
) -> Optional[Tuple[TaskType, ModelType, Optional[str]]]:
    if cached is None:
        return None
    if len(cached) >= 3:
        return cached[0], cached[1], cached[2]
    if len(cached) == 2:
        return cached[0], cached[1], None
    return None


def _get_model_metadata_from_cache(
    dataset_id: Union[DatasetID, ModelID],
    version_id: Optional[VersionID],
    api_key: Optional[str] = None,
) -> Optional[Tuple[TaskType, ModelType, Optional[str]]]:
    model_id = _combine_model_id(dataset_id=dataset_id, version_id=version_id)
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
        cache_dir_root = get_cache_dir()
        current_cache_key = get_model_id_cache_path(
            model_id=model_id,
            cache_dir_root=cache_dir_root,
        )
        cached_metadata = _load_model_metadata_from_path(
            path=model_type_cache_path,
            # Ownerless historical entries remain compatible only when the
            # current model ID still maps to its unchanged raw path. Generated
            # V2 paths can collide with old raw IDs and therefore require exact
            # attribution.
            required_model_id=model_id,
            allow_ownerless=current_cache_key == model_id,
        )
        if cached_metadata is not None:
            return cached_metadata

    # Slugged paths written before the digest was widened remain readable only
    # when the metadata proves which exact model ID owns the directory.
    try:
        legacy_model_type_cache_path = _construct_legacy_model_type_cache_path(
            model_id=model_id
        )
    except ValueError as error:
        logger.warning(
            "Could not load legacy model description from an unsafe cache path "
            "for %s: %s",
            model_id,
            error,
        )
    else:
        if legacy_model_type_cache_path is not None:
            cached_metadata = _load_model_metadata_from_path(
                path=legacy_model_type_cache_path,
                required_model_id=model_id,
            )
            if cached_metadata is not None:
                return cached_metadata

    # Layout 2: `inference-models` model_config.json
    return _get_model_metadata_from_inference_models_cache(
        model_id=model_id,
        api_key=api_key,
    )


def _load_model_metadata_from_path(
    path: str,
    required_model_id: Optional[str] = None,
    allow_ownerless: bool = False,
) -> Optional[Tuple[TaskType, ModelType, Optional[str]]]:
    try:
        model_metadata = _read_model_metadata_json(path=path)
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as error:
        logger.warning(
            "Could not load model description from cache under path: "
            "%s - read or decoding issue: %s.",
            path,
            error,
        )
        return None
    if model_metadata_content_is_invalid(content=model_metadata):
        return None
    if required_model_id is not None:
        owner_is_absent = MODEL_ID_METADATA_KEY not in model_metadata
        cached_model_id = model_metadata.get(MODEL_ID_METADATA_KEY)
        if not (allow_ownerless and owner_is_absent) and (
            not isinstance(cached_model_id, str)
            or not cached_model_id
            or cached_model_id != required_model_id
        ):
            logger.warning(
                "Refusing cached model metadata under path %s because its "
                "model_id does not exactly match %s.",
                path,
                required_model_id,
            )
            return None
    model_variant = model_metadata.get(MODEL_VARIANT_KEY)
    if not isinstance(model_variant, str) or not model_variant:
        model_variant = None
    return (
        model_metadata[PROJECT_TASK_TYPE_KEY],
        model_metadata[MODEL_TYPE_KEY],
        model_variant,
    )


def _get_model_metadata_from_inference_models_cache(
    model_id: str,
    api_key: Optional[str] = None,
) -> Optional[Tuple[TaskType, ModelType, Optional[str]]]:
    """Check the `inference-models` cache layout for model metadata.

    Best-effort fallback used when the traditional ``model_type.json`` is
    absent (e.g. cache warmed through `inference-models` directly). The
    ``model_architecture`` stored in ``model_config.json`` is used as the
    model type - architecture-level keys are registered in
    ``ROBOFLOW_MODEL_TYPES``.
    """
    if not USE_INFERENCE_MODELS:
        return None
    cached_dir = find_cached_model_package_dir(
        model_id=model_id,
        api_key=api_key,
    )
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
        return task_type, model_architecture, None
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
    api_key: Optional[str] = None,
    model_variant: Optional[str] = None,
) -> None:
    model_id = _combine_model_id(dataset_id=dataset_id, version_id=version_id)
    validate_model_id_for_cache(model_id=model_id)
    if LAMBDA:
        _save_model_metadata_in_cache(
            dataset_id=dataset_id,
            version_id=version_id,
            project_task_type=project_task_type,
            model_type=model_type,
            model_variant=model_variant,
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
                model_variant=model_variant,
            )
    _in_process_metadata_cache.set(
        _get_in_process_metadata_cache_key(
            dataset_id=dataset_id,
            version_id=version_id,
            api_key=api_key,
        ),
        (project_task_type, model_type, model_variant),
    )


def _save_model_metadata_in_cache(
    dataset_id: Union[DatasetID, ModelID],
    version_id: Optional[VersionID],
    project_task_type: TaskType,
    model_type: ModelType,
    model_variant: Optional[str] = None,
) -> None:
    model_id = _combine_model_id(dataset_id=dataset_id, version_id=version_id)
    model_type_cache_path = construct_model_type_cache_path(
        dataset_id=dataset_id, version_id=version_id
    )
    _ensure_current_model_cache_path_can_be_claimed(
        model_id=model_id,
        model_type_cache_path=model_type_cache_path,
    )
    metadata = {
        PROJECT_TASK_TYPE_KEY: project_task_type,
        MODEL_TYPE_KEY: model_type,
        MODEL_ID_METADATA_KEY: model_id,
    }
    if model_variant:
        metadata[MODEL_VARIANT_KEY] = model_variant
    dump_json_atomic(
        path=model_type_cache_path, content=metadata, allow_override=True, indent=4
    )


def _ensure_current_model_cache_path_can_be_claimed(
    model_id: str,
    model_type_cache_path: str,
) -> None:
    """Prevent a generated V2 path from claiming an older raw cache tree."""

    cache_dir_root = get_cache_dir()
    current_cache_key = get_model_id_cache_path(
        model_id=model_id,
        cache_dir_root=cache_dir_root,
    )
    if current_cache_key == model_id:
        # Portable raw paths are already injective and retain historical
        # ownerless compatibility. A present attribution must still agree:
        # explicit empty or contradictory owners are corruption, not legacy
        # ownerless metadata.
        if not os.path.lexists(model_type_cache_path):
            return
        try:
            existing_metadata = _read_model_metadata_json(path=model_type_cache_path)
        except (OSError, ValueError) as error:
            raise ModelArtefactError(
                f"Refusing to replace unreadable metadata in raw cache path "
                f"for model {model_id}."
            ) from error
        if not isinstance(existing_metadata, dict):
            raise ModelArtefactError(
                f"Refusing to replace malformed metadata in raw cache path "
                f"for model {model_id}."
            )
        if MODEL_ID_METADATA_KEY not in existing_metadata:
            return
        existing_owner = existing_metadata.get(MODEL_ID_METADATA_KEY)
        if existing_owner != model_id:
            raise ModelArtefactError(
                f"Refusing to claim raw cache path for model {model_id}; "
                "the existing metadata declares a different or invalid owner."
            )
        return

    model_cache_dir = os.path.dirname(model_type_cache_path)
    if not os.path.lexists(model_cache_dir):
        return
    if not os.path.isdir(model_cache_dir) or os.path.islink(model_cache_dir):
        raise ModelArtefactError(
            f"Refusing to claim unsafe generated cache path for model {model_id}."
        )

    if os.path.lexists(model_type_cache_path):
        try:
            existing_metadata = _read_model_metadata_json(path=model_type_cache_path)
        except (OSError, ValueError) as error:
            raise ModelArtefactError(
                f"Refusing to replace unreadable metadata in generated cache "
                f"path for model {model_id}."
            ) from error
        existing_owner = (
            existing_metadata.get(MODEL_ID_METADATA_KEY)
            if isinstance(existing_metadata, dict)
            else None
        )
        if existing_owner != model_id:
            raise ModelArtefactError(
                f"Refusing to claim generated cache path for model {model_id}; "
                "the existing metadata does not prove the same owner."
            )
        return

    try:
        directory_entries = os.listdir(model_cache_dir)
    except OSError as error:
        raise ModelArtefactError(
            f"Refusing to inspect generated cache path for model {model_id}."
        ) from error
    if directory_entries:
        raise ModelArtefactError(
            f"Refusing to claim non-empty generated cache path for model "
            f"{model_id} without exact ownership metadata."
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
    model_id = _combine_model_id(dataset_id=dataset_id, version_id=version_id)
    cache_dir = get_cache_dir(model_id=model_id)
    model_type_cache_path = os.path.join(cache_dir, "model_type.json")
    return _validate_model_type_cache_path(
        model_id=model_id, model_type_cache_path=model_type_cache_path
    )


def _construct_legacy_model_type_cache_path(model_id: str) -> Optional[str]:
    cache_dir_root = get_cache_dir()
    legacy_model_cache_path = get_legacy_model_id_cache_path(
        model_id=model_id, cache_dir_root=cache_dir_root
    )
    if legacy_model_cache_path is None:
        return None
    model_type_cache_path = os.path.join(
        cache_dir_root, legacy_model_cache_path, "model_type.json"
    )
    return _validate_model_type_cache_path(
        model_id=model_id, model_type_cache_path=model_type_cache_path
    )


def _combine_model_id(
    dataset_id: Union[DatasetID, ModelID], version_id: Optional[VersionID]
) -> str:
    return str(dataset_id) if version_id is None else f"{dataset_id}/{version_id}"


def _validate_model_type_cache_path(model_id: str, model_type_cache_path: str) -> str:

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

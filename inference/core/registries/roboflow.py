import os
from dataclasses import dataclass
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
from inference.core.utils.file_system import dump_json, read_json
from inference.core.utils.roboflow import get_model_id_chunks
from inference.models.aliases import resolve_roboflow_model_alias
from inference_models.models.auto_loaders.core import parse_model_config
from inference_models.models.auto_loaders.entities import MODEL_CONFIG_FILE_NAME

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

# In-process cache for model metadata to avoid Redis lock contention on every request.
_in_process_metadata_cache = LRUCache(capacity=1000)

FINE_TUNED_SAM3_DEPLOYMENT_ERROR = (
    "Fine-tuned SAM3 models are not supported on this deployment. "
    "Please use a workflow or self-host the server."
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

    model_id = resolve_roboflow_model_alias(model_id=model_id)
    local_model_type = _get_local_model_type(model_id=model_id)
    if local_model_type is not None:
        return local_model_type
    pipeline_definition = _get_model_pipeline_definition(model_id=model_id)
    if pipeline_definition is not None:
        logger.debug(f"Loading model pipeline: {model_id}.")
        return pipeline_definition.task_type, pipeline_definition.model_type
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
    model_id = dataset_id if version_id is None else f"{dataset_id}/{version_id}"
    cache_dir = get_cache_dir(model_id=model_id)
    return os.path.join(cache_dir, "model_type.json")

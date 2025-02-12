import hashlib
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from inference.core import logger
from inference.core.active_learning.entities import (
    ActiveLearningConfiguration,
    RoboflowProjectMetadata,
    SamplingMethod,
)
from inference.core.active_learning.samplers.close_to_threshold import (
    initialize_close_to_threshold_sampling,
)
from inference.core.active_learning.samplers.contains_classes import (
    initialize_classes_based_sampling,
)
from inference.core.active_learning.samplers.number_of_detections import (
    initialize_detections_number_based_sampling,
)
from inference.core.active_learning.samplers.random import initialize_random_sampling
from inference.core.cache.base import BaseCache
from inference.core.constants import CLASSIFICATION_TASK
from inference.core.exceptions import (
    ActiveLearningConfigurationDecodingError,
    ActiveLearningConfigurationError,
    RoboflowAPINotAuthorizedError,
    RoboflowAPINotNotFoundError,
)
from inference.core.roboflow_api import (
    get_roboflow_active_learning_configuration,
    get_roboflow_dataset_type,
    get_roboflow_workspace,
)

TYPE2SAMPLING_INITIALIZERS = {
    "random": initialize_random_sampling,
    "close_to_threshold": initialize_close_to_threshold_sampling,
    "classes_based": initialize_classes_based_sampling,
    "detections_number_based": initialize_detections_number_based_sampling,
}
ACTIVE_LEARNING_CONFIG_CACHE_EXPIRE = 900  # 15 min


def prepare_active_learning_configuration(
    api_key: str,
    target_dataset: str,
    model_id: str,
    cache: BaseCache,
) -> Optional[ActiveLearningConfiguration]:
    try:
        project_metadata = get_roboflow_project_metadata(
            api_key=api_key,
            target_dataset=target_dataset,
            model_id=model_id,
            cache=cache,
        )
    except Exception as e:
        logger.warn(
            f"Failed to initialise Active Learning configuration. Active Learning will not be enabled for this session. Cause: {str(e)}"
        )
        return None
    if not project_metadata.active_learning_configuration.get("enabled", False):
        return None
    logger.info(
        f"Configuring active learning for workspace: {project_metadata.workspace_id}, "
        f"project: {project_metadata.dataset_id} of type: {project_metadata.dataset_type}. "
        f"AL configuration: {project_metadata.active_learning_configuration}"
    )
    return initialise_active_learning_configuration(
        project_metadata=project_metadata,
        model_id=model_id,
    )


def prepare_active_learning_configuration_inplace(
    api_key: str,
    target_dataset: str,
    model_id: str,
    active_learning_configuration: Optional[dict],
) -> Optional[ActiveLearningConfiguration]:
    if (
        active_learning_configuration is None
        or active_learning_configuration.get("enabled", False) is False
    ):
        return None
    workspace_id = get_roboflow_workspace(api_key=api_key)
    dataset_type = get_roboflow_dataset_type(
        api_key=api_key,
        workspace_id=workspace_id,
        dataset_id=target_dataset,
    )
    model_type = dataset_type
    if not model_id.startswith(target_dataset):
        model_type = get_model_type(model_id=model_id, api_key=api_key)
    if predictions_incompatible_with_dataset(
        model_type=model_type, dataset_type=dataset_type
    ):
        logger.warning(
            f"Attempted to register predictions from model {model_id} (type: {model_type}) "
            f"into dataset {target_dataset} (of type {dataset_type}) which have incompatible types."
        )
        return None
    project_metadata = RoboflowProjectMetadata(
        dataset_id=target_dataset,
        workspace_id=workspace_id,
        dataset_type=dataset_type,
        active_learning_configuration=active_learning_configuration,
    )
    return initialise_active_learning_configuration(
        project_metadata=project_metadata,
        model_id=model_id,
    )


def get_roboflow_project_metadata(
    api_key: str,
    target_dataset: str,
    model_id: str,
    cache: BaseCache,
) -> RoboflowProjectMetadata:
    logger.info(f"Fetching active learning configuration.")
    config_cache_key = construct_cache_key_for_active_learning_config(
        api_key=api_key,
        target_dataset=target_dataset,
        model_id=model_id,
    )
    cached_config = cache.get(config_cache_key)
    if cached_config is not None:
        logger.info("Found Active Learning configuration in cache.")
        return parse_cached_roboflow_project_metadata(cached_config=cached_config)
    workspace_id = get_roboflow_workspace(api_key=api_key)
    dataset_type = get_roboflow_dataset_type(
        api_key=api_key,
        workspace_id=workspace_id,
        dataset_id=target_dataset,
    )
    model_type = dataset_type
    if not model_id.startswith(target_dataset):
        model_type = get_model_type(model_id=model_id, api_key=api_key)
    if predictions_incompatible_with_dataset(
        model_type=model_type, dataset_type=dataset_type
    ):
        logger.warning(
            f"Attempted to register predictions from model {model_id} (type: {model_type}) "
            f"into dataset {target_dataset} (of type {dataset_type}) which have incompatible types."
        )
        roboflow_api_configuration = {"enabled": False}
    else:
        roboflow_api_configuration = safe_get_roboflow_active_learning_configuration(
            api_key=api_key,
            workspace_id=workspace_id,
            dataset_id=target_dataset,
        )
    configuration = RoboflowProjectMetadata(
        dataset_id=target_dataset,
        workspace_id=workspace_id,
        dataset_type=dataset_type,
        active_learning_configuration=roboflow_api_configuration,
    )
    cache.set(
        key=config_cache_key,
        value=asdict(configuration),
        expire=ACTIVE_LEARNING_CONFIG_CACHE_EXPIRE,
    )
    return configuration


def construct_cache_key_for_active_learning_config(
    api_key: str, target_dataset: str, model_id: str
) -> str:
    api_key_hash = hashlib.md5(api_key.encode("utf-8")).hexdigest()
    return f"active_learning:configurations:{api_key_hash}:{target_dataset}:{model_id}"


def parse_cached_roboflow_project_metadata(
    cached_config: dict,
) -> RoboflowProjectMetadata:
    try:
        return RoboflowProjectMetadata(
            dataset_id=cached_config["dataset_id"],
            workspace_id=cached_config["workspace_id"],
            dataset_type=cached_config["dataset_type"],
            active_learning_configuration=cached_config[
                "active_learning_configuration"
            ],
        )
    except Exception as error:
        raise ActiveLearningConfigurationDecodingError(
            f"Failed to initialise Active Learning configuration. Cause: {str(error)}"
        ) from error


def get_model_type(model_id: str, api_key: str) -> str:
    model_dataset = model_id.split("/")[0]
    model_workspace = get_roboflow_workspace(api_key=api_key)
    return get_roboflow_dataset_type(
        api_key=api_key,
        workspace_id=model_workspace,
        dataset_id=model_dataset,
    )


def predictions_incompatible_with_dataset(
    model_type: str,
    dataset_type: str,
) -> bool:
    """
    The incompatibility occurs when we mix classification with detection - as detection-based
    predictions are partially compatible (for instance - for key-points detection we may register bboxes
    from object detection and manually provide key-points annotations)
    """
    model_is_classifier = CLASSIFICATION_TASK in model_type
    dataset_is_of_type_classification = CLASSIFICATION_TASK in dataset_type
    return model_is_classifier != dataset_is_of_type_classification


def safe_get_roboflow_active_learning_configuration(
    api_key: str,
    workspace_id: str,
    dataset_id: str,
) -> dict:
    try:
        return get_roboflow_active_learning_configuration(
            api_key=api_key, workspace_id=workspace_id, dataset_id=dataset_id
        )
    except (RoboflowAPINotAuthorizedError, RoboflowAPINotNotFoundError):
        # currently backend returns HTTP 404 if dataset does not exist
        # or workspace_id from api_key indicate that the owner is different,
        # so in the situation when we query for Universe dataset.
        # We want the owner of public dataset to be able to set AL configs
        # and use them, but not other people. At this point it's known
        # that HTTP 404 means not authorised (which will probably change
        # in future iteration of backend) - so on both NotAuth and NotFound
        # errors we assume that we simply cannot use AL with this model and
        # this api_key.
        return {"enabled": False}


def initialise_active_learning_configuration(
    project_metadata: RoboflowProjectMetadata,
    model_id: str,
) -> ActiveLearningConfiguration:
    sampling_methods = initialize_sampling_methods(
        sampling_strategies_configs=project_metadata.active_learning_configuration[
            "sampling_strategies"
        ],
    )
    target_workspace_id = project_metadata.active_learning_configuration.get(
        "target_workspace", project_metadata.workspace_id
    )
    target_dataset_id = project_metadata.active_learning_configuration.get(
        "target_project", project_metadata.dataset_id
    )
    return ActiveLearningConfiguration.init(
        roboflow_api_configuration=project_metadata.active_learning_configuration,
        sampling_methods=sampling_methods,
        workspace_id=target_workspace_id,
        dataset_id=target_dataset_id,
        model_id=model_id,
    )


def initialize_sampling_methods(
    sampling_strategies_configs: List[Dict[str, Any]],
) -> List[SamplingMethod]:
    result = []
    for sampling_strategy_config in sampling_strategies_configs:
        sampling_type = sampling_strategy_config["type"]
        if sampling_type not in TYPE2SAMPLING_INITIALIZERS:
            logger.warn(
                f"Could not identify sampling method `{sampling_type}` - skipping initialisation."
            )
            continue
        initializer = TYPE2SAMPLING_INITIALIZERS[sampling_type]
        result.append(initializer(sampling_strategy_config))
    names = set(m.name for m in result)
    if len(names) != len(result):
        raise ActiveLearningConfigurationError(
            "Detected duplication of Active Learning strategies names."
        )
    return result

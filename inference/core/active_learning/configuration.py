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
from inference.core.env import ACTIVE_LEARNING_ENABLED
from inference.core.exceptions import (
    ActiveLearningConfigurationDecodingError,
    ActiveLearningConfigurationError,
)
from inference.core.roboflow_api import (
    get_roboflow_active_learning_configuration,
    get_roboflow_dataset_type,
    get_roboflow_workspace,
)
from inference.core.utils.roboflow import get_model_id_chunks

TYPE2SAMPLING_INITIALIZERS = {
    "random": initialize_random_sampling,
    "close_to_threshold": initialize_close_to_threshold_sampling,
    "classes_based": initialize_classes_based_sampling,
    "detections_number_based": initialize_detections_number_based_sampling,
}
ACTIVE_LEARNING_CONFIG_CACHE_EXPIRE = 300  # 5 min


def prepare_active_learning_configuration(
    api_key: str,
    model_id: str,
    cache: BaseCache,
) -> Optional[ActiveLearningConfiguration]:
    if not ACTIVE_LEARNING_ENABLED:
        return None
    project_metadata = get_roboflow_project_metadata(
        api_key=api_key,
        model_id=model_id,
        cache=cache,
    )
    if not project_metadata.active_learning_configuration.get("enabled", False):
        return None
    logger.info(
        f"Configuring active learning for workspace: {project_metadata.workspace_id}, "
        f"project: {project_metadata.dataset_id} of type: {project_metadata.dataset_type}. "
        f"AL configuration: {project_metadata.active_learning_configuration}"
    )
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


def get_roboflow_project_metadata(
    api_key: str,
    model_id: str,
    cache: BaseCache,
) -> RoboflowProjectMetadata:
    logger.info(f"Fetching active learning configuration.")
    config_cache_key = construct_cache_key_for_active_learning_config(model_id=model_id)
    cached_config = cache.get(config_cache_key)
    if cached_config is not None:
        logger.info("Found Active Learning configuration in cache.")
        return parse_cached_roboflow_project_metadata(cached_config=cached_config)
    dataset_id, version_id = get_model_id_chunks(model_id=model_id)
    workspace_id = get_roboflow_workspace(api_key=api_key)
    dataset_type = get_roboflow_dataset_type(
        api_key=api_key,
        workspace_id=workspace_id,
        dataset_id=dataset_id,
    )
    roboflow_api_configuration = get_roboflow_active_learning_configuration(
        api_key=api_key, workspace_id=workspace_id, dataset_id=dataset_id
    )
    configuration = RoboflowProjectMetadata(
        dataset_id=dataset_id,
        version_id=version_id,
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


def construct_cache_key_for_active_learning_config(model_id: str) -> str:
    dataset_id = model_id.split("/")[0]
    return f"active_learning:configurations:{dataset_id}"


def parse_cached_roboflow_project_metadata(
    cached_config: dict,
) -> RoboflowProjectMetadata:
    try:
        return RoboflowProjectMetadata(**cached_config)
    except Exception as error:
        raise ActiveLearningConfigurationDecodingError(
            f"Failed to initialise Active Learning configuration. Cause: {str(error)}"
        ) from error


def initialize_sampling_methods(
    sampling_strategies_configs: List[Dict[str, Any]]
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

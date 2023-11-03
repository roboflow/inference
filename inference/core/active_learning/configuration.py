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
from inference.core.env import ACTIVE_LEARNING_ENABLED
from inference.core.exceptions import ActiveLearningConfigurationError
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


def prepare_active_learning_configuration(
    api_key: str,
    model_id: str,
) -> Optional[ActiveLearningConfiguration]:
    if not ACTIVE_LEARNING_ENABLED:
        return None
    project_metadata = get_roboflow_project_metadata(
        api_key=api_key,
        model_id=model_id,
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
) -> RoboflowProjectMetadata:
    logger.info(f"Fetching active learning configuration.")
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
    return RoboflowProjectMetadata(
        dataset_id=dataset_id,
        version_id=version_id,
        workspace_id=workspace_id,
        dataset_type=dataset_type,
        active_learning_configuration=roboflow_api_configuration,
    )


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

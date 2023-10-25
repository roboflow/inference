from typing import Optional, List, Dict, Any

from inference.core import logger
from inference.core.active_learning.entities import (
    ActiveLearningConfiguration,
    SamplingMethod,
)
from inference.core.active_learning.sampling import initialize_random_sampling
from inference.core.env import ACTIVE_LEARNING_ENABLED
from inference.core.roboflow_api import (
    get_roboflow_workspace,
    get_roboflow_dataset_type,
    get_roboflow_active_learning_configuration,
)
from inference.core.utils.roboflow import get_model_id_chunks


TYPE2SAMPLING_INITIALIZERS = {"random_sampling": initialize_random_sampling}


def prepare_active_learning_configuration(
    api_key: str,
    model_id: str,
) -> Optional[ActiveLearningConfiguration]:
    if not ACTIVE_LEARNING_ENABLED:
        return None
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
    if not roboflow_api_configuration.get("enabled", False):
        return None
    logger.info(
        f"Configuring active learning for workspace: {workspace_id}, project: {dataset_id} "
        f"of type: {dataset_type}. AL configuration: {roboflow_api_configuration}"
    )
    sampling_methods = initialize_sampling_methods(
        sampling_strategies_configs=roboflow_api_configuration["sampling_strategies"],
    )
    return ActiveLearningConfiguration.init(
        roboflow_api_configuration=roboflow_api_configuration,
        sampling_methods=sampling_methods,
        workspace_id=workspace_id,
        dataset_id=dataset_id,
        model_id=model_id,
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
    return result

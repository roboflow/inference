"""Loading of the active learning configuration of a Roboflow project."""

import hashlib
import logging
from dataclasses import asdict
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from roboflow_workflows.prototypes.platform_errors import (
    RoboflowAPINotAuthorizedError,
    RoboflowAPINotNotFoundError,
)

from inference_server.active_learning.entities import (
    CLASSIFICATION_TASK,
    ActiveLearningConfiguration,
    ActiveLearningConfigurationDecodingError,
    ActiveLearningConfigurationError,
    RoboflowProjectMetadata,
    SamplingMethod,
)
from inference_server.active_learning.samplers.close_to_threshold import (
    initialize_close_to_threshold_sampling,
)
from inference_server.active_learning.samplers.contains_classes import (
    initialize_classes_based_sampling,
)
from inference_server.active_learning.samplers.number_of_detections import (
    initialize_detections_number_based_sampling,
)
from inference_server.active_learning.samplers.random import initialize_random_sampling

if TYPE_CHECKING:
    from inference_server.workflows.host import ServerRoboflowPlatformClient

logger = logging.getLogger(__name__)

TYPE2SAMPLING_INITIALIZERS = {
    "random": initialize_random_sampling,
    "close_to_threshold": initialize_close_to_threshold_sampling,
    "classes_based": initialize_classes_based_sampling,
    "detections_number_based": initialize_detections_number_based_sampling,
}
ACTIVE_LEARNING_CONFIG_CACHE_EXPIRE = 900


def prepare_active_learning_configuration(
    api_key: str,
    target_dataset: str,
    model_id: str,
    cache: Any,
    platform_client: "ServerRoboflowPlatformClient",
) -> Optional[ActiveLearningConfiguration]:
    """Load the active learning configuration of a target project.

    Args:
        api_key: Roboflow API key of the request.
        target_dataset: Project datapoints are registered in.
        model_id: Model whose predictions are registered.
        cache: Cache with ``get`` and ``set(key, value, expire)``.
        platform_client: Roboflow platform client of the server.

    Returns:
        The configuration, or None when active learning is disabled for the
        project or the project metadata could not be loaded.

    Raises:
        ActiveLearningConfigurationError: If the configuration is inconsistent.
        ActiveLearningConfigurationDecodingError: If the configuration cannot
            be decoded.
    """
    try:
        project_metadata = get_roboflow_project_metadata(
            api_key=api_key,
            target_dataset=target_dataset,
            model_id=model_id,
            cache=cache,
            platform_client=platform_client,
        )
    except Exception as error:
        logger.warning(
            "Failed to initialise Active Learning configuration. Active Learning "
            "will not be enabled for this session. Cause: %s",
            type(error).__name__,
        )
        return None

    if not project_metadata.active_learning_configuration.get("enabled", False):
        return None

    logger.info(
        "Configuring active learning for project: %s, enabled: %s, "
        "sampling strategies: %d",
        target_dataset,
        True,
        len(
            project_metadata.active_learning_configuration.get(
                "sampling_strategies", []
            )
        ),
    )
    configuration = initialise_active_learning_configuration(
        project_metadata=project_metadata,
        model_id=model_id,
    )

    return configuration


def prepare_active_learning_configuration_inplace(
    api_key: str,
    target_dataset: str,
    model_id: str,
    active_learning_configuration: Optional[dict],
    platform_client: "ServerRoboflowPlatformClient",
) -> Optional[ActiveLearningConfiguration]:
    """Build the configuration from a document the caller already holds.

    Args:
        api_key: Roboflow API key of the request.
        target_dataset: Project datapoints are registered in.
        model_id: Model whose predictions are registered.
        active_learning_configuration: Configuration document, or None.
        platform_client: Roboflow platform client of the server.

    Returns:
        The configuration, or None when the document is missing, disabled, or
        the model and the project have incompatible types.

    Raises:
        RoboflowAPIRequestError: If the platform cannot be reached or rejects
            a request.
        ActiveLearningConfigurationError: If the configuration is inconsistent.
        ActiveLearningConfigurationDecodingError: If the configuration cannot
            be decoded.
    """
    if (
        active_learning_configuration is None
        or active_learning_configuration.get("enabled", False) is False
    ):
        return None

    workspace_id = platform_client.get_roboflow_workspace(api_key=api_key)
    dataset_type = platform_client.get_roboflow_dataset_type(
        api_key=api_key,
        workspace_id=workspace_id,
        dataset_id=target_dataset,
    )
    model_type = dataset_type
    if not model_id.startswith(target_dataset):
        model_type = get_model_type(
            model_id=model_id, api_key=api_key, platform_client=platform_client
        )
    if predictions_incompatible_with_dataset(
        model_type=model_type, dataset_type=dataset_type
    ):
        logger.warning(
            "Attempted to register predictions from model %s into dataset %s "
            "which have incompatible types.",
            model_id,
            target_dataset,
        )
        return None

    project_metadata = RoboflowProjectMetadata(
        dataset_id=target_dataset,
        workspace_id=workspace_id,
        dataset_type=dataset_type,
        active_learning_configuration=active_learning_configuration,
    )
    configuration = initialise_active_learning_configuration(
        project_metadata=project_metadata,
        model_id=model_id,
    )

    return configuration


def get_roboflow_project_metadata(
    api_key: str,
    target_dataset: str,
    model_id: str,
    cache: Any,
    platform_client: "ServerRoboflowPlatformClient",
) -> RoboflowProjectMetadata:
    """Load the project metadata, from the cache when it holds them.

    Args:
        api_key: Roboflow API key of the request.
        target_dataset: Project datapoints are registered in.
        model_id: Model whose predictions are registered.
        cache: Cache with ``get`` and ``set(key, value, expire)``.
        platform_client: Roboflow platform client of the server.

    Returns:
        The project metadata; its configuration is ``{"enabled": False}`` when
        the model and the project have incompatible types or the key may not
        read the configuration.

    Raises:
        RoboflowAPIRequestError: If the platform cannot be reached or rejects
            a request.
        ActiveLearningConfigurationDecodingError: If the cached entry is
            malformed.
    """
    logger.info("Fetching active learning configuration.")
    config_cache_key = construct_cache_key_for_active_learning_config(
        api_key=api_key,
        target_dataset=target_dataset,
        model_id=model_id,
    )
    cached_config = cache.get(config_cache_key)
    if cached_config is not None:
        logger.info("Found Active Learning configuration in cache.")
        cached_metadata = parse_cached_roboflow_project_metadata(
            cached_config=cached_config
        )

        return cached_metadata

    workspace_id = platform_client.get_roboflow_workspace(api_key=api_key)
    dataset_type = platform_client.get_roboflow_dataset_type(
        api_key=api_key,
        workspace_id=workspace_id,
        dataset_id=target_dataset,
    )
    model_type = dataset_type
    if not model_id.startswith(target_dataset):
        model_type = get_model_type(
            model_id=model_id, api_key=api_key, platform_client=platform_client
        )
    if predictions_incompatible_with_dataset(
        model_type=model_type, dataset_type=dataset_type
    ):
        logger.warning(
            "Attempted to register predictions from model %s into dataset %s "
            "which have incompatible types.",
            model_id,
            target_dataset,
        )
        roboflow_api_configuration = {"enabled": False}
    else:
        roboflow_api_configuration = safe_get_roboflow_active_learning_configuration(
            api_key=api_key,
            workspace_id=workspace_id,
            dataset_id=target_dataset,
            platform_client=platform_client,
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
    """Build the cache key of a project configuration.

    Args:
        api_key: Roboflow API key of the request; only its MD5 enters the key.
        target_dataset: Project datapoints are registered in.
        model_id: Model whose predictions are registered.

    Returns:
        The cache key.
    """
    api_key_hash = hashlib.md5(api_key.encode("utf-8")).hexdigest()

    return f"active_learning:configurations:{api_key_hash}:{target_dataset}:{model_id}"


def parse_cached_roboflow_project_metadata(
    cached_config: dict,
) -> RoboflowProjectMetadata:
    """Decode the project metadata kept in the cache.

    Args:
        cached_config: Cached entry.

    Returns:
        The project metadata.

    Raises:
        ActiveLearningConfigurationDecodingError: If the entry is malformed.
    """
    try:
        project_metadata = RoboflowProjectMetadata(
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

    return project_metadata


def get_model_type(
    model_id: str, api_key: str, platform_client: "ServerRoboflowPlatformClient"
) -> str:
    """Fetch the task type of the project a model belongs to.

    Args:
        model_id: Model identifier, ``project/version``.
        api_key: Roboflow API key of the request.
        platform_client: Roboflow platform client of the server.

    Returns:
        The project type of the model.

    Raises:
        RoboflowAPIRequestError: If the platform cannot be reached or rejects
            a request.
    """
    model_dataset = model_id.split("/")[0]
    model_workspace = platform_client.get_roboflow_workspace(api_key=api_key)
    model_type = platform_client.get_roboflow_dataset_type(
        api_key=api_key,
        workspace_id=model_workspace,
        dataset_id=model_dataset,
    )

    return model_type


def predictions_incompatible_with_dataset(
    model_type: str,
    dataset_type: str,
) -> bool:
    """Tell whether predictions of a model cannot be registered in a project.

    Classification and detection do not mix; detection-like types are
    compatible with each other.

    Args:
        model_type: Project type of the model.
        dataset_type: Project type of the target project.

    Returns:
        True when exactly one of the two types is a classification type.
    """
    model_is_classifier = CLASSIFICATION_TASK in model_type
    dataset_is_of_type_classification = CLASSIFICATION_TASK in dataset_type

    return model_is_classifier != dataset_is_of_type_classification


def safe_get_roboflow_active_learning_configuration(
    api_key: str,
    workspace_id: str,
    dataset_id: str,
    platform_client: "ServerRoboflowPlatformClient",
) -> dict:
    """Fetch the configuration, treating a refused key as disabled.

    Args:
        api_key: Roboflow API key of the request.
        workspace_id: Workspace of the key.
        dataset_id: Project the configuration belongs to.
        platform_client: Roboflow platform client of the server.

    Returns:
        The configuration document, ``{"enabled": False}`` when the platform
        answers not authorised or not found.

    Raises:
        RoboflowAPIRequestError: If the platform cannot be reached or rejects
            the request for another reason.
    """
    try:
        active_learning_configuration = (
            platform_client.get_roboflow_active_learning_configuration(
                api_key=api_key, workspace_id=workspace_id, dataset_id=dataset_id
            )
        )
    except (RoboflowAPINotAuthorizedError, RoboflowAPINotNotFoundError):
        return {"enabled": False}

    return active_learning_configuration


def initialise_active_learning_configuration(
    project_metadata: RoboflowProjectMetadata,
    model_id: str,
) -> ActiveLearningConfiguration:
    """Build the configuration from the project metadata.

    Args:
        project_metadata: Metadata of the target project.
        model_id: Model whose predictions are registered.

    Returns:
        The configuration, registering into ``target_workspace`` and
        ``target_project`` of the document when it names them.

    Raises:
        ActiveLearningConfigurationError: If the configuration is inconsistent.
        ActiveLearningConfigurationDecodingError: If the configuration cannot
            be decoded.
    """
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
    configuration = ActiveLearningConfiguration.init(
        roboflow_api_configuration=project_metadata.active_learning_configuration,
        sampling_methods=sampling_methods,
        workspace_id=target_workspace_id,
        dataset_id=target_dataset_id,
        model_id=model_id,
    )

    return configuration


def initialize_sampling_methods(
    sampling_strategies_configs: List[Dict[str, Any]],
) -> List[SamplingMethod]:
    """Build the sampling methods of the configured strategies.

    Strategies of an unknown type are skipped.

    Args:
        sampling_strategies_configs: Strategy configurations of the project.

    Returns:
        The sampling methods, in the order of the configuration.

    Raises:
        ActiveLearningConfigurationError: If two strategies share a name or a
            strategy misses a required key.
    """
    result = []
    for sampling_strategy_config in sampling_strategies_configs:
        sampling_type = sampling_strategy_config["type"]
        if sampling_type not in TYPE2SAMPLING_INITIALIZERS:
            logger.warning(
                "Could not identify sampling method - skipping initialisation."
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

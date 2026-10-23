"""Sampling of a datapoint and its registration at Roboflow."""

import logging
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, List, Optional
from uuid import uuid4

import numpy as np
from roboflow_workflows.core_steps.sinks.roboflow.dataset_upload.active_learning import (
    PreparedRegistrationImage,
    prepare_image_to_registration_with_metadata,
)

from inference_server import configuration as server_configuration
from inference_server.active_learning.cache_operations import (
    return_strategy_credit,
    use_credit_of_matching_strategy,
)
from inference_server.active_learning.entities import (
    ActiveLearningConfiguration,
    Prediction,
    PredictionType,
    SamplingMethod,
)
from inference_server.active_learning.post_processing import (
    adjust_prediction_to_client_scaling_factor,
    encode_prediction,
    prediction_has_supported_annotation_format,
)

if TYPE_CHECKING:
    from inference_server.workflows.host import ServerRoboflowPlatformClient

__all__ = [
    "PreparedRegistrationImage",
    "collect_tags",
    "execute_datapoint_registration",
    "execute_sampling",
    "is_prediction_registration_forbidden",
    "prepare_image_to_registration_with_metadata",
    "register_datapoint_at_roboflow",
    "safe_register_image_at_roboflow",
]

logger = logging.getLogger(__name__)


def execute_sampling(
    image: np.ndarray,
    prediction: Prediction,
    prediction_type: PredictionType,
    sampling_methods: List[SamplingMethod],
) -> List[str]:
    """Run every sampling method on a datapoint.

    Args:
        image: Decoded image the prediction was made for.
        prediction: Prediction in the response format of the server.
        prediction_type: Task type of the model.
        sampling_methods: Sampling methods of the configuration.

    Returns:
        Names of the strategies that selected the datapoint, in order.
    """
    matching_strategies = []
    for method in sampling_methods:
        sampling_result = method.sample(image, prediction, prediction_type)
        if sampling_result:
            matching_strategies.append(method.name)

    return matching_strategies


def execute_datapoint_registration(
    cache: Any,
    matching_strategies: List[str],
    image: np.ndarray,
    prediction: Prediction,
    prediction_type: PredictionType,
    configuration: ActiveLearningConfiguration,
    api_key: str,
    batch_name: str,
    platform_client: "ServerRoboflowPlatformClient",
    inference_id: Optional[str] = None,
) -> None:
    """Register a sampled datapoint when a matching strategy has credit left.

    The prediction is rescaled in place to the size of the registered image.

    Args:
        cache: Cache holding the strategy usage counters.
        matching_strategies: Strategies that selected the datapoint.
        image: Decoded BGR image.
        prediction: Prediction in the response format of the server.
        prediction_type: Task type of the model.
        configuration: Active learning configuration of the target project.
        api_key: Roboflow API key of the request.
        batch_name: Labeling batch the image goes to.
        platform_client: Roboflow platform client of the server.
        inference_id: Identifier of the inference, sent with the image.

    Raises:
        ValueError: If the image has no pixels.
        RoboflowAPIRequestError: If the platform cannot be reached or rejects
            a request.
    """
    local_image_id = str(uuid4())
    prepared_image = prepare_image_to_registration_with_metadata(
        image=image,
        desired_size=configuration.max_image_size,
        jpeg_compression_level=configuration.jpeg_compression_level,
    )
    prediction = adjust_prediction_to_client_scaling_factor(
        prediction=prediction,
        scaling_factor=prepared_image.scaling_factor,
        prediction_type=prediction_type,
    )
    matching_strategies_limits = OrderedDict(
        (strategy_name, configuration.strategies_limits[strategy_name])
        for strategy_name in matching_strategies
    )
    strategy_with_spare_credit = use_credit_of_matching_strategy(
        cache=cache,
        workspace=configuration.workspace_id,
        project=configuration.dataset_id,
        matching_strategies_limits=matching_strategies_limits,
    )
    if strategy_with_spare_credit is None:
        logger.debug("Limit on Active Learning strategy reached.")
        return None

    register_datapoint_at_roboflow(
        cache=cache,
        strategy_with_spare_credit=strategy_with_spare_credit,
        encoded_image=prepared_image.encoded_image,
        local_image_id=local_image_id,
        prediction=prediction,
        prediction_type=prediction_type,
        configuration=configuration,
        api_key=api_key,
        batch_name=batch_name,
        inference_id=inference_id,
        platform_client=platform_client,
    )


def register_datapoint_at_roboflow(
    cache: Any,
    strategy_with_spare_credit: str,
    encoded_image: bytes,
    local_image_id: str,
    prediction: Prediction,
    prediction_type: PredictionType,
    configuration: ActiveLearningConfiguration,
    api_key: str,
    batch_name: str,
    inference_id: Optional[str],
    platform_client: "ServerRoboflowPlatformClient",
) -> None:
    """Upload an image and, when allowed, its prediction as an annotation.

    A prediction without a supported annotation format leaves the image
    registered without an annotation.

    Args:
        cache: Cache holding the strategy usage counters.
        strategy_with_spare_credit: Strategy whose credit was consumed.
        encoded_image: JPEG bytes of the image.
        local_image_id: Identifier the image and annotation files are named by.
        prediction: Prediction in the response format of the server.
        prediction_type: Task type of the model.
        configuration: Active learning configuration of the target project.
        api_key: Roboflow API key of the request.
        batch_name: Labeling batch the image goes to.
        inference_id: Identifier of the inference, sent with the image.
        platform_client: Roboflow platform client of the server.

    Raises:
        RoboflowAPIRequestError: If the platform cannot be reached or rejects
            a request.
    """
    tags = collect_tags(
        configuration=configuration,
        sampling_strategy=strategy_with_spare_credit,
    )
    roboflow_image_id = safe_register_image_at_roboflow(
        cache=cache,
        strategy_with_spare_credit=strategy_with_spare_credit,
        encoded_image=encoded_image,
        local_image_id=local_image_id,
        configuration=configuration,
        api_key=api_key,
        batch_name=batch_name,
        tags=tags,
        inference_id=inference_id,
        platform_client=platform_client,
    )
    if is_prediction_registration_forbidden(
        prediction=prediction,
        persist_predictions=configuration.persist_predictions,
        roboflow_image_id=roboflow_image_id,
    ):
        return None

    if not prediction_has_supported_annotation_format(
        prediction=prediction, prediction_type=prediction_type
    ):
        logger.warning(
            "Image registered without annotation: prediction format of task "
            "type %s is not supported.",
            prediction_type,
        )
        return None

    encoded_prediction, prediction_file_type = encode_prediction(
        prediction=prediction, prediction_type=prediction_type
    )
    _ = platform_client.annotate_image_at_roboflow(
        api_key=api_key,
        dataset_id=configuration.dataset_id,
        local_image_id=local_image_id,
        roboflow_image_id=roboflow_image_id,
        annotation_content=encoded_prediction,
        annotation_file_type=prediction_file_type,
        is_prediction=True,
    )


def collect_tags(
    configuration: ActiveLearningConfiguration, sampling_strategy: str
) -> List[str]:
    """Collect the tags a registered image carries.

    Args:
        configuration: Active learning configuration of the target project.
        sampling_strategy: Strategy whose credit was consumed.

    Returns:
        Tags of ``ACTIVE_LEARNING_TAGS``, of the configuration and of the
        strategy, followed by the model identifier when predictions are kept.
    """
    env_tags = server_configuration.ACTIVE_LEARNING_TAGS
    tags = list(env_tags) if env_tags is not None else []
    tags.extend(configuration.tags)
    tags.extend(configuration.strategies_tags[sampling_strategy])
    if configuration.persist_predictions:
        tags.append(configuration.model_id.replace("/", "-"))

    return tags


def safe_register_image_at_roboflow(
    cache: Any,
    strategy_with_spare_credit: str,
    encoded_image: bytes,
    local_image_id: str,
    configuration: ActiveLearningConfiguration,
    api_key: str,
    batch_name: str,
    tags: List[str],
    inference_id: Optional[str],
    platform_client: "ServerRoboflowPlatformClient",
) -> Optional[str]:
    """Upload an image, returning the strategy credit when nothing was added.

    Args:
        cache: Cache holding the strategy usage counters.
        strategy_with_spare_credit: Strategy whose credit was consumed.
        encoded_image: JPEG bytes of the image.
        local_image_id: Identifier the image file is named by.
        configuration: Active learning configuration of the target project.
        api_key: Roboflow API key of the request.
        batch_name: Labeling batch the image goes to.
        tags: Tags the image carries.
        inference_id: Identifier of the inference, sent with the image.
        platform_client: Roboflow platform client of the server.

    Returns:
        The Roboflow image identifier, None when the image is a duplicate.

    Raises:
        RoboflowAPIRequestError: If the platform cannot be reached or rejects
            the upload.
    """
    credit_to_be_returned = False
    try:
        registration_response = platform_client.register_image_at_roboflow(
            api_key=api_key,
            dataset_id=configuration.dataset_id,
            local_image_id=local_image_id,
            image_bytes=encoded_image,
            batch_name=batch_name,
            tags=tags,
            inference_id=inference_id,
        )
        image_duplicated = registration_response.get("duplicate", False)
        if image_duplicated:
            credit_to_be_returned = True
            logger.warning("Image duplication detected.")
            return None

        return registration_response["id"]
    except Exception as error:
        credit_to_be_returned = True
        raise error
    finally:
        if credit_to_be_returned:
            return_strategy_credit(
                cache=cache,
                workspace=configuration.workspace_id,
                project=configuration.dataset_id,
                strategy_name=strategy_with_spare_credit,
            )


def is_prediction_registration_forbidden(
    prediction: Prediction,
    persist_predictions: bool,
    roboflow_image_id: Optional[str],
) -> bool:
    """Tell whether a prediction must not be registered as an annotation.

    Args:
        prediction: Prediction in the response format of the server.
        persist_predictions: Whether the configuration keeps predictions.
        roboflow_image_id: Identifier of the registered image, None when the
            image was not added.

    Returns:
        True when the image was not added, predictions are not kept, the
        prediction is a stub, or it is empty.
    """
    return (
        roboflow_image_id is None
        or persist_predictions is False
        or prediction.get("is_stub", False) is True
        or (len(prediction.get("predictions", [])) == 0 and "top" not in prediction)
    )

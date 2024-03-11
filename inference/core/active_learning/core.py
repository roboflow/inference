from collections import OrderedDict
from typing import List, Optional, Tuple
from uuid import uuid4

import numpy as np

from inference.core import logger
from inference.core.active_learning.cache_operations import (
    return_strategy_credit,
    use_credit_of_matching_strategy,
)
from inference.core.active_learning.entities import (
    ActiveLearningConfiguration,
    ImageDimensions,
    Prediction,
    PredictionType,
    SamplingMethod,
)
from inference.core.active_learning.post_processing import (
    adjust_prediction_to_client_scaling_factor,
    encode_prediction,
)
from inference.core.cache.base import BaseCache
from inference.core.env import ACTIVE_LEARNING_TAGS
from inference.core.roboflow_api import (
    annotate_image_at_roboflow,
    register_image_at_roboflow,
)
from inference.core.utils.image_utils import encode_image_to_jpeg_bytes
from inference.core.utils.preprocess import downscale_image_keeping_aspect_ratio


def execute_sampling(
    image: np.ndarray,
    prediction: Prediction,
    prediction_type: PredictionType,
    sampling_methods: List[SamplingMethod],
) -> List[str]:
    matching_strategies = []
    for method in sampling_methods:
        sampling_result = method.sample(image, prediction, prediction_type)
        if sampling_result:
            matching_strategies.append(method.name)
    return matching_strategies


def execute_datapoint_registration(
    cache: BaseCache,
    matching_strategies: List[str],
    image: np.ndarray,
    prediction: Prediction,
    prediction_type: PredictionType,
    configuration: ActiveLearningConfiguration,
    api_key: str,
    batch_name: str,
    inference_id: Optional[str] = None,
) -> None:
    local_image_id = str(uuid4())
    encoded_image, scaling_factor = prepare_image_to_registration(
        image=image,
        desired_size=configuration.max_image_size,
        jpeg_compression_level=configuration.jpeg_compression_level,
    )
    prediction = adjust_prediction_to_client_scaling_factor(
        prediction=prediction,
        scaling_factor=scaling_factor,
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
        logger.debug(f"Limit on Active Learning strategy reached.")
        return None
    register_datapoint_at_roboflow(
        cache=cache,
        strategy_with_spare_credit=strategy_with_spare_credit,
        encoded_image=encoded_image,
        local_image_id=local_image_id,
        prediction=prediction,
        prediction_type=prediction_type,
        configuration=configuration,
        api_key=api_key,
        batch_name=batch_name,
        inference_id=inference_id,
    )


def prepare_image_to_registration(
    image: np.ndarray,
    desired_size: Optional[ImageDimensions],
    jpeg_compression_level: int,
) -> Tuple[bytes, float]:
    scaling_factor = 1.0
    if desired_size is not None:
        height_before_scale = image.shape[0]
        image = downscale_image_keeping_aspect_ratio(
            image=image,
            desired_size=desired_size.to_wh(),
        )
        scaling_factor = image.shape[0] / height_before_scale
    return (
        encode_image_to_jpeg_bytes(image=image, jpeg_quality=jpeg_compression_level),
        scaling_factor,
    )


def register_datapoint_at_roboflow(
    cache: BaseCache,
    strategy_with_spare_credit: str,
    encoded_image: bytes,
    local_image_id: str,
    prediction: Prediction,
    prediction_type: PredictionType,
    configuration: ActiveLearningConfiguration,
    api_key: str,
    batch_name: str,
    inference_id: Optional[str],
) -> None:
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
    )
    if is_prediction_registration_forbidden(
        prediction=prediction,
        persist_predictions=configuration.persist_predictions,
        roboflow_image_id=roboflow_image_id,
    ):
        return None
    encoded_prediction, prediction_file_type = encode_prediction(
        prediction=prediction, prediction_type=prediction_type
    )
    _ = annotate_image_at_roboflow(
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
    tags = ACTIVE_LEARNING_TAGS if ACTIVE_LEARNING_TAGS is not None else []
    tags.extend(configuration.tags)
    tags.extend(configuration.strategies_tags[sampling_strategy])
    if configuration.persist_predictions:
        # this replacement is needed due to backend input validation
        tags.append(configuration.model_id.replace("/", "-"))
    return tags


def safe_register_image_at_roboflow(
    cache: BaseCache,
    strategy_with_spare_credit: str,
    encoded_image: bytes,
    local_image_id: str,
    configuration: ActiveLearningConfiguration,
    api_key: str,
    batch_name: str,
    tags: List[str],
    inference_id: Optional[str],
) -> Optional[str]:
    credit_to_be_returned = False
    try:
        registration_response = register_image_at_roboflow(
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
            logger.warning(f"Image duplication detected: {registration_response}.")
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
    return (
        roboflow_image_id is None
        or persist_predictions is False
        or prediction.get("is_stub", False) is True
        or (len(prediction.get("predictions", [])) == 0 and "top" not in prediction)
    )

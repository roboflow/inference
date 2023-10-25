import json
from collections import OrderedDict
from typing import Any, List, Optional
from uuid import uuid4

import numpy as np

from inference.core import logger
from inference.core.active_learning.accounting import image_can_be_submitted_to_batch
from inference.core.active_learning.batching import generate_batch_name
from inference.core.active_learning.cache import (
    find_strategy_with_spare_usage_limit,
    generate_cache_key_for_active_learning_usage_lock,
    MAX_LOCK_TIME,
    increment_strategy_usage,
    increment_strategies_usage,
)
from inference.core.active_learning.configuration import (
    prepare_active_learning_configuration,
)
from inference.core.active_learning.entities import (
    ActiveLearningConfiguration,
    SamplingMethod,
    Prediction,
    PredictionType,
    ImageDimensions,
)
from inference.core.cache.base import BaseCache
from inference.core.env import ACTIVE_LEARNING_TAGS
from inference.core.roboflow_api import (
    register_image_at_roboflow,
    annotate_image_at_roboflow,
)
from inference.core.utils.image_utils import encode_image_to_jpeg_bytes, load_image
from inference.core.utils.preprocess import downscale_image_keeping_aspect_ratio


class ActiveLearningMiddleware:
    @classmethod
    def init(
        cls, api_key: str, model_id: str, cache: BaseCache
    ) -> "ActiveLearningMiddleware":
        configuration = prepare_active_learning_configuration(
            api_key=api_key,
            model_id=model_id,
        )
        return cls(
            api_key=api_key,
            configuration=configuration,
            cache=cache,
        )

    def __init__(
        self,
        api_key: str,
        configuration: ActiveLearningConfiguration,
        cache: BaseCache,
    ):
        self._api_key = api_key
        self._configuration = configuration
        self._cache = cache

    def register_batch(
        self,
        inference_inputs: List[Any],
        predictions: List[Prediction],
        prediction_type: PredictionType,
        disable_preproc_auto_orient: bool = False,
    ) -> None:
        for inference_input, prediction in zip(inference_inputs, predictions):
            self.register(
                inference_input=inference_input,
                prediction=prediction,
                prediction_type=prediction_type,
                disable_preproc_auto_orient=disable_preproc_auto_orient,
            )

    def register(
        self,
        inference_input: Any,
        prediction: dict,
        prediction_type: PredictionType,
        disable_preproc_auto_orient: bool = False,
    ) -> None:
        image, is_bgr = load_image(
            value=inference_input,
            disable_preproc_auto_orient=disable_preproc_auto_orient,
        )
        if not is_bgr:
            image = image[:, :, ::-1]
        matching_strategies = execute_sampling(
            image=image,
            prediction=prediction,
            prediction_type=prediction_type,
            sampling_methods=self._configuration.sampling_methods,
        )
        if len(matching_strategies) == 0:
            return None
        batch_name = generate_batch_name(configuration=self._configuration)
        if not image_can_be_submitted_to_batch(
            batch_name=batch_name,
            workspace_id=self._configuration.workspace_id,
            dataset_id=self._configuration.dataset_id,
            max_batch_images=self._configuration.max_batch_images,
            api_key=self._api_key,
        ):
            logger.warning(f"Limit on Active Learning batch size reached.")
            return None
        execute_datapoint_registration(
            cache=self._cache,
            matching_strategies=matching_strategies,
            image=image,
            prediction=prediction,
            configuration=self._configuration,
            api_key=self._api_key,
            batch_name=batch_name,
        )


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
    configuration: ActiveLearningConfiguration,
    api_key: str,
    batch_name: str,
) -> None:
    local_image_id = str(uuid4())
    encoded_image = prepare_image_to_registration(
        image=image,
        desired_size=configuration.max_image_size,
        jpeg_compression_level=configuration.jpeg_compression_level,
    )
    limits_lock_key = generate_cache_key_for_active_learning_usage_lock(
        workspace=configuration.workspace_id,
        project=configuration.dataset_id,
    )
    matching_strategies_limits = OrderedDict(
        (strategy_name, configuration.strategies_limits[strategy_name])
        for strategy_name in matching_strategies
    )
    with cache.lock(key=limits_lock_key, expire=MAX_LOCK_TIME):
        strategy_with_spare_limit = find_strategy_with_spare_usage_limit(
            cache=cache,
            workspace=configuration.workspace_id,
            project=configuration.dataset_id,
            matching_strategies_limits=matching_strategies_limits,
        )
        if strategy_with_spare_limit is None:
            logger.warning(f"Limit on Active Learning strategy reached.")
            return None
        tags = collect_tags(
            configuration=configuration, sampling_strategy=strategy_with_spare_limit
        )
        register_datapoint_at_roboflow(
            cache=cache,
            strategy_with_spare_limit=strategy_with_spare_limit,
            encoded_image=encoded_image,
            local_image_id=local_image_id,
            prediction=prediction,
            configuration=configuration,
            api_key=api_key,
            batch_name=batch_name,
            tags=tags,
        )


def prepare_image_to_registration(
    image: np.ndarray,
    desired_size: Optional[ImageDimensions],
    jpeg_compression_level: int,
) -> bytes:
    if desired_size is not None:
        image = downscale_image_keeping_aspect_ratio(
            image=image,
            desired_size=desired_size.to_wh(),
        )
    return encode_image_to_jpeg_bytes(image=image, jpeg_quality=jpeg_compression_level)


def collect_tags(
    configuration: ActiveLearningConfiguration, sampling_strategy: str
) -> List[str]:
    tags = ACTIVE_LEARNING_TAGS if ACTIVE_LEARNING_TAGS is not None else []
    tags.extend(configuration.tags)
    tags.extend(configuration.strategies_tags[sampling_strategy])
    if configuration.persist_predictions:
        tags.append(configuration.model_id.replace("/", "-"))
    return tags


def register_datapoint_at_roboflow(
    cache: BaseCache,
    strategy_with_spare_limit: str,
    encoded_image: bytes,
    local_image_id: str,
    prediction: Prediction,
    configuration: ActiveLearningConfiguration,
    api_key: str,
    batch_name: str,
    tags: List[str],
) -> None:
    registration_response = register_image_at_roboflow(
        api_key=api_key,
        dataset_id=configuration.dataset_id,
        local_image_id=local_image_id,
        image_bytes=encoded_image,
        batch_name=batch_name,
        tags=tags,
    )
    duplication_status = registration_response.get("duplicate", False)
    if duplication_status is False:
        increment_strategies_usage(
            cache=cache,
            workspace=configuration.workspace_id,
            project=configuration.dataset_id,
            strategy_name=strategy_with_spare_limit,
        )
    logger.info(f"Image duplication status: {duplication_status}")
    if configuration.persist_predictions and not duplication_status:
        encoded_prediction = json.dumps(prediction)
        _ = annotate_image_at_roboflow(
            api_key=api_key,
            dataset_id=configuration.dataset_id,
            local_image_id=local_image_id,
            roboflow_image_id=registration_response["id"],
            annotation_content=encoded_prediction,
            is_prediction=True,
        )

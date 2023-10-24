from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from inference.core import logger
from inference.core.active_learning.accounting import image_can_be_submitted_to_batch
from inference.core.active_learning.batching import generate_batch_name
from inference.core.active_learning.entities import (
    LocalImageIdentifier,
    ActiveLearningConfiguration,
    SamplingMethod,
    SamplingResult,
    Prediction,
    PredictionType,
    ImageDimensions,
)
from inference.core.env import ACTIVE_LEARNING_TAGS
from inference.core.roboflow_api import (
    register_image_at_roboflow,
    annotate_image_at_roboflow,
)
from inference.core.utils.image_utils import encode_image_to_jpeg_bytes
from inference.core.utils.preprocess import downscale_image_keeping_aspect_ratio


class BaseActiveLearningMiddleware:
    def __int__(
        self,
        api_key: str,
        configuration: ActiveLearningConfiguration,
        registered_images: Dict[str, np.ndarray],
    ):
        self._api_key = api_key
        self._configuration = configuration
        self._registered_images = registered_images

    def register_image(self, image_id: LocalImageIdentifier, image: np.ndarray) -> None:
        self._registered_images[image_id] = image

    def register_prediction(
        self,
        image_id: LocalImageIdentifier,
        prediction: Any,
        prediction_type: str,
    ) -> None:
        if image_id not in self._registered_images:
            logger.warn(
                f"Could not find image with id {image_id} registered in Active Learning middleware."
            )
            return None
        image = self._registered_images[image_id]
        del self._registered_images[image_id]
        sampling_details = execute_sampling(
            image=image,
            prediction=prediction,
            prediction_type=prediction_type,
            sampling_methods=self._configuration.sampling_methods,
        )
        if sampling_details is None:
            return None
        sampling_method, sampling_result = sampling_details
        if sampling_result.datapoint_selected is False:
            return None
        batch_name = generate_batch_name(configuration=self._configuration)
        if not image_can_be_submitted_to_batch(
            batch_name=batch_name,
            workspace_id=self._configuration.workspace_id,
            dataset_id=self._configuration.dataset_id,
            max_batch_images=self._configuration.max_batch_images,
            api_key=self._api_key,
        ):
            return None
        env_defined_tags = (
            ACTIVE_LEARNING_TAGS if ACTIVE_LEARNING_TAGS is not None else []
        )
        execute_datapoint_registration(
            image=image,
            local_image_id=image_id,
            prediction=prediction,
            prediction_type=prediction_type,
            configuration=self._configuration,
            api_key=self._api_key,
            split=sampling_result.target_split,
            batch_name=batch_name,
            tags=[sampling_method] + env_defined_tags,
        )


def execute_sampling(
    image: np.ndarray,
    prediction: Prediction,
    prediction_type: PredictionType,
    sampling_methods: List[SamplingMethod],
) -> Optional[Tuple[str, SamplingResult]]:
    for method in sampling_methods:
        sampling_result = method.sample(image, prediction, prediction_type)
        if sampling_result.datapoint_selected:
            return method.name, sampling_result
    return None


def execute_datapoint_registration(
    image: np.ndarray,
    local_image_id: str,
    prediction: Prediction,
    prediction_type: PredictionType,
    configuration: ActiveLearningConfiguration,
    api_key: str,
    split: str,
    batch_name: str,
    tags: List[str],
) -> None:
    encoded_image = prepare_image_to_registration(
        image=image,
        desired_size=configuration.max_image_size,
        jpeg_compression_level=configuration.jpeg_compression_level,
    )
    registration_response = register_image_at_roboflow(
        api_key=api_key,
        dataset_id=configuration.dataset_id,
        local_image_id=local_image_id,
        image_bytes=encoded_image,
        split=split,
        batch_name=batch_name,
        tags=tags,
    )
    if configuration.persist_predictions:
        encoded_prediction = prepare_prediction_to_registration(
            prediction=prediction,
            prediction_type=prediction_type,
        )
        _ = annotate_image_at_roboflow(
            api_key=api_key,
            dataset_id=configuration.dataset_id,
            local_image_id=local_image_id,
            roboflow_image_id=registration_response["id"],
            annotation_content=encoded_prediction,
            is_prediction=True,
        )


def prepare_image_to_registration(
    image: np.ndarray,
    desired_size: ImageDimensions,
    jpeg_compression_level: int,
) -> bytes:
    resized_image = downscale_image_keeping_aspect_ratio(
        image=image,
        desired_size=desired_size.to_wh(),
    )
    return encode_image_to_jpeg_bytes(
        image=resized_image, jpeg_quality=jpeg_compression_level
    )


def prepare_prediction_to_registration(
    prediction: Prediction, prediction_type: PredictionType
) -> str:
    # TODO: implement
    return ""

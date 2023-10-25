import json
import logging
from typing import Any, List, Optional, Tuple
from uuid import uuid4

import numpy as np

from inference.core import logger
from inference.core.active_learning.accounting import image_can_be_submitted_to_batch
from inference.core.active_learning.batching import generate_batch_name
from inference.core.active_learning.configuration import (
    prepare_active_learning_configuration,
)
from inference.core.active_learning.entities import (
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
from inference.core.utils.image_utils import encode_image_to_jpeg_bytes, load_image
from inference.core.utils.preprocess import downscale_image_keeping_aspect_ratio


class ActiveLearningMiddleware:
    @classmethod
    def init(cls, api_key: str, model_id: str) -> "ActiveLearningMiddleware":
        configuration = prepare_active_learning_configuration(
            api_key=api_key,
            model_id=model_id,
        )
        return cls(
            api_key=api_key,
            configuration=configuration,
        )

    def __init__(self, api_key: str, configuration: ActiveLearningConfiguration):
        self._api_key = api_key
        self._configuration = configuration

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
        local_image_id = str(uuid4())
        execute_datapoint_registration(
            image=image,
            local_image_id=local_image_id,
            prediction=prediction,
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
    if configuration.persist_predictions:
        tags.append(configuration.model_id)
    registration_response = register_image_at_roboflow(
        api_key=api_key,
        dataset_id=configuration.dataset_id,
        local_image_id=local_image_id,
        image_bytes=encoded_image,
        split=split,
        batch_name=batch_name,
        tags=tags,
    )
    duplication_status = registration_response.get("duplicate", False)
    logger.info(f"Image duplication status: {duplication_status}")
    if configuration.persist_predictions and duplication_status:
        encoded_prediction = json.dumps(prediction)
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
    desired_size: Optional[ImageDimensions],
    jpeg_compression_level: int,
) -> bytes:
    if desired_size is not None:
        image = downscale_image_keeping_aspect_ratio(
            image=image,
            desired_size=desired_size.to_wh(),
        )
    return encode_image_to_jpeg_bytes(image=image, jpeg_quality=jpeg_compression_level)

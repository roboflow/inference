"""Registration of predictions of one model into one target project."""

import logging
from typing import TYPE_CHECKING, Any, List, Optional

import numpy as np

from inference_server.active_learning.accounting import image_can_be_submitted_to_batch
from inference_server.active_learning.batching import generate_batch_name
from inference_server.active_learning.configuration import (
    prepare_active_learning_configuration,
    prepare_active_learning_configuration_inplace,
)
from inference_server.active_learning.core import (
    execute_datapoint_registration,
    execute_sampling,
)
from inference_server.active_learning.entities import (
    ActiveLearningConfiguration,
    Prediction,
    PredictionType,
)

if TYPE_CHECKING:
    from inference_server.workflows.host import ServerRoboflowPlatformClient

logger = logging.getLogger(__name__)


class ActiveLearningMiddleware:
    """Holds the configuration of one target and registers datapoints into it."""

    @classmethod
    def init(
        cls,
        api_key: str,
        target_dataset: str,
        model_id: str,
        cache: Any,
        platform_client: "ServerRoboflowPlatformClient",
    ) -> "ActiveLearningMiddleware":
        """Build the middleware from the configuration kept at Roboflow.

        Args:
            api_key: Roboflow API key used for every platform call.
            target_dataset: Project datapoints are registered in.
            model_id: Model whose predictions are registered.
            cache: Cache with ``get`` and ``set(key, value, expire)``.
            platform_client: Roboflow platform client of the server.

        Returns:
            The middleware, inactive when active learning is disabled for the
            project or its metadata could not be loaded.

        Raises:
            ActiveLearningConfigurationError: If the configuration is
                inconsistent.
            ActiveLearningConfigurationDecodingError: If the configuration
                cannot be decoded.
        """
        configuration = prepare_active_learning_configuration(
            api_key=api_key,
            target_dataset=target_dataset,
            model_id=model_id,
            cache=cache,
            platform_client=platform_client,
        )
        middleware = cls(
            api_key=api_key,
            configuration=configuration,
            cache=cache,
            platform_client=platform_client,
        )

        return middleware

    @classmethod
    def init_from_config(
        cls,
        api_key: str,
        target_dataset: str,
        model_id: str,
        cache: Any,
        config: Optional[dict],
        platform_client: "ServerRoboflowPlatformClient",
    ) -> "ActiveLearningMiddleware":
        """Build the middleware from a configuration document.

        Args:
            api_key: Roboflow API key used for every platform call.
            target_dataset: Project datapoints are registered in.
            model_id: Model whose predictions are registered.
            cache: Cache holding the strategy usage counters.
            config: Configuration document, or None.
            platform_client: Roboflow platform client of the server.

        Returns:
            The middleware, inactive when the document is missing or disabled.

        Raises:
            RoboflowAPIRequestError: If the platform cannot be reached or
                rejects a request.
            ActiveLearningConfigurationError: If the configuration is
                inconsistent.
            ActiveLearningConfigurationDecodingError: If the configuration
                cannot be decoded.
        """
        configuration = prepare_active_learning_configuration_inplace(
            api_key=api_key,
            target_dataset=target_dataset,
            model_id=model_id,
            active_learning_configuration=config,
            platform_client=platform_client,
        )
        middleware = cls(
            api_key=api_key,
            configuration=configuration,
            cache=cache,
            platform_client=platform_client,
        )

        return middleware

    def __init__(
        self,
        api_key: str,
        configuration: Optional[ActiveLearningConfiguration],
        cache: Any,
        platform_client: "ServerRoboflowPlatformClient",
    ):
        self._api_key = api_key
        self._configuration = configuration
        self._cache = cache
        self._platform_client = platform_client

    @property
    def active(self) -> bool:
        """Whether the middleware holds a configuration and can register."""
        return self._configuration is not None

    def register_batch(
        self,
        images: List[np.ndarray],
        predictions: List[Prediction],
        prediction_type: PredictionType,
        inference_id: Optional[str] = None,
    ) -> None:
        """Register each image with its prediction.

        Args:
            images: Decoded BGR images.
            predictions: Predictions in the response format of the server; they
                are modified in place when the registered image is downscaled.
            prediction_type: Task type of the model.
            inference_id: Identifier of the inference, sent with the images.

        Raises:
            ValueError: If an image has no pixels.
            RoboflowAPIRequestError: If the platform cannot be reached or
                rejects a request.
        """
        for image, prediction in zip(images, predictions):
            self.register(
                image=image,
                prediction=prediction,
                prediction_type=prediction_type,
                inference_id=inference_id,
            )

    def register(
        self,
        image: np.ndarray,
        prediction: dict,
        prediction_type: PredictionType,
        inference_id: Optional[str] = None,
    ) -> None:
        """Sample one datapoint and register it when selected.

        Args:
            image: Decoded BGR image.
            prediction: Prediction in the response format of the server; it is
                modified in place when the registered image is downscaled.
            prediction_type: Task type of the model.
            inference_id: Identifier of the inference, sent with the image.

        Raises:
            ValueError: If the image has no pixels.
            RoboflowAPIRequestError: If the platform cannot be reached or
                rejects a request.
        """
        if self._configuration is None:
            return None

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
            platform_client=self._platform_client,
        ):
            logger.debug("Limit on Active Learning batch size reached.")
            return None

        execute_datapoint_registration(
            cache=self._cache,
            matching_strategies=matching_strategies,
            image=image,
            prediction=prediction,
            prediction_type=prediction_type,
            configuration=self._configuration,
            api_key=self._api_key,
            batch_name=batch_name,
            platform_client=self._platform_client,
            inference_id=inference_id,
        )

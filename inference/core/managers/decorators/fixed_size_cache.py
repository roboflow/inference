from collections import deque
from typing import List, Optional

from inference.core import logger
from inference.core.entities.requests.inference import InferenceRequest
from inference.core.entities.responses.inference import InferenceResponse
from inference.core.managers.base import Model, ModelManager
from inference.core.managers.decorators.base import ModelManagerDecorator
from inference.core.managers.entities import ModelDescription


class WithFixedSizeCache(ModelManagerDecorator):
    def __init__(self, model_manager: ModelManager, max_size: int = 8):
        """Cache decorator, models will be evicted based on the last utilization (`.infer` call). Internally, a [double-ended queue](https://docs.python.org/3/library/collections.html#collections.deque) is used to keep track of model utilization.

        Args:
            model_manager (ModelManager): Instance of a ModelManager.
            max_size (int, optional): Max number of models at the same time. Defaults to 8.
        """
        super().__init__(model_manager)
        self.max_size = max_size
        self._key_queue = deque(self.model_manager.keys())

    def add_model(
        self, model_id: str, api_key: str, model_id_alias: Optional[str] = None
    ) -> None:
        """Adds a model to the manager and evicts the least recently used if the cache is full.

        Args:
            model_id (str): The identifier of the model.
            model (Model): The model instance.
        """
        queue_id = self._resolve_queue_id(
            model_id=model_id, model_id_alias=model_id_alias
        )
        if queue_id in self:
            logger.debug(
                f"Detected {queue_id} in WithFixedSizeCache models queue -> marking as most recently used."
            )
            self._key_queue.remove(queue_id)
            self._key_queue.append(queue_id)
            return None

        logger.debug(f"Current capacity of ModelManager: {len(self)}/{self.max_size}")
        while len(self) >= self.max_size:
            to_remove_model_id = self._key_queue.popleft()
            logger.debug(
                f"Reached maximum capacity of ModelManager. Unloading model {to_remove_model_id}"
            )
            super().remove(to_remove_model_id)
            logger.debug(f"Model {to_remove_model_id} successfully unloaded.")
        logger.debug(f"Marking new model {queue_id} as most recently used.")
        self._key_queue.append(queue_id)
        try:
            return super().add_model(model_id, api_key, model_id_alias=model_id_alias)
        except Exception as error:
            logger.debug(
                f"Could not initialise model {queue_id}. Removing from WithFixedSizeCache models queue."
            )
            self._key_queue.remove(queue_id)
            raise error

    def clear(self) -> None:
        """Removes all models from the manager."""
        for model_id in list(self.keys()):
            self.remove(model_id)

    def remove(self, model_id: str) -> Model:
        try:
            self._key_queue.remove(model_id)
        except ValueError:
            logger.warning(
                f"Could not successfully purge model {model_id} from  WithFixedSizeCache models queue"
            )
        return super().remove(model_id)

    async def infer_from_request(
        self, model_id: str, request: InferenceRequest, **kwargs
    ) -> InferenceResponse:
        """Processes a complete inference request and updates the cache.

        Args:
            model_id (str): The identifier of the model.
            request (InferenceRequest): The request to process.

        Returns:
            InferenceResponse: The response from the inference.
        """
        self._key_queue.remove(model_id)
        self._key_queue.append(model_id)
        return await super().infer_from_request(model_id, request, **kwargs)

    def infer_from_request_sync(
        self, model_id: str, request: InferenceRequest, **kwargs
    ) -> InferenceResponse:
        """Processes a complete inference request and updates the cache.

        Args:
            model_id (str): The identifier of the model.
            request (InferenceRequest): The request to process.

        Returns:
            InferenceResponse: The response from the inference.
        """
        self._key_queue.remove(model_id)
        self._key_queue.append(model_id)
        return super().infer_from_request_sync(model_id, request, **kwargs)

    def infer_only(self, model_id: str, request, img_in, img_dims, batch_size=None):
        """Performs only the inference part of a request and updates the cache.

        Args:
            model_id (str): The identifier of the model.
            request: The request to process.
            img_in: Input image.
            img_dims: Image dimensions.
            batch_size (int, optional): Batch size.

        Returns:
            Response from the inference-only operation.
        """
        self._key_queue.remove(model_id)
        self._key_queue.append(model_id)
        return super().infer_only(model_id, request, img_in, img_dims, batch_size)

    def preprocess(self, model_id: str, request):
        """Processes the preprocessing part of a request and updates the cache.

        Args:
            model_id (str): The identifier of the model.
            request (InferenceRequest): The request to preprocess.
        """
        self._key_queue.remove(model_id)
        self._key_queue.append(model_id)
        return super().preprocess(model_id, request)

    def describe_models(self) -> List[ModelDescription]:
        return self.model_manager.describe_models()

    def _resolve_queue_id(
        self, model_id: str, model_id_alias: Optional[str] = None
    ) -> str:
        return model_id if model_id_alias is None else model_id_alias

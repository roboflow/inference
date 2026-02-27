import gc
from collections import deque
from threading import Lock
from typing import List, Optional

from inference.core import logger
from inference.core.entities.requests.inference import InferenceRequest
from inference.core.entities.responses.inference import InferenceResponse
from inference.core.env import (
    DISK_CACHE_CLEANUP,
    HOT_MODELS_QUEUE_LOCK_ACQUIRE_TIMEOUT,
    MEMORY_FREE_THRESHOLD,
    MODELS_CACHE_AUTH_ENABLED,
    USE_INFERENCE_MODELS,
)
from inference.core.exceptions import (
    ModelManagerLockAcquisitionError,
    RoboflowAPINotAuthorizedError,
)
from inference.core.managers.base import Model, ModelManager, acquire_with_timeout
from inference.core.managers.model_load_collector import request_model_ids
from inference.core.managers.decorators.base import ModelManagerDecorator
from inference.core.managers.entities import ModelDescription
from inference.core.registries.roboflow import (
    ModelEndpointType,
    _check_if_api_key_has_access_to_model,
)


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
        self._queue_lock = Lock()
        self._pinned_models: set = set()

    def pin_model(self, model_id: str) -> None:
        """Mark a model as pinned so it won't be evicted by the LRU cache.

        Pinned models (typically preloaded models) are protected from eviction
        when the cache is full or under memory pressure.
        """
        self._pinned_models.add(model_id)
        logger.debug(f"Model '{model_id}' pinned — will not be evicted from cache.")

    def add_model(
        self,
        model_id: str,
        api_key: str,
        model_id_alias: Optional[str] = None,
        endpoint_type: ModelEndpointType = ModelEndpointType.ORT,
        countinference: Optional[bool] = None,
        service_secret: Optional[str] = None,
    ) -> None:
        """Adds a model to the manager and evicts the least recently used if the cache is full.

        Args:
            model_id (str): The identifier of the model.
            model (Model): The model instance.
            endpoint_type (ModelEndpointType, optional): The endpoint type to use for the model.
        """
        if MODELS_CACHE_AUTH_ENABLED:
            if not _check_if_api_key_has_access_to_model(
                api_key=api_key,
                model_id=model_id,
                endpoint_type=endpoint_type,
                countinference=countinference,
                service_secret=service_secret,
            ):
                raise RoboflowAPINotAuthorizedError(
                    f"API key {api_key} does not have access to model {model_id}"
                )

        queue_id = self._resolve_queue_id(
            model_id=model_id, model_id_alias=model_id_alias
        )
        ids_collector = request_model_ids.get(None)
        if ids_collector is not None:
            ids_collector.add(queue_id)
        if queue_id in self:
            logger.debug(
                f"Detected {queue_id} in WithFixedSizeCache models queue -> marking as most recently used."
            )
            self._refresh_model_position_in_a_queue(model_id=queue_id)
            return None

        logger.debug(f"Current capacity of ModelManager: {len(self)}/{self.max_size}")
        with acquire_with_timeout(
            lock=self._queue_lock, timeout=HOT_MODELS_QUEUE_LOCK_ACQUIRE_TIMEOUT
        ) as acquired:
            if not acquired:
                raise ModelManagerLockAcquisitionError(
                    "Could not acquire lock on Model Manager state to add model from active models queue."
                )
            while self._key_queue and (
                len(self) >= self.max_size
                or (MEMORY_FREE_THRESHOLD and self.memory_pressure_detected())
            ):
                # To prevent flapping around the threshold, remove up to 3 models to make some space.
                evicted_count = 0
                skipped_pinned = []
                for _ in range(3):
                    if not self._key_queue:
                        logger.error(
                            "Tried to remove model from cache even though key queue is already empty!"
                            "(max_size: %s, len(self): %s, MEMORY_FREE_THRESHOLD: %s)",
                            self.max_size,
                            len(self),
                            MEMORY_FREE_THRESHOLD,
                        )
                        break
                    to_remove_model_id = self._key_queue.popleft()
                    if to_remove_model_id in self._pinned_models:
                        skipped_pinned.append(to_remove_model_id)
                        continue
                    super().remove(
                        to_remove_model_id, delete_from_disk=DISK_CACHE_CLEANUP
                    )  # LRU model overflow cleanup may or maynot need the weights removed from disk
                    logger.debug(f"Model {to_remove_model_id} successfully unloaded.")
                    evicted_count += 1
                # Put pinned models back at the front of the queue
                for mid in reversed(skipped_pinned):
                    self._key_queue.appendleft(mid)
                if evicted_count == 0:
                    logger.warning(
                        "Cannot free model cache space — all remaining models are pinned (preloaded). "
                        "Proceeding with cache exceeding max_size."
                    )
                    break
                gc.collect()
            logger.debug(f"Marking new model {queue_id} as most recently used.")
            self._key_queue.append(queue_id)
        try:
            return super().add_model(
                model_id,
                api_key,
                model_id_alias=model_id_alias,
                endpoint_type=endpoint_type,
                countinference=countinference,
                service_secret=service_secret,
            )
        except Exception as error:
            logger.debug(
                f"Could not initialise model {queue_id}. Removing from WithFixedSizeCache models queue."
            )
            with acquire_with_timeout(
                lock=self._queue_lock, timeout=HOT_MODELS_QUEUE_LOCK_ACQUIRE_TIMEOUT
            ) as acquired:
                if not acquired:
                    raise ModelManagerLockAcquisitionError(
                        "Could not acquire lock on Model Manager state to remove model from active models queue."
                    )
                self._safe_remove_model_from_queue(queue_id)
            raise error

    def clear(self) -> None:
        """Removes all models from the manager."""
        for model_id in list(self.keys()):
            self.remove(model_id)

    def remove(self, model_id: str, delete_from_disk: bool = True) -> Model:
        with acquire_with_timeout(
            lock=self._queue_lock, timeout=HOT_MODELS_QUEUE_LOCK_ACQUIRE_TIMEOUT
        ) as acquired:
            if not acquired:
                raise ModelManagerLockAcquisitionError(
                    "Could not acquire lock on Model Manager state to remove model from active models queue."
                )
            self._safe_remove_model_from_queue(model_id=model_id)
        return super().remove(model_id, delete_from_disk=delete_from_disk)

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
        self._refresh_model_position_in_a_queue(model_id=model_id)
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
        self._refresh_model_position_in_a_queue(model_id=model_id)
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
        self._refresh_model_position_in_a_queue(model_id=model_id)
        return super().infer_only(model_id, request, img_in, img_dims, batch_size)

    def preprocess(self, model_id: str, request):
        """Processes the preprocessing part of a request and updates the cache.

        Args:
            model_id (str): The identifier of the model.
            request (InferenceRequest): The request to preprocess.
        """
        self._refresh_model_position_in_a_queue(model_id=model_id)
        return super().preprocess(model_id, request)

    def describe_models(self) -> List[ModelDescription]:
        return self.model_manager.describe_models()

    def _resolve_queue_id(
        self, model_id: str, model_id_alias: Optional[str] = None
    ) -> str:
        return model_id if model_id_alias is None else model_id_alias

    def _refresh_model_position_in_a_queue(self, model_id: str) -> None:
        with acquire_with_timeout(
            lock=self._queue_lock, timeout=HOT_MODELS_QUEUE_LOCK_ACQUIRE_TIMEOUT
        ) as acquired:
            if not acquired:
                raise ModelManagerLockAcquisitionError(
                    "Could not acquire lock on Model Manager state to refresh model position in active models queue."
                )
            self._safe_remove_model_from_queue(model_id=model_id)
            self._key_queue.append(model_id)

    def _safe_remove_model_from_queue(self, model_id: str) -> None:
        try:
            while model_id in self._key_queue:
                self._key_queue.remove(model_id)
        except ValueError:
            logger.warning(
                f"Could not successfully purge model {model_id} from  WithFixedSizeCache models queue - "
                f"model id not found."
            )

    def memory_pressure_detected(self) -> bool:
        return_boolean = False
        try:
            import torch

            if torch.cuda.is_available():
                free_memory, total_memory = torch.cuda.mem_get_info()
                return_boolean = (
                    float(free_memory / total_memory) < MEMORY_FREE_THRESHOLD
                )
                if return_boolean and USE_INFERENCE_MODELS:
                    # we only enable this under condition that USE_INFERENCE_MODELS is True
                    # and we are about to remove a model
                    # just to make sure we are not flapping around the threshold for no reason
                    torch.cuda.empty_cache()
                    free_memory, total_memory = torch.cuda.mem_get_info()
                    return_boolean = (
                        float(free_memory / total_memory) < MEMORY_FREE_THRESHOLD
                    )
                logger.debug(
                    f"Free memory: {free_memory}, Total memory: {total_memory}, threshold: {MEMORY_FREE_THRESHOLD}, return_boolean: {return_boolean}"
                )
            # TODO: Add memory calculation for other non-CUDA devices
        except Exception as e:
            logger.error(
                f"Failed to check CUDA memory pressure: {e}, returning {return_boolean}"
            )
        return return_boolean

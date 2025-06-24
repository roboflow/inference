from collections import deque
from typing import List, Optional

from inference.core import logger
from inference.core.entities.requests.inference import InferenceRequest
from inference.core.entities.responses.inference import InferenceResponse
from inference.core.env import (
    DISK_CACHE_CLEANUP,
    MEMORY_FREE_THRESHOLD,
    MODELS_CACHE_AUTH_ENABLED,
)
from inference.core.exceptions import RoboflowAPINotAuthorizedError
from inference.core.managers.base import Model, ModelManager
from inference.core.managers.decorators.base import ModelManagerDecorator
from inference.core.managers.entities import ModelDescription
from inference.core.registries.roboflow import (
    ModelEndpointType,
    _check_if_api_key_has_access_to_model,
)
from inference.core.roboflow_api import ModelEndpointType


class WithFixedSizeCache(ModelManagerDecorator):
    def __init__(self, model_manager: ModelManager, max_size: int = 8):
        """Cache decorator, models will be evicted based on the last utilization (`.infer` call). Internally, a [double-ended queue](https://docs.python.org/3/library/collections.html#collections.deque) is used to keep track of model utilization.

        Args:
            model_manager (ModelManager): Instance of a ModelManager.
            max_size (int, optional): Max number of models at the same time. Defaults to 8.
        """
        # LRU cache with O(1) item moving using deque for keys, for fast eviction/refresh of use order
        super().__init__(model_manager)
        self.max_size = max_size
        self._key_queue = deque(self.model_manager.keys())

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

        # Fast-path: skip access check if not enabled
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

        queue_id = model_id if model_id_alias is None else model_id_alias

        # Fast check: Model already present
        if queue_id in self:
            # Move already-present model to MRU position
            try:
                self._key_queue.remove(queue_id)
            except ValueError:
                # Defensive: This should not happen, but just in case, sync the queue with actual models
                self._key_queue = deque(k for k in self.model_manager.keys())
                if queue_id in self._key_queue:
                    self._key_queue.remove(queue_id)
            self._key_queue.append(queue_id)
            return None

        # Only log if necessary due to performance during profiling
        # logger.debug(f"Current capacity: {len(self)}/{self.max_size}")

        need_evict = len(self) >= self.max_size or (
            MEMORY_FREE_THRESHOLD and self.memory_pressure_detected()
        )

        # Evict as many models as needed. Batch removals so we call gc only once.
        keys_to_remove = []
        # While check handles both scenarios (LRU + memory pressure)
        while self._key_queue and need_evict:
            # Remove up to 3 models per policy for one pass, then re-check exit condition
            removals_this_pass = min(3, len(self._key_queue))
            for _ in range(removals_this_pass):
                if not self._key_queue:
                    logger.error(
                        "Tried to remove model from cache but queue is empty! (max_size: %s, len(self): %s, MEMORY_FREE_THRESHOLD: %s)",
                        self.max_size,
                        len(self),
                        MEMORY_FREE_THRESHOLD,
                    )
                    break
                to_remove_model_id = self._key_queue.popleft()
                super().remove(
                    to_remove_model_id, delete_from_disk=DISK_CACHE_CLEANUP
                )  # Also calls clear_cache
                # logger.debug(f"Model {to_remove_model_id} successfully unloaded.")  # Perf: can be commented
            # Re-test need_evict after removals (memory pressure may be gone, size may now be under limit)
            need_evict = len(self) >= self.max_size or (
                MEMORY_FREE_THRESHOLD and self.memory_pressure_detected()
            )

        # Only now, after batch eviction, trigger gc.collect() ONCE if anything was evicted
        if self._key_queue and len(self) < self.max_size:
            # No recent eviction: no gc necessary
            pass
        else:
            # Import gc only if required
            import gc

            gc.collect()

        self._key_queue.append(queue_id)
        try:
            super().add_model(
                model_id,
                api_key,
                model_id_alias=model_id_alias,
                endpoint_type=endpoint_type,
                countinference=countinference,
                service_secret=service_secret,
            )
        except Exception as error:
            # Defensive: Only remove queue_id if present. Use try-except to avoid further exceptions.
            try:
                self._key_queue.remove(queue_id)
            except ValueError:
                pass
            raise error

    def clear(self) -> None:
        """Removes all models from the manager."""
        for model_id in list(self.keys()):
            self.remove(model_id)

    def remove(self, model_id: str, delete_from_disk: bool = True) -> Model:
        try:
            self._key_queue.remove(model_id)
        except ValueError:
            logger.warning(
                f"Could not successfully purge model {model_id} from  WithFixedSizeCache models queue"
            )
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
        # Used only by legacy callers, now inlined for speed above
        return model_id if model_id_alias is None else model_id_alias

    def memory_pressure_detected(self) -> bool:
        # Only check CUDA memory if threshold is enabled, and torch is present
        return_boolean = False
        try:
            import torch

            if torch.cuda.is_available():
                free_memory, total_memory = torch.cuda.mem_get_info()
                return_boolean = (
                    float(free_memory / total_memory) < MEMORY_FREE_THRESHOLD
                )
                # logger.debug(...)    # For perf, skip logging
        except Exception:
            # Silently ignore errors here, default: not under pressure
            pass
        return return_boolean

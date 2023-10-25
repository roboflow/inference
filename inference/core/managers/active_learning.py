import time
from typing import Optional, Dict

from inference.core import logger
from inference.core.active_learning.core import ActiveLearningMiddleware
from inference.core.cache.base import BaseCache
from inference.core.entities.requests.inference import InferenceRequest
from inference.core.entities.responses.inference import InferenceResponse
from inference.core.managers.base import ModelManager
from inference.core.registries.base import ModelRegistry


class ActiveLearningManager(ModelManager):
    def __init__(
        self,
        model_registry: ModelRegistry,
        cache: BaseCache,
        middlewares: Optional[Dict[str, ActiveLearningMiddleware]] = None,
    ):
        super().__init__(model_registry=model_registry)
        self._cache = cache
        self._middlewares = middlewares if middlewares is not None else {}

    def infer_from_request(
        self, model_id: str, request: InferenceRequest
    ) -> InferenceResponse:
        if model_id not in self._middlewares:
            start = time.perf_counter()
            logger.info(f"Initialising AL middleware for {model_id}")
            self._middlewares[model_id] = ActiveLearningMiddleware.init(
                api_key=request.api_key,
                model_id=model_id,
                cache=self._cache,
            )
            end = time.perf_counter()
            logger.info(f"Middleware init latency: {(end - start) * 1000} ms")
        result = super().infer_from_request(model_id=model_id, request=request)
        start = time.perf_counter()
        inference_inputs = getattr(request, "image", None)
        if inference_inputs is None:
            logger.warning(
                "Could not register datapoint, as inference input has no `image` field."
            )
            return result
        if not issubclass(type(inference_inputs), list):
            inference_inputs = [inference_inputs]
        if not issubclass(type(result), list):
            results_dicts = [result.dict(by_alias=True, exclude={"visualization"})]
        else:
            results_dicts = [
                e.dict(by_alias=True, exclude={"visualization"}) for e in result
            ]
        prediction_type = self.get_task_type(model_id=model_id)
        self._middlewares[model_id].register_batch(
            inference_inputs=inference_inputs,
            predictions=results_dicts,
            prediction_type=prediction_type,
        )
        end = time.perf_counter()
        logger.info(f"Registration: {(end - start) * 1000} ms")
        return result

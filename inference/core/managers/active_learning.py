import time
from typing import Dict, Optional

from fastapi import BackgroundTasks

from inference.core import logger
from inference.core.active_learning.middlewares import ActiveLearningMiddleware
from inference.core.cache.base import BaseCache
from inference.core.entities.requests.inference import InferenceRequest
from inference.core.entities.responses.inference import InferenceResponse
from inference.core.env import DISABLE_PREPROC_AUTO_ORIENT
from inference.core.managers.base import ModelManager
from inference.core.registries.base import ModelRegistry

ACTIVE_LEARNING_ELIGIBLE_PARAM = "active_learning_eligible"
DISABLE_ACTIVE_LEARNING_PARAM = "disable_active_learning"
BACKGROUND_TASKS_PARAM = "background_tasks"


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

    async def infer_from_request(
        self, model_id: str, request: InferenceRequest, **kwargs
    ) -> InferenceResponse:
        prediction = await super().infer_from_request(
            model_id=model_id, request=request, **kwargs
        )
        active_learning_eligible = kwargs.get(ACTIVE_LEARNING_ELIGIBLE_PARAM, False)
        active_learning_disabled_for_request = getattr(
            request, DISABLE_ACTIVE_LEARNING_PARAM, False
        )
        if (
            not active_learning_eligible
            or active_learning_disabled_for_request
            or request.api_key is None
        ):
            return prediction
        self.register(prediction=prediction, model_id=model_id, request=request)
        return prediction

    def register(
        self, prediction: InferenceResponse, model_id: str, request: InferenceRequest
    ) -> None:
        try:
            self.ensure_middleware_initialised(model_id=model_id, request=request)
            self.register_datapoint(
                prediction=prediction,
                model_id=model_id,
                request=request,
            )
        except Exception as error:
            # Error handling to be decided
            logger.warning(
                f"Error in datapoint registration for Active Learning. Details: {error}. "
                f"Error is suppressed in favour of normal operations of API."
            )

    def ensure_middleware_initialised(
        self, model_id: str, request: InferenceRequest
    ) -> None:
        if model_id in self._middlewares:
            return None
        start = time.perf_counter()
        logger.debug(f"Initialising AL middleware for {model_id}")
        self._middlewares[model_id] = ActiveLearningMiddleware.init(
            api_key=request.api_key,
            model_id=model_id,
            cache=self._cache,
        )
        end = time.perf_counter()
        logger.debug(f"Middleware init latency: {(end - start) * 1000} ms")

    def register_datapoint(
        self, prediction: InferenceResponse, model_id: str, request: InferenceRequest
    ) -> None:
        start = time.perf_counter()
        inference_inputs = getattr(request, "image", None)
        if inference_inputs is None:
            logger.warning(
                "Could not register datapoint, as inference input has no `image` field."
            )
            return None
        if not issubclass(type(inference_inputs), list):
            inference_inputs = [inference_inputs]
        if not issubclass(type(prediction), list):
            results_dicts = [prediction.dict(by_alias=True, exclude={"visualization"})]
        else:
            results_dicts = [
                e.dict(by_alias=True, exclude={"visualization"}) for e in prediction
            ]
        prediction_type = self.get_task_type(model_id=model_id)
        disable_preproc_auto_orient = (
            getattr(request, "disable_preproc_auto_orient", False)
            or DISABLE_PREPROC_AUTO_ORIENT
        )
        self._middlewares[model_id].register_batch(
            inference_inputs=inference_inputs,
            predictions=results_dicts,
            prediction_type=prediction_type,
            disable_preproc_auto_orient=disable_preproc_auto_orient,
        )
        end = time.perf_counter()
        logger.debug(f"Registration: {(end - start) * 1000} ms")


class BackgroundTaskActiveLearningManager(ActiveLearningManager):
    async def infer_from_request(
        self, model_id: str, request: InferenceRequest, **kwargs
    ) -> InferenceResponse:
        active_learning_eligible = kwargs.get(ACTIVE_LEARNING_ELIGIBLE_PARAM, False)
        active_learning_disabled_for_request = getattr(
            request, DISABLE_ACTIVE_LEARNING_PARAM, False
        )
        kwargs[ACTIVE_LEARNING_ELIGIBLE_PARAM] = False  # disabling AL in super-classes
        prediction = await super().infer_from_request(
            model_id=model_id, request=request, **kwargs
        )
        if (
            not active_learning_eligible
            or active_learning_disabled_for_request
            or request.api_key is None
        ):
            return prediction
        if BACKGROUND_TASKS_PARAM not in kwargs:
            logger.warning(
                "BackgroundTaskActiveLearningManager used against rules - `background_tasks` argument not "
                "provided making Active Learning registration running sequentially."
            )
            self.register(prediction=prediction, model_id=model_id, request=request)
        else:
            background_tasks: BackgroundTasks = kwargs["background_tasks"]
            background_tasks.add_task(
                self.register, prediction=prediction, model_id=model_id, request=request
            )
        return prediction

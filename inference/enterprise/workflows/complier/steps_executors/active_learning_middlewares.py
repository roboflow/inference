from typing import Dict, List, Optional

from fastapi import BackgroundTasks

from inference.core import logger
from inference.core.active_learning.middlewares import ActiveLearningMiddleware
from inference.core.cache.base import BaseCache
from inference.core.env import DISABLE_PREPROC_AUTO_ORIENT


class WorkflowsActiveLearningMiddleware:

    def __init__(
        self,
        cache: BaseCache,
        middlewares: Optional[Dict[str, ActiveLearningMiddleware]] = None,
    ):
        self._cache = cache
        self._middlewares = middlewares if middlewares is not None else {}

    def register(
        self,
        dataset_name: str,
        images: List[dict],
        predictions: List[dict],
        api_key: Optional[str],
        prediction_type: str,
        active_learning_disabled_for_request: bool,
        background_tasks: Optional[BackgroundTasks] = None,
    ) -> None:
        model_id = f"{dataset_name}/workflows"
        if api_key is None or active_learning_disabled_for_request:
            return None
        if background_tasks is None:
            self._register(
                model_id=model_id,
                images=images,
                predictions=predictions,
                api_key=api_key,
                prediction_type=prediction_type,
            )
            return None
        background_tasks.add_task(
            self._register,
            model_id=model_id,
            images=images,
            predictions=predictions,
            api_key=api_key,
            prediction_type=prediction_type,
        )

    def _register(
        self,
        model_id: str,
        images: List[dict],
        predictions: List[dict],
        api_key: str,
        prediction_type: str,
    ) -> None:
        try:
            self._ensure_middleware_initialised(model_id=model_id, api_key=api_key)
            self._middlewares[model_id].register_batch(
                inference_inputs=images,
                predictions=predictions,
                prediction_type=prediction_type,
                disable_preproc_auto_orient=DISABLE_PREPROC_AUTO_ORIENT,
            )
        except Exception as error:
            # Error handling to be decided
            logger.warning(
                f"Error in datapoint registration for Active Learning. Details: {error}. "
                f"Error is suppressed in favour of normal operations of API."
            )

    def _ensure_middleware_initialised(
        self,
        model_id: str,
        api_key: str,
    ) -> None:
        if model_id in self._middlewares:
            return None
        self._middlewares[model_id] = ActiveLearningMiddleware.init(
            api_key=api_key,
            model_id=model_id,
            cache=self._cache,
        )

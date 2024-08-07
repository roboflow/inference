import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from fastapi.encoders import jsonable_encoder

from inference.core.cache import cache
from inference.core.cache.serializers import to_cachable_inference_item
from inference.core.devices.utils import GLOBAL_INFERENCE_SERVER_ID
from inference.core.entities.requests.inference import InferenceRequest
from inference.core.entities.responses.inference import InferenceResponse
from inference.core.env import (
    DISABLE_INFERENCE_CACHE,
    METRICS_ENABLED,
    METRICS_INTERVAL,
    ROBOFLOW_SERVER_UUID,
)
from inference.core.exceptions import InferenceModelNotFound
from inference.core.logger import logger
from inference.core.managers.entities import ModelDescription
from inference.core.managers.pingback import PingbackInfo
from inference.core.models.base import Model, PreprocessReturnMetadata
from inference.core.registries.base import ModelRegistry


class ModelManager:
    """Model managers keep track of a dictionary of Model objects and is responsible for passing requests to the right model using the infer method."""

    def __init__(self, model_registry: ModelRegistry, models: Optional[dict] = None):
        self.model_registry = model_registry
        self._models: Dict[str, Model] = models if models is not None else {}
        self.pingback = None

    def init_pingback(self):
        """Initializes pingback mechanism."""
        self.num_errors = 0  # in the device
        self.uuid = ROBOFLOW_SERVER_UUID
        if METRICS_ENABLED:
            self.pingback = PingbackInfo(self)
            self.pingback.start()

    def add_model(
        self, model_id: str, api_key: str, model_id_alias: Optional[str] = None
    ) -> None:
        """Adds a new model to the manager.

        Args:
            model_id (str): The identifier of the model.
            model (Model): The model instance.
        """
        logger.debug(
            f"ModelManager - Adding model with model_id={model_id}, model_id_alias={model_id_alias}"
        )
        resolved_identifier = model_id if model_id_alias is None else model_id_alias
        if resolved_identifier in self._models:
            logger.debug(
                f"ModelManager - model with model_id={resolved_identifier} is already loaded."
            )
            return
        logger.debug("ModelManager - model initialisation...")
        model = self.model_registry.get_model(resolved_identifier, api_key)(
            model_id=model_id,
            api_key=api_key,
        )
        logger.debug("ModelManager - model successfully loaded.")
        self._models[resolved_identifier] = model

    def check_for_model(self, model_id: str) -> None:
        """Checks whether the model with the given ID is in the manager.

        Args:
            model_id (str): The identifier of the model.

        Raises:
            InferenceModelNotFound: If the model is not found in the manager.
        """
        if model_id not in self:
            raise InferenceModelNotFound(f"Model with id {model_id} not loaded.")

    async def infer_from_request(
        self, model_id: str, request: InferenceRequest, **kwargs
    ) -> InferenceResponse:
        """Runs inference on the specified model with the given request.

        Args:
            model_id (str): The identifier of the model.
            request (InferenceRequest): The request to process.

        Returns:
            InferenceResponse: The response from the inference.
        """
        logger.debug(
            f"ModelManager - inference from request started for model_id={model_id}."
        )
        if METRICS_ENABLED and self.pingback:
            logger.debug("ModelManager - setting pingback fallback api key...")
            self.pingback.fallback_api_key = request.api_key
        try:
            rtn_val = await self.model_infer(
                model_id=model_id, request=request, **kwargs
            )
            logger.debug(
                f"ModelManager - inference from request finished for model_id={model_id}."
            )
            finish_time = time.time()
            if not DISABLE_INFERENCE_CACHE:
                logger.debug(
                    f"ModelManager - caching inference request started for model_id={model_id}"
                )
                cache.zadd(
                    f"models",
                    value=f"{GLOBAL_INFERENCE_SERVER_ID}:{request.api_key}:{model_id}",
                    score=finish_time,
                    expire=METRICS_INTERVAL * 2,
                )
                if (
                    hasattr(request, "image")
                    and hasattr(request.image, "type")
                    and request.image.type == "numpy"
                ):
                    request.image.value = str(request.image.value)
                cache.zadd(
                    f"inference:{GLOBAL_INFERENCE_SERVER_ID}:{model_id}",
                    value=to_cachable_inference_item(request, rtn_val),
                    score=finish_time,
                    expire=METRICS_INTERVAL * 2,
                )
                logger.debug(
                    f"ModelManager - caching inference request finished for model_id={model_id}"
                )
            return rtn_val
        except Exception as e:
            finish_time = time.time()
            if not DISABLE_INFERENCE_CACHE:
                cache.zadd(
                    f"models",
                    value=f"{GLOBAL_INFERENCE_SERVER_ID}:{request.api_key}:{model_id}",
                    score=finish_time,
                    expire=METRICS_INTERVAL * 2,
                )
                cache.zadd(
                    f"error:{GLOBAL_INFERENCE_SERVER_ID}:{model_id}",
                    value={
                        "request": jsonable_encoder(
                            request.dict(exclude={"image", "subject", "prompt"})
                        ),
                        "error": str(e),
                    },
                    score=finish_time,
                    expire=METRICS_INTERVAL * 2,
                )
            raise

    def infer_from_request_sync(
        self, model_id: str, request: InferenceRequest, **kwargs
    ) -> InferenceResponse:
        """Runs inference on the specified model with the given request.

        Args:
            model_id (str): The identifier of the model.
            request (InferenceRequest): The request to process.

        Returns:
            InferenceResponse: The response from the inference.
        """
        logger.debug(
            f"ModelManager - inference from request started for model_id={model_id}."
        )
        if METRICS_ENABLED and self.pingback:
            logger.debug("ModelManager - setting pingback fallback api key...")
            self.pingback.fallback_api_key = request.api_key
        try:
            rtn_val = self.model_infer_sync(
                model_id=model_id, request=request, **kwargs
            )
            logger.debug(
                f"ModelManager - inference from request finished for model_id={model_id}."
            )
            finish_time = time.time()
            if not DISABLE_INFERENCE_CACHE:
                logger.debug(
                    f"ModelManager - caching inference request started for model_id={model_id}"
                )
                cache.zadd(
                    f"models",
                    value=f"{GLOBAL_INFERENCE_SERVER_ID}:{request.api_key}:{model_id}",
                    score=finish_time,
                    expire=METRICS_INTERVAL * 2,
                )
                if (
                    hasattr(request, "image")
                    and hasattr(request.image, "type")
                    and request.image.type == "numpy"
                ):
                    request.image.value = str(request.image.value)
                cache.zadd(
                    f"inference:{GLOBAL_INFERENCE_SERVER_ID}:{model_id}",
                    value=to_cachable_inference_item(request, rtn_val),
                    score=finish_time,
                    expire=METRICS_INTERVAL * 2,
                )
                logger.debug(
                    f"ModelManager - caching inference request finished for model_id={model_id}"
                )
            return rtn_val
        except Exception as e:
            finish_time = time.time()
            if not DISABLE_INFERENCE_CACHE:
                cache.zadd(
                    f"models",
                    value=f"{GLOBAL_INFERENCE_SERVER_ID}:{request.api_key}:{model_id}",
                    score=finish_time,
                    expire=METRICS_INTERVAL * 2,
                )
                cache.zadd(
                    f"error:{GLOBAL_INFERENCE_SERVER_ID}:{model_id}",
                    value={
                        "request": jsonable_encoder(
                            request.dict(exclude={"image", "subject", "prompt"})
                        ),
                        "error": str(e),
                    },
                    score=finish_time,
                    expire=METRICS_INTERVAL * 2,
                )
            raise

    async def model_infer(self, model_id: str, request: InferenceRequest, **kwargs):
        self.check_for_model(model_id)
        return self._models[model_id].infer_from_request(request)

    def model_infer_sync(
        self, model_id: str, request: InferenceRequest, **kwargs
    ) -> Union[List[InferenceResponse], InferenceResponse]:
        self.check_for_model(model_id)
        return self._models[model_id].infer_from_request(request)

    def make_response(
        self, model_id: str, predictions: List[List[float]], *args, **kwargs
    ) -> InferenceResponse:
        """Creates a response object from the model's predictions.

        Args:
            model_id (str): The identifier of the model.
            predictions (List[List[float]]): The model's predictions.

        Returns:
            InferenceResponse: The created response object.
        """
        self.check_for_model(model_id)
        return self._models[model_id].make_response(predictions, *args, **kwargs)

    def postprocess(
        self,
        model_id: str,
        predictions: Tuple[np.ndarray, ...],
        preprocess_return_metadata: PreprocessReturnMetadata,
        *args,
        **kwargs,
    ) -> List[List[float]]:
        """Processes the model's predictions after inference.

        Args:
            model_id (str): The identifier of the model.
            predictions (np.ndarray): The model's predictions.

        Returns:
            List[List[float]]: The post-processed predictions.
        """
        self.check_for_model(model_id)
        return self._models[model_id].postprocess(
            predictions, preprocess_return_metadata, *args, **kwargs
        )

    def predict(self, model_id: str, *args, **kwargs) -> Tuple[np.ndarray, ...]:
        """Runs prediction on the specified model.

        Args:
            model_id (str): The identifier of the model.

        Returns:
            np.ndarray: The predictions from the model.
        """
        self.check_for_model(model_id)
        self._models[model_id].metrics["num_inferences"] += 1
        tic = time.perf_counter()
        res = self._models[model_id].predict(*args, **kwargs)
        toc = time.perf_counter()
        self._models[model_id].metrics["avg_inference_time"] += toc - tic
        return res

    def preprocess(
        self, model_id: str, request: InferenceRequest
    ) -> Tuple[np.ndarray, PreprocessReturnMetadata]:
        """Preprocesses the request before inference.

        Args:
            model_id (str): The identifier of the model.
            request (InferenceRequest): The request to preprocess.

        Returns:
            Tuple[np.ndarray, List[Tuple[int, int]]]: The preprocessed data.
        """
        self.check_for_model(model_id)
        return self._models[model_id].preprocess(**request.dict())

    def get_class_names(self, model_id):
        """Retrieves the class names for a given model.

        Args:
            model_id (str): The identifier of the model.

        Returns:
            List[str]: The class names of the model.
        """
        self.check_for_model(model_id)
        return self._models[model_id].class_names

    def get_task_type(self, model_id: str, api_key: str = None) -> str:
        """Retrieves the task type for a given model.

        Args:
            model_id (str): The identifier of the model.

        Returns:
            str: The task type of the model.
        """
        self.check_for_model(model_id)
        return self._models[model_id].task_type

    def remove(self, model_id: str) -> None:
        """Removes a model from the manager.

        Args:
            model_id (str): The identifier of the model.
        """
        try:
            logger.debug(f"Removing model {model_id} from base model manager")
            self.check_for_model(model_id)
            self._models[model_id].clear_cache()
            del self._models[model_id]
        except InferenceModelNotFound:
            logger.warning(
                f"Attempted to remove model with id {model_id}, but it is not loaded. Skipping..."
            )

    def clear(self) -> None:
        """Removes all models from the manager."""
        for model_id in list(self.keys()):
            self.remove(model_id)

    def __contains__(self, model_id: str) -> bool:
        """Checks if the model is contained in the manager.

        Args:
            model_id (str): The identifier of the model.

        Returns:
            bool: Whether the model is in the manager.
        """
        return model_id in self._models

    def __getitem__(self, key: str) -> Model:
        """Retrieve a model from the manager by key.

        Args:
            key (str): The identifier of the model.

        Returns:
            Model: The model corresponding to the key.
        """
        self.check_for_model(model_id=key)
        return self._models[key]

    def __len__(self) -> int:
        """Retrieve the number of models in the manager.

        Returns:
            int: The number of models in the manager.
        """
        return len(self._models)

    def keys(self):
        """Retrieve the keys (model identifiers) from the manager.

        Returns:
            List[str]: The keys of the models in the manager.
        """
        return self._models.keys()

    def models(self) -> Dict[str, Model]:
        """Retrieve the models dictionary from the manager.

        Returns:
            Dict[str, Model]: The keys of the models in the manager.
        """
        return self._models

    def describe_models(self) -> List[ModelDescription]:
        return [
            ModelDescription(
                model_id=model_id,
                task_type=model.task_type,
                batch_size=getattr(model, "batch_size", None),
                input_width=getattr(model, "img_size_w", None),
                input_height=getattr(model, "img_size_h", None),
            )
            for model_id, model in self._models.items()
        ]

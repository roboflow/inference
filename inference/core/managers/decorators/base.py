from typing import List, Optional, Tuple

import numpy as np

from inference.core import logger
from inference.core.entities.requests.inference import InferenceRequest
from inference.core.entities.responses.inference import InferenceResponse
from inference.core.env import API_KEY
from inference.core.managers.base import Model, ModelManager
from inference.core.models.types import PreprocessReturnMetadata


class ModelManagerDecorator(ModelManager):
    """Basic decorator, it acts like a `ModelManager` and contains a `ModelManager`.

    Args:
        model_manager (ModelManager): Instance of a ModelManager.

    Methods:
        add_model: Adds a model to the manager.
        infer: Processes a complete inference request.
        infer_only: Performs only the inference part of a request.
        preprocess: Processes the preprocessing part of a request.
        get_task_type: Gets the task type associated with a model.
        get_class_names: Gets the class names for a given model.
        remove: Removes a model from the manager.
        __len__: Returns the number of models in the manager.
        __getitem__: Retrieves a model by its ID.
        __contains__: Checks if a model exists in the manager.
        keys: Returns the keys (model IDs) from the manager.
    """

    @property
    def _models(self):
        raise ValueError("Should only be accessing self.model_manager._models")

    @property
    def model_registry(self):
        raise ValueError("Should only be accessing self.model_manager.model_registry")

    def __init__(self, model_manager: ModelManager):
        """Initializes the decorator with an instance of a ModelManager."""
        self.model_manager = model_manager

    def init_pingback(self):
        self.model_manager.init_pingback()

    @property
    def pingback(self):
        return self.model_manager.pingback

    def add_model(
        self, model_id: str, api_key: str, model_id_alias: Optional[str] = None
    ):
        """Adds a model to the manager.

        Args:
            model_id (str): The identifier of the model.
            model (Model): The model instance.
        """
        if model_id in self:
            return
        self.model_manager.add_model(model_id, api_key, model_id_alias=model_id_alias)

    async def infer_from_request(
        self, model_id: str, request: InferenceRequest, **kwargs
    ) -> InferenceResponse:
        """Processes a complete inference request.

        Args:
            model_id (str): The identifier of the model.
            request (InferenceRequest): The request to process.

        Returns:
            InferenceResponse: The response from the inference.
        """
        return await self.model_manager.infer_from_request(model_id, request, **kwargs)

    def infer_from_request_sync(
        self, model_id: str, request: InferenceRequest, **kwargs
    ) -> InferenceResponse:
        """Processes a complete inference request.

        Args:
            model_id (str): The identifier of the model.
            request (InferenceRequest): The request to process.

        Returns:
            InferenceResponse: The response from the inference.
        """
        return self.model_manager.infer_from_request_sync(model_id, request, **kwargs)

    def infer_only(self, model_id: str, request, img_in, img_dims, batch_size=None):
        """Performs only the inference part of a request.

        Args:
            model_id (str): The identifier of the model.
            request: The request to process.
            img_in: Input image.
            img_dims: Image dimensions.
            batch_size (int, optional): Batch size.

        Returns:
            Response from the inference-only operation.
        """
        return self.model_manager.infer_only(
            model_id, request, img_in, img_dims, batch_size
        )

    def preprocess(self, model_id: str, request: InferenceRequest):
        """Processes the preprocessing part of a request.

        Args:
            model_id (str): The identifier of the model.
            request (InferenceRequest): The request to preprocess.
        """
        return self.model_manager.preprocess(model_id, request)

    def get_task_type(self, model_id: str, api_key: str = None) -> str:
        """Gets the task type associated with a model.

        Args:
            model_id (str): The identifier of the model.

        Returns:
            str: The task type.
        """
        if api_key is None:
            api_key = API_KEY
        return self.model_manager.get_task_type(model_id, api_key=api_key)

    def get_class_names(self, model_id):
        """Gets the class names for a given model.

        Args:
            model_id: The identifier of the model.

        Returns:
            List of class names.
        """
        return self.model_manager.get_class_names(model_id)

    def remove(self, model_id: str, delete_from_disk: bool = True) -> Model:
        """Removes a model from the manager.

        Args:
            model_id (str): The identifier of the model.

        Returns:
            Model: The removed model.
        """
        return self.model_manager.remove(model_id, delete_from_disk=delete_from_disk)

    def __len__(self) -> int:
        """Returns the number of models in the manager.

        Returns:
            int: Number of models.
        """
        return len(self.model_manager)

    def __getitem__(self, key: str) -> Model:
        """Retrieves a model by its ID.

        Args:
            key (str): The identifier of the model.

        Returns:
            Model: The model instance.
        """
        return self.model_manager[key]

    def __contains__(self, model_id: str):
        """Checks if a model exists in the manager.

        Args:
            model_id (str): The identifier of the model.

        Returns:
            bool: True if the model exists, False otherwise.
        """
        return model_id in self.model_manager

    def keys(self):
        """Returns the keys (model IDs) from the manager.

        Returns:
            List of keys (model IDs).
        """
        return self.model_manager.keys()

    def models(self):
        return self.model_manager.models()

    def predict(self, model_id: str, *args, **kwargs) -> Tuple[np.ndarray, ...]:
        return self.model_manager.predict(model_id, *args, **kwargs)

    def postprocess(
        self,
        model_id: str,
        predictions: Tuple[np.ndarray, ...],
        preprocess_return_metadata: PreprocessReturnMetadata,
        *args,
        **kwargs
    ) -> List[List[float]]:
        return self.model_manager.postprocess(
            model_id, predictions, preprocess_return_metadata, *args, **kwargs
        )

    def make_response(
        self, model_id: str, predictions: List[List[float]], *args, **kwargs
    ) -> InferenceResponse:
        return self.model_manager.make_response(model_id, predictions, *args, **kwargs)

    @property
    def num_errors(self):
        return self.model_manager.num_errors

    @num_errors.setter
    def num_errors(self, value):
        self.model_manager.num_errors = value

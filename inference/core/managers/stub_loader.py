from typing import Optional

from inference.core.managers.base import ModelManager
from inference.core.roboflow_api import ModelEndpointType


class StubLoaderManager(ModelManager):
    def add_model(
        self,
        model_id: str,
        api_key: str,
        model_id_alias=None,
        endpoint_type: ModelEndpointType = ModelEndpointType.ORT,
        countinference: Optional[bool] = None,
        service_secret: Optional[str] = None,
    ) -> None:
        """Adds a new model to the manager.

        Args:
            model_id (str): The identifier of the model.
            model (Model): The model instance.
            endpoint_type (ModelEndpointType, optional): The endpoint type to use for the model.
        """
        if model_id in self._models:
            return
        model_class = self.model_registry.get_model(
            model_id_alias if model_id_alias is not None else model_id, api_key
        )
        model = model_class(model_id=model_id, api_key=api_key, load_weights=False)
        self._models[model_id] = model

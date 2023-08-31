from inference.core.data_models import InferenceRequest, InferenceResponse
from inference.core.logger import logger
from inference.core.managers.base import Model
from inference.core.managers.decorators.base import ModelManagerDecorator


class WithLogger(ModelManagerDecorator):
    """Logger Decorator, it logs what's going on inside the manager."""

    def add_model(self, model_id: str, model: Model):
        """Adds a model to the manager and logs the action.

        Args:
            model_id (str): The identifier of the model.
            model (Model): The model instance.

        Returns:
            The result of the add_model method from the superclass.
        """
        logger.info(f"🤖 {model_id} added.")
        return super().add_model(model_id, model)

    def infer_from_request(
        self, model_id: str, request: InferenceRequest
    ) -> InferenceResponse:
        """Processes a complete inference request and logs both the request and response.

        Args:
            model_id (str): The identifier of the model.
            request (InferenceRequest): The request to process.

        Returns:
            InferenceResponse: The response from the inference.
        """
        logger.info(f"📥 [{model_id}] request={request}.")
        res = super().infer_from_request(model_id, request)
        logger.info(f"📥 [{model_id}] res={res}.")
        return res

    def remove(self, model_id: str) -> Model:
        """Removes a model from the manager and logs the action.

        Args:
            model_id (str): The identifier of the model to remove.

        Returns:
            Model: The removed model.
        """
        res = super().remove(model_id)
        logger.info(f"❌ removed {model_id}")
        return res

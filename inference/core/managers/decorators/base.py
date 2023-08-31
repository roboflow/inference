from inference.core.data_models import InferenceRequest, InferenceResponse
from inference.core.managers.base import Model, ModelManager


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

    def __init__(self, model_manager: ModelManager):
        """Initializes the decorator with an instance of a ModelManager."""
        self.model_manager = model_manager

    def add_model(self, model_id: str, model: Model):
        """Adds a model to the manager.

        Args:
            model_id (str): The identifier of the model.
            model (Model): The model instance.
        """
        self.model_manager.add_model(model_id, model)

    def infer_from_request(
        self, model_id: str, request: InferenceRequest
    ) -> InferenceResponse:
        """Processes a complete inference request.

        Args:
            model_id (str): The identifier of the model.
            request (InferenceRequest): The request to process.

        Returns:
            InferenceResponse: The response from the inference.
        """
        return self.model_manager.infer_from_request(model_id, request)

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

    def get_task_type(self, model_id: str) -> str:
        """Gets the task type associated with a model.

        Args:
            model_id (str): The identifier of the model.

        Returns:
            str: The task type.
        """
        return self.model_manager.get_task_type(model_id)

    def get_class_names(self, model_id):
        """Gets the class names for a given model.

        Args:
            model_id: The identifier of the model.

        Returns:
            List of class names.
        """
        return self.model_manager.get_class_names(model_id)

    def remove(self, model_id: str) -> Model:
        """Removes a model from the manager.

        Args:
            model_id (str): The identifier of the model.

        Returns:
            Model: The removed model.
        """
        return self.model_manager.remove(model_id)

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

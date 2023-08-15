from inference.core.models.base import Model
from inference.core.registries.base import ModelRegistry


class ModelRegistryDecorator(ModelRegistry):
    """Model Registry Decorator class that acts as a wrapper around an instance of ModelRegistry.

    This class provides a way to extend the functionalities of the ModelRegistry without modifying its code.
    It delegates the `get_model` method to the encapsulated ModelRegistry object.

    Attributes:
        registry (ModelRegistry): The underlying ModelRegistry instance being decorated.

    Inherits:
        ModelRegistry: Base class for Model Registry.
    """

    def __init__(self, registry: ModelRegistry) -> None:
        """Initializes the ModelRegistryDecorator instance.

        Args:
            registry (ModelRegistry): The underlying ModelRegistry instance to be decorated.
        """
        self.registry = registry

    def get_model(self, model_id: str, api_key: str) -> Model:
        """Delegates the get_model call to the encapsulated ModelRegistry instance.

        Args:
            model_id (str): The unique identifier for the model.
            api_key (str): The API key used for authentication.

        Returns:
            Model: The model instance associated with the given model_id.
        """
        return self.registry.get_model(model_id, api_key)

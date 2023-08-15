from inference.core.env import DEVICE_ID
from inference.core.registries.base import ModelRegistry
from inference.core.registries.decorators.base import ModelRegistryDecorator


class WithFixedDeviceId(ModelRegistryDecorator):
    """Decorator class that returns models from the internal registry dictionary but modified so that the device ID is fixed.

    This class is used to enforce a fixed device ID across all models retrieved through the registry.

    Attributes:
        device_id (str): The fixed device ID to be used for all models.

    Inherits:
        ModelRegistryDecorator: Base decorator class for Model Registry.
    """

    def __init__(self, registry: ModelRegistry, device_id: str) -> None:
        """Initializes the WithFixedDeviceId instance.

        Args:
            registry (ModelRegistry): The underlying ModelRegistry instance to be decorated.
            device_id (str): The fixed device ID to be used.
        """
        super().__init__(registry)
        self.device_id = device_id

    def get_model(self, model_id: str, api_key: str):
        """Retrieves the model from the underlying registry and modifies it to use the fixed device ID.

        Args:
            model_id (str): The unique identifier for the model.
            api_key (str): The API key used for authentication.

        Returns:
            Type[Model]: The modified model class with the fixed device ID.
        """
        model_class = self.registry.get_model(model_id, api_key)

        class model(model_class):
            def __init__(model_self, *args, **kwargs) -> None:
                super().__init__(*args, device_id=self.device_id, **kwargs)

        return model


class WithEnvVarDeviceId(WithFixedDeviceId):
    """Decorator class that extends the WithFixedDeviceId decorator so that the fixed device ID can be provided via an environment variable.

    This class allows the fixed device ID to be specified through the DEVICE_ID environment variable,
    providing flexibility in configuration.

    Inherits:
        WithFixedDeviceId: Base class that enforces a fixed device ID.
    """

    def __init__(self, registry: ModelRegistry) -> None:
        """Initializes the WithEnvVarDeviceId instance, setting the device ID from the DEVICE_ID environment variable.

        Args:
            registry (ModelRegistry): The underlying ModelRegistry instance to be decorated.
        """
        super().__init__(registry, DEVICE_ID)

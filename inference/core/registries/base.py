from typing import List, Optional

from inference.core.exceptions import ModelNotRecognisedError
from inference.core.models.base import Model
from inference.core.roboflow_api import ModelEndpointType


class ModelRegistry:
    """An object which is able to return model classes based on given model IDs and model types.

    Attributes:
        registry_dict (dict): A dictionary mapping model types to model classes.
    """

    def __init__(self, registry_dict) -> None:
        """Initializes the ModelRegistry with the given dictionary of registered models.

        Args:
            registry_dict (dict): A dictionary mapping model types to model classes.
        """
        self.registry_dict = registry_dict

    def get_model(
        self,
        model_type: str,
        model_id: str,
        **kwargs,
    ) -> Model:
        """Returns the model class based on the given model type.

        Args:
            model_type (str): The type of the model to be retrieved.
            model_id (str): The ID of the model to be retrieved (unused in the current implementation).

        Returns:
            Model: The model class corresponding to the given model type.

        Raises:
            ModelNotRecognisedError: If the model_type is not found in the registry_dict.
        """
        if model_type not in self.registry_dict:
            raise ModelNotRecognisedError(
                f"Could not find model of type: {model_type} in configured registry."
            )
        return self.registry_dict[model_type]

    def get_model_auth_targets(
        self,
        model_id: str,
        endpoint_type: ModelEndpointType = ModelEndpointType.ORT,
        api_key: Optional[str] = None,
        countinference: Optional[bool] = None,
        service_secret: Optional[str] = None,
    ) -> List[str]:
        """Return concrete model IDs that must be authorized for a load."""
        return [model_id]

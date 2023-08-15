import os

from inference.core.devices.utils import (
    get_gpu_serial_num,
    get_gpu_serial_num_uid,
    get_tegra_serial,
)
from inference.core.registries.decorators.base import ModelRegistryDecorator


class WithTrtDeviceId(ModelRegistryDecorator):
    """Decorator class that returns models using TensorRT (TRT) device ID logic.

    This class modifies models retrieved from the registry to use a device ID based on TensorRT logic.
    The device ID is determined from the Tegra serial number or GPU serial number or from a dynamic environment variable.

    Inherits:
        ModelRegistryDecorator: Base decorator class for Model Registry.
    """

    def get_model(self, model_id: str, api_key: str):
        """Retrieves the model from the underlying registry and modifies it to use the TRT device ID logic.

        The device ID is obtained from the Tegra serial number or GPU serial number. If not available, it uses the GPU serial number UID or the value of the "DYNAMIC_ENVIRONMENT" environment variable.

        Args:
            model_id (str): The unique identifier for the model.
            api_key (str): The API key used for authentication.

        Returns:
            Type[Model]: The modified model class with the TRT device ID logic applied.
        """
        model_class = self.registry.get_model(model_id, api_key)
        if not os.environ.get("DYNAMIC_ENVIRONMENT", None):
            try:
                device_id = get_tegra_serial()
            except Exception as e:
                device_id = get_gpu_serial_num()
                if device_id == "N/A":
                    device_id = get_gpu_serial_num_uid()
        else:
            device_id = os.environ["DYNAMIC_ENVIRONMENT"]

        class model(model_class):
            def __init__(model_self, *args, **kwargs) -> None:
                """Initializes the model instance with the TRT device ID.

                Args:
                    *args: Variable length argument list.
                    **kwargs: Arbitrary keyword arguments.

                Keyword Args:
                    device_id (str): The TRT device ID to be used.
                """
                super().__init__(*args, device_id=device_id, **kwargs)

        return model

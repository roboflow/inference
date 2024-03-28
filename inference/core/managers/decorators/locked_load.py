from inference.core.cache import cache
from inference.core.managers.decorators.base import ModelManagerDecorator
from inference.core.models.utils.quantization import QuantizationMode

lock_str = lambda z: f"locks:model-load:{z}"


class LockedLoadModelManagerDecorator(ModelManagerDecorator):
    """Must acquire lock to load model"""

    def add_model(
        self,
        model_id: str,
        api_key: str,
        model_id_alias=None,
        quantization: QuantizationMode = QuantizationMode.unquantized,
    ):
        with cache.lock(lock_str(model_id), expire=180.0):
            return super().add_model(
                model_id,
                api_key,
                model_id_alias=model_id_alias,
                quantization=quantization,
            )

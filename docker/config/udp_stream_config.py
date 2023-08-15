from inference.core.interfaces.udp.udp_stream import UdpStream
from inference.core.managers.base import ModelManager
from inference.core.registries.decorators.fixed_device_id import WithEnvVarDeviceId
from inference.core.registries.roboflow import (
    RoboflowModelRegistry,
)
from inference.models.utils import ROBOFLOW_MODEL_TYPES

model_registry = WithEnvVarDeviceId(RoboflowModelRegistry(ROBOFLOW_MODEL_TYPES))
model_manager = ModelManager()
# model_manager.model_manager.init_pingback()
interface = UdpStream(model_manager, model_registry=model_registry)
app = interface.run_thread()

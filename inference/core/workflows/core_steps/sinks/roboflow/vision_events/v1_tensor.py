from inference.core.workflows.core_steps.common.wrap_consumer import (
    make_tensor_wrapper_block,
)
from inference.core.workflows.core_steps.sinks.roboflow.vision_events.v1 import (
    RoboflowVisionEventsBlockV1 as _NumpyImpl,
)

RoboflowVisionEventsBlockV1 = make_tensor_wrapper_block(_NumpyImpl)

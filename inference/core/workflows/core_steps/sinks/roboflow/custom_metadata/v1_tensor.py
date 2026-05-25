from inference.core.workflows.core_steps.common.wrap_consumer import (
    make_tensor_wrapper_block,
)
from inference.core.workflows.core_steps.sinks.roboflow.custom_metadata.v1 import (
    RoboflowCustomMetadataBlockV1 as _NumpyImpl,
)

RoboflowCustomMetadataBlockV1 = make_tensor_wrapper_block(_NumpyImpl)

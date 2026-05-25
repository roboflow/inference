from inference.core.workflows.core_steps.common.wrap_consumer import (
    make_tensor_wrapper_block,
)
from inference.core.workflows.core_steps.sinks.onvif_movement.v1 import (
    ONVIFSinkBlockV1 as _NumpyImpl,
)

ONVIFSinkBlockV1 = make_tensor_wrapper_block(_NumpyImpl)

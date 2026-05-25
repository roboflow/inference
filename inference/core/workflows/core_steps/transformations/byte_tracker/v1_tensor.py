from inference.core.workflows.core_steps.common.wrap_consumer import (
    make_tensor_wrapper_block,
)
from inference.core.workflows.core_steps.transformations.byte_tracker.v1 import (
    ByteTrackerBlockV1 as _NumpyImpl,
)

ByteTrackerBlockV1 = make_tensor_wrapper_block(_NumpyImpl)

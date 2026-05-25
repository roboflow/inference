from inference.core.workflows.core_steps.common.wrap_consumer import (
    make_tensor_wrapper_block,
)
from inference.core.workflows.core_steps.transformations.byte_tracker.v3 import (
    ByteTrackerBlockV3 as _NumpyImpl,
)

ByteTrackerBlockV3 = make_tensor_wrapper_block(_NumpyImpl)

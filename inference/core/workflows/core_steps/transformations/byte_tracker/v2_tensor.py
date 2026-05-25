from inference.core.workflows.core_steps.common.wrap_consumer import (
    make_tensor_wrapper_block,
)
from inference.core.workflows.core_steps.transformations.byte_tracker.v2 import (
    ByteTrackerBlockV2 as _NumpyImpl,
)

ByteTrackerBlockV2 = make_tensor_wrapper_block(_NumpyImpl)

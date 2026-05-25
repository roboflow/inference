from inference.core.workflows.core_steps.common.wrap_consumer import (
    make_tensor_wrapper_block,
)
from inference.core.workflows.core_steps.transformations.detection_offset.v1 import (
    DetectionOffsetBlockV1 as _NumpyImpl,
)

DetectionOffsetBlockV1 = make_tensor_wrapper_block(_NumpyImpl)

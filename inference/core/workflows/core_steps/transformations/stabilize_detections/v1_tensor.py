from inference.core.workflows.core_steps.common.wrap_consumer import (
    make_tensor_wrapper_block,
)
from inference.core.workflows.core_steps.transformations.stabilize_detections.v1 import (
    StabilizeTrackedDetectionsBlockV1 as _NumpyImpl,
)

StabilizeTrackedDetectionsBlockV1 = make_tensor_wrapper_block(_NumpyImpl)

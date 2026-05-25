from inference.core.workflows.core_steps.common.wrap_consumer import (
    make_tensor_wrapper_block,
)
from inference.core.workflows.core_steps.transformations.detections_filter.v1 import (
    DetectionsFilterBlockV1 as _NumpyImpl,
)

DetectionsFilterBlockV1 = make_tensor_wrapper_block(_NumpyImpl)

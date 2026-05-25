from inference.core.workflows.core_steps.common.wrap_consumer import (
    make_tensor_wrapper_block,
)
from inference.core.workflows.core_steps.models.foundation.depth_estimation.v1 import (
    DepthEstimationBlockV1 as _NumpyImpl,
)

DepthEstimationBlockV1 = make_tensor_wrapper_block(_NumpyImpl)

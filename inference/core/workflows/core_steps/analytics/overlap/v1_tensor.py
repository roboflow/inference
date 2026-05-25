from inference.core.workflows.core_steps.common.wrap_consumer import (
    make_tensor_wrapper_block,
)
from inference.core.workflows.core_steps.analytics.overlap.v1 import (
    OverlapBlockV1 as _NumpyImpl,
)

OverlapBlockV1 = make_tensor_wrapper_block(_NumpyImpl)

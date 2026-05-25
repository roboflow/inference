from inference.core.workflows.core_steps.common.wrap_consumer import (
    make_tensor_wrapper_block,
)
from inference.core.workflows.core_steps.transformations.perspective_correction.v1 import (
    PerspectiveCorrectionBlockV1 as _NumpyImpl,
)

PerspectiveCorrectionBlockV1 = make_tensor_wrapper_block(_NumpyImpl)

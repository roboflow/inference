from inference.core.workflows.core_steps.common.wrap_consumer import (
    make_tensor_wrapper_block,
)
from inference.core.workflows.core_steps.analytics.line_counter.v1 import (
    LineCounterBlockV1 as _NumpyImpl,
)

LineCounterBlockV1 = make_tensor_wrapper_block(_NumpyImpl)

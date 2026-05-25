from inference.core.workflows.core_steps.common.wrap_consumer import (
    make_tensor_wrapper_block,
)
from inference.core.workflows.core_steps.analytics.line_counter.v2 import (
    LineCounterBlockV2 as _NumpyImpl,
)

LineCounterBlockV2 = make_tensor_wrapper_block(_NumpyImpl)

from inference.core.workflows.core_steps.common.wrap_consumer import (
    make_tensor_wrapper_block,
)
from inference.core.workflows.core_steps.models.foundation.florence2.v1 import (
    Florence2BlockV1 as _NumpyImpl,
)

Florence2BlockV1 = make_tensor_wrapper_block(_NumpyImpl)

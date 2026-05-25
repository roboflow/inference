from inference.core.workflows.core_steps.common.wrap_consumer import (
    make_tensor_wrapper_block,
)
from inference.core.workflows.core_steps.models.foundation.florence2.v2 import (
    Florence2BlockV2 as _NumpyImpl,
)

Florence2BlockV2 = make_tensor_wrapper_block(_NumpyImpl)

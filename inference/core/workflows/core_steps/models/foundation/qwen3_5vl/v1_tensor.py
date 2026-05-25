from inference.core.workflows.core_steps.common.wrap_consumer import (
    make_tensor_wrapper_block,
)
from inference.core.workflows.core_steps.models.foundation.qwen3_5vl.v1 import (
    Qwen35VLBlockV1 as _NumpyImpl,
)

Qwen35VLBlockV1 = make_tensor_wrapper_block(_NumpyImpl)

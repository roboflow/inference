from inference.core.workflows.core_steps.common.wrap_consumer import (
    make_tensor_wrapper_block,
)
from inference.core.workflows.core_steps.models.foundation.qwen3vl.v1 import (
    Qwen3VLBlockV1 as _NumpyImpl,
)

Qwen3VLBlockV1 = make_tensor_wrapper_block(_NumpyImpl)

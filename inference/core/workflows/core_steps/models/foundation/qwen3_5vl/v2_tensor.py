from inference.core.workflows.core_steps.common.wrap_consumer import (
    make_tensor_wrapper_block,
)
from inference.core.workflows.core_steps.models.foundation.qwen3_5vl.v2 import (
    Qwen35VLBlockV2 as _NumpyImpl,
)

Qwen35VLBlockV2 = make_tensor_wrapper_block(_NumpyImpl)

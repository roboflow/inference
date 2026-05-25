from inference.core.workflows.core_steps.common.wrap_consumer import (
    make_tensor_wrapper_block,
)
from inference.core.workflows.core_steps.models.foundation.qwen.v1 import (
    Qwen25VLBlockV1 as _NumpyImpl,
)

Qwen25VLBlockV1 = make_tensor_wrapper_block(_NumpyImpl)

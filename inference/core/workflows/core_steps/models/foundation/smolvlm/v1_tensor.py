from inference.core.workflows.core_steps.common.wrap_consumer import (
    make_tensor_wrapper_block,
)
from inference.core.workflows.core_steps.models.foundation.smolvlm.v1 import (
    SmolVLM2BlockV1 as _NumpyImpl,
)

SmolVLM2BlockV1 = make_tensor_wrapper_block(_NumpyImpl)

from inference.core.workflows.core_steps.common.wrap_consumer import (
    make_tensor_wrapper_block,
)
from inference.core.workflows.core_steps.models.foundation.clip.v1 import (
    ClipModelBlockV1 as _NumpyImpl,
)

ClipModelBlockV1 = make_tensor_wrapper_block(_NumpyImpl)

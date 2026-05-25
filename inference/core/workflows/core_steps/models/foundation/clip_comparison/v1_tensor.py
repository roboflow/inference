from inference.core.workflows.core_steps.common.wrap_consumer import (
    make_tensor_wrapper_block,
)
from inference.core.workflows.core_steps.models.foundation.clip_comparison.v1 import (
    ClipComparisonBlockV1 as _NumpyImpl,
)

ClipComparisonBlockV1 = make_tensor_wrapper_block(_NumpyImpl)

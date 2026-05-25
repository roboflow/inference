from inference.core.workflows.core_steps.common.wrap_consumer import (
    make_tensor_wrapper_block,
)
from inference.core.workflows.core_steps.models.foundation.clip_comparison.v2 import (
    ClipComparisonBlockV2 as _NumpyImpl,
)

ClipComparisonBlockV2 = make_tensor_wrapper_block(_NumpyImpl)

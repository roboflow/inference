from inference.core.workflows.core_steps.common.wrap_consumer import (
    make_tensor_wrapper_block,
)
from inference.core.workflows.core_steps.models.foundation.segment_anything3.v2 import (
    SegmentAnything3BlockV2 as _NumpyImpl,
)

SegmentAnything3BlockV2 = make_tensor_wrapper_block(_NumpyImpl)

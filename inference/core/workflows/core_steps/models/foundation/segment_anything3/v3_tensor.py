from inference.core.workflows.core_steps.common.wrap_consumer import (
    make_tensor_wrapper_block,
)
from inference.core.workflows.core_steps.models.foundation.segment_anything3.v3 import (
    SegmentAnything3BlockV3 as _NumpyImpl,
)

SegmentAnything3BlockV3 = make_tensor_wrapper_block(_NumpyImpl)

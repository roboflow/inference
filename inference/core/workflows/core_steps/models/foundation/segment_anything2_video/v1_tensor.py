from inference.core.workflows.core_steps.common.wrap_consumer import (
    make_tensor_wrapper_block,
)
from inference.core.workflows.core_steps.models.foundation.segment_anything2_video.v1 import (
    SegmentAnything2VideoBlockV1 as _NumpyImpl,
)

SegmentAnything2VideoBlockV1 = make_tensor_wrapper_block(_NumpyImpl)
